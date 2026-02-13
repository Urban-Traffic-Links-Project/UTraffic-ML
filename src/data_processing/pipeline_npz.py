# src/data_processing/pipeline_npz.py
import pandas as pd
import numpy as np
import json
from typing import Dict, Any, Optional, List
from pathlib import Path

from .collectors.tomtom_collector import TomTomTrafficDataCollector
from .streaming.kafka_producer import TrafficDataProducer
from .streaming.kafka_consumer import RawDataConsumer, ValidatedDataConsumer
from .preprocessors.data_validator import DataValidator
from .preprocessors.data_cleaner import DataCleaner
from .preprocessors.feature_extractor import FeatureExtractor
from .preprocessors.categorical_encoder import CategoricalFeatureEncoder
from .preprocessors.spatial_processor import SpatialFeatureProcessor
from .preprocessors.data_normalizer import DataNormalizer
from .storage.npz_storage import NPZWriter, NPZReader

from utils.config import config
from utils.logger import setup_logger

class TrafficDataPipelineNPZ:
    """
    Pipeline xử lý dữ liệu giao thông theo Kappa Architecture
    Lưu trữ vào NPZ files thay vì database
    """
    
    def __init__(self):
        self.logger = setup_logger('TrafficDataPipelineNPZ', config.log_file, config.log_level)
        
        # Initialize components
        self.collector = TomTomTrafficDataCollector()
        self.validator = DataValidator()
        self.cleaner = DataCleaner()
        self.feature_extractor = FeatureExtractor()
        self.categorical_encoder = CategoricalFeatureEncoder(encoding_strategy='ordinal')
        self.spatial_processor = SpatialFeatureProcessor(normalize=True, create_features=True)
        self.normalizer = DataNormalizer()
        
        # Storage
        self.npz_writer = NPZWriter()
        self.npz_reader = NPZReader()
        
        # Batch accumulator
        self.accumulated_data = []
        self.batch_size = 10000
        
        self.logger.info("✅ Pipeline initialized with NPZ storage")
    
    def run_31_days_collection(
        self,
        geometry: Dict,
        start_date: str,
    ) -> Optional[List[str]]:
        """
        Stage 1 (31 ngày): Thu thập dữ liệu 31 ngày liên tiếp từ TomTom.
        Mỗi ngày 1 job riêng, không gộp trung bình — phù hợp time series cho T-GCN.

        Args:
            geometry: Polygon geometry cho vùng phân tích.
            start_date: Ngày bắt đầu (YYYY-MM-DD). Sẽ thu 31 ngày từ ngày này.

        Returns:
            Danh sách job_id tương ứng 31 ngày, hoặc None nếu thất bại.
        """
        self.logger.info("=" * 60)
        self.logger.info("STAGE 1: 31-DAY SEQUENTIAL DATA COLLECTION")
        self.logger.info("=" * 60)

        job_ids, all_results = self.collector.collect_31_days_sequential(
            geometry=geometry,
            start_date_str=start_date,
        )

        if not job_ids:
            self.logger.error("No jobs created")
            return None

        self.logger.info(f"✅ Stage 1 complete: {len(job_ids)} jobs (31 days)")
        return job_ids

    def run_batch_collection(
        self,
        geometry: Dict,
        date_from: str,
        date_to: str,
        job_name: str = "Traffic Analysis"
    ) -> Optional[str]:
        """
        Stage 1 (legacy): Thu thập dữ liệu batch một khoảng ngày từ TomTom.
        Dùng run_31_days_collection nếu cần chuỗi 31 ngày riêng từng ngày.
        """
        self.logger.info("=" * 60)
        self.logger.info("STAGE 1: BATCH DATA COLLECTION")
        self.logger.info("=" * 60)

        job_id = self.collector.create_area_analysis_job(
            geometry=geometry,
            date_from=date_from,
            date_to=date_to,
            job_name=job_name
        )

        if not job_id:
            self.logger.error("Failed to create job")
            return None

        status = self.collector.wait_for_job_completion(job_id, max_wait_minutes=60)

        if not status or status.get('jobState') != 'DONE':
            self.logger.error("Job did not complete successfully")
            return None

        results = self.collector.download_results(job_id)

        if not results:
            self.logger.error("Failed to download results")
            return None

        self.logger.info(f"✅ Stage 1 complete: Job {job_id}")
        return job_id
    
    def run_streaming_ingestion(self, job_id: str):
        """
        Stage 2: Đưa dữ liệu từ 1 file job vào Kafka stream.
        """
        self.logger.info("=" * 60)
        self.logger.info("STAGE 2: STREAMING INGESTION")
        self.logger.info("=" * 60)
        n = self._ingest_one_job_to_stream(job_id)
        self.logger.info(f"✅ Stage 2 complete: {n} segments")

    def run_streaming_ingestion_from_jobs(self, job_ids: List[str]):
        """
        Stage 2 (31 ngày): Đưa dữ liệu từ nhiều file job (31 ngày) vào Kafka stream.
        Mỗi job tương ứng 1 ngày; thứ tự job_ids theo thứ tự ngày.
        """
        self.logger.info("=" * 60)
        self.logger.info("STAGE 2: STREAMING INGESTION (31 DAYS)")
        self.logger.info("=" * 60)

        total_sent = 0
        for idx, job_id in enumerate(job_ids):
            n = self._ingest_one_job_to_stream(job_id, day_index=idx + 1, total_days=len(job_ids))
            total_sent += n

        self.logger.info(f"✅ Stage 2 complete: {total_sent} segments from {len(job_ids)} days")

    def _ingest_one_job_to_stream(
        self,
        job_id: str,
        day_index: Optional[int] = None,
        total_days: Optional[int] = None,
    ) -> int:
        """Đọc 1 file job_*_results.json và gửi segments vào Kafka. Trả về số segment đã gửi."""
        result_file = config.data.raw_dir / "tomtom_stats" / f"job_{job_id}_results.json"

        if not result_file.exists():
            self.logger.error(f"Result file not found: {result_file}")
            return 0

        with open(result_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        sent_count = 0
        with TrafficDataProducer() as producer:
            network = data.get('network', {})
            segment_results = network.get('segmentResults', [])

            for segment in segment_results:
                message = {
                    'job_id': job_id,
                    'segment': segment,
                    'job_name': data.get('jobName'),
                    'date_ranges': data.get('dateRanges'),
                    'time_sets': data.get('timeSets'),
                }

                key = str(segment.get('segmentId'))
                producer.send_raw_data(message, key=key)
                sent_count += 1

            producer.flush()

        label = f" (day {day_index}/{total_days})" if day_index and total_days else ""
        self.logger.info(f"✅ Sent {sent_count} segments for job {job_id}{label}")
        return sent_count
    
    def run_validation_processing(self):
        """
        Stage 3: Validate và gửi vào traffic.validated
        """
        self.logger.info("=" * 60)
        self.logger.info("STAGE 3: VALIDATION PROCESSING")
        self.logger.info("=" * 60)
        
        validated_count = 0
        
        def validate_processor(key, value, topic, partition, offset):
            nonlocal validated_count
            
            try:
                segment = value.get('segment', {})
                
                is_valid, errors = self.validator.validate_segment(segment)
                
                if not is_valid:
                    self.logger.warning(f"Invalid segment {key}: {errors}")
                    return True
                
                time_results = segment.get('segmentTimeResults', [])
                valid_time_results = []
                
                for time_result in time_results:
                    is_valid, errors = self.validator.validate_time_result(time_result)
                    if is_valid:
                        valid_time_results.append(time_result)
                
                if not valid_time_results:
                    return True
                
                value['segment']['segmentTimeResults'] = valid_time_results
                
                with TrafficDataProducer() as producer:
                    producer.send_validated_data(value, key=key)
                
                validated_count += 1
                return True
                
            except Exception as e:
                self.logger.error(f"Error in validate_processor: {e}")
                return False
        
        with RawDataConsumer() as consumer:
            consumer.consume(validate_processor, max_messages=None)
        
        self.logger.info(f"✅ Stage 3 complete: {validated_count} validated")
    
    def run_feature_extraction(self):
        """
        Stage 4: Extract features và lưu NPZ
        """
        self.logger.info("=" * 60)
        self.logger.info("STAGE 4: FEATURE EXTRACTION & NPZ STORAGE")
        self.logger.info("=" * 60)
        
        self.accumulated_data = []
        processed_count = 0
        
        def feature_processor(key, value, topic, partition, offset):
            nonlocal processed_count
            
            try:
                segment = value.get('segment', {})
                time_results = segment.get('segmentTimeResults', [])

                # Trong feature_processor callback
                time_sets_map = {}
                for ts in value.get('time_sets', []):
                    time_sets_map[ts.get('@id')] = ts.get('name')

                date_ranges_map = {}
                for dr in value.get('date_ranges', []):
                    date_ranges_map[dr.get('@id')] = dr.get('from')

                
                for time_result in time_results:
                    record = {
                        'segment_id': segment.get('segmentId'),
                        'new_segment_id': segment.get('newSegmentId'),
                        'street_name': segment.get('streetName'),
                        'distance': segment.get('distance'),
                        'frc': segment.get('frc'),
                        'speed_limit': segment.get('speedLimit'),
                        
                        'time_set': time_sets_map.get(time_result.get('timeSet')),  # "Slot_0700"
                        'date_from': date_ranges_map.get(time_result.get('dateRange')),  # "2024-08-01"
                        
                        'harmonic_average_speed': time_result.get('harmonicAverageSpeed'),
                        'median_speed': time_result.get('medianSpeed'),
                        'average_speed': time_result.get('averageSpeed'),
                        'std_speed': time_result.get('standardDeviationSpeed'),
                        
                        'average_travel_time': time_result.get('averageTravelTime'),
                        'median_travel_time': time_result.get('medianTravelTime'),
                        'travel_time_std': time_result.get('travelTimeStandardDeviation'),
                        'travel_time_ratio': time_result.get('travelTimeRatio'),
                        
                        'sample_size': time_result.get('sampleSize')
                    }
                    
                    shape = segment.get('shape', [])
                    if shape:
                        record['latitude'] = shape[0].get('latitude')
                        record['longitude'] = shape[0].get('longitude')
                    
                    self.accumulated_data.append(record)
                
                if len(self.accumulated_data) >= self.batch_size:
                    self._process_and_save_batch()
                    processed_count += len(self.accumulated_data)
                    self.accumulated_data.clear()
                
                return True
                
            except Exception as e:
                self.logger.error(f"Error in feature_processor: {e}")
                return False
        
        with ValidatedDataConsumer() as consumer:
            consumer.consume(feature_processor, max_messages=None)
        
        if self.accumulated_data:
            self._process_and_save_batch()
            processed_count += len(self.accumulated_data)
        
        self.logger.info(f"✅ Stage 4 complete: {processed_count} records processed")
    
    def _process_and_save_batch(self):
        """Xử lý và lưu batch vào NPZ - FIXED to preserve metadata columns"""
        if not self.accumulated_data:
            return
        
        df = pd.DataFrame(self.accumulated_data)
        
        # 1. Clean (GIỮ nguyên các cột metadata để không bị xóa trục thời gian)
        #    - _remove_duplicates sẽ dùng segment_id + time_set + date_from
        df = self.cleaner.clean(df)
        
        # ===== CRITICAL: Preserve metadata columns TRƯỚC KHI transform features =====
        metadata_cols = ['time_set', 'date_from']
        preserved_data = {}
        for col in metadata_cols:
            if col in df.columns:
                preserved_data[col] = df[col].copy()
        
        # 2. Extract features
        df = self.feature_extractor.extract_all_features(df)
        
        # 3. Encode categorical
        df = self.categorical_encoder.fit_transform(df)
        
        # 4. Process spatial
        df = self.spatial_processor.fit_transform(df)
        
        # 5. Normalize
        df = self.normalizer.fit_transform(df, method='standard')
        
        # ===== CRITICAL: Restore metadata columns AFTER processing =====
        for col, data in preserved_data.items():
            df[col] = data
        
        # 6. Save to NPZ
        self.npz_writer.write_features(df)
        
        self.logger.info(f"Processed and saved batch of {len(df)} records (with preserved metadata)")

    def export_for_model_training(
        self,
        output_name: str = None,
        sequence_length: int = 12,
        prediction_horizon: int = 12
    ):
        """
        Stage 5: Export data với ĐÚNG SHAPE 4D cho T-GCN
        FIX: Properly concatenate 31 days × 24 slots = 744 timesteps
        """
        from datetime import datetime
        import re
        
        self.logger.info("=" * 60)
        self.logger.info("STAGE 5: EXPORT FOR MODEL TRAINING (31-DAY SEQUENTIAL FIX)")
        self.logger.info("=" * 60)
        
        df = self.npz_reader.read_features()
        
        if df is None or df.empty:
            self.logger.error("No features found to export")
            return
        
        self.logger.info(f"Loaded {len(df)} records")
        
        # ===== CRITICAL FIX: Check for correct column names =====
        if 'time_set' not in df.columns:
            self.logger.error("❌ Missing 'time_set' column!")
            self.logger.info(f"Available columns: {df.columns.tolist()}")
            return

        # Detect date column (flexible - accept either name)
        date_col = None
        if 'date_from' in df.columns:
            date_col = 'date_from'
        elif 'date_range' in df.columns:
            date_col = 'date_range'
        else:
            self.logger.error("❌ Missing date column! Need 'date_from' or 'date_range'")
            self.logger.info(f"Available columns: {df.columns.tolist()}")
            return

        self.logger.info(f"Using date column: {date_col}")

        # Extract date from "Single Day 2024-08-01" format
        def extract_date(date_str):
            date_str = str(date_str)
            # If already in YYYY-MM-DD format
            if re.match(r'^\d{4}-\d{2}-\d{2}$', date_str):
                return date_str
            # Extract from "Single Day 2024-08-01" format
            match = re.search(r'\d{4}-\d{2}-\d{2}', date_str)
            return match.group(0) if match else None

        # Extract slot index from "Slot_0700" format
        def extract_slot_index(time_str):
            match = re.search(r'Slot_(\d{4})', str(time_str))
            if match:
                time_code = match.group(1)
                hour = int(time_code[:2])
                minute = int(time_code[2:])
                
                # Morning: 07:00-09:45 = indices 0-11
                # Evening: 15:00-17:45 = indices 12-23
                if 7 <= hour < 10:
                    return (hour - 7) * 4 + minute // 15
                elif 15 <= hour < 18:
                    return 12 + (hour - 15) * 4 + minute // 15
            return None

        df['date'] = df[date_col].apply(extract_date)
        df['slot_index'] = df['time_set'].apply(extract_slot_index)

        # Debug info
        self.logger.info(f"Sample dates: {df['date'].head(3).tolist()}")
        self.logger.info(f"Sample time_sets: {df['time_set'].head(3).tolist()}")
        self.logger.info(f"Sample slot_indices: {df['slot_index'].head(3).tolist()}")

        # Remove invalid rows
        df = df.dropna(subset=['date', 'slot_index'])
        df['slot_index'] = df['slot_index'].astype(int)

        self.logger.info(f"After cleaning: {len(df)} records remain")

        # Create day index (0, 1, 2, ..., 30)
        unique_dates = sorted(df['date'].unique())
        num_days = len(unique_dates)

        if num_days == 0:
            self.logger.error("❌ No valid dates found!")
            return

        self.logger.info(f"Found {num_days} days: {unique_dates[0]} to {unique_dates[-1]}")

        date_to_idx = {date: idx for idx, date in enumerate(unique_dates)}
        df['day_index'] = df['date'].map(date_to_idx)
        
        # Create global timestamp: day * 24 + slot
        df['global_timestamp'] = df['day_index'] * 24 + df['slot_index']
        
        # Sort by segment and global timestamp
        df = df.sort_values(['segment_id', 'global_timestamp']).reset_index(drop=True)
        
        self.logger.info(f"Global timestamps range: {df['global_timestamp'].min()} - {df['global_timestamp'].max()}")
        
        # Get segments (tổng trước khi lọc)
        all_segment_ids = sorted(df['segment_id'].unique())
        self.logger.info(f"Found {len(all_segment_ids)} segments before filtering for completeness")
        
        # ===== Select Features =====
        exclude_patterns = [
            'segment_id', 'new_segment_id', 'street_name',
            'date_range', 'time_set', 'date', 'slot_index', 'day_index', 'global_timestamp',
            'latitude', 'longitude',
            'grid_lat', 'grid_lon', 'grid_cell',
            'congestion_level', 'distance_category',
            'frc_encoded', 'time_set_encoded', 'frc_level'
        ]
        
        feature_cols = [col for col in df.columns 
                       if col not in exclude_patterns
                       and pd.api.types.is_numeric_dtype(df[col])
                       and not df[col].isnull().all()
                       and df[col].nunique() > 1]
        
        num_features = len(feature_cols)
        self.logger.info(f"Selected {num_features} features: {feature_cols[:10]}")
        
        if num_features == 0:
            self.logger.error("❌ No features!")
            return
        
        # ===== Check timesteps =====
        timesteps_per_seg = df.groupby('segment_id')['global_timestamp'].count().to_dict()
        min_ts = min(timesteps_per_seg.values())
        max_ts = max(timesteps_per_seg.values())
        expected_ts = num_days * 24
        
        self.logger.info(f"Timesteps: min={min_ts}, max={max_ts}, expected={expected_ts}")
        
        # Segments có đầy đủ 31 ngày × 24 slots
        full_data_segs = [s for s, c in timesteps_per_seg.items() if c == expected_ts]
        self.logger.info(
            f"Segments with full {expected_ts} timesteps: {len(full_data_segs)}/{len(timesteps_per_seg)}"
        )
        
        # Số timestep tối thiểu cần cho 1 mẫu (seq + pred)
        min_needed = sequence_length + prediction_horizon
        
        if full_data_segs:
            # Ưu tiên dùng các segment có đủ 744 timestep
            selected_segs = full_data_segs
            num_timesteps = expected_ts
            self.logger.info(
                f"Using {len(selected_segs)} segments with FULL {num_timesteps} timesteps "
                f"({num_days} days × 24 slots)"
            )
        else:
            # Nếu không có segment nào đủ 744, dùng các segment có ít nhất min_needed timesteps
            candidate_segs = [s for s, c in timesteps_per_seg.items() if c >= min_needed]
            if not candidate_segs:
                self.logger.error(
                    f"❌ Not enough timesteps in ANY segment: "
                    f"max available per segment = {max_ts}, need at least {min_needed}"
                )
                return
            
            selected_segs = candidate_segs
            # Lấy số timestep nhỏ nhất trong các segment được chọn để làm trục thời gian chung
            num_timesteps = min(timesteps_per_seg[s] for s in selected_segs)
            self.logger.info(
                f"Using {len(selected_segs)} segments with at least {num_timesteps} timesteps "
                f"(min_needed={min_needed})"
            )
        
        # Cập nhật lại danh sách nodes sau khi lọc theo độ đầy đủ
        segment_ids = sorted(selected_segs)
        num_nodes = len(segment_ids)
        self.logger.info(f"Final selected segments: {num_nodes}")
        
        # ===== Create 3D tensor: (timesteps, nodes, features) =====
        self.logger.info(f"Creating 3D: ({num_timesteps}, {num_nodes}, {num_features})")
        
        data_3d = np.zeros((num_timesteps, num_nodes, num_features), dtype=np.float32)
        
        for node_idx, seg_id in enumerate(segment_ids):
            seg_df = df[df['segment_id'] == seg_id].sort_values('global_timestamp')
            seg_data = seg_df[feature_cols].values.astype(np.float32)
            
            # Truncate or pad
            if len(seg_data) > num_timesteps:
                seg_data = seg_data[:num_timesteps]
            elif len(seg_data) < num_timesteps:
                if len(seg_data) > 0:
                    pad = np.repeat([seg_data[-1]], num_timesteps - len(seg_data), axis=0)
                    seg_data = np.vstack([seg_data, pad])
                else:
                    seg_data = np.zeros((num_timesteps, num_features), dtype=np.float32)
            
            # Clean
            seg_data = np.nan_to_num(seg_data, nan=0.0, posinf=0.0, neginf=0.0)
            data_3d[:, node_idx, :] = seg_data
        
        self.logger.info(f"✓ 3D tensor: {data_3d.shape}")
        if num_timesteps == expected_ts:
            self.logger.info(f"✅ PERFECT! {expected_ts} timesteps = {num_days} days × 24 slots")
        
        # ===== Create sliding windows =====
        sequences, targets = [], []
        max_start = num_timesteps - sequence_length - prediction_horizon + 1
        
        self.logger.info(f"Creating {max_start} sliding window samples")
        
        for i in range(max_start):
            sequences.append(data_3d[i:i+sequence_length, :, :])
            targets.append(data_3d[i+sequence_length:i+sequence_length+prediction_horizon, :, :])
        
        X = np.array(sequences, dtype=np.float32)
        y = np.array(targets, dtype=np.float32)
        
        self.logger.info(f"✓ X: {X.shape}, y: {y.shape}")
        assert len(X.shape) == 4 and len(y.shape) == 4
        
        # ===== Train/val/test split =====
        n = len(X)
        train_size = max(1, int(0.7 * n))
        val_size = max(1, int(0.15 * n))
        
        X_train, y_train = X[:train_size], y[:train_size]
        X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
        X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]
        
        # ===== Save =====
        if output_name is None:
            output_name = f"model_ready_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        data = {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test,
            'segment_ids': np.array(segment_ids),
            'feature_names': np.array(feature_cols)
        }
        
        metadata = {
            'sequence_length': sequence_length,
            'prediction_horizon': prediction_horizon,
            'num_features': num_features,
            'num_nodes': num_nodes,
            'num_timesteps': num_timesteps,
            'num_days': num_days,
            'expected_timesteps': expected_ts,
            'train_size': len(X_train),
            'val_size': len(X_val),
            'test_size': len(X_test),
            'date_range': f"{unique_dates[0]} to {unique_dates[-1]}"
        }
        
        self.npz_writer.write_batch(data, output_name, metadata)
        
        self.logger.info(f"✅ Saved: {output_name}")
        self.logger.info(f"   Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        self.logger.info(f"   Date: {unique_dates[0]} to {unique_dates[-1]} ({num_days} days)")

    def build_graph_structure(
        self,
        distance_threshold: float = 1.0,
        min_degree: int = 3,
        output_name: str = 'graph_structure'
    ):
        """Stage 6: Build graph structure"""
        self.logger.info("=" * 60)
        self.logger.info("STAGE 6: BUILD GRAPH STRUCTURE (IMPROVED)")
        self.logger.info("=" * 60)
        
        df = self.npz_reader.read_features()
        
        if df is None or df.empty:
            self.logger.error("No features found")
            return
        
        # Get coordinates
        if 'latitude' not in df.columns or 'longitude' not in df.columns:
            self.logger.error("Coordinates not found")
            return
        
        segment_info = df.groupby('segment_id').agg({
            'latitude': 'first',
            'longitude': 'first'
        }).reset_index()
        
        segment_ids = segment_info['segment_id'].values
        coords = segment_info[['latitude', 'longitude']].values
        
        num_nodes = len(segment_ids)
        self.logger.info(f"Building graph for {num_nodes} nodes")
        
        # ===== FIX: Tạo edges với distance threshold cao hơn =====
        edge_list = []
        
        for i in range(len(coords)):
            for j in range(i + 1, len(coords)):
                dist = self._haversine_distance(
                    coords[i, 0], coords[i, 1],
                    coords[j, 0], coords[j, 1]
                )
                
                if dist <= distance_threshold:
                    edge_list.append([i, j])
                    edge_list.append([j, i])
        
        # ===== FIX: Đảm bảo minimum degree cho mỗi node =====
        # Tính degree hiện tại
        degree_count = {}
        for edge in edge_list:
            src = edge[0]
            degree_count[src] = degree_count.get(src, 0) + 1
        
        # Tính distance matrix
        dist_matrix = np.zeros((num_nodes, num_nodes))
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    dist_matrix[i, j] = self._haversine_distance(
                        coords[i, 0], coords[i, 1],
                        coords[j, 0], coords[j, 1]
                    )
                else:
                    dist_matrix[i, j] = np.inf
        
        # Add edges to nodes with degree < min_degree
        adjacency_set = set(tuple(edge) for edge in edge_list)
        
        for i in range(num_nodes):
            current_degree = degree_count.get(i, 0)
            
            if current_degree < min_degree:
                needed = min_degree - current_degree
                
                # Find nearest neighbors not already connected
                candidates = []
                for j in range(num_nodes):
                    if i != j and (i, j) not in adjacency_set:
                        candidates.append((j, dist_matrix[i, j]))
                
                # Sort by distance and add nearest
                candidates.sort(key=lambda x: x[1])
                
                for j, _ in candidates[:needed]:
                    edge_list.append([i, j])
                    edge_list.append([j, i])
                    adjacency_set.add((i, j))
                    adjacency_set.add((j, i))
                    degree_count[i] = degree_count.get(i, 0) + 1
                    degree_count[j] = degree_count.get(j, 0) + 1
        
        # ===== LOG kết quả =====
        num_edges = len(edge_list)
        self.logger.info(f"Created {num_edges} edges")
        
        if num_edges > 0:
            avg_degree = num_edges / num_nodes
            self.logger.info(f"Average degree: {avg_degree:.2f}")
        else:
            self.logger.warning("No edges created!")
        
        # Convert to edge_index
        if edge_list:
            edge_index = np.array(edge_list, dtype=np.int64).T
        else:
            edge_index = np.array([[], []], dtype=np.int64)
        
        # Select features
        exclude_cols = [
            'segment_id', 'new_segment_id', 'street_name',
            'latitude', 'longitude', 'date_range', 'time_set'
        ]
        feature_cols = [col for col in df.columns 
                    if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])]
        
        # Create node features
        node_features_list = []
        for seg_id in segment_ids:
            seg_data = df[df['segment_id'] == seg_id][feature_cols]
            node_feat = seg_data.select_dtypes(include=np.number).mean().values
            node_features_list.append(node_feat)
        
        node_features = np.array(node_features_list, dtype=np.float32)
        
        # Create adjacency matrix
        adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        if len(edge_index) > 0 and edge_index.shape[1] > 0:
            adj[edge_index[0], edge_index[1]] = 1.0
        
        # Save
        graph_data = {
            'node_features': node_features,
            'edge_index': edge_index,
            'adjacency_matrix': adj,  # ← THÊM adjacency matrix
            'segment_ids': segment_ids,
            'coordinates': coords,
            'feature_names': np.array(feature_cols)
        }
        
        metadata = {
            'num_nodes': len(segment_ids),
            'num_edges': edge_index.shape[1] if len(edge_index) > 0 else 0,
            'num_features': node_features.shape[1],
            'distance_threshold': distance_threshold,
            'min_degree': min_degree,
            'avg_degree': num_edges / num_nodes if num_nodes > 0 else 0
        }
        
        self.npz_writer.write_batch(graph_data, output_name, metadata)
        
        self.logger.info(f"✅ Graph structure saved to {output_name}")
        self.logger.info(f"   Nodes: {len(segment_ids)}")
        self.logger.info(f"   Edges: {num_edges}")
    
    @staticmethod
    def _haversine_distance(lat1, lon1, lat2, lon2):
        """Tính khoảng cách Haversine (km)"""
        R = 6371
        
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        return R * c
    
    def run_full_pipeline(
        self,
        geometry: Dict,
        start_date: str,
        use_31_days: bool = True,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        job_name: str = "Traffic Analysis",
    ):
        """
        Chạy toàn bộ pipeline từ collection đến export.

        Args:
            geometry: Polygon geometry cho vùng phân tích.
            start_date: Ngày bắt đầu (YYYY-MM-DD). Dùng khi use_31_days=True.
            use_31_days: True = thu 31 ngày liên tiếp (mỗi ngày 1 job); False = dùng date_from/date_to.
            date_from: Ngày bắt đầu (legacy, khi use_31_days=False).
            date_to: Ngày kết thúc (legacy, khi use_31_days=False).
            job_name: Tên job (chủ yếu cho legacy batch).
        """
        self.logger.info("=" * 60)
        self.logger.info("STARTING FULL PIPELINE WITH NPZ STORAGE")
        self.logger.info("=" * 60)

        if use_31_days:
            # Stage 1: 31 ngày, mỗi ngày 1 job
            job_ids = self.run_31_days_collection(geometry=geometry, start_date=start_date)
            if not job_ids:
                self.logger.error("Pipeline failed at Stage 1 (31-day collection)")
                return
            # Stage 2: Ingestion từ nhiều file
            self.run_streaming_ingestion_from_jobs(job_ids)
        else:
            date_f = date_from or start_date
            date_t = date_to or start_date
            job_id = self.run_batch_collection(geometry, date_f, date_t, job_name)
            if not job_id:
                self.logger.error("Pipeline failed at Stage 1")
                return
            self.run_streaming_ingestion(job_id)

        # Stage 3: Validation
        self.run_validation_processing()

        # Stage 4: Feature Extraction
        self.run_feature_extraction()

        # Stage 5: Export for training
        self.export_for_model_training()

        # Stage 6: Build graph
        self.build_graph_structure()

        self.logger.info("=" * 60)
        self.logger.info("✅ FULL PIPELINE COMPLETE")
        self.logger.info("=" * 60)