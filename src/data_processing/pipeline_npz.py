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
        Stage 2 (31 ngày): Đưa dữ liệu từ nhiều file job vào Kafka stream.
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
        """Đọc 1 file job_*_results.json và gửi segments vào Kafka."""
        result_file = config.data.raw_dir / "tomtom_stats_frc5" / f"job_{job_id}_results.json"

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
                        
                        'time_set': time_sets_map.get(time_result.get('timeSet')),
                        'date_from': date_ranges_map.get(time_result.get('dateRange')),
                        
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
        """
        Xử lý và lưu batch vào NPZ.
        
        FIX: Luôn lưu tọa độ gốc vào cột riêng `raw_latitude/raw_longitude`.
        Tránh việc `read_features()` gộp nhiều file (cũ/mới) khiến graph builder
        đôi lúc dùng nhầm lat/lon đã MinMax normalize ([0,1]) để tính Haversine.
        """
        if not self.accumulated_data:
            return
        
        df = pd.DataFrame(self.accumulated_data)
        
        # 1. Clean
        df = self.cleaner.clean(df)
        
        # ===== Preserve metadata + raw coordinates TRƯỚC KHI transform =====
        metadata_cols = ['time_set', 'date_from', 'latitude', 'longitude']
        preserved_data = {}
        for col in metadata_cols:
            if col in df.columns:
                preserved_data[col] = df[col].copy()
        
        # 2. Extract features
        df = self.feature_extractor.extract_all_features(df)
        
        # 3. Encode categorical
        df = self.categorical_encoder.fit_transform(df)
        
        # 4. Process spatial (normalize=True sẽ biến lat/lon thành [0,1])
        df = self.spatial_processor.fit_transform(df)
        
        # 5. Normalize
        df = self.normalizer.fit_transform(df, method='standard')
        
        # ===== Restore metadata AFTER processing =====
        for col in ['time_set', 'date_from']:
            if col in preserved_data:
                df[col] = preserved_data[col]

        # ===== Store RAW coordinates in dedicated columns =====
        # Keep normalized `latitude/longitude` (if created by spatial processor) for modeling/visualization,
        # and store real degrees in `raw_latitude/raw_longitude` for graph construction.
        if 'latitude' in preserved_data:
            df['raw_latitude'] = preserved_data['latitude']
        if 'longitude' in preserved_data:
            df['raw_longitude'] = preserved_data['longitude']
        
        # 6. Save to NPZ
        self.npz_writer.write_features(df)
        
        self.logger.info(f"Processed and saved batch of {len(df)} records")

    def export_for_model_training(
        self,
        output_name: str = None,
        sequence_length: int = 12,
        prediction_horizon: int = 12
    ):
        """
        Stage 5: Export data với ĐÚNG SHAPE 4D cho T-GCN
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
        
        if 'time_set' not in df.columns:
            self.logger.error("❌ Missing 'time_set' column!")
            self.logger.info(f"Available columns: {df.columns.tolist()}")
            return

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

        def extract_date(date_str):
            date_str = str(date_str)
            if re.match(r'^\d{4}-\d{2}-\d{2}$', date_str):
                return date_str
            match = re.search(r'\d{4}-\d{2}-\d{2}', date_str)
            return match.group(0) if match else None

        def extract_slot_index(time_str):
            match = re.search(r'Slot_(\d{4})', str(time_str))
            if match:
                time_code = match.group(1)
                hour = int(time_code[:2])
                minute = int(time_code[2:])
                if 7 <= hour < 10:
                    return (hour - 7) * 4 + minute // 15
                elif 15 <= hour < 18:
                    return 12 + (hour - 15) * 4 + minute // 15
            return None

        df['date'] = df[date_col].apply(extract_date)
        df['slot_index'] = df['time_set'].apply(extract_slot_index)

        self.logger.info(f"Sample dates: {df['date'].head(3).tolist()}")
        self.logger.info(f"Sample time_sets: {df['time_set'].head(3).tolist()}")
        self.logger.info(f"Sample slot_indices: {df['slot_index'].head(3).tolist()}")

        df = df.dropna(subset=['date', 'slot_index'])
        df['slot_index'] = df['slot_index'].astype(int)

        self.logger.info(f"After cleaning: {len(df)} records remain")

        unique_dates = sorted(df['date'].unique())
        num_days = len(unique_dates)

        if num_days == 0:
            self.logger.error("❌ No valid dates found!")
            return

        self.logger.info(f"Found {num_days} days: {unique_dates[0]} to {unique_dates[-1]}")

        date_to_idx = {date: idx for idx, date in enumerate(unique_dates)}
        df['day_index'] = df['date'].map(date_to_idx)
        df['global_timestamp'] = df['day_index'] * 24 + df['slot_index']
        df = df.sort_values(['segment_id', 'global_timestamp']).reset_index(drop=True)
        
        self.logger.info(f"Global timestamps range: {df['global_timestamp'].min()} - {df['global_timestamp'].max()}")
        
        all_segment_ids = sorted(df['segment_id'].unique())
        self.logger.info(f"Found {len(all_segment_ids)} segments before filtering for completeness")
        
        exclude_patterns = [
            'segment_id', 'new_segment_id', 'street_name',
            'date_range', 'time_set', 'date', 'slot_index', 'day_index', 'global_timestamp',
            'latitude', 'longitude',
            'raw_latitude', 'raw_longitude',
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
        
        timesteps_per_seg = df.groupby('segment_id')['global_timestamp'].count().to_dict()
        min_ts = min(timesteps_per_seg.values())
        max_ts = max(timesteps_per_seg.values())
        expected_ts = num_days * 24
        
        self.logger.info(f"Timesteps: min={min_ts}, max={max_ts}, expected={expected_ts}")
        
        full_data_segs = [s for s, c in timesteps_per_seg.items() if c == expected_ts]
        self.logger.info(
            f"Segments with full {expected_ts} timesteps: {len(full_data_segs)}/{len(timesteps_per_seg)}"
        )
        
        min_needed = sequence_length + prediction_horizon
        
        if full_data_segs:
            selected_segs = full_data_segs
            num_timesteps = expected_ts
            self.logger.info(
                f"Using {len(selected_segs)} segments with FULL {num_timesteps} timesteps "
                f"({num_days} days × 24 slots)"
            )
        else:
            candidate_segs = [s for s, c in timesteps_per_seg.items() if c >= min_needed]
            if not candidate_segs:
                self.logger.error(
                    f"❌ Not enough timesteps in ANY segment: "
                    f"max available = {max_ts}, need at least {min_needed}"
                )
                return
            
            selected_segs = candidate_segs
            num_timesteps = min(timesteps_per_seg[s] for s in selected_segs)
            self.logger.info(
                f"Using {len(selected_segs)} segments with at least {num_timesteps} timesteps"
            )
        
        segment_ids = sorted(selected_segs)
        num_nodes = len(segment_ids)
        self.logger.info(f"Final selected segments: {num_nodes}")
        
        self.logger.info(f"Creating 3D: ({num_timesteps}, {num_nodes}, {num_features})")
        
        data_3d = np.zeros((num_timesteps, num_nodes, num_features), dtype=np.float32)
        
        for node_idx, seg_id in enumerate(segment_ids):
            seg_df = df[df['segment_id'] == seg_id].sort_values('global_timestamp')
            seg_data = seg_df[feature_cols].values.astype(np.float32)
            
            if len(seg_data) > num_timesteps:
                seg_data = seg_data[:num_timesteps]
            elif len(seg_data) < num_timesteps:
                if len(seg_data) > 0:
                    pad = np.repeat([seg_data[-1]], num_timesteps - len(seg_data), axis=0)
                    seg_data = np.vstack([seg_data, pad])
                else:
                    seg_data = np.zeros((num_timesteps, num_features), dtype=np.float32)
            
            seg_data = np.nan_to_num(seg_data, nan=0.0, posinf=0.0, neginf=0.0)
            data_3d[:, node_idx, :] = seg_data
        
        self.logger.info(f"✓ 3D tensor: {data_3d.shape}")
        
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
        
        n = len(X)
        train_size = max(1, int(0.7 * n))
        val_size = max(1, int(0.15 * n))
        
        X_train, y_train = X[:train_size], y[:train_size]
        X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
        X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]
        
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

    @staticmethod
    def _haversine_distance(lat1, lon1, lat2, lon2) -> float:
        """Tính khoảng cách Haversine (km). Yêu cầu tọa độ thực (degrees)."""
        R = 6371.0
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        return R * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

    @staticmethod
    def _bearing(lat1, lon1, lat2, lon2) -> float:
        """
        Tính bearing (góc hướng) từ điểm 1 → điểm 2, đơn vị độ [0, 360).
        Dùng để kiểm tra 2 nodes có cùng hướng đường hay không.
        """
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlon = lon2 - lon1
        x = np.sin(dlon) * np.cos(lat2)
        y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
        return (np.degrees(np.arctan2(x, y)) + 360) % 360

    @staticmethod
    def _bearing_diff(b1: float, b2: float) -> float:
        """
        Góc lệch nhỏ nhất giữa 2 bearing, tính cả chiều ngược (0–90°).
        Ví dụ: bearing 10° và 190° → diff = 0° (cùng trục đường, 2 chiều).
        """
        diff = abs(b1 - b2) % 360
        if diff > 180:
            diff = 360 - diff
        # Đường 2 chiều: bearing ngược nhau (diff ≈ 180°) vẫn là cùng đường
        return min(diff, 180 - diff) if diff > 90 else diff

# =========================================================================
    # STATIC HELPERS
    # =========================================================================

    @staticmethod
    def _haversine_distance(lat1, lon1, lat2, lon2) -> float:
        """Haversine distance in km."""
        R = 6371.0
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat, dlon = lat2 - lat1, lon2 - lon1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        return R * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

    @staticmethod
    def _bearing(lat1, lon1, lat2, lon2) -> float:
        """Forward bearing from point-1 → point-2, degrees [0, 360)."""
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlon = lon2 - lon1
        x = np.sin(dlon) * np.cos(lat2)
        y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
        return (np.degrees(np.arctan2(x, y)) + 360) % 360

    @staticmethod
    def _bearing_diff(b1: float, b2: float) -> float:
        """
        Smallest angular difference accounting for bidirectionality (0–90°).
        e.g. 10° vs 190° → 0° (same road axis, opposite directions).
        """
        diff = abs(b1 - b2) % 360
        if diff > 180:
            diff = 360 - diff
        return min(diff, abs(180 - diff))

    @staticmethod
    def _project_onto_axis(
        lat_i, lon_i,   # reference node (origin of road axis)
        lat_j, lon_j,   # candidate node
        axis_bearing: float,  # road axis bearing in degrees
    ) -> tuple[float, float]:
        """
        Project vector i→j onto the road axis and its perpendicular.
        Returns (along_road_km, cross_road_km).

        Uses a local flat-earth approximation (valid for < 1 km distances).
        """
        R = 6371.0
        dlat = np.radians(lat_j - lat_i)
        dlon = np.radians(lon_j - lon_i)
        avg_lat = np.radians((lat_i + lat_j) / 2)

        dy = R * dlat            # north component km
        dx = R * dlon * np.cos(avg_lat)  # east component km

        # Road unit vector (axis_bearing measured clockwise from north)
        axis_rad = np.radians(axis_bearing)
        ux = np.sin(axis_rad)   # east component of road unit vector
        uy = np.cos(axis_rad)   # north component

        along = dx * ux + dy * uy    # projection along road
        cross = abs(dx * uy - dy * ux)  # perpendicular distance (always positive)
        return along, cross

    # =========================================================================
    # MAIN METHOD
    # =========================================================================

    def build_graph_structure(
        self,
        distance_threshold: float = 0.2,   # max straight-line dist km (~120 m)
        min_distance: float = 0.01,         # min dist km (~8 m) — skip same-detector
        max_neighbors: int = 6,              # only connect nearest 2 along-road neighbors
        bearing_threshold: float = 20.0,     # max bearing deviation (degrees)
        cross_road_max_km: float = 0.015,    # max perpendicular offset km (~15 m)
        use_street_name: bool = True,        # enforce same street_name when available
        output_name: str = 'graph_structure',
    ):
        """
        Stage 6 v3 — Chain-based sequential graph construction.

        Key improvements over v2:
        ─────────────────────────
        1. **Cross-road filter**: After bearing check, compute the perpendicular
           component of i→j. If it exceeds `cross_road_max_km` the nodes are on
           parallel / adjacent roads and the edge is rejected.  This is the main
           fix for roundabout cliques and diagonal cross-street edges.

        2. **Sequential chain (max_neighbors=2)**: Each node connects only to its
           1 or 2 nearest neighbours along the road axis (not 4–8 arbitrary ones).
           This naturally produces a chain topology along each road.

        3. **Tighter defaults**: distance_threshold 0.15→0.12 km, bearing 25→20°.

        Parameters
        ----------
        distance_threshold : float
            Maximum straight-line distance (km) to consider an edge candidate.
        min_distance : float
            Minimum distance (km); closer nodes treated as duplicates.
        max_neighbors : int
            Maximum along-road neighbours per node (recommended: 2).
        bearing_threshold : float
            Maximum deviation (°) between candidate bearing and road axis.
        cross_road_max_km : float
            Maximum perpendicular offset (km) allowed before rejecting edge.
        use_street_name : bool
            When True, nodes with matching street_name skip bearing/cross checks
            (they are definitely on the same road). Nodes with *different* names
            are rejected outright (cross-road).
        output_name : str
            Output NPZ file name.
        """
        self.logger.info("=" * 60)
        self.logger.info("STAGE 6: BUILD GRAPH STRUCTURE v3 (chain + cross-road filter)")
        self.logger.info("=" * 60)

        df = self.npz_reader.read_features()
        if df is None or df.empty:
            self.logger.error("No features found")
            return

        # ── Select coordinate columns ──────────────────────────────────────────
        lat_col, lon_col = None, None
        if 'raw_latitude' in df.columns and 'raw_longitude' in df.columns:
            lat_col, lon_col = 'raw_latitude', 'raw_longitude'
        elif 'latitude' in df.columns and 'longitude' in df.columns:
            lat_col, lon_col = 'latitude', 'longitude'
        else:
            self.logger.error("Coordinates not found")
            return

        # ── Build per-segment lookup ───────────────────────────────────────────
        extra_cols = [lat_col, lon_col, 'segment_id']
        if use_street_name and 'street_name' in df.columns:
            extra_cols.append('street_name')

        coords_df = df[extra_cols].dropna(subset=[lat_col, lon_col, 'segment_id'])

        # Drop normalized [0,1] coords from stale batches
        norm_mask = (
            coords_df[lat_col].between(-0.01, 1.01) &
            coords_df[lon_col].between(-0.01, 1.01)
        )
        if norm_mask.any():
            self.logger.warning(f"⚠️  {int(norm_mask.sum())} rows with normalized coords dropped")
            coords_df = coords_df.loc[~norm_mask].copy()

        if coords_df.empty:
            self.logger.error("No valid degree-like coordinates after filtering")
            return

        agg = {lat_col: 'first', lon_col: 'first'}
        if 'street_name' in coords_df.columns:
            agg['street_name'] = lambda x: x.dropna().iloc[0] if not x.dropna().empty else None

        seg_info     = coords_df.groupby('segment_id').agg(agg).reset_index()
        segment_ids  = seg_info['segment_id'].values
        coords       = seg_info[[lat_col, lon_col]].values
        street_names = (
            seg_info['street_name'].values
            if 'street_name' in seg_info.columns
            else np.full(len(segment_ids), None)
        )
        num_nodes = len(segment_ids)

        # Sanity check
        lat_min, lat_max = float(coords[:, 0].min()), float(coords[:, 0].max())
        lon_min, lon_max = float(coords[:, 1].min()), float(coords[:, 1].max())
        in_vietnam = (5.0 <= lat_min <= 25.0 and 90.0 <= lon_min <= 130.0)
        if not in_vietnam:
            self.logger.warning(
                f"⚠️  Coords not degree-like: lat=[{lat_min:.4f},{lat_max:.4f}], "
                f"lon=[{lon_min:.4f},{lon_max:.4f}]"
            )
        else:
            self.logger.info(
                f"Coord range: lat=[{lat_min:.4f},{lat_max:.4f}], "
                f"lon=[{lon_min:.4f},{lon_max:.4f}] | nodes={num_nodes}"
            )

        self.logger.info(
            f"Params: dist=[{min_distance},{distance_threshold}] km, "
            f"max_neighbors={max_neighbors}, bearing≤{bearing_threshold}°, "
            f"cross_road≤{cross_road_max_km*1000:.0f}m, street_name={use_street_name}"
        )

        # ── Edge building ──────────────────────────────────────────────────────
        #
        # For each node i:
        #   1. Collect candidates j within [min_distance, distance_threshold].
        #   2. Determine road axis from 3 nearest candidates (circular mean bearing).
        #   3. For each candidate j:
        #        a. If use_street_name and BOTH names known:
        #             - same name  → accept (skip bearing/cross checks)
        #             - diff name  → reject
        #        b. Bearing check: bearing(i→j) must be within bearing_threshold
        #           of road axis (bidirectional).
        #        c. Cross-road check: perpendicular offset of j from road axis
        #           must be ≤ cross_road_max_km.
        #   4. Sort surviving candidates by ALONG-ROAD distance; keep max_neighbors.

        neighbor_sets: list[set] = [set() for _ in range(num_nodes)]
        cnt_name_rej    = 0
        cnt_bearing_rej = 0
        cnt_cross_rej   = 0

        for i in range(num_nodes):
            lat_i, lon_i = coords[i, 0], coords[i, 1]
            name_i       = street_names[i]

            # Step 1 — distance filter
            raw_candidates: list[tuple[float, int, float]] = []  # (dist, j, bearing)
            for j in range(num_nodes):
                if i == j:
                    continue
                dist = self._haversine_distance(lat_i, lon_i, coords[j, 0], coords[j, 1])
                if min_distance <= dist <= distance_threshold:
                    b = self._bearing(lat_i, lon_i, coords[j, 0], coords[j, 1])
                    raw_candidates.append((dist, j, b))

            if not raw_candidates:
                continue

            # Step 2 — estimate road axis from 3 nearest candidates
            raw_candidates.sort(key=lambda x: x[0])
            ref_bearings = [b for _, _, b in raw_candidates[:3]]
            sin_mean = np.mean([np.sin(np.radians(b)) for b in ref_bearings])
            cos_mean = np.mean([np.cos(np.radians(b)) for b in ref_bearings])
            road_axis = (np.degrees(np.arctan2(sin_mean, cos_mean)) + 360) % 360

            # Step 3 — apply filters
            accepted: list[tuple[float, int]] = []   # (along_road_km, j)

            for dist, j, b in raw_candidates:
                name_j = street_names[j]
                lat_j, lon_j = coords[j, 0], coords[j, 1]

                # (a) Street name gate
                if use_street_name and name_i and name_j:
                    if name_i != name_j:
                        cnt_name_rej += 1
                        continue
                    # Same name → accept without further checks
                    along, _ = self._project_onto_axis(lat_i, lon_i, lat_j, lon_j, road_axis)
                    accepted.append((abs(along), j))
                    continue

                # (b) Bearing check
                if self._bearing_diff(b, road_axis) > bearing_threshold:
                    cnt_bearing_rej += 1
                    continue

                # (c) Cross-road check — reject nodes on parallel/adjacent roads
                along, cross = self._project_onto_axis(lat_i, lon_i, lat_j, lon_j, road_axis)
                if cross > cross_road_max_km:
                    cnt_cross_rej += 1
                    continue

                accepted.append((abs(along), j))

            # Step 4 — keep max_neighbors nearest along road axis
            accepted.sort(key=lambda x: x[0])
            for _, j in accepted[:max_neighbors]:
                neighbor_sets[i].add(j)
                neighbor_sets[j].add(i)

        self.logger.info(
            f"Rejected — name={cnt_name_rej}, bearing={cnt_bearing_rej}, "
            f"cross_road={cnt_cross_rej}"
        )

        # ── Build directed edge list → undirected ─────────────────────────────
        edge_set = set()
        for i, nbrs in enumerate(neighbor_sets):
            for j in nbrs:
                edge_set.add((min(i, j), max(i, j)))

        edge_list = [[i, j] for i, j in edge_set] + [[j, i] for i, j in edge_set]
        self.logger.info(f"Edges after all filters: {len(edge_set)} undirected")

        # ── Remove isolated nodes ──────────────────────────────────────────────
        nodes_with_edges = set(e[0] for e in edge_list)
        isolated_indices = [i for i in range(num_nodes) if i not in nodes_with_edges]
        if isolated_indices:
            self.logger.warning(f"{len(isolated_indices)} isolated nodes removed")

        valid_indices = sorted(nodes_with_edges)
        segment_ids   = segment_ids[valid_indices]
        coords        = coords[valid_indices]
        num_nodes     = len(segment_ids)

        old_to_new = {old: new for new, old in enumerate(valid_indices)}
        remapped_edges = [
            [old_to_new[e[0]], old_to_new[e[1]]]
            for e in edge_list
            if e[0] in old_to_new and e[1] in old_to_new
        ]

        self.logger.info(f"Nodes (non-isolated): {num_nodes}")
        self.logger.info(f"Final edges: {len(remapped_edges) // 2} undirected")

        # ── Adjacency matrix ──────────────────────────────────────────────────
        if remapped_edges:
            edge_index = np.array(remapped_edges, dtype=np.int64).T
        else:
            edge_index = np.array([[], []], dtype=np.int64)
            self.logger.warning("No edges! Consider relaxing distance_threshold or bearing_threshold.")

        adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        if edge_index.shape[1] > 0:
            adj[edge_index[0], edge_index[1]] = 1.0
            degrees = (adj > 0).sum(axis=1).astype(int)
            self.logger.info(
                f"Degree — min={degrees.min()}, max={degrees.max()}, "
                f"mean={degrees.mean():.2f}, median={np.median(degrees):.1f}"
            )

        # ── Node features ──────────────────────────────────────────────────────
        exclude_cols = {
            'segment_id', 'new_segment_id', 'street_name',
            'latitude', 'longitude', 'raw_latitude', 'raw_longitude',
            'date_range', 'time_set', 'date_from',
        }
        feature_cols = [
            c for c in df.columns
            if c not in exclude_cols and pd.api.types.is_numeric_dtype(df[c])
        ]

        node_features = np.array(
            [
                np.nan_to_num(
                    df[df['segment_id'] == sid][feature_cols]
                    .select_dtypes(include=np.number)
                    .mean()
                    .values,
                    nan=0.0, posinf=0.0, neginf=0.0,
                )
                for sid in segment_ids
            ],
            dtype=np.float32,
        )

        # ── Save ───────────────────────────────────────────────────────────────
        num_edges  = edge_index.shape[1] if edge_index.ndim == 2 else 0
        avg_degree = num_edges / num_nodes if num_nodes > 0 else 0.0

        self.npz_writer.write_batch(
            {
                'node_features':    node_features,
                'edge_index':       edge_index,
                'adjacency_matrix': adj,
                'segment_ids':      segment_ids,
                'coordinates':      coords,
                'feature_names':    np.array(feature_cols),
            },
            output_name,
            {
                'num_nodes':              num_nodes,
                'num_edges':              num_edges,
                'avg_degree':             avg_degree,
                'distance_threshold_km':  distance_threshold,
                'min_distance_km':        min_distance,
                'max_neighbors':          max_neighbors,
                'bearing_threshold_deg':  bearing_threshold,
                'cross_road_max_km':      cross_road_max_km,
                'use_street_name':        use_street_name,
                'isolated_nodes_removed': len(isolated_indices),
            },
        )

        self.logger.info(f"✅ Graph saved → '{output_name}'")
        self.logger.info(f"   Nodes: {num_nodes}  |  Edges: {num_edges // 2}  |  Avg degree: {avg_degree:.2f}")
        self.logger.info(f"   Isolated removed: {len(isolated_indices)}")

    @staticmethod
    def _haversine_distance(lat1, lon1, lat2, lon2) -> float:
        """Tính khoảng cách Haversine (km). Yêu cầu tọa độ thực (degrees)."""
        R = 6371.0
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        return R * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

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
        """
        self.logger.info("=" * 60)
        self.logger.info("STARTING FULL PIPELINE WITH NPZ STORAGE")
        self.logger.info("=" * 60)

        if use_31_days:
            job_ids = self.run_31_days_collection(geometry=geometry, start_date=start_date)
            if not job_ids:
                self.logger.error("Pipeline failed at Stage 1 (31-day collection)")
                return
            self.run_streaming_ingestion_from_jobs(job_ids)
        else:
            date_f = date_from or start_date
            date_t = date_to or start_date
            job_id = self.run_batch_collection(geometry, date_f, date_t, job_name)
            if not job_id:
                self.logger.error("Pipeline failed at Stage 1")
                return
            self.run_streaming_ingestion(job_id)

        self.run_validation_processing()
        self.run_feature_extraction()
        self.export_for_model_training()
        self.build_graph_structure()

        self.logger.info("=" * 60)
        self.logger.info("✅ FULL PIPELINE COMPLETE")
        self.logger.info("=" * 60)