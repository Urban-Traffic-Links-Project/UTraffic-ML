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
                
                for time_result in time_results:
                    record = {
                        'segment_id': segment.get('segmentId'),
                        'new_segment_id': segment.get('newSegmentId'),
                        'street_name': segment.get('streetName'),
                        'distance': segment.get('distance'),
                        'frc': segment.get('frc'),
                        'speed_limit': segment.get('speedLimit'),
                        
                        'time_set': time_result.get('timeSet'),
                        'date_range': time_result.get('dateRange'),
                        
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
        """Xử lý và lưu batch vào NPZ"""
        if not self.accumulated_data:
            return
        
        df = pd.DataFrame(self.accumulated_data)
        
        # 1. Clean
        df = self.cleaner.clean(df)
        
        # 2. Extract features
        df = self.feature_extractor.extract_all_features(df)
        
        # 3. Encode categorical
        df = self.categorical_encoder.fit_transform(df)
        
        # 4. Process spatial
        df = self.spatial_processor.fit_transform(df)
        
        # 5. Normalize
        df = self.normalizer.fit_transform(df, method='standard')
        
        # 6. Save to NPZ
        self.npz_writer.write_features(df)
        
        self.logger.info(f"Processed and saved batch of {len(df)} records")
    
    def export_for_model_training(
        self,
        output_name: str = 'model_ready_data',
        sequence_length: int = 12,
        prediction_horizon: int = 12
    ):
        """
        Stage 5: Export data sẵn sàng cho model training
        
        Args:
            output_name: Tên file output
            sequence_length: Độ dài input sequence (số timesteps)
            prediction_horizon: Độ dài prediction (số timesteps)
        """
        self.logger.info("=" * 60)
        self.logger.info("STAGE 5: EXPORT FOR MODEL TRAINING")
        self.logger.info("=" * 60)
        
        # Read processed features
        df = self.npz_reader.read_features()
        
        if df is None or df.empty:
            self.logger.error("No features found to export")
            return
        
        self.logger.info(f"Loaded {len(df)} records with {len(df.columns)} features")
        
        # Sort by segment, date (31 ngày), rồi time_set để chuỗi thời gian đúng thứ tự
        sort_cols = ['segment_id']
        if 'date_range' in df.columns:
            sort_cols.append('date_range')
        if 'time_set' in df.columns:
            sort_cols.append('time_set')
        if len(sort_cols) > 1:
            df = df.sort_values(sort_cols)
        
        # Get unique segments
        if 'segment_id' not in df.columns:
            self.logger.error("segment_id column not found")
            return
        
        segment_ids = df['segment_id'].unique()
        self.logger.info(f"Found {len(segment_ids)} unique segments")
        
        # Select feature columns (exclude IDs and text)
        exclude_cols = [
            'segment_id', 'new_segment_id', 'street_name',
            'date_range', 'time_set'
        ]
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        self.logger.info(f"Using {len(feature_cols)} features for training")
        
        # Create sequences for each segment
        sequences = []
        targets = []
        segment_indices = []
        
        for seg_id in segment_ids:
            seg_data = df[df['segment_id'] == seg_id][feature_cols].values
            
            if len(seg_data) < sequence_length + prediction_horizon:
                continue
            
            # Create sliding windows
            for i in range(len(seg_data) - sequence_length - prediction_horizon + 1):
                seq = seg_data[i:i + sequence_length]
                target = seg_data[i + sequence_length:i + sequence_length + prediction_horizon]
                
                sequences.append(seq)
                targets.append(target)
                segment_indices.append(seg_id)
        
        if not sequences:
            self.logger.error("No sequences created")
            return
        
        # Convert to numpy arrays
        X = np.array(sequences)  # [num_samples, sequence_length, num_features]
        y = np.array(targets)    # [num_samples, prediction_horizon, num_features]
        segment_indices = np.array(segment_indices)
        
        self.logger.info(f"Created {len(X)} sequences")
        self.logger.info(f"X shape: {X.shape}, y shape: {y.shape}")
        
        # Create train/val/test split (temporal split)
        n_samples = len(X)
        train_size = int(0.7 * n_samples)
        val_size = int(0.15 * n_samples)
        
        X_train = X[:train_size]
        y_train = y[:train_size]
        
        X_val = X[train_size:train_size + val_size]
        y_val = y[train_size:train_size + val_size]
        
        X_test = X[train_size + val_size:]
        y_test = y[train_size + val_size:]
        
        # Save to NPZ
        train_data = {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test,
            'segment_ids': segment_indices,
            'feature_names': np.array(feature_cols)
        }
        
        metadata = {
            'sequence_length': sequence_length,
            'prediction_horizon': prediction_horizon,
            'num_features': len(feature_cols),
            'num_segments': len(segment_ids),
            'train_size': len(X_train),
            'val_size': len(X_val),
            'test_size': len(X_test),
            'feature_columns': feature_cols
        }
        
        self.npz_writer.write_batch(train_data, output_name, metadata)
        
        self.logger.info(f"✅ Exported training data to {output_name}")
        self.logger.info(f"   Train: {X_train.shape}")
        self.logger.info(f"   Val:   {X_val.shape}")
        self.logger.info(f"   Test:  {X_test.shape}")
    
    def build_graph_structure(
        self,
        distance_threshold: float = 0.2,  # km
        output_name: str = 'graph_structure'
    ):
        """
        Stage 6: Xây dựng graph structure cho GNN
        
        Args:
            distance_threshold: Khoảng cách tối đa để tạo edge (km)
            output_name: Tên file output
        """
        self.logger.info("=" * 60)
        self.logger.info("STAGE 6: BUILD GRAPH STRUCTURE")
        self.logger.info("=" * 60)
        
        df = self.npz_reader.read_features()
        
        if df is None or df.empty:
            self.logger.error("No features found")
            return
        
        # Get unique segments with their coordinates
        if 'latitude' not in df.columns or 'longitude' not in df.columns:
            self.logger.error("Coordinates not found")
            return
        
        # Group by segment to get representative coordinates
        segment_info = df.groupby('segment_id').agg({
            'latitude': 'first',
            'longitude': 'first'
        }).reset_index()
        
        segment_ids = segment_info['segment_id'].values
        coords = segment_info[['latitude', 'longitude']].values
        
        self.logger.info(f"Building graph for {len(segment_ids)} nodes")
        
        # Create edge index based on distance
        edge_list = []
        
        for i in range(len(coords)):
            for j in range(i + 1, len(coords)):
                dist = self._haversine_distance(
                    coords[i, 0], coords[i, 1],
                    coords[j, 0], coords[j, 1]
                )
                
                if dist <= distance_threshold:
                    # Bidirectional edges
                    edge_list.append([i, j])
                    edge_list.append([j, i])
        
        if not edge_list:
            self.logger.warning("No edges created with current threshold")
            edge_index = np.array([[], []], dtype=np.int64)
        else:
            edge_index = np.array(edge_list, dtype=np.int64).T
        
        self.logger.info(f"Created {edge_index.shape[1]} edges")
        
        # Create node features (segment-level aggregated features)
        feature_cols = [col for col in df.columns 
                       if col not in ['segment_id', 'new_segment_id', 'street_name',
                                     'latitude', 'longitude', 'date_range', 'time_set']]
        
        node_features_list = []
        for seg_id in segment_ids:
            seg_data = df[df['segment_id'] == seg_id][feature_cols]
            # Use mean as representative features
            node_features_list.append(seg_data.select_dtypes(include=np.number).mean().values)
        
        node_features = np.array(node_features_list)
        
        # Save graph data
        graph_data = {
            'node_features': node_features,
            'edge_index': edge_index,
            'segment_ids': segment_ids,
            'coordinates': coords,
            'feature_names': np.array(feature_cols)
        }
        
        metadata = {
            'num_nodes': len(segment_ids),
            'num_edges': edge_index.shape[1],
            'num_features': node_features.shape[1],
            'distance_threshold': distance_threshold,
            'avg_degree': edge_index.shape[1] / len(segment_ids) if len(segment_ids) > 0 else 0
        }
        
        self.npz_writer.write_batch(graph_data, output_name, metadata)
        
        self.logger.info(f"✅ Graph structure saved to {output_name}")
        self.logger.info(f"   Nodes: {len(segment_ids)}")
        self.logger.info(f"   Edges: {edge_index.shape[1]}")
        self.logger.info(f"   Avg degree: {metadata['avg_degree']:.2f}")
    
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