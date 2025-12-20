# src/data_processing/pipeline.py
import pandas as pd
import json
from typing import Dict, Any, Optional
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
from .storage.parquet_writer import ParquetWriter, ParquetReader

from utils.config import config
from utils.logger import setup_logger

class TrafficDataPipeline:
    """
    Pipeline chính xử lý dữ liệu giao thông theo Kappa Architecture
    """
    
    def __init__(self):
        self.logger = setup_logger('TrafficDataPipeline', config.log_file, config.log_level)
        
        # Initialize components
        self.collector = TomTomTrafficDataCollector()
        self.validator = DataValidator()
        self.cleaner = DataCleaner()
        self.feature_extractor = FeatureExtractor()
        self.categorical_encoder = CategoricalFeatureEncoder(encoding_strategy='ordinal')
        self.spatial_processor = SpatialFeatureProcessor(normalize=True, create_features=True)
        self.normalizer = DataNormalizer()
        self.parquet_writer = ParquetWriter()
        self.parquet_reader = ParquetReader()
        
        self.logger.info("✅ Pipeline initialized")
    
    def run_batch_collection(
        self,
        geometry: Dict,
        date_from: str,
        date_to: str,
        job_name: str = "Traffic Analysis"
    ) -> Optional[str]:
        """
        Stage 1: Thu thập dữ liệu batch từ TomTom
        
        Returns:
            job_id nếu thành công
        """
        self.logger.info("=" * 60)
        self.logger.info("STAGE 1: BATCH DATA COLLECTION")
        self.logger.info("=" * 60)
        
        # Create analysis job
        job_id = self.collector.create_area_analysis_job(
            geometry=geometry,
            date_from=date_from,
            date_to=date_to,
            job_name=job_name
        )
        
        if not job_id:
            self.logger.error("Failed to create job")
            return None
        
        # Wait for completion
        status = self.collector.wait_for_job_completion(job_id, max_wait_minutes=60)
        
        if not status or status.get('jobState') != 'DONE':
            self.logger.error("Job did not complete successfully")
            return None
        
        # Download results
        results = self.collector.download_results(job_id)
        
        if not results:
            self.logger.error("Failed to download results")
            return None
        
        self.logger.info(f"✅ Stage 1 complete: Job {job_id}")
        return job_id
    
    def run_streaming_ingestion(self, job_id: str):
        """
        Stage 2: Đưa dữ liệu vào Kafka stream (traffic.raw)
        """
        self.logger.info("=" * 60)
        self.logger.info("STAGE 2: STREAMING INGESTION")
        self.logger.info("=" * 60)
        
        # Load results from file
        result_file = config.data.raw_dir / "tomtom_stats" / f"job_{job_id}_results.json"
        
        if not result_file.exists():
            self.logger.error(f"Result file not found: {result_file}")
            return
        
        with open(result_file, 'r') as f:
            data = json.load(f)
        
        # Send to Kafka
        sent_count = 0

        with TrafficDataProducer() as producer:
            if data.get('type') == 'FeatureCollection':
                features = data.get('features', [])

                for feature in features:
                    props = feature.get('properties', {})

                    # Bỏ feature metadata (không có segmentId)
                    if 'segmentId' not in props:
                        continue

                    message = {
                        'job_id': job_id,
                        'segment_id': props.get('segmentId'),
                        'new_segment_id': props.get('newSegmentId'),
                        'street_name': props.get('streetName'),
                        'distance': props.get('distance'),
                        'speed_limit': props.get('speedLimit'),
                        'frc': props.get('frc'),
                        'segment_time_results': props.get('segmentTimeResults'),
                    }

                    key = str(props.get('segmentId'))
                    producer.send_raw_data(message, key=key)
                    sent_count += 1

                producer.flush()
                self.logger.info(f"✅ Sent {sent_count} segments to Kafka")

            # Handle JSON format (not GeoJSON)
            else:
                network = data.get('network', {})
                segment_results = network.get('segmentResults', [])
                
                for segment in segment_results:
                    message = {
                        'job_id': job_id,
                        'segment': segment,
                        'job_name': data.get('jobName'),
                        'date_ranges': data.get('dateRanges'),
                        'time_sets': data.get('timeSets')
                    }
                    
                    key = str(segment.get('segmentId'))
                    producer.send_raw_data(message, key=key)
                
                producer.flush()
                self.logger.info(f"✅ Sent {len(segment_results)} segments to Kafka")
        
        self.logger.info("✅ Stage 2 complete")
    
    def run_validation_processing(self):
        """
        Stage 3: Consume raw data, validate và gửi vào traffic.validated
        """
        self.logger.info("=" * 60)
        self.logger.info("STAGE 3: VALIDATION PROCESSING")
        self.logger.info("=" * 60)
        
        def validate_processor(key, value, topic, partition, offset):
            """Processor function cho validation"""
            try:
                segment = value.get('segment', {})
                
                # Validate segment
                is_valid, errors = self.validator.validate_segment(segment)
                
                if not is_valid:
                    self.logger.warning(f"Invalid segment {key}: {errors}")
                    return True  # Commit nhưng skip
                
                # Validate time results
                time_results = segment.get('segmentTimeResults', [])
                valid_time_results = []
                
                for time_result in time_results:
                    is_valid, errors = self.validator.validate_time_result(time_result)
                    if is_valid:
                        valid_time_results.append(time_result)
                    else:
                        self.logger.debug(f"Invalid time result: {errors}")
                
                if not valid_time_results:
                    return True  # No valid time results, skip
                
                # Update with valid time results only
                value['segment']['segmentTimeResults'] = valid_time_results
                
                # Send to validated topic
                with TrafficDataProducer() as producer:
                    producer.send_validated_data(value, key=key)
                
                return True
                
            except Exception as e:
                self.logger.error(f"Error in validate_processor: {e}")
                return False
        
        # Consume and process
        with RawDataConsumer() as consumer:
            consumer.consume(validate_processor, max_messages=None)
        
        self.logger.info("✅ Stage 3 complete")
    
    def run_feature_extraction(self):
        """
        Stage 4: Consume validated data, extract features, normalize và lưu Parquet
        """
        self.logger.info("=" * 60)
        self.logger.info("STAGE 4: FEATURE EXTRACTION & STORAGE")
        self.logger.info("=" * 60)
        
        accumulated_data = []
        
        def feature_processor(key, value, topic, partition, offset):
            """Processor function cho feature extraction"""
            try:
                segment = value.get('segment', {})
                time_results = segment.get('segmentTimeResults', [])
                
                # Convert to flat records
                for time_result in time_results:
                    record = {
                        'segment_id': segment.get('segmentId'),
                        'new_segment_id': segment.get('newSegmentId'),
                        'street_name': segment.get('streetName'),
                        'distance': segment.get('distance'),
                        'frc': segment.get('frc'),
                        'speed_limit': segment.get('speedLimit'),
                        
                        # Time information
                        'time_set': time_result.get('timeSet'),
                        'date_range': time_result.get('dateRange'),
                        
                        # Speed metrics
                        'harmonic_average_speed': time_result.get('harmonicAverageSpeed'),
                        'median_speed': time_result.get('medianSpeed'),
                        'average_speed': time_result.get('averageSpeed'),
                        'std_speed': time_result.get('standardDeviationSpeed'),
                        
                        # Travel time
                        'average_travel_time': time_result.get('averageTravelTime'),
                        'median_travel_time': time_result.get('medianTravelTime'),
                        'travel_time_std': time_result.get('travelTimeStandardDeviation'),
                        'travel_time_ratio': time_result.get('travelTimeRatio'),
                        
                        # Sample info
                        'sample_size': time_result.get('sampleSize')
                    }
                    
                    # Add coordinates (first point)
                    shape = segment.get('shape', [])
                    if shape:
                        record['latitude'] = shape[0].get('latitude')
                        record['longitude'] = shape[0].get('longitude')
                    
                    accumulated_data.append(record)
                
                # Batch process every 10000 records
                if len(accumulated_data) >= 10000:
                    self._process_and_save_batch(accumulated_data)
                    accumulated_data.clear()
                    consumer.commit()
                
                return True
                
            except Exception as e:
                self.logger.error(f"Error in feature_processor: {e}")
                return False
        
        # Consume and process
        with ValidatedDataConsumer() as consumer:
            consumer.consume(feature_processor, max_messages=None)
        
        # Process remaining data
        if accumulated_data:
            self._process_and_save_batch(accumulated_data)
        
        self.logger.info("✅ Stage 4 complete")
    
    def _process_and_save_batch(self, records: list):
        """Helper để xử lý và lưu một batch records"""
        if not records:
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(records)
        
        # 1. Clean data
        df = self.cleaner.clean(df)
        
        # 2. Extract features (numerical features)
        df = self.feature_extractor.extract_all_features(df)
        
        # 3. Encode categorical features (BEFORE normalization)
        df = self.categorical_encoder.fit_transform(df)
        
        # 4. Process spatial features (BEFORE normalization)
        df = self.spatial_processor.fit_transform(df)
        
        # 5. Normalize numerical features (excludes categorical and spatial)
        df = self.normalizer.fit_transform(df, method='standard')
        
        # Save to Parquet
        self.parquet_writer.write_features(df)
        
        self.logger.info(f"Processed and saved batch of {len(df)} records")
    
    def run_full_pipeline(
        self,
        geometry: Dict,
        date_from: str,
        date_to: str,
        job_name: str = "Traffic Analysis"
    ):
        """
        Chạy toàn bộ pipeline từ collection đến storage
        """
        self.logger.info("=" * 60)
        self.logger.info("STARTING FULL PIPELINE")
        self.logger.info("=" * 60)
        
        # Stage 1: Collection
        job_id = self.run_batch_collection(geometry, date_from, date_to, job_name)
        if not job_id:
            self.logger.error("Pipeline failed at Stage 1")
            return
        
        # Stage 2: Ingestion
        self.run_streaming_ingestion(job_id)
        
        # Stage 3: Validation
        self.run_validation_processing()
        
        # Stage 4: Feature Extraction
        self.run_feature_extraction()
        
        self.logger.info("=" * 60)
        self.logger.info("✅ FULL PIPELINE COMPLETE")
        self.logger.info("=" * 60)