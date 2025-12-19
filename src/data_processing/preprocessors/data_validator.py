# src/data_processing/preprocessors/data_validator.py
from typing import Dict, Any, List, Tuple
import pandas as pd

from utils.config import config
from utils.logger import LoggerMixin

class DataValidator(LoggerMixin):
    """
    Validator kiểm tra tính hợp lệ của dữ liệu giao thông
    """
    
    def __init__(self):
        self.min_speed = config.data.min_speed
        self.max_speed = config.data.max_speed
        self.confidence_threshold = config.data.confidence_threshold
    
    def validate_tomtom_result(self, data) -> Tuple[bool, List[str]]:
        """
        Validate dữ liệu kết quả từ TomTom API
        
        Returns:
            (is_valid, error_messages)
        """
        errors = []
        
        # Check structure
        if not isinstance(data, dict):
            errors.append("Data must be a dictionary")
            return False, errors
        
        # Check required fields
        required_fields = ['jobName', 'network']
        for field in required_fields:
            if field not in data:
                errors.append(f"Missing required field: {field}")
        
        # Check network structure
        network = data.get('network', {})
        if 'segmentResults' not in network:
            errors.append("Missing segmentResults in network")
        else:
            segments = network['segmentResults']
            if not isinstance(segments, list) or len(segments) == 0:
                errors.append("segmentResults must be a non-empty list")
        
        is_valid = len(errors) == 0
        return is_valid, errors
    
    def validate_segment(self, segment: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate một segment data
        
        Returns:
            (is_valid, error_messages)
        """
        errors = []
        
        # Required fields
        required_fields = ['segmentId', 'distance', 'shape', 'segmentTimeResults']
        for field in required_fields:
            if field not in segment:
                errors.append(f"Missing field: {field}")
        
        # Validate distance
        distance = segment.get('distance')
        if distance is not None and (distance <= 0 or distance > 100):
            errors.append(f"Invalid distance: {distance} km")
        
        # Validate shape
        shape = segment.get('shape', [])
        if not isinstance(shape, list) or len(shape) < 2:
            errors.append("Shape must have at least 2 points")
        else:
            for point in shape:
                if not self._validate_coordinate(point):
                    errors.append(f"Invalid coordinate: {point}")
        
        # Validate time results
        time_results = segment.get('segmentTimeResults', [])
        if not isinstance(time_results, list) or len(time_results) == 0:
            errors.append("segmentTimeResults must be a non-empty list")
        
        is_valid = len(errors) == 0
        return is_valid, errors
    
    def validate_time_result(self, time_result: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate dữ liệu thời gian của một segment
        
        Returns:
            (is_valid, error_messages)
        """
        errors = []
        
        # Check speeds
        speeds = ['harmonicAverageSpeed', 'medianSpeed', 'averageSpeed']
        for speed_field in speeds:
            speed = time_result.get(speed_field)
            if speed is not None:
                if not (self.min_speed <= speed <= self.max_speed):
                    errors.append(
                        f"Invalid {speed_field}: {speed} km/h "
                        f"(must be between {self.min_speed} and {self.max_speed})"
                    )
        
        # Check travel times
        travel_time = time_result.get('averageTravelTime')
        if travel_time is not None and travel_time < 0:
            errors.append(f"Invalid averageTravelTime: {travel_time}")
        
        # Check sample size
        sample_size = time_result.get('sampleSize')
        if sample_size is not None and sample_size < 0:
            errors.append(f"Invalid sampleSize: {sample_size}")
        
        # Check standard deviation
        std = time_result.get('standardDeviationSpeed')
        if std is not None and std < 0:
            errors.append(f"Invalid standardDeviationSpeed: {std}")
        
        is_valid = len(errors) == 0
        return is_valid, errors
    
    def validate_dataframe(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate pandas DataFrame
        
        Returns:
            (is_valid, error_messages)
        """
        errors = []
        
        # Check required columns
        required_cols = [
            'segment_id', 'time_set', 'date_range',
            'average_speed', 'sample_size'
        ]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            errors.append(f"Missing columns: {missing_cols}")
        
        # Check data types
        if 'average_speed' in df.columns:
            if not pd.api.types.is_numeric_dtype(df['average_speed']):
                errors.append("average_speed must be numeric")
        
        # Check for null values in critical columns
        critical_cols = ['segment_id', 'average_speed']
        for col in critical_cols:
            if col in df.columns:
                null_count = df[col].isnull().sum()
                if null_count > 0:
                    errors.append(f"{col} has {null_count} null values")
        
        # Check value ranges
        if 'average_speed' in df.columns:
            invalid_speeds = df[
                (df['average_speed'] < self.min_speed) | 
                (df['average_speed'] > self.max_speed)
            ]
            if len(invalid_speeds) > 0:
                errors.append(
                    f"Found {len(invalid_speeds)} rows with invalid speeds"
                )
        
        # Check missing data ratio
        missing_ratio = config.data.missing_threshold
        for col in df.columns:
            null_pct = df[col].isnull().sum() / len(df)
            if null_pct > missing_ratio:
                errors.append(
                    f"Column {col} has {null_pct:.1%} missing data "
                    f"(threshold: {missing_ratio:.1%})"
                )
        
        is_valid = len(errors) == 0
        return is_valid, errors
    
    def _validate_coordinate(self, point: Dict[str, float]) -> bool:
        """Validate tọa độ GPS"""
        if not isinstance(point, dict):
            return False
        
        lat = point.get('latitude')
        lon = point.get('longitude')
        
        if lat is None or lon is None:
            return False
        
        # Check valid ranges
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            return False
        
        return True