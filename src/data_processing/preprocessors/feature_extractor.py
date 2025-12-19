# src/data_processing/preprocessors/feature_extractor.py
import pandas as pd
import numpy as np

from utils.config import config
from utils.logger import LoggerMixin

class FeatureExtractor(LoggerMixin):
    """
    Trích xuất features từ dữ liệu giao thông
    """
    
    def __init__(self):
        self.time_windows = config.features.time_windows
    
    def extract_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Trích xuất tất cả features
        
        Args:
            df: DataFrame với dữ liệu đã clean
            
        Returns:
            DataFrame với features mới
        """
        self.logger.info(f"Extracting features from {len(df)} rows...")
        
        df_features = df.copy()
        
        # 1. Speed-based features
        df_features = self._extract_speed_features(df_features)
        
        # 2. Temporal features
        if 'timestamp' in df_features.columns:
            df_features = self._extract_temporal_features(df_features)
        
        # 3. Congestion features
        df_features = self._extract_congestion_features(df_features)
        
        # 4. Dynamic features (derivatives)
        if 'segment_id' in df_features.columns and 'timestamp' in df_features.columns:
            df_features = self._extract_dynamic_features(df_features)
        
        # 5. Statistical features
        df_features = self._extract_statistical_features(df_features)
        
        self.logger.info(f"Feature extraction complete. Total features: {len(df_features.columns)}")
        
        return df_features
    
    def _extract_speed_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Trích xuất speed-based features"""
        
        # Relative speed
        if 'average_speed' in df.columns and 'free_flow_speed' in df.columns:
            df['relative_speed'] = df['average_speed'] / (df['free_flow_speed'] + 1e-6)
            df['speed_reduction_ratio'] = 1 - df['relative_speed']
        
        # Speed metrics
        # Phân biệt chậm vì kẹt hay chậm vì luật
        if 'average_speed' in df.columns:
            if 'speed_limit' in df.columns:
                df['speed_limit_ratio'] = df['average_speed'] / (df['speed_limit'] + 1e-6)
            
            if 'median_speed' in df.columns:
                df['speed_skewness'] = (df['average_speed'] - df['median_speed']) / (df['std_speed'] + 1e-6)
        
        return df
    
    def _extract_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Trích xuất temporal features"""
        
        if 'timestamp' not in df.columns:
            return df
        
        # Convert to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Hour of day
        df['hour'] = df['timestamp'].dt.hour
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        # Day of week
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Weekday vs weekend
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Peak hours
        df['is_morning_peak'] = ((df['hour'] >= 6) & (df['hour'] < 9)).astype(int)
        df['is_evening_peak'] = ((df['hour'] >= 16) & (df['hour'] < 19)).astype(int)
        df['is_peak'] = (df['is_morning_peak'] | df['is_evening_peak']).astype(int)
        
        return df
    
    def _extract_congestion_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Trích xuất congestion features"""
        
        # Congestion Index
        if 'average_travel_time' in df.columns and 'free_flow_travel_time' in df.columns:
            df['congestion_index'] = (
                df['average_travel_time'] / (df['free_flow_travel_time'] + 1e-6) - 1
            )
        
        # Travel time ratio
        if 'travel_time_ratio' not in df.columns:
            if 'average_travel_time' in df.columns and 'free_flow_travel_time' in df.columns:
                df['travel_time_ratio'] = df['average_travel_time'] / (df['free_flow_travel_time'] + 1e-6)
        
        # Congestion level (categorical)
        if 'congestion_index' in df.columns:
            df['congestion_level'] = pd.cut(
                df['congestion_index'],
                bins=[-np.inf, 0.2, 0.5, 0.8, np.inf],
                labels=['free_flow', 'moderate', 'heavy', 'severe']
            )
        
        return df
    
    def _extract_dynamic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Trích xuất dynamic features (derivatives)"""
        
        # Sort by segment and time
        df = df.sort_values(['segment_id', 'timestamp'])
        
        # Speed derivatives
        if 'average_speed' in df.columns:
            # First derivative (velocity)
            df['speed_derivative'] = df.groupby('segment_id')['average_speed'].diff()
            
            # Second derivative (acceleration)
            df['speed_acceleration'] = df.groupby('segment_id')['speed_derivative'].diff()
            
            # Rate of change
            df['speed_rate_of_change'] = df['speed_derivative'] / (df['average_speed'] + 1e-6)
        
        # Moving averages
        for window in [3, 6, 12]:  # 15min, 30min, 1h at 5-min resolution
            if 'average_speed' in df.columns:
                # Moving Average
                df[f'speed_ma_{window}'] = df.groupby('segment_id')['average_speed'].transform(
                    lambda x: x.rolling(window, min_periods=1).mean()
                )
        
        # Exponential moving average
        if 'average_speed' in df.columns:
            df['speed_ema'] = df.groupby('segment_id')['average_speed'].transform(
                lambda x: x.ewm(span=6, adjust=False).mean()
            )
        
        # Volatility (rolling std)
        if 'average_speed' in df.columns:
            df['speed_volatility'] = df.groupby('segment_id')['average_speed'].transform(
                lambda x: x.rolling(12, min_periods=1).std()
            )
        
        return df
    
    def _extract_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Trích xuất statistical features"""
        
        # Coefficient of variation
        if 'average_speed' in df.columns and 'std_speed' in df.columns:
            df['speed_cv'] = df['std_speed'] / (df['average_speed'] + 1e-6)
        
        # Z-score (normalized deviation from mean)
        if 'average_speed' in df.columns and 'segment_id' in df.columns:
            df['speed_zscore'] = df.groupby('segment_id')['average_speed'].transform(
                lambda x: (x - x.mean()) / (x.std() + 1e-6)
            )
        
        # Percentile rank
        if 'average_speed' in df.columns and 'segment_id' in df.columns:
            df['speed_percentile'] = df.groupby('segment_id')['average_speed'].rank(pct=True)
        
        return df