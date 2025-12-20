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
        
        # 0. Calculate derived base features (free_flow_speed, free_flow_travel_time)
        df_features = self._calculate_base_features(df_features)
        
        # 1. Speed-based features
        df_features = self._extract_speed_features(df_features)
        
        # 2. Temporal features (from timestamp or time_set)
        df_features = self._extract_temporal_features(df_features)
        
        # 3. Congestion features
        df_features = self._extract_congestion_features(df_features)
        
        # 4. Distance and spatial-based features
        df_features = self._extract_distance_features(df_features)
        
        # 5. Dynamic features (derivatives) - only if we have time ordering
        if 'segment_id' in df_features.columns:
            df_features = self._extract_dynamic_features(df_features)
        
        # 6. Statistical features
        df_features = self._extract_statistical_features(df_features)
        
        # 7. FRC-based features
        df_features = self._extract_frc_features(df_features)
        
        self.logger.info(f"Feature extraction complete. Total features: {len(df_features.columns)}")
        
        return df_features
    
    def _calculate_base_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Tính toán các base features như free_flow_speed, free_flow_travel_time"""
        
        # Calculate free_flow_speed if not present
        # Free flow speed is typically the maximum speed observed for a segment
        if 'free_flow_speed' not in df.columns and 'average_speed' in df.columns:
            if 'segment_id' in df.columns:
                # Use 95th percentile as free flow speed
                df['free_flow_speed'] = df.groupby('segment_id')['average_speed'].transform(
                    lambda x: x.quantile(0.95) if len(x) > 0 else x.max()
                )
            else:
                df['free_flow_speed'] = df['average_speed'].quantile(0.95)
        
        # Calculate free_flow_travel_time from distance and free_flow_speed
        if 'free_flow_travel_time' not in df.columns:
            if 'distance' in df.columns and 'free_flow_speed' in df.columns:
                # Travel time = distance / speed (convert km to hours if needed)
                df['free_flow_travel_time'] = df['distance'] / (df['free_flow_speed'] + 1e-6)
            elif 'distance' in df.columns and 'speed_limit' in df.columns:
                # Use speed_limit as proxy for free_flow_speed
                df['free_flow_travel_time'] = df['distance'] / (df['speed_limit'] + 1e-6)
        
        return df
    
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
            
            if 'median_speed' in df.columns and 'std_speed' in df.columns:
                df['speed_skewness'] = (df['average_speed'] - df['median_speed']) / (df['std_speed'] + 1e-6)
            
            # Speed range
            if 'harmonic_average_speed' in df.columns:
                df['speed_range'] = df['average_speed'] - df['harmonic_average_speed']
        
        return df
    
    def _extract_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Trích xuất temporal features từ timestamp hoặc time_set"""
        
        # If timestamp exists, use it
        if 'timestamp' in df.columns:
            # Convert to datetime if needed
            if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            
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
        
        # Extract features from time_set if available (even if already normalized)
        if 'time_set' in df.columns:
            # time_set might be normalized, so we'll create features based on unique values
            unique_times = df['time_set'].unique()
            if len(unique_times) > 1:
                # Create time_set encoding (0, 1, 2, ...)
                time_mapping = {val: idx for idx, val in enumerate(sorted(unique_times))}
                df['time_set_encoded'] = df['time_set'].map(time_mapping)
                
                # Cyclic encoding for time_set
                n_times = len(unique_times)
                if n_times > 1:
                    df['time_set_sin'] = np.sin(2 * np.pi * df['time_set_encoded'] / n_times)
                    df['time_set_cos'] = np.cos(2 * np.pi * df['time_set_encoded'] / n_times)
        
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
        
        # Determine sort columns
        sort_cols = ['segment_id']
        if 'timestamp' in df.columns:
            sort_cols.append('timestamp')
        elif 'time_set' in df.columns:
            sort_cols.append('time_set')
        
        if len(sort_cols) > 1:
            df = df.sort_values(sort_cols)
        
        # Speed derivatives
        if 'average_speed' in df.columns and 'segment_id' in df.columns:
            # First derivative (velocity)
            df['speed_derivative'] = df.groupby('segment_id')['average_speed'].diff()
            
            # Second derivative (acceleration)
            df['speed_acceleration'] = df.groupby('segment_id')['speed_derivative'].diff()
            
            # Rate of change
            df['speed_rate_of_change'] = df['speed_derivative'] / (df['average_speed'] + 1e-6)
            
            # Moving averages
            for window in [3, 6, 12]:  # 15min, 30min, 1h at 5-min resolution
                df[f'speed_ma_{window}'] = df.groupby('segment_id')['average_speed'].transform(
                    lambda x: x.rolling(window, min_periods=1).mean()
                )
            
            # Exponential moving average
            df['speed_ema'] = df.groupby('segment_id')['average_speed'].transform(
                lambda x: x.ewm(span=6, adjust=False).mean()
            )
            
            # Volatility (rolling std)
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
        
        # Travel time statistics
        if 'average_travel_time' in df.columns and 'median_travel_time' in df.columns:
            df['travel_time_skewness'] = (df['average_travel_time'] - df['median_travel_time']) / (df['travel_time_std'] + 1e-6)
        
        return df
    
    def _extract_distance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Trích xuất features dựa trên distance"""
        
        if 'distance' in df.columns:
            # Distance-based speed (if distance and travel time available)
            if 'average_travel_time' in df.columns:
                df['distance_speed'] = df['distance'] / (df['average_travel_time'] + 1e-6)
            
            # Distance categories
            if df['distance'].notna().any():
                df['distance_category'] = pd.cut(
                    df['distance'],
                    bins=[0, 0.1, 0.5, 1.0, 5.0, np.inf],
                    labels=['very_short', 'short', 'medium', 'long', 'very_long']
                )
        
        return df
    
    def _extract_frc_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Trích xuất features dựa trên FRC (Functional Road Class)"""
        
        if 'frc' in df.columns:
            # FRC might be normalized, so we'll create encoding
            unique_frcs = df['frc'].unique()
            if len(unique_frcs) > 1:
                # Create FRC encoding
                frc_mapping = {val: idx for idx, val in enumerate(sorted(unique_frcs))}
                df['frc_encoded'] = df['frc'].map(frc_mapping)
                
                # FRC is ordinal (higher = better road), so we can use it directly
                # But if normalized, we'll use encoded version
                df['frc_level'] = df['frc_encoded'] if 'frc_encoded' in df.columns else df['frc']
        
        return df