# src/data_processing/preprocessors/data_normalizer.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from typing import Dict, List, Optional
import joblib

from utils.logger import LoggerMixin

class DataNormalizer(LoggerMixin):
    """
    Chuẩn hóa dữ liệu giao thông
    """
    
    def __init__(self):
        self.scalers: Dict[str, object] = {}
        self.feature_ranges: Dict[str, Dict] = {}
        # Separate scaler for distance and speed_limit (always MinMaxScaler)
        self.minmax_scaler: Optional[MinMaxScaler] = None
        self.minmax_cols: List[str] = ['distance', 'speed_limit']
    
    def fit_transform(
        self, 
        df: pd.DataFrame,
        method: str = 'standard',
        exclude_cols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Fit và transform data
        
        Args:
            df: DataFrame cần normalize
            method: 'standard', 'minmax', or 'robust'
            exclude_cols: Các cột không normalize
            
        Returns:
            DataFrame đã normalize
        """
        self.logger.info(f"Normalizing data using {method} scaler...")
        
        df_norm = df.copy()
        
        # Identify numeric columns
        numeric_cols = df_norm.select_dtypes(include=[np.number]).columns.tolist()
        
        # Exclude certain columns
        if exclude_cols is None:
            exclude_cols = [
                # ID columns
                'segment_id', 'new_segment_id',
                # Categorical columns (should be encoded, not normalized)
                'time_set', 'date_range', 'frc', 'hour', 'day_of_week',
                'time_set_encoded', 'frc_encoded', 'frc_level',
                'distance_category', 'congestion_level',
                # Note: distance and speed_limit will be normalized separately with MinMaxScaler
                # Text columns
                'street_name', 'timestamp',
                # Spatial columns (should be processed separately)
                'latitude', 'longitude', 'distance_from_center', 
                'grid_lat', 'grid_lon', 'grid_cell',
                'lat_sin', 'lat_cos', 'lon_sin', 'lon_cos',
                # Cyclic encoded features (already normalized)
                'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
                'time_set_sin', 'time_set_cos',
                # Binary/boolean features
                'is_weekend', 'is_morning_peak', 'is_evening_peak', 'is_peak',
                # One-hot encoded columns (if any)
            ]
            # Also exclude any one-hot encoded columns
            onehot_cols = [col for col in df_norm.columns 
                          if any(cat_col in col for cat_col in ['time_set_', 'frc_', 'date_range_'])]
            exclude_cols.extend(onehot_cols)
        
        # Exclude distance and speed_limit from main normalization (they use MinMaxScaler separately)
        exclude_cols.extend(self.minmax_cols)
        
        cols_to_normalize = [col for col in numeric_cols if col not in exclude_cols]
        
        if not cols_to_normalize:
            self.logger.warning("No columns to normalize")
            return df_norm
        
        # Create scaler
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Fit and transform
        df_norm[cols_to_normalize] = scaler.fit_transform(df_norm[cols_to_normalize])
        
        # Save scaler and feature info
        self.scalers[method] = scaler
        self.feature_ranges[method] = {
            'columns': cols_to_normalize,
            'mean': df[cols_to_normalize].mean().to_dict(),
            'std': df[cols_to_normalize].std().to_dict(),
            'min': df[cols_to_normalize].min().to_dict(),
            'max': df[cols_to_normalize].max().to_dict()
        }
        
        self.logger.info(f"Normalized {len(cols_to_normalize)} columns")
        
        # Normalize distance and speed_limit with MinMaxScaler separately
        minmax_cols_to_normalize = [col for col in self.minmax_cols 
                                     if col in df_norm.columns and col in numeric_cols]
        
        if minmax_cols_to_normalize:
            self.minmax_scaler = MinMaxScaler()
            df_norm[minmax_cols_to_normalize] = self.minmax_scaler.fit_transform(
                df_norm[minmax_cols_to_normalize]
            )
            self.logger.info(f"Normalized {minmax_cols_to_normalize} with MinMaxScaler")
        
        return df_norm
    
    def transform(
        self, 
        df: pd.DataFrame,
        method: str = 'standard'
    ) -> pd.DataFrame:
        """
        Transform data using fitted scaler
        
        Args:
            df: DataFrame cần transform
            method: Scaler method đã fit
            
        Returns:
            DataFrame đã transform
        """
        if method not in self.scalers:
            raise ValueError(f"Scaler for method {method} not fitted yet")
        
        df_norm = df.copy()
        scaler = self.scalers[method]
        cols = self.feature_ranges[method]['columns']
        
        # Only transform columns that exist
        cols_to_transform = [col for col in cols if col in df_norm.columns]
        
        if cols_to_transform:
            df_norm[cols_to_transform] = scaler.transform(df_norm[cols_to_transform])
        
        # Transform distance and speed_limit with MinMaxScaler if fitted
        if self.minmax_scaler is not None:
            minmax_cols_to_transform = [col for col in self.minmax_cols 
                                        if col in df_norm.columns]
            if minmax_cols_to_transform:
                df_norm[minmax_cols_to_transform] = self.minmax_scaler.transform(
                    df_norm[minmax_cols_to_transform]
                )
        
        return df_norm
    
    def inverse_transform(
        self, 
        df: pd.DataFrame,
        method: str = 'standard'
    ) -> pd.DataFrame:
        """
        Inverse transform về dữ liệu gốc
        """
        if method not in self.scalers:
            raise ValueError(f"Scaler for method {method} not fitted yet")
        
        df_orig = df.copy()
        scaler = self.scalers[method]
        cols = self.feature_ranges[method]['columns']
        
        cols_to_transform = [col for col in cols if col in df_orig.columns]
        
        if cols_to_transform:
            df_orig[cols_to_transform] = scaler.inverse_transform(df_orig[cols_to_transform])
        
        # Inverse transform distance and speed_limit with MinMaxScaler if fitted
        if self.minmax_scaler is not None:
            minmax_cols_to_transform = [col for col in self.minmax_cols 
                                        if col in df_orig.columns]
            if minmax_cols_to_transform:
                df_orig[minmax_cols_to_transform] = self.minmax_scaler.inverse_transform(
                    df_orig[minmax_cols_to_transform]
                )
        
        return df_orig
    
    def save_scalers(self, path: str):
        """Lưu scalers ra file"""
        save_data = {
            'scalers': self.scalers,
            'feature_ranges': self.feature_ranges,
            'minmax_scaler': self.minmax_scaler,
            'minmax_cols': self.minmax_cols
        }
        joblib.dump(save_data, path)
        self.logger.info(f"Saved scalers to {path}")
    
    def load_scalers(self, path: str):
        """Load scalers từ file"""
        save_data = joblib.load(path)
        self.scalers = save_data['scalers']
        self.feature_ranges = save_data['feature_ranges']
        # Handle backward compatibility
        self.minmax_scaler = save_data.get('minmax_scaler', None)
        self.minmax_cols = save_data.get('minmax_cols', ['distance', 'speed_limit'])
        self.logger.info(f"Loaded scalers from {path}")
    
    def normalize_by_segment(
        self, 
        df: pd.DataFrame,
        value_col: str = 'average_speed'
    ) -> pd.DataFrame:
        """
        Normalize theo từng segment (z-score normalization)
        
        Args:
            df: DataFrame
            value_col: Cột cần normalize
            
        Returns:
            DataFrame với cột normalized
        """
        if 'segment_id' not in df.columns or value_col not in df.columns:
            return df
        
        df_norm = df.copy()
        
        # Z-score normalization by segment
        df_norm[f'{value_col}_normalized'] = df_norm.groupby('segment_id')[value_col].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-6)
        )
        
        return df_norm