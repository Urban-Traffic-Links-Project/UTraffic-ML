# src/data_processing/preprocessors/spatial_processor.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Dict, List, Optional
import joblib

from utils.logger import LoggerMixin

class SpatialFeatureProcessor(LoggerMixin):
    """
    Xử lý spatial features (latitude, longitude)
    - MinMax scaling để giữ trong khoảng [0, 1] (phù hợp cho tọa độ)
    - Tạo spatial features mới: distance từ center, grid cells, etc.
    """
    
    def __init__(self, normalize: bool = True, create_features: bool = True):
        """
        Args:
            normalize: Có normalize latitude/longitude bằng MinMax không
            create_features: Có tạo thêm spatial features không
        """
        self.normalize = normalize
        self.create_features = create_features
        self.scaler = MinMaxScaler() if normalize else None
        self.spatial_columns = ['latitude', 'longitude']
        self.center_lat: Optional[float] = None
        self.center_lon: Optional[float] = None
        self.bounds: Optional[Dict[str, float]] = None
    
    def fit_transform(
        self, 
        df: pd.DataFrame,
        spatial_cols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Fit và transform spatial features
        
        Args:
            df: DataFrame cần xử lý
            spatial_cols: Danh sách cột spatial (mặc định: latitude, longitude)
            
        Returns:
            DataFrame với spatial features đã được xử lý
        """
        self.logger.info("Processing spatial features...")
        
        df_spatial = df.copy()
        
        if spatial_cols is None:
            spatial_cols = [col for col in self.spatial_columns if col in df.columns]
        
        if not spatial_cols:
            self.logger.warning("No spatial columns found")
            return df_spatial
        
        # Tính center và bounds
        if 'latitude' in df.columns and 'longitude' in df.columns:
            valid_coords = df[['latitude', 'longitude']].dropna()
            if len(valid_coords) > 0:
                self.center_lat = valid_coords['latitude'].mean()
                self.center_lon = valid_coords['longitude'].mean()
                self.bounds = {
                    'min_lat': valid_coords['latitude'].min(),
                    'max_lat': valid_coords['latitude'].max(),
                    'min_lon': valid_coords['longitude'].min(),
                    'max_lon': valid_coords['longitude'].max()
                }
                self.logger.info(
                    f"  Center: ({self.center_lat:.6f}, {self.center_lon:.6f})"
                )
        
        # Normalize spatial coordinates
        if self.normalize and self.scaler:
            # Kiểm tra xem đã normalize chưa
            needs_normalization = False
            for col in spatial_cols:
                if col in df.columns:
                    # Nếu giá trị nằm ngoài [0, 1] hoặc có giá trị âm/dương lớn, cần normalize
                    col_min = df[col].min()
                    col_max = df[col].max()
                    if col_min < 0 or col_max > 1 or (col_max - col_min) > 1:
                        needs_normalization = True
                        break
            
            if needs_normalization:
                # Fit scaler
                spatial_data = df[spatial_cols].dropna()
                if len(spatial_data) > 0:
                    self.scaler.fit(spatial_data)
                    # Transform
                    df_spatial[spatial_cols] = self.scaler.transform(
                        df[spatial_cols].fillna(df[spatial_cols].mean())
                    )
                    self.logger.info(
                        f"  Normalized {len(spatial_cols)} spatial columns using MinMaxScaler"
                    )
            else:
                self.logger.info(
                    "  Spatial columns appear to be already normalized, skipping"
                )
        
        # Tạo spatial features mới
        if self.create_features:
            df_spatial = self._create_spatial_features(df_spatial, spatial_cols)
        
        self.logger.info("✅ Spatial feature processing complete")
        return df_spatial
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted scaler
        """
        df_spatial = df.copy()
        spatial_cols = [col for col in self.spatial_columns if col in df.columns]
        
        if not spatial_cols:
            return df_spatial
        
        # Normalize
        if self.normalize and self.scaler:
            df_spatial[spatial_cols] = self.scaler.transform(
                df[spatial_cols].fillna(df[spatial_cols].mean())
            )
        
        # Create features
        if self.create_features:
            df_spatial = self._create_spatial_features(df_spatial, spatial_cols)
        
        return df_spatial
    
    def _create_spatial_features(
        self, 
        df: pd.DataFrame, 
        spatial_cols: List[str]
    ) -> pd.DataFrame:
        """Tạo các spatial features mới"""
        
        if 'latitude' not in df.columns or 'longitude' not in df.columns:
            return df
        
        # 1. Distance from center (nếu có center)
        if self.center_lat is not None and self.center_lon is not None:
            df['distance_from_center'] = self._haversine_distance(
                df['latitude'],
                df['longitude'],
                self.center_lat,
                self.center_lon
            )
            # Normalize distance
            if df['distance_from_center'].max() > 0:
                df['distance_from_center'] = (
                    df['distance_from_center'] / df['distance_from_center'].max()
                )
        
        # 2. Grid cells (chia thành grid để tạo spatial bins)
        if self.bounds:
            n_grids = 10  # 10x10 grid
            lat_range = self.bounds['max_lat'] - self.bounds['min_lat']
            lon_range = self.bounds['max_lon'] - self.bounds['min_lon']
            
            if lat_range > 0 and lon_range > 0:
                df['grid_lat'] = (
                    ((df['latitude'] - self.bounds['min_lat']) / lat_range * n_grids)
                    .fillna(0)
                    .astype(int)
                    .clip(0, n_grids - 1)
                )
                df['grid_lon'] = (
                    ((df['longitude'] - self.bounds['min_lon']) / lon_range * n_grids)
                    .fillna(0)
                    .astype(int)
                    .clip(0, n_grids - 1)
                )
                df['grid_cell'] = df['grid_lat'] * n_grids + df['grid_lon']
        
        # 3. Spatial encoding (sin/cos cho cyclic nature của tọa độ)
        # Latitude: -90 to 90
        df['lat_sin'] = np.sin(2 * np.pi * (df['latitude'] - df['latitude'].min()) / 
                               (df['latitude'].max() - df['latitude'].min() + 1e-6))
        df['lat_cos'] = np.cos(2 * np.pi * (df['latitude'] - df['latitude'].min()) / 
                               (df['latitude'].max() - df['latitude'].min() + 1e-6))
        
        # Longitude: -180 to 180
        df['lon_sin'] = np.sin(2 * np.pi * (df['longitude'] - df['longitude'].min()) / 
                               (df['longitude'].max() - df['longitude'].min() + 1e-6))
        df['lon_cos'] = np.cos(2 * np.pi * (df['longitude'] - df['longitude'].min()) / 
                               (df['longitude'].max() - df['longitude'].min() + 1e-6))
        
        self.logger.info("  Created additional spatial features: distance_from_center, grid_cell, lat/lon sin/cos")
        
        return df
    
    @staticmethod
    def _haversine_distance(
        lat1: pd.Series, 
        lon1: pd.Series, 
        lat2: float, 
        lon2: float
    ) -> pd.Series:
        """
        Tính khoảng cách Haversine giữa các điểm và một điểm center
        """
        R = 6371  # Bán kính Trái Đất (km)
        
        # Convert to radians
        lat1_rad = np.radians(lat1)
        lon1_rad = np.radians(lon1)
        lat2_rad = np.radians(lat2)
        lon2_rad = np.radians(lon2)
        
        # Haversine formula
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        a = (
            np.sin(dlat / 2) ** 2 +
            np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
        )
        c = 2 * np.arcsin(np.sqrt(a))
        distance = R * c
        
        return distance
    
    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Inverse transform về tọa độ gốc
        """
        if not self.normalize or not self.scaler:
            return df
        
        df_original = df.copy()
        spatial_cols = [col for col in self.spatial_columns if col in df.columns]
        
        if spatial_cols:
            df_original[spatial_cols] = self.scaler.inverse_transform(
                df[spatial_cols]
            )
        
        return df_original
    
    def save_processor(self, path: str):
        """Lưu processor ra file"""
        save_data = {
            'scaler': self.scaler,
            'center_lat': self.center_lat,
            'center_lon': self.center_lon,
            'bounds': self.bounds,
            'normalize': self.normalize,
            'create_features': self.create_features,
            'spatial_columns': self.spatial_columns
        }
        joblib.dump(save_data, path)
        self.logger.info(f"Saved spatial processor to {path}")
    
    def load_processor(self, path: str):
        """Load processor từ file"""
        save_data = joblib.load(path)
        self.scaler = save_data['scaler']
        self.center_lat = save_data['center_lat']
        self.center_lon = save_data['center_lon']
        self.bounds = save_data['bounds']
        self.normalize = save_data['normalize']
        self.create_features = save_data['create_features']
        self.spatial_columns = save_data.get('spatial_columns', ['latitude', 'longitude'])
        self.logger.info(f"Loaded spatial processor from {path}")
