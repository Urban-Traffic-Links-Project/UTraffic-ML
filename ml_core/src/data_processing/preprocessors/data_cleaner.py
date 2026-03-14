# src/data_processing/preprocessors/data_cleaner.py
import pandas as pd
import numpy as np
from scipy import stats

from utils.config import config
from utils.logger import LoggerMixin

class DataCleaner(LoggerMixin):
    """
    Làm sạch dữ liệu giao thông: xử lý missing values và outliers
    """
    
    def __init__(self):
        self.z_threshold = config.data.z_score_threshold
        self.iqr_multiplier = config.data.iqr_multiplier
        self.interpolation_gap_max = config.data.interpolation_gap_max
    
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Pipeline làm sạch toàn bộ
        
        Args:
            df: DataFrame cần clean
            
        Returns:
            DataFrame đã được clean
        """
        self.logger.info(f"Cleaning data: {len(df)} rows")
        
        df_clean = df.copy()
        
        # 0. Convert distance from meters to kilometers
        if 'distance' in df_clean.columns:
            # Check if distance is likely in meters (values > 1 typically)
            if df_clean['distance'].notna().any():
                median_distance = df_clean['distance'].median()
                if median_distance > 50:
                    df_clean['distance'] = df_clean['distance'] / 1000.0
                    self.logger.info("Converted distance from meters to kilometers")
        
        # 1. Remove duplicates
        df_clean = self._remove_duplicates(df_clean)
        
        # 2. Handle missing values
        df_clean = self._handle_missing_values(df_clean)
        
        # 3. Detect and handle outliers
        df_clean = self._handle_outliers(df_clean)
        
        # 4. Remove invalid rows
        df_clean = self._remove_invalid_rows(df_clean)
        
        self.logger.info(
            f"Cleaning complete: {len(df_clean)} rows "
            f"({len(df) - len(df_clean)} removed)"
        )
        
        return df_clean
    
    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Xóa các dòng trùng lặp"""
        before = len(df)
        
        # Identify columns for duplicate check
        id_cols = ['segment_id', 'time_set', 'date_range']
        id_cols = [col for col in id_cols if col in df.columns]
        
        if id_cols:
            df = df.drop_duplicates(subset=id_cols, keep='first')
        
        removed = before - len(df)
        if removed > 0:
            self.logger.info(f"Removed {removed} duplicate rows")
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Xử lý missing values theo các phương pháp khác nhau
        """
        self.logger.info("Handling missing values...")
        
        # Speed columns
        speed_cols = [col for col in df.columns if 'speed' in col.lower()]
        
        for col in speed_cols:
            if col in df.columns:
                missing_count = df[col].isnull().sum()
                if missing_count > 0:
                    self.logger.info(f"  {col}: {missing_count} missing values")
                    
                    # Linear interpolation for small gaps
                    df[col] = self._interpolate_with_limit(
                        df[col], 
                        limit=self.interpolation_gap_max
                    )
                    
                    # Fill remaining with median by segment
                    if 'segment_id' in df.columns:
                        df[col] = df.groupby('segment_id')[col].transform(
                            lambda x: x.fillna(x.median())
                        )
                    
                    # Fill remaining with global median
                    df[col].fillna(df[col].median())
        
        # Travel time columns
        time_cols = [col for col in df.columns if 'travel_time' in col.lower()]
        for col in time_cols:
            if col in df.columns:
                # Similar strategy
                df[col] = self._interpolate_with_limit(df[col], limit=self.interpolation_gap_max)
                if 'segment_id' in df.columns:
                    df[col] = df.groupby('segment_id')[col].transform(
                        lambda x: x.fillna(x.median())
                    )
                df[col].fillna(df[col].median())
        
        # Sample size: fill with 0 or median
        if 'sample_size' in df.columns:
            df['sample_size'].fillna(0, inplace=True)
        
        return df
    
    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Phát hiện và xử lý outliers
        """
        self.logger.info("Handling outliers...")

        speed_cols = [col for col in df.columns if 'speed' in col.lower()]

        for col in speed_cols:
            # Z-score method
            z_scores = np.abs(stats.zscore(df[col], nan_policy='omit'))
            outliers_z = z_scores > self.z_threshold

            # IQR method
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - self.iqr_multiplier * IQR
            upper_bound = Q3 + self.iqr_multiplier * IQR
            outliers_iqr = (df[col] < lower_bound) | (df[col] > upper_bound)
            
            # Combined outlier detection
            outliers = outliers_z & outliers_iqr # Giảm false positive
            outlier_count = outliers.sum()
            
            if outlier_count > 0:
                self.logger.info(f"  {col}: {outlier_count} outliers detected")

                # Cap outliers instead of removing
                df.loc[outliers, col] = df.loc[outliers, col].clip(
                    lower=lower_bound,
                    upper=upper_bound
                )

        return df
    
    def _remove_invalid_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Xóa các dòng có dữ liệu không hợp lệ
        """
        before = len(df)
        
        # Remove rows with speed = 0 (likely errors)
        speed_cols = [col for col in df.columns if 'speed' in col.lower()]
        for col in speed_cols:
            if col in df.columns:
                df = df[df[col] > 0]
        
        # Remove rows with very low sample size
        if 'sample_size' in df.columns:
            min_sample_size = 10
            df = df[df['sample_size'] >= min_sample_size]
        
        removed = before - len(df)
        if removed > 0:
            self.logger.info(f"Removed {removed} invalid rows")
        
        return df
    
    def _interpolate_with_limit(
        self, 
        series: pd.Series, 
        limit: int = 3
    ) -> pd.Series:
        """
        Linear interpolation with gap limit
        
        Args:
            series: Series to interpolate
            limit: Maximum consecutive NaNs to interpolate
        """
        return series.interpolate(
            method='linear',
            limit=limit,
            limit_direction='both'
        )