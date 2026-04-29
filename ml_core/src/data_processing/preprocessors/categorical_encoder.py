# src/data_processing/preprocessors/categorical_encoder.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
from typing import Dict, List, Optional, Union
import joblib

from utils.logger import LoggerMixin

class CategoricalFeatureEncoder(LoggerMixin):
    """
    Encode các categorical features đúng cách
    - time_set: Ordinal encoding (giữ thứ tự thời gian)
    - frc: Ordinal encoding (Functional Road Class có thứ tự)
    - date_range: One-hot hoặc Label encoding (nếu có nhiều giá trị)
    """
    
    def __init__(self, encoding_strategy: str = 'ordinal'):
        """
        Args:
            encoding_strategy: 'ordinal', 'onehot', hoặc 'label'
        """
        self.encoding_strategy = encoding_strategy
        self.encoders: Dict[str, object] = {}
        self.categorical_columns = ['time_set', 'frc', 'date_range']
        self.original_values: Dict[str, List] = {}
    
    def fit_transform(
        self, 
        df: pd.DataFrame,
        categorical_cols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Fit và transform categorical features
        
        Args:
            df: DataFrame cần encode
            categorical_cols: Danh sách cột categorical (mặc định: time_set, frc, date_range)
            
        Returns:
            DataFrame với categorical features đã được encode
        """
        self.logger.info("Encoding categorical features...")
        
        df_encoded = df.copy()
        
        if categorical_cols is None:
            categorical_cols = [col for col in self.categorical_columns if col in df.columns]
        
        # Kiểm tra xem các cột có phải đã được normalize chưa
        for col in categorical_cols:
            if col not in df.columns:
                continue
                
            # Nếu cột đã là số thực (có thể đã bị normalize), cần decode về giá trị gốc trước
            if df[col].dtype in [np.float64, np.float32]:
                # Kiểm tra xem có phải đã normalize không (có giá trị âm, std ~ 1, mean ~ 0)
                col_mean = df[col].mean()
                col_std = df[col].std()
                
                # Nếu mean gần 0 và std gần 1, có thể đã normalize
                if abs(col_mean) < 0.1 and 0.9 < col_std < 1.1:
                    self.logger.warning(
                        f"Column '{col}' appears to be already normalized. "
                        f"Will treat unique values as categories."
                    )
                    # Lấy các giá trị unique và map lại
                    unique_vals = sorted(df[col].unique())
                    mapping = {val: idx for idx, val in enumerate(unique_vals)}
                    df_encoded[col] = df[col].map(mapping)
                    self.original_values[col] = unique_vals
                    continue
            
            # Lưu giá trị gốc để có thể inverse transform
            self.original_values[col] = sorted(df[col].dropna().unique().tolist())
            
            # Xử lý missing values trước
            if df[col].isnull().any():
                # Fill với mode hoặc giá trị đặc biệt
                mode_val = df[col].mode()[0] if not df[col].mode().empty else -1
                df_encoded[col] = df[col].fillna(mode_val)
                self.logger.info(f"  {col}: Filled {df[col].isnull().sum()} missing values with mode")
            
            # Encode theo strategy
            if self.encoding_strategy == 'ordinal':
                encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
                df_encoded[[col]] = encoder.fit_transform(df_encoded[[col]])
                self.encoders[col] = encoder
                
            elif self.encoding_strategy == 'label':
                encoder = LabelEncoder()
                df_encoded[col] = encoder.fit_transform(df_encoded[col].astype(str))
                self.encoders[col] = encoder
                
            elif self.encoding_strategy == 'onehot':
                # One-hot encoding tạo nhiều cột mới
                encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop='first')
                encoded = encoder.fit_transform(df_encoded[[col]])
                encoded_df = pd.DataFrame(
                    encoded,
                    columns=[f"{col}_{i}" for i in range(encoded.shape[1])],
                    index=df_encoded.index
                )
                # Xóa cột gốc và thêm các cột one-hot
                df_encoded = df_encoded.drop(columns=[col])
                df_encoded = pd.concat([df_encoded, encoded_df], axis=1)
                self.encoders[col] = encoder
            else:
                raise ValueError(f"Unknown encoding strategy: {self.encoding_strategy}")
            
            self.logger.info(
                f"  {col}: Encoded {len(self.original_values[col])} unique values "
                f"using {self.encoding_strategy} encoding"
            )
        
        self.logger.info(f"✅ Categorical encoding complete for {len(categorical_cols)} columns")
        return df_encoded
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted encoders
        """
        df_encoded = df.copy()
        
        for col, encoder in self.encoders.items():
            if col not in df.columns:
                continue
            
            # Handle missing values
            if df[col].isnull().any():
                mode_val = df[col].mode()[0] if not df[col].mode().empty else -1
                df_encoded[col] = df[col].fillna(mode_val)
            
            # Transform
            if self.encoding_strategy == 'ordinal':
                df_encoded[[col]] = encoder.transform(df_encoded[[col]])
            elif self.encoding_strategy == 'label':
                # LabelEncoder cần xử lý unknown values
                try:
                    df_encoded[col] = encoder.transform(df_encoded[col].astype(str))
                except ValueError:
                    # Nếu có giá trị mới, gán -1
                    known_classes = set(encoder.classes_)
                    df_encoded[col] = df_encoded[col].apply(
                        lambda x: encoder.transform([x])[0] if str(x) in known_classes else -1
                    )
            elif self.encoding_strategy == 'onehot':
                encoded = encoder.transform(df_encoded[[col]])
                encoded_df = pd.DataFrame(
                    encoded,
                    columns=[f"{col}_{i}" for i in range(encoded.shape[1])],
                    index=df_encoded.index
                )
                df_encoded = df_encoded.drop(columns=[col])
                df_encoded = pd.concat([df_encoded, encoded_df], axis=1)
        
        return df_encoded
    
    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Inverse transform về giá trị gốc
        """
        df_original = df.copy()
        
        for col, encoder in self.encoders.items():
            if col not in df.columns and not any(c.startswith(f"{col}_") for c in df.columns):
                continue
            
            if self.encoding_strategy == 'ordinal':
                df_original[[col]] = encoder.inverse_transform(df_original[[col]])
            elif self.encoding_strategy == 'label':
                df_original[col] = encoder.inverse_transform(df_original[col].astype(int))
            elif self.encoding_strategy == 'onehot':
                # One-hot inverse: tìm cột có giá trị 1
                onehot_cols = [c for c in df.columns if c.startswith(f"{col}_")]
                if onehot_cols:
                    encoded = df[onehot_cols].values
                    df_original[col] = encoder.inverse_transform(encoded)
                    df_original = df_original.drop(columns=onehot_cols)
        
        return df_original
    
    def save_encoders(self, path: str):
        """Lưu encoders ra file"""
        save_data = {
            'encoders': self.encoders,
            'original_values': self.original_values,
            'encoding_strategy': self.encoding_strategy,
            'categorical_columns': self.categorical_columns
        }
        joblib.dump(save_data, path)
        self.logger.info(f"Saved encoders to {path}")
    
    def load_encoders(self, path: str):
        """Load encoders từ file"""
        save_data = joblib.load(path)
        self.encoders = save_data['encoders']
        self.original_values = save_data['original_values']
        self.encoding_strategy = save_data['encoding_strategy']
        self.categorical_columns = save_data.get('categorical_columns', ['time_set', 'frc', 'date_range'])
        self.logger.info(f"Loaded encoders from {path}")
