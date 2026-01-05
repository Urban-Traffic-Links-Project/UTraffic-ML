# src/data_processing/storage/parquet_writer.py
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime

from utils.config import config
from utils.logger import LoggerMixin

class ParquetWriter(LoggerMixin):
    """
    Writer để lưu dữ liệu vào Parquet files với partitioning
    """
    
    def __init__(self, base_path: Optional[Path] = None):
        self.base_path = base_path or config.data.parquet_dir
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def write_batch(
        self,
        df: pd.DataFrame,
        table_name: str,
        partition_cols: Optional[List[str]] = None,
        compression: str = 'snappy'
    ):
        """
        Ghi batch data vào Parquet
        
        Args:
            df: DataFrame cần ghi
            table_name: Tên bảng/dataset
            partition_cols: Các cột dùng để partition (e.g., ['date', 'time_set'])
            compression: Compression method ('snappy', 'gzip', 'brotli', 'zstd')
        """
        if df.empty:
            self.logger.warning(f"Empty dataframe for {table_name}, skipping")
            return
        
        output_path = self.base_path / table_name
        output_path.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(
            f"Writing {len(df)} rows to {table_name} "
            f"(partitions: {partition_cols or 'none'})"
        )
        
        # Convert to PyArrow Table
        table = pa.Table.from_pandas(df, preserve_index=False)
        
        # Write with partitioning
        if partition_cols:
            pq.write_to_dataset(
                table,
                root_path=str(output_path),
                partition_cols=partition_cols,
                compression=compression,
                use_dictionary=True,
                write_statistics=True,
                version='2.6'
            )
        else:
            # Single file write
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = output_path / f"{table_name}_{timestamp}.parquet"
            
            pq.write_table(
                table,
                str(file_path),
                compression=compression,
                use_dictionary=True,
                write_statistics=True,
                version='2.6'
            )
        
        self.logger.info(f"✅ Successfully wrote to {output_path}")
    
    def append_to_table(
        self,
        df: pd.DataFrame,
        table_name: str,
        partition_cols: Optional[List[str]] = None
    ):
        """
        Append data vào existing table
        """
        self.write_batch(df, table_name, partition_cols)
    
    def write_raw_data(self, df: pd.DataFrame):
        """Ghi raw data"""
        partition_cols = ['date_range'] if 'date_range' in df.columns else None
        self.write_batch(df, 'raw_traffic_data', partition_cols)
    
    def write_validated_data(self, df: pd.DataFrame):
        """Ghi validated data"""
        partition_cols = ['date_range', 'time_set'] if all(
            col in df.columns for col in ['date_range', 'time_set']
        ) else None
        self.write_batch(df, 'validated_traffic_data', partition_cols)
    
    def write_features(self, df: pd.DataFrame):
        """Ghi feature data"""
        # Partition by date if timestamp exists
        if 'timestamp' in df.columns:
            df['date'] = pd.to_datetime(df['timestamp']).dt.date
            partition_cols = ['date']
        else:
            partition_cols = None
        
        self.write_batch(df, 'traffic_features', partition_cols)
    
    def write_metadata(self, metadata: Dict, table_name: str):
        """
        Ghi metadata (schema info, statistics, etc.)
        
        Args:
            metadata: Dictionary chứa metadata
            table_name: Tên table tương ứng
        """
        metadata_path = self.base_path / f"{table_name}_metadata.json"
        
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        self.logger.info(f"Wrote metadata to {metadata_path}")


class ParquetReader(LoggerMixin):
    """
    Reader để đọc dữ liệu từ Parquet files
    """
    
    def __init__(self, base_path: Optional[Path] = None):
        self.base_path = base_path or config.data.parquet_dir
    
    def read_table(
        self,
        table_name: str,
        filters: Optional[List] = None,
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Đọc toàn bộ table hoặc với filters
        
        Args:
            table_name: Tên table
            filters: PyArrow filters (e.g., [('date', '=', '2024-08-01')])
            columns: Specific columns to read
            
        Returns:
            DataFrame
        """
        table_path = self.base_path / table_name
        
        if not table_path.exists():
            self.logger.error(f"Table {table_name} not found at {table_path}")
            return pd.DataFrame()
        
        self.logger.info(f"Reading {table_name} from {table_path}")
        
        try:
            # Read partitioned dataset
            dataset = ds.dataset(
                table_path,
                format="parquet"
            )

            table = dataset.to_table(
                filter=filters,
                columns=columns
            )
            
            df = table.to_pandas()
            self.logger.info(f"Read {len(df)} rows")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error reading {table_name}: {e}")
            return pd.DataFrame()
    
    def read_raw_data(self, filters: Optional[List] = None) -> pd.DataFrame:
        """Đọc raw data"""
        return self.read_table('raw_traffic_data', filters)
    
    def read_validated_data(self, filters: Optional[List] = None) -> pd.DataFrame:
        """Đọc validated data"""
        return self.read_table('validated_traffic_data', filters)
    
    def read_features(self, filters: Optional[List] = None) -> pd.DataFrame:
        """Đọc features"""
        return self.read_table('traffic_features', filters)
    
    def list_tables(self) -> List[str]:
        """List tất cả tables available"""
        tables = [d.name for d in self.base_path.iterdir() if d.is_dir()]
        return tables
    
    def get_table_info(self, table_name: str) -> Dict:
        """
        Lấy thông tin về table (size, schema, partitions)
        """
        table_path = self.base_path / table_name
        
        if not table_path.exists():
            return {}
        
        dataset = ds.dataset(
            table_path,
            format="parquet"
        )
        
        try:
            fragments = list(dataset.get_fragments())
            num_fragments = len(fragments)
        except Exception:
            num_fragments = None

        info = {
            "path": str(table_path),
            "schema": str(dataset.schema),
            "num_fragments": num_fragments,
            "partitioning": str(dataset.partitioning) if dataset.partitioning else None,
        }
        
        return info