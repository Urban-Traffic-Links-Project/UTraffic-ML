# src/data_processing/storage/npz_storage.py
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

from utils.logger import LoggerMixin

class NPZWriter(LoggerMixin):
    """
    Writer để lưu dữ liệu vào NPZ files (compressed numpy format)
    Phù hợp cho time series và graph data
    """
    
    def __init__(self, base_path: Optional[Path] = None):
        from utils.config import config
        self.base_path = base_path or config.data.processed_dir
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def write_batch(
        self,
        data: Dict[str, np.ndarray],
        dataset_name: str,
        metadata: Optional[Dict[str, Any]] = None,
        compress: bool = True
    ):
        """
        Ghi batch data vào NPZ
        
        Args:
            data: Dictionary chứa arrays {key: array}
            dataset_name: Tên dataset
            metadata: Metadata (sẽ được serialize)
            compress: Dùng savez_compressed (mặc định True)
        """
        if not data:
            self.logger.warning(f"Empty data for {dataset_name}, skipping")
            return
        
        output_path = self.base_path / dataset_name
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = output_path / f"{dataset_name}_{timestamp}.npz"
        
        # Add metadata to data dict
        save_dict = data.copy()
        if metadata:
            # Serialize metadata as JSON string
            import json
            save_dict['_metadata'] = np.array([json.dumps(metadata)])
        
        # Write
        if compress:
            np.savez_compressed(str(file_path), **save_dict)
        else:
            np.savez(str(file_path), **save_dict)
        
        # Calculate total size
        total_items = sum(arr.size for arr in data.values())
        self.logger.info(
            f"✅ Wrote {len(data)} arrays ({total_items:,} total elements) "
            f"to {file_path.name}"
        )
    
    def write_dataframe(
        self,
        df: pd.DataFrame,
        dataset_name: str,
        include_index: bool = False
    ):
        """
        Convert DataFrame to NPZ format
        
        Args:
            df: DataFrame to save
            dataset_name: Dataset name
            include_index: Include DataFrame index
        """
        data_dict = {}
        
        # Save each column as array
        for col in df.columns:
            data_dict[col] = df[col].to_numpy()
        
        # Save index if requested
        if include_index:
            data_dict['_index'] = df.index.to_numpy()
        
        # Save metadata
        metadata = {
            'columns': list(df.columns),
            'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
            'shape': df.shape,
            'has_index': include_index
        }
        
        self.write_batch(data_dict, dataset_name, metadata)
    
    def write_features(self, df: pd.DataFrame):
        """Ghi traffic features"""
        self.write_dataframe(df, 'traffic_features', include_index=False)
    
    def write_graph_data(
        self,
        node_features: np.ndarray,
        edge_index: np.ndarray,
        edge_attr: Optional[np.ndarray] = None,
        timestamps: Optional[np.ndarray] = None,
        dataset_name: str = 'graph_data'
    ):
        """
        Ghi dữ liệu graph structure
        
        Args:
            node_features: [num_nodes, num_features]
            edge_index: [2, num_edges] - source và target nodes
            edge_attr: [num_edges, edge_features] - thuộc tính edges
            timestamps: [num_snapshots] - timestamps cho temporal graphs
        """
        data = {
            'node_features': node_features,
            'edge_index': edge_index
        }
        
        if edge_attr is not None:
            data['edge_attr'] = edge_attr
        
        if timestamps is not None:
            data['timestamps'] = timestamps
        
        metadata = {
            'num_nodes': node_features.shape[0],
            'num_features': node_features.shape[1] if node_features.ndim > 1 else 1,
            'num_edges': edge_index.shape[1],
            'has_edge_attr': edge_attr is not None,
            'has_timestamps': timestamps is not None
        }
        
        self.write_batch(data, dataset_name, metadata)
    
    def write_time_series(
        self,
        data: np.ndarray,
        timestamps: np.ndarray,
        segment_ids: Optional[np.ndarray] = None,
        dataset_name: str = 'time_series'
    ):
        """
        Ghi time series data
        
        Args:
            data: [num_segments, num_timesteps, num_features]
            timestamps: [num_timesteps]
            segment_ids: [num_segments] - IDs của các segments
        """
        save_dict = {
            'data': data,
            'timestamps': timestamps
        }
        
        if segment_ids is not None:
            save_dict['segment_ids'] = segment_ids
        
        metadata = {
            'shape': data.shape,
            'num_segments': data.shape[0] if data.ndim > 1 else 1,
            'num_timesteps': data.shape[1] if data.ndim > 2 else data.shape[0],
            'num_features': data.shape[2] if data.ndim > 2 else 1,
            'has_segment_ids': segment_ids is not None
        }
        
        self.write_batch(save_dict, dataset_name, metadata)


class NPZReader(LoggerMixin):
    """
    Reader để đọc dữ liệu từ NPZ files
    """
    
    def __init__(self, base_path: Optional[Path] = None):
        from utils.config import config
        self.base_path = base_path or config.data.processed_dir
    
    def read_latest(self, dataset_name: str) -> Optional[Dict[str, np.ndarray]]:
        """
        Đọc file mới nhất của dataset
        
        Returns:
            Dictionary chứa arrays
        """
        dataset_path = self.base_path / dataset_name
        
        if not dataset_path.exists():
            self.logger.error(f"Dataset {dataset_name} not found at {dataset_path}")
            return None
        
        # Find latest file
        npz_files = sorted(dataset_path.glob(f"{dataset_name}_*.npz"))
        if not npz_files:
            self.logger.error(f"No NPZ files found in {dataset_path}")
            return None
        
        latest_file = npz_files[-1]
        self.logger.info(f"Reading {latest_file.name}")
        
        return self.read_file(latest_file)
    
    def read_file(self, file_path: Path) -> Optional[Dict[str, np.ndarray]]:
        """Đọc một file NPZ cụ thể"""
        try:
            data = np.load(str(file_path), allow_pickle=True)
            result = {key: data[key] for key in data.files}
            
            # Extract metadata if present
            if '_metadata' in result:
                import json
                metadata_str = str(result['_metadata'][0])
                metadata = json.loads(metadata_str)
                result['_metadata'] = metadata
                self.logger.debug(f"Metadata: {metadata}")
            
            self.logger.info(f"Loaded {len(result)} arrays from {file_path.name}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error reading {file_path}: {e}")
            return None
    
    def read_as_dataframe(self, dataset_name: str) -> Optional[pd.DataFrame]:
        """
        Đọc NPZ và convert về DataFrame
        """
        dataset_path = self.base_path / dataset_name
        
        if not dataset_path.exists():
            self.logger.error(f"Dataset {dataset_name} not found at {dataset_path}")
            return None
        
        # Đọc TẤT CẢ các file NPZ của dataset và concat lại
        npz_files = sorted(dataset_path.glob(f"{dataset_name}_*.npz"))
        if not npz_files:
            self.logger.error(f"No NPZ files found in {dataset_path}")
            return None
        
        dfs = []
        for file_path in npz_files:
            data = self.read_file(file_path)
            if not data:
                continue
            
            # Extract metadata
            metadata = data.get('_metadata', {})
            if isinstance(metadata, np.ndarray):
                import json
                metadata = json.loads(str(metadata[0]))
            
            # Remove metadata and internal keys
            data_arrays = {
                k: v for k, v in data.items()
                if not k.startswith('_') and k not in ['_index']
            }
            
            # Handle index if present
            index = data.get('_index', None)
            
            df_part = pd.DataFrame(data_arrays, index=index)
            
            # Restore dtypes if available
            if isinstance(metadata, dict) and 'dtypes' in metadata:
                for col, dtype_str in metadata['dtypes'].items():
                    if col in df_part.columns:
                        try:
                            df_part[col] = df_part[col].astype(dtype_str)
                        except Exception:
                            # Nếu cast lỗi thì giữ nguyên
                            pass
            
            dfs.append(df_part)
        
        if not dfs:
            self.logger.error(f"Failed to read any NPZ files for {dataset_name}")
            return None
        
        # Concatenate tất cả batches lại
        full_df = pd.concat(dfs, ignore_index=True)
        return full_df
    
    def read_features(self) -> Optional[pd.DataFrame]:
        """
        Đọc traffic features từ TẤT CẢ các file traffic_features_*.npz
        (gộp toàn bộ 31 ngày × 24 time slots lại)
        """
        return self.read_as_dataframe('traffic_features')
    
    def read_graph_data(self, dataset_name: str = 'graph_data') -> Optional[Dict]:
        """
        Đọc graph structure data
        
        Returns:
            Dictionary chứa node_features, edge_index, edge_attr, timestamps
        """
        return self.read_latest(dataset_name)
    
    def read_time_series(self, dataset_name: str = 'time_series') -> Optional[Dict]:
        """
        Đọc time series data
        
        Returns:
            Dictionary chứa data, timestamps, segment_ids
        """
        return self.read_latest(dataset_name)
    
    def list_datasets(self) -> List[str]:
        """List tất cả datasets có sẵn"""
        if not self.base_path.exists():
            return []
        
        datasets = [d.name for d in self.base_path.iterdir() if d.is_dir()]
        return datasets
    
    def get_dataset_info(self, dataset_name: str) -> Dict:
        """
        Lấy thông tin về dataset (số files, size, metadata)
        """
        dataset_path = self.base_path / dataset_name
        
        if not dataset_path.exists():
            return {}
        
        npz_files = list(dataset_path.glob(f"{dataset_name}_*.npz"))
        
        if not npz_files:
            return {}
        
        # Get info from latest file
        latest_file = sorted(npz_files)[-1]
        data = self.read_file(latest_file)
        
        if not data:
            return {}
        
        metadata = data.get('_metadata', {})
        if isinstance(metadata, np.ndarray):
            import json
            metadata = json.loads(str(metadata[0]))
        
        total_size = sum(f.stat().st_size for f in npz_files)
        
        info = {
            'num_files': len(npz_files),
            'total_size_mb': total_size / (1024 * 1024),
            'latest_file': latest_file.name,
            'arrays': list(k for k in data.keys() if not k.startswith('_')),
            'metadata': metadata
        }
        
        return info