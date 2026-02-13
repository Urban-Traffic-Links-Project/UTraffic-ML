"""
Data Loader for Urban Traffic Links Dataset
Load and prepare data from .npz files
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import os
import sys


class TrafficDataset(Dataset):
    """PyTorch Dataset for traffic data"""
    
    def __init__(self, X, y):
        """
        Args:
            X: Input sequences (samples, seq_len, nodes, features)
            y: Target sequences (samples, pred_len, nodes, features)
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_graph_structure(filepath):
    """
    Load graph structure from .npz file
    """
    data = np.load(filepath, allow_pickle=True)
    
    print("\n=== Loading Graph Structure ===")
    print(f"File: {filepath}")
    
    if 'adjacency_matrix' in data:
        adj = data['adjacency_matrix']
        print(f"Loaded adjacency matrix: {adj.shape}")
    elif 'edge_index' in data:
        edge_index = data['edge_index']

        if 'num_nodes' in data:
            num_nodes = int(data['num_nodes'])
        elif 'node_features' in data:
            num_nodes = data['node_features'].shape[0]
        else:
            num_nodes = int(edge_index.max()) + 1

        # Create adjacency matrix from edge_index
        adj = np.zeros((num_nodes, num_nodes))
        if edge_index.shape[1] > 0:
            adj[edge_index[0], edge_index[1]] = 1
        
        print(f"Created adjacency from edge_index: {adj.shape}")
        print(f"Number of edges: {edge_index.shape[1]}")
    else:
        adj = None
        print("No adjacency data found")
    
    # Get other data
    node_features = data['node_features'] if 'node_features' in data else None
    coordinates = data['coordinates'] if 'coordinates' in data else None
    segment_ids = data['segment_ids'] if 'segment_ids' in data else None
    
    if node_features is not None:
        print(f"Node features shape: {node_features.shape}")
    if coordinates is not None:
        print(f"Coordinates shape: {coordinates.shape}")
    
    # ===== THÊM: In graph statistics =====
    if adj is not None:
        num_edges = np.count_nonzero(adj)
        avg_degree = num_edges / adj.shape[0] if adj.shape[0] > 0 else 0
        print(f"Graph stats: {num_edges} edges, avg degree: {avg_degree:.2f}")
    
    return adj, node_features, coordinates, segment_ids


def load_model_ready_data(filepath):
    """
    Load preprocessed model-ready data
    
    Args:
        filepath: Path to model_ready_data_*.npz
    
    Returns:
        Dictionary with train/val/test splits
    """
    data = np.load(filepath, allow_pickle=True)
    print("Available keys in npz:", data.files)
    
    print("\n=== Loading Model Ready Data ===")
    print(f"File: {filepath}")
    
    # Extract data
    result = {}
    for key in ['X_train', 'y_train', 'X_val', 'y_val', 'X_test', 'y_test']:
        if key in data:
            arr = data[key]
            
            # Convert object array to float
            if arr.dtype == object:
                # Try to convert object array containing nested arrays
                try:
                    # First, check if it's an array of arrays
                    if len(arr) > 0:
                        first_elem = arr[0]
                        if isinstance(first_elem, np.ndarray):
                            # It's an array of numpy arrays - check if they contain strings
                            if first_elem.dtype == object:
                                # Nested object arrays - likely contains strings
                                print(f"Warning: {key} contains nested object arrays (likely string values)")
                                print(f"  Attempting to filter numeric values only...")
                                
                                # Try to convert each sample, filtering out non-numeric values
                                arr_list = []
                                for i, sample in enumerate(arr):
                                    if isinstance(sample, np.ndarray) and sample.dtype == object:
                                        # Filter numeric values from each sample
                                        sample_flat = sample.flatten()
                                        numeric_values = []
                                        for val in sample_flat:
                                            try:
                                                numeric_values.append(float(val))
                                            except (ValueError, TypeError):
                                                # Skip non-numeric values
                                                continue
                                        
                                        if len(numeric_values) > 0:
                                            # Reshape to original shape if possible, otherwise use flat
                                            try:
                                                sample_numeric = np.array(numeric_values, dtype=np.float32).reshape(sample.shape)
                                            except ValueError:
                                                # If reshape fails, pad or truncate
                                                target_size = np.prod(sample.shape)
                                                if len(numeric_values) >= target_size:
                                                    sample_numeric = np.array(numeric_values[:target_size], dtype=np.float32).reshape(sample.shape)
                                                else:
                                                    # Pad with zeros or repeat last value
                                                    padded = list(numeric_values) + [numeric_values[-1]] * (target_size - len(numeric_values))
                                                    sample_numeric = np.array(padded[:target_size], dtype=np.float32).reshape(sample.shape)
                                            arr_list.append(sample_numeric)
                                        else:
                                            raise ValueError(f"Sample {i} contains no numeric values")
                                    elif isinstance(sample, np.ndarray):
                                        # Already numeric array
                                        arr_list.append(sample.astype(np.float32))
                                    else:
                                        raise ValueError(f"Unexpected element type in sample {i}: {type(sample)}")
                                
                                if arr_list:
                                    arr = np.array(arr_list, dtype=np.float32)
                                else:
                                    raise ValueError("No valid numeric arrays found")
                            else:
                                # Arrays are numeric, just convert dtype
                                arr_list = [sample.astype(np.float32) if isinstance(sample, np.ndarray) else np.array(sample, dtype=np.float32) 
                                           for sample in arr]
                                arr = np.array(arr_list, dtype=np.float32)
                        else:
                            # Try direct conversion
                            arr = np.array(arr, dtype=np.float32)
                    else:
                        raise ValueError("Empty array")
                except Exception as e:
                    print(f"Error converting {key} from object to float: {e}")
                    print(f"  Array shape: {arr.shape}, dtype: {arr.dtype}")
                    if len(arr) > 0:
                        print(f"  First element type: {type(arr[0])}")
                        if isinstance(arr[0], np.ndarray):
                            print(f"  First element shape: {arr[0].shape}, dtype: {arr[0].dtype}")
                            if arr[0].dtype == object and len(arr[0]) > 0:
                                print(f"  First element of first array: {arr[0].flat[0]} (type: {type(arr[0].flat[0])})")
                    raise ValueError(f"Failed to load {key}: {e}")
            
            # Ensure numeric dtype
            if arr.dtype not in [np.float32, np.float64, np.int32, np.int64]:
                try:
                    arr = arr.astype(np.float32)
                except (ValueError, TypeError) as e:
                    print(f"Warning: Could not convert {key} to float32: {e}")
                    raise ValueError(f"Data in {key} is not numeric")
            
            result[key] = arr
            print(f"{key}: {arr.shape}, dtype={arr.dtype}")
        else:
            print(f"Warning: {key} not found in npz file")
    
    # Optional metadata: segment_ids, feature_names
    if 'segment_ids' in data:
        result['segment_ids'] = data['segment_ids']
    if 'feature_names' in data:
        result['feature_names'] = data['feature_names']
    
    # Check if all required keys are present
    required_keys = ['X_train', 'y_train', 'X_val', 'y_val', 'X_test', 'y_test']
    missing_keys = [key for key in required_keys if key not in result]
    if missing_keys:
        raise ValueError(f"Missing required keys in data: {missing_keys}")
    
    return result


def load_traffic_features(filepath, segment_ids=None, max_samples=None):
    """
    Load traffic features from .npz file
    
    Args:
        filepath: Path to traffic_features_*.npz
        segment_ids: List of segment IDs to filter (optional)
        max_samples: Maximum number of samples to load
    
    Returns:
        features: Traffic features array
        metadata: Metadata dictionary
    """
    data = np.load(filepath, allow_pickle=True)
    
    print("\n=== Loading Traffic Features ===")
    print(f"File: {filepath}")
    
    # Get numeric features
    numeric_features = []
    feature_names = []
    
    for key in data.files:
        if key.startswith('_') or key in ['segment_id', 'new_segment_id', 'street_name', 
                                           'congestion_level', 'distance_category']:
            continue
        
        arr = data[key]
        if arr.dtype in [np.float64, np.float32, np.int64, np.int32]:
            numeric_features.append(arr)
            feature_names.append(key)
    
    # Stack features
    if numeric_features:
        features = np.column_stack(numeric_features)
        print(f"Features shape: {features.shape}")
        print(f"Number of features: {len(feature_names)}")
        
        if max_samples:
            features = features[:max_samples]
            print(f"Limited to {max_samples} samples")
    else:
        features = None
    
    metadata = {
        'feature_names': feature_names,
        'segment_ids': data['segment_id'] if 'segment_id' in data else None
    }
    
    return features, metadata


def create_sequences(data, seq_len=12, pred_len=12, stride=1):
    """
    Create sequences from time series data
    
    Args:
        data: Time series data (timesteps, nodes, features)
        seq_len: Input sequence length
        pred_len: Prediction sequence length
        stride: Stride for sliding window
    
    Returns:
        X: Input sequences (samples, seq_len, nodes, features)
        y: Target sequences (samples, pred_len, nodes, features)
    """
    X, y = [], []
    
    for i in range(0, len(data) - seq_len - pred_len + 1, stride):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len:i+seq_len+pred_len])
    
    return np.array(X), np.array(y)


def prepare_data_loaders(X_train, y_train, X_val, y_val, X_test, y_test, batch_size=32):
    """
    Create PyTorch DataLoaders
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        X_test, y_test: Test data
        batch_size: Batch size
    
    Returns:
        train_loader, val_loader, test_loader
    """
    train_dataset = TrafficDataset(X_train, y_train)
    val_dataset = TrafficDataset(X_val, y_val)
    test_dataset = TrafficDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader


def normalize_adj(adj):
    """
    Normalize adjacency matrix: D^(-1/2) * A * D^(-1/2)
    where A = A + I (with self-loops)
    
    Args:
        adj: (num_nodes, num_nodes) adjacency matrix (numpy array)
    Returns:
        normalized adjacency matrix (numpy array)
    """
    # Convert to Tensor
    is_numpy = False
    if not torch.is_tensor(adj):
        is_numpy = True
        adj_tensor = torch.FloatTensor(adj)
    else:
        adj_tensor = adj
    
    # 1. Add self-loops (A = A + I)
    adj_tensor += torch.eye(adj_tensor.size(0))
    
    # 2. Calculate degree matrix (D)
    rowsum = adj_tensor.sum(1)
    
    # 3. Calculate D^(-1/2)
    d_inv_sqrt = torch.pow(rowsum, -1/2).flatten()
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.  # Handle division by zero
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    
    # 4. Multiply: D^(-1/2) * A * D^(-1/2)
    adj_normalized = d_mat_inv_sqrt @ adj_tensor @ d_mat_inv_sqrt
    
    # Convert back to numpy if input was numpy
    if is_numpy:
        return adj_normalized.numpy()
    return adj_normalized


def normalize_data(X_train, X_val, X_test):
    """
    Normalize data using StandardScaler
    
    Args:
        X_train, X_val, X_test: Data arrays
    
    Returns:
        Normalized arrays and scaler
    """
    # Reshape to 2D for scaling
    original_shape_train = X_train.shape
    original_shape_val = X_val.shape
    original_shape_test = X_test.shape
    
    X_train_2d = X_train.reshape(-1, X_train.shape[-1])
    X_val_2d = X_val.reshape(-1, X_val.shape[-1])
    X_test_2d = X_test.reshape(-1, X_test.shape[-1])
    
    # Fit scaler on training data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_2d)
    X_val_scaled = scaler.transform(X_val_2d)
    X_test_scaled = scaler.transform(X_test_2d)
    
    # Reshape back
    X_train_scaled = X_train_scaled.reshape(original_shape_train)
    X_val_scaled = X_val_scaled.reshape(original_shape_val)
    X_test_scaled = X_test_scaled.reshape(original_shape_test)
    
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


class DataManager:
    """
    Manager class to handle all data loading and preparation
    """
    def __init__(self, data_dir=None):
        if data_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            data_dir = os.path.join(base_dir, 'data', 'processed')
            
        self.data_dir = data_dir
        self.adj = None
        self.scaler = None
        
    def load_all(self):
        """Load all data files"""
        # Find files
        import glob
        
        graph_dir = os.path.join(self.data_dir, 'graph_structure')
        model_dir = os.path.join(self.data_dir, 'model_ready_data')

        graph_files = glob.glob(os.path.join(graph_dir, 'graph_structure_*.npz'))
        model_files = glob.glob(os.path.join(model_dir, 'model_ready_data_*.npz'))
        
        if not graph_files:
            print("Warning: No graph structure file found")
        if not model_files:
            print("Warning: No model ready data file found")
        
        # Load graph (lấy file mới nhất)
        if graph_files:
            graph_files = sorted(graph_files)
            graph_path = graph_files[-1]
            self.adj, self.node_features, self.coordinates, self.graph_segment_ids = load_graph_structure(graph_path)
        
        # Load model data (lấy file mới nhất)
        if model_files:
            model_files = sorted(model_files)
            model_path = model_files[-1]
            self.data = load_model_ready_data(model_path)
        
        return self
    
    def prepare_for_training(self, batch_size=32, normalize=False):
        """
        Prepare data for training
        
        Returns:
            train_loader, val_loader, test_loader, adj_tensor
        """
        if not hasattr(self, 'data'):
            raise ValueError("Data not loaded. Call load_all() first.")
        
        # Check for required keys
        required_keys = ['X_train', 'y_train', 'X_val', 'y_val', 'X_test', 'y_test']
        missing_keys = [key for key in required_keys if key not in self.data]
        if missing_keys:
            raise ValueError(
                f"Missing required data keys: {missing_keys}. "
                f"Available keys: {list(self.data.keys())}. "
                f"This might be due to data conversion errors during loading."
            )
        
        X_train = self.data['X_train']
        y_train = self.data['y_train']
        X_val = self.data['X_val']
        y_val = self.data['y_val']
        X_test = self.data['X_test']
        y_test = self.data['y_test']
        
        # Nếu có thông tin segment_id từ cả graph và model, đồng bộ số node
        if self.adj is not None and hasattr(self, 'graph_segment_ids') and 'segment_ids' in self.data:
            graph_ids = np.array(self.graph_segment_ids).reshape(-1)
            model_ids = np.array(self.data['segment_ids']).reshape(-1)
            
            id_to_idx = {int(sid): idx for idx, sid in enumerate(graph_ids)}
            selected_indices = []
            missing = []
            for sid in model_ids:
                sid_int = int(sid)
                if sid_int in id_to_idx:
                    selected_indices.append(id_to_idx[sid_int])
                else:
                    missing.append(sid_int)
            
            if missing:
                print(f"Warning: {len(missing)} model segment_ids not found in graph, they will be ignored in adjacency.")
            
            if selected_indices:
                selected_indices = np.array(selected_indices, dtype=int)
                # Cắt adjacency theo thứ tự segment_ids của model_ready_data
                self.adj = self.adj[selected_indices[:, None], selected_indices]
                print(f"Aligned adjacency matrix to model data: {self.adj.shape}")
            else:
                print("Warning: No overlapping segment_ids between graph and model data; using original adjacency.")
        
        # Normalize
        if normalize:
            X_train, X_val, X_test, self.scaler = normalize_data(X_train, X_val, X_test)
            y_train_scaled = self.scaler.transform(y_train.reshape(-1, y_train.shape[-1])).reshape(y_train.shape)
            y_val_scaled = self.scaler.transform(y_val.reshape(-1, y_val.shape[-1])).reshape(y_val.shape)
            y_test_scaled = self.scaler.transform(y_test.reshape(-1, y_test.shape[-1])).reshape(y_test.shape)
        else:
            y_train_scaled = y_train
            y_val_scaled = y_val
            y_test_scaled = y_test
        
        # Create loaders
        train_loader, val_loader, test_loader = prepare_data_loaders(
            X_train, y_train_scaled,
            X_val, y_val_scaled,
            X_test, y_test_scaled,
            batch_size=batch_size
        )
        
        # Prepare adjacency matrix
        if self.adj is not None:
            adj_normalized = normalize_adj(self.adj)
            adj_tensor = torch.FloatTensor(adj_normalized)
        else:
            # Create fully connected graph if no adjacency matrix
            num_nodes = X_train.shape[2]
            adj_tensor = torch.ones(num_nodes, num_nodes)
            print(f"Warning: Creating fully connected graph with {num_nodes} nodes")
        
        return train_loader, val_loader, test_loader, adj_tensor


if __name__ == "__main__":
    # Test data loading
    print("Testing Data Manager...")
    
    dm = DataManager()
    dm.load_all()
    
    if hasattr(dm, 'data'):
        train_loader, val_loader, test_loader, adj = dm.prepare_for_training(batch_size=4, normalize=False)
        
        print("\n=== Data Loaders Ready ===")
        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")
        print(f"Test batches: {len(test_loader)}")
        print(f"Adjacency matrix: {adj.shape}")
        
        # Test one batch
        for X, y in train_loader:
            print(f"\nBatch shapes:")
            print(f"X: {X.shape}")
            print(f"y: {y.shape}")
            break
        
        print("\n✓ Data loading successful!")