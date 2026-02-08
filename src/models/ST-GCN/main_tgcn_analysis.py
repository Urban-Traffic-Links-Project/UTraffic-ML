"""
Main script for T-GCN training and correlation analysis
Uses real traffic data from NPZ files
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import pandas as pd
import os
import json
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from t_gcn_model import TGCN, TGCNTrainer, normalize_adjacency_matrix
from correlation_analysis import CorrelationAnalyzer


class TrafficDataset(Dataset):
    """
    Dataset class for traffic data
    """
    def __init__(self, features, seq_len=2, horizon=1):
        """
        Args:
            features: Traffic features (num_samples, num_nodes)
            seq_len: Length of input sequence
            horizon: Prediction horizon
        """
        self.features = features
        self.seq_len = seq_len
        self.horizon = horizon
        
        # Create sequences
        self.X, self.y = self._create_sequences()
    
    def _create_sequences(self):
        """Create input-output sequences"""
        num_timesteps = len(self.features)
        num_nodes = self.features.shape[1]
        
        # Calculate number of samples
        num_samples = num_timesteps - self.seq_len - self.horizon + 1
        
        # Validate
        if num_samples <= 0:
            raise ValueError(
                f"Insufficient timesteps: {num_timesteps} < {self.seq_len + self.horizon}. "
                f"Need at least {self.seq_len + self.horizon} timesteps."
            )
        
        X = np.zeros((num_samples, self.seq_len, num_nodes, 1))
        y = np.zeros((num_samples, num_nodes, self.horizon))
        
        for i in range(num_samples):
            # Input sequence: [i, i+seq_len)
            X[i] = self.features[i:i+self.seq_len, :, np.newaxis]
            
            # Target: [i+seq_len, i+seq_len+horizon)
            if self.horizon == 1:
                y[i] = self.features[i+self.seq_len, :]
            else:
                y[i] = self.features[i+self.seq_len:i+self.seq_len+self.horizon, :].T
        
        return X, y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.X[idx]), torch.FloatTensor(self.y[idx])


def load_traffic_data(graph_path, traffic_path):
    """
    Load traffic data from NPZ files
    
    Args:
        graph_path: Path to graph structure NPZ file
        traffic_path: Path to traffic features NPZ file
        
    Returns:
        graph_data: Dictionary containing graph information
        traffic_data: Dictionary containing traffic features (can be used as DataFrame)
    """
    print("Loading data from NPZ files...")
    
    # Load graph structure
    graph_npz = np.load(graph_path, allow_pickle=True)
    graph_data = {}
    
    required_keys = ['node_features', 'edge_index', 'segment_ids', 'coordinates', 'feature_names']
    for key in required_keys:
        if key in graph_npz.files:
            graph_data[key] = graph_npz[key]
        else:
            print(f"  Warning: {key} not found in graph NPZ file")
    
    if '_metadata' in graph_npz.files:
        try:
            metadata = graph_npz['_metadata']
            if isinstance(metadata, np.ndarray) and len(metadata) > 0:
                graph_data['metadata'] = json.loads(str(metadata[0]))
            else:
                graph_data['metadata'] = metadata
        except Exception as e:
            print(f"  Warning: Could not parse graph metadata: {e}")
    
    # Load traffic features
    traffic_npz = np.load(traffic_path, allow_pickle=True)
    traffic_data = {}
    
    # Load all arrays from NPZ
    for key in traffic_npz.files:
        if key != '_metadata':
            traffic_data[key] = traffic_npz[key]
    
    # Load metadata
    if '_metadata' in traffic_npz.files:
        try:
            metadata = traffic_npz['_metadata']
            if isinstance(metadata, np.ndarray) and len(metadata) > 0:
                traffic_data['metadata'] = json.loads(str(metadata[0]))
            else:
                traffic_data['metadata'] = metadata
        except Exception as e:
            print(f"  Warning: Could not parse traffic metadata: {e}")
    
    # Print summary
    if 'node_features' in graph_data:
        num_nodes = graph_data['node_features'].shape[0]
    else:
        num_nodes = 0
    
    if 'edge_index' in graph_data:
        num_edges = graph_data['edge_index'].shape[1]
    else:
        num_edges = 0
    
    print(f"✓ Graph data loaded: {num_nodes} nodes, {num_edges} edges")
    
    # Count non-metadata features
    num_features = len([k for k in traffic_data.keys() if k != 'metadata'])
    print(f"✓ Traffic data loaded: {num_features} features")
    
    # Print available features
    if num_features > 0:
        feature_list = [k for k in traffic_data.keys() if k != 'metadata'][:10]
        print(f"  Sample features: {', '.join(feature_list)}")
        if num_features > 10:
            print(f"  ... and {num_features - 10} more")
    
    return graph_data, traffic_data


def construct_adjacency_matrix(edge_index, num_nodes):
    """
    Construct adjacency matrix from edge index
    
    Args:
        edge_index: Edge index array (2, num_edges)
        num_nodes: Number of nodes
        
    Returns:
        adj_matrix: Adjacency matrix (num_nodes, num_nodes)
    """
    adj_matrix = np.zeros((num_nodes, num_nodes))
    
    if edge_index.shape[1] == 0:
        print("  Warning: Empty edge_index, creating identity matrix")
        adj_matrix = np.eye(num_nodes)
        return adj_matrix
    
    # Check if edge_index is 0-indexed or 1-indexed
    min_idx = edge_index.min()
    max_idx = edge_index.max()
    
    if min_idx == 0:
        # 0-indexed
        offset = 0
    elif min_idx == 1:
        # 1-indexed
        offset = 1
    else:
        # Unknown, assume 0-indexed
        offset = 0
        print(f"  Warning: Unexpected edge_index range [{min_idx}, {max_idx}], assuming 0-indexed")
    
    for i in range(edge_index.shape[1]):
        src = edge_index[0, i] - offset
        dst = edge_index[1, i] - offset
        
        if 0 <= src < num_nodes and 0 <= dst < num_nodes:
            adj_matrix[src, dst] = 1
            adj_matrix[dst, src] = 1  # Make symmetric
        else:
            print(f"  Warning: Edge index out of bounds: ({src}, {dst})")
    
    return adj_matrix


def prepare_data_for_tgcn(traffic_data, target_feature='average_speed', 
                          train_ratio=0.8, normalize=True, graph_data=None):
    """
    Prepare traffic data for T-GCN model
    
    Args:
        traffic_data: Dictionary containing traffic features (from NPZ) or DataFrame
        target_feature: Target feature to predict
        train_ratio: Ratio of training data
        normalize: Whether to normalize data
        graph_data: Graph data dictionary (optional, for node alignment)
        
    Returns:
        train_data: Training data (num_timesteps, num_nodes)
        test_data: Testing data (num_timesteps, num_nodes)
        scaler_params: Parameters for denormalization
    """
    print(f"\nPreparing data for T-GCN (target feature: {target_feature})...")
    
    # Reconstruct DataFrame from NPZ dictionary if needed
    if isinstance(traffic_data, dict):
        # Check if we have segment_id and time_set
        if 'segment_id' not in traffic_data or 'time_set' not in traffic_data:
            raise ValueError(
                "traffic_data must contain 'segment_id' and 'time_set' columns. "
                "Available keys: " + ", ".join([k for k in traffic_data.keys() if k != 'metadata'])
            )
        
        # Check if target feature exists
        if target_feature not in traffic_data:
            print(f"Warning: {target_feature} not found. Available features:")
            available = [k for k in traffic_data.keys() if k not in ['metadata', 'segment_id', 'time_set']]
            for key in available[:20]:  # Show first 20
                print(f"  - {key}")
            if len(available) > 20:
                print(f"  ... and {len(available) - 20} more")
            raise ValueError(f"Feature {target_feature} not found in traffic_data")
        
        # Reconstruct DataFrame from NPZ arrays
        # All arrays should have the same length
        array_lengths = {k: len(v) for k, v in traffic_data.items() 
                        if k != 'metadata' and isinstance(v, np.ndarray)}
        
        if not array_lengths:
            raise ValueError("No valid arrays found in traffic_data")
        
        # Check all arrays have same length
        lengths = list(array_lengths.values())
        if len(set(lengths)) > 1:
            raise ValueError(f"Inconsistent array lengths: {set(lengths)}")
        
        # Create DataFrame
        df_dict = {k: v for k, v in traffic_data.items() 
                  if k != 'metadata' and isinstance(v, np.ndarray)}
        df = pd.DataFrame(df_dict)
        print(f"  Reconstructed DataFrame: {df.shape}")
    else:
        # Assume it's already a DataFrame
        df = traffic_data
        if target_feature not in df.columns:
            raise ValueError(f"Feature {target_feature} not found in DataFrame")
        
        if 'segment_id' not in df.columns or 'time_set' not in df.columns:
            raise ValueError("DataFrame must have 'segment_id' and 'time_set' columns")
    
    # Sort by time_set and segment_id
    df = df.sort_values(['time_set', 'segment_id'])
    
    # Pivot to create (timesteps, nodes) structure
    # timesteps = unique time_sets (rows)
    # nodes = unique segment_ids (columns)
    df_pivot = df.pivot_table(
        index='time_set',
        columns='segment_id',
        values=target_feature,
        aggfunc='mean'  # Use mean if there are duplicates
    )
    
    # Align with graph segment_ids if provided
    if graph_data and 'segment_ids' in graph_data:
        graph_segments = graph_data['segment_ids']
        # Reindex columns to match graph segment_ids, fill missing with 0
        df_pivot = df_pivot.reindex(columns=graph_segments, fill_value=0)
        print(f"  Aligned with graph: {len(graph_segments)} nodes")
    
    features_reshaped = df_pivot.values
    num_timesteps, num_nodes = features_reshaped.shape
    
    print(f"  Data shape: {features_reshaped.shape} (timesteps={num_timesteps}, nodes={num_nodes})")
    
    # Validate minimum requirements
    min_timesteps = 3  # seq_len (12) + horizon (1)
    if num_timesteps < min_timesteps:
        raise ValueError(
            f"Insufficient timesteps: {num_timesteps} < {min_timesteps}. "
            f"Need at least {min_timesteps} timesteps for seq_len=2 and horizon=1. "
            f"Consider reducing seq_len or using more data."
        )
    
    # Normalize data
    scaler_params = {}
    if normalize:
        scaler_params['mean'] = np.mean(features_reshaped)
        scaler_params['std'] = np.std(features_reshaped)
        
        if scaler_params['std'] > 0:
            features_normalized = (features_reshaped - scaler_params['mean']) / scaler_params['std']
        else:
            features_normalized = features_reshaped - scaler_params['mean']
        
        print(f"  Normalized: mean={scaler_params['mean']:.4f}, std={scaler_params['std']:.4f}")
    else:
        features_normalized = features_reshaped
        scaler_params['mean'] = 0
        scaler_params['std'] = 1
    
    # Split train/test (temporal split)
    split_idx = int(len(features_normalized) * train_ratio)
    
    # Ensure we have enough data for both train and test
    if split_idx < min_timesteps:
        split_idx = min_timesteps
        print(f"  Warning: Adjusted split_idx to {split_idx} to meet minimum requirements")
    
    if len(features_normalized) - split_idx < min_timesteps:
        split_idx = len(features_normalized) - min_timesteps
        print(f"  Warning: Adjusted split_idx to {split_idx} to ensure test set has enough data")
    
    train_data = features_normalized[:split_idx]
    test_data = features_normalized[split_idx:]
    
    print(f"  Train data: {train_data.shape}")
    print(f"  Test data: {test_data.shape}")
    
    return train_data, test_data, scaler_params


def train_tgcn_model(train_data, test_data, adj_matrix, 
                     seq_len=2, horizon=1, hidden_dim=64,
                     num_epochs=100, batch_size=32, learning_rate=0.001,
                     device='cuda', save_dir='./tgcn_results'):
    """
    Train T-GCN model
    
    Args:
        train_data: Training data (num_timesteps, num_nodes)
        test_data: Testing data (num_timesteps, num_nodes)
        adj_matrix: Adjacency matrix
        seq_len: Sequence length
        horizon: Prediction horizon
        hidden_dim: Hidden dimension
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        device: Device to use
        save_dir: Directory to save results
        
    Returns:
        model: Trained model
        results: Training results
    """
    os.makedirs(save_dir, exist_ok=True)
    
    num_nodes = train_data.shape[1]
    
    print("\n" + "=" * 70)
    print("TRAINING T-GCN MODEL")
    print("=" * 70)
    print(f"Model configuration:")
    print(f"  - Nodes: {num_nodes}")
    print(f"  - Sequence length: {seq_len}")
    print(f"  - Prediction horizon: {horizon}")
    print(f"  - Hidden dimension: {hidden_dim}")
    print(f"  - Epochs: {num_epochs}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Learning rate: {learning_rate}")
    print(f"  - Device: {device}")
    
    # Create datasets
    train_dataset = TrafficDataset(train_data, seq_len=seq_len, horizon=horizon)
    test_dataset = TrafficDataset(test_data, seq_len=seq_len, horizon=horizon)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"\nDataset created:")
    print(f"  - Training samples: {len(train_dataset)}")
    print(f"  - Testing samples: {len(test_dataset)}")
    
    # Normalize adjacency matrix
    adj_normalized = normalize_adjacency_matrix(adj_matrix)
    
    # Create model
    model = TGCN(
        num_nodes=num_nodes,
        input_dim=1,
        hidden_dim=hidden_dim,
        output_dim=horizon,
        num_layers=1,
        gcn_depth=2
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters:")
    print(f"  - Total: {total_params:,}")
    print(f"  - Trainable: {trainable_params:,}")
    
    # Create trainer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    trainer = TGCNTrainer(model, optimizer, criterion, device)
    
    # Training loop
    print("\nStarting training...")
    train_losses = []
    test_losses = []
    best_test_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Train
        train_loss = trainer.train_epoch(train_loader, adj_normalized)
        train_losses.append(train_loss)
        
        # Evaluate
        test_loss, _, _ = trainer.evaluate(test_loader, adj_normalized)
        test_losses.append(test_loss)
        
        # Save best model
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
        
        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] "
                  f"Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}")
    
    print(f"\n✓ Training complete! Best test loss: {best_test_loss:.6f}")
    
    # Load best model
    model.load_state_dict(torch.load(os.path.join(save_dir, 'best_model.pth')))
    
    # Final evaluation
    test_loss, predictions, targets = trainer.evaluate(test_loader, adj_normalized)
    
    # Calculate metrics
    mse = mean_squared_error(targets.flatten(), predictions.flatten())
    mae = mean_absolute_error(targets.flatten(), predictions.flatten())
    rmse = np.sqrt(mse)
    
    # R² score
    r2 = r2_score(targets.reshape(-1, num_nodes), predictions.reshape(-1, num_nodes))
    
    print(f"\nFinal Test Metrics:")
    print(f"  - MSE: {mse:.6f}")
    print(f"  - MAE: {mae:.6f}")
    print(f"  - RMSE: {rmse:.6f}")
    print(f"  - R²: {r2:.6f}")
    
    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss', alpha=0.8)
    plt.plot(test_losses, label='Test Loss', alpha=0.8)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('T-GCN Training Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=300)
    plt.close()
    
    # Save results
    results = {
        'train_losses': train_losses,
        'test_losses': test_losses,
        'best_test_loss': best_test_loss,
        'final_metrics': {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'r2': r2
        },
        'predictions': predictions,
        'targets': targets
    }
    
    # Save predictions
    np.save(os.path.join(save_dir, 'predictions.npy'), predictions)
    np.save(os.path.join(save_dir, 'targets.npy'), targets)
    
    return model, results


def main():
    """Main execution function"""
    
    # Paths to data files
    BASE_DIR = Path.cwd()  # notebooks/
    TRAFFIC_PATH = (
        BASE_DIR
        / "data"
        / "processed"
        / "traffic_features"
        / "traffic_features_20260122_191403.npz"
    ).resolve()
    # Load Graph Structure NPZ
    GRAPH_PATH = (
        BASE_DIR
        / "data"
        / "processed"
        / "graph_structure"
        / "graph_structure_20260122_191428.npz"
    ).resolve()
    
    # Output directory
    OUTPUT_DIR = './tgcn_traffic_analysis'
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("=" * 70)
    print("T-GCN TRAFFIC PREDICTION AND CORRELATION ANALYSIS")
    print("=" * 70)
    
    # 1. Load data
    graph_data, traffic_data = load_traffic_data(GRAPH_PATH, TRAFFIC_PATH)
    
    # 2. Construct adjacency matrix
    num_nodes = graph_data['node_features'].shape[0]
    adj_matrix = construct_adjacency_matrix(graph_data['edge_index'], num_nodes)
    
    print(f"\nAdjacency matrix constructed:")
    print(f"  - Shape: {adj_matrix.shape}")
    print(f"  - Edges: {np.sum(adj_matrix) / 2:.0f}")
    print(f"  - Density: {np.sum(adj_matrix) / (num_nodes * (num_nodes - 1)):.4f}")
    
    # 3. Prepare data
    try:
        train_data, test_data, scaler_params = prepare_data_for_tgcn(
            traffic_data,
            target_feature='average_speed',
            train_ratio=0.8,
            normalize=True,
            graph_data=graph_data
        )
    except Exception as e:
        print(f"\n❌ Error preparing data: {e}")
        print("\nTroubleshooting:")
        print("  1. Check if traffic_features NPZ file contains 'segment_id' and 'time_set'")
        print("  2. Check if 'average_speed' feature exists")
        print("  3. Ensure you have at least 13 timesteps (for seq_len=2, horizon=1)")
        raise
    
    # 4. Train model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    model, results = train_tgcn_model(
        train_data=train_data,
        test_data=test_data,
        adj_matrix=adj_matrix,
        seq_len=2,
        horizon=1,
        # seq_len = 1,
        # horizon = 0,
        hidden_dim=64,
        num_epochs=100,
        batch_size=32,
        learning_rate=0.001,
        device=device,
        save_dir=OUTPUT_DIR
    )
    
    # 5. Correlation Analysis
    print("\n" + "=" * 70)
    print("CORRELATION ANALYSIS")
    print("=" * 70)
    
    # Prepare data for correlation analysis
    all_data = np.concatenate([train_data, test_data], axis=0)
    all_data_expanded = all_data[:, :, np.newaxis]
    
    # Create analyzer
    analyzer = CorrelationAnalyzer(model=model, device=device)
    
    # Generate comprehensive correlation report
    correlation_results = analyzer.generate_correlation_report(
        data=all_data_expanded,
        adj_matrix=adj_matrix,
        predictions=results['predictions'],
        ground_truth=results['targets'],
        save_dir=os.path.join(OUTPUT_DIR, 'correlation_analysis')
    )
    
    # 6. Summary
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE!")
    print("=" * 70)
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print(f"\nKey files:")
    print(f"  - Model: {OUTPUT_DIR}/best_model.pth")
    print(f"  - Training curves: {OUTPUT_DIR}/training_curves.png")
    print(f"  - Predictions: {OUTPUT_DIR}/predictions.npy")
    print(f"  - Correlation analysis: {OUTPUT_DIR}/correlation_analysis/")
    print(f"  - Correlation summary: {OUTPUT_DIR}/correlation_analysis/correlation_summary.txt")
    
    return model, results, correlation_results


if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Run main analysis
    model, results, correlation_results = main()