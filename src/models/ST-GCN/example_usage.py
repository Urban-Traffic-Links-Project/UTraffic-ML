"""
Example: Quick Start with T-GCN Model
This script demonstrates how to use the T-GCN model with synthetic data
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from t_gcn_model import TGCN, normalize_adjacency_matrix
from correlation_analysis import CorrelationAnalyzer


def create_synthetic_traffic_data(num_nodes=20, num_timesteps=200, seed=42):
    """
    Create synthetic traffic data for demonstration
    
    Args:
        num_nodes: Number of nodes in the network
        num_timesteps: Number of timesteps
        seed: Random seed
        
    Returns:
        data: Synthetic traffic data
        adj_matrix: Adjacency matrix
    """
    np.random.seed(seed)
    
    # Create random adjacency matrix (connected graph)
    adj_matrix = np.random.rand(num_nodes, num_nodes)
    adj_matrix = (adj_matrix + adj_matrix.T) / 2  # Make symmetric
    adj_matrix = (adj_matrix > 0.7).astype(float)  # Threshold
    
    # Ensure connectivity
    for i in range(num_nodes - 1):
        adj_matrix[i, i+1] = 1
        adj_matrix[i+1, i] = 1
    
    # Create traffic data with temporal and spatial correlation
    data = np.zeros((num_timesteps, num_nodes))
    
    # Initialize with random values
    data[0] = np.random.randn(num_nodes) * 10 + 50
    
    # Generate time series with temporal correlation
    for t in range(1, num_timesteps):
        # Temporal component (AR process)
        data[t] = 0.8 * data[t-1] + np.random.randn(num_nodes) * 5
        
        # Spatial component (diffusion on graph)
        spatial_influence = adj_matrix @ data[t]
        data[t] = 0.7 * data[t] + 0.3 * spatial_influence
        
        # Add periodic pattern (daily/weekly cycle)
        cycle = 20 * np.sin(2 * np.pi * t / 24)
        data[t] += cycle
    
    # Normalize to [0, 1]
    data = (data - data.min()) / (data.max() - data.min())
    
    return data, adj_matrix


def example_model_training():
    """
    Example 1: Train T-GCN model on synthetic data
    """
    print("=" * 70)
    print("EXAMPLE 1: MODEL TRAINING")
    print("=" * 70)
    
    # Create synthetic data
    num_nodes = 20
    num_timesteps = 200
    data, adj_matrix = create_synthetic_traffic_data(num_nodes, num_timesteps)
    
    print(f"\nData created:")
    print(f"  - Nodes: {num_nodes}")
    print(f"  - Timesteps: {num_timesteps}")
    print(f"  - Data shape: {data.shape}")
    print(f"  - Edges: {np.sum(adj_matrix) / 2:.0f}")
    
    # Split train/test
    train_size = int(0.8 * num_timesteps)
    train_data = data[:train_size]
    test_data = data[train_size:]
    
    # Create sequences
    seq_len = 12
    horizon = 1
    
    X_train = []
    y_train = []
    for i in range(len(train_data) - seq_len - horizon + 1):
        X_train.append(train_data[i:i+seq_len])
        y_train.append(train_data[i+seq_len:i+seq_len+horizon].T)
    
    X_train = torch.FloatTensor(np.array(X_train)).unsqueeze(-1)
    y_train = torch.FloatTensor(np.array(y_train))
    
    print(f"\nSequences created:")
    print(f"  - X_train shape: {X_train.shape}")
    print(f"  - y_train shape: {y_train.shape}")
    
    # Create model
    model = TGCN(
        num_nodes=num_nodes,
        input_dim=1,
        hidden_dim=32,
        output_dim=horizon
    )
    
    print(f"\nModel created:")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  - Parameters: {total_params:,}")
    
    # Normalize adjacency matrix
    adj_normalized = normalize_adjacency_matrix(adj_matrix)
    adj_tensor = torch.FloatTensor(adj_normalized)
    
    # Training
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()
    
    print("\nTraining...")
    num_epochs = 50
    losses = []
    
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        
        output = model(X_train, adj_tensor)
        loss = criterion(output, y_train)
        
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}")
    
    # Plot training loss
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Training Loss')
    plt.grid(True, alpha=0.3)
    plt.savefig('example_training_loss.png', dpi=300, bbox_inches='tight')
    print("\n✓ Training loss plot saved: example_training_loss.png")
    
    # Make predictions
    model.eval()
    with torch.no_grad():
        predictions = model(X_train, adj_tensor)
        predictions = predictions.numpy()
    
    print(f"\n✓ Example 1 complete!")
    print(f"  Final loss: {losses[-1]:.6f}")
    
    return model, data, adj_matrix, predictions, y_train.numpy()


def example_correlation_analysis():
    """
    Example 2: Correlation analysis
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 2: CORRELATION ANALYSIS")
    print("=" * 70)
    
    # Create synthetic data
    num_nodes = 15
    num_timesteps = 150
    data, adj_matrix = create_synthetic_traffic_data(num_nodes, num_timesteps)
    
    print(f"\nData created for correlation analysis:")
    print(f"  - Nodes: {num_nodes}")
    print(f"  - Timesteps: {num_timesteps}")
    
    # Create analyzer
    analyzer = CorrelationAnalyzer()
    
    # 1. Node correlation
    print("\n1. Computing node-to-node correlation...")
    data_expanded = data[:, :, np.newaxis]
    corr_matrix, p_values = analyzer.compute_node_correlation_matrix(data)
    
    print(f"   Mean correlation: {corr_matrix[~np.eye(num_nodes, dtype=bool)].mean():.4f}")
    print(f"   Max correlation: {corr_matrix[~np.eye(num_nodes, dtype=bool)].max():.4f}")
    
    # Visualize
    analyzer.visualize_correlation_matrix(
        corr_matrix,
        title="Node-to-Node Correlation",
        save_path='example_node_correlation.png'
    )
    
    # 2. Spatial correlation
    print("\n2. Computing spatial correlation...")
    spatial_corr = analyzer.compute_spatial_correlation(data_expanded, adj_matrix)
    
    non_zero = spatial_corr[spatial_corr != 0]
    if len(non_zero) > 0:
        print(f"   Mean spatial correlation: {non_zero.mean():.4f}")
    
    analyzer.visualize_correlation_matrix(
        spatial_corr,
        title="Spatial Correlation (Weighted by Adjacency)",
        save_path='example_spatial_correlation.png'
    )
    
    # 3. Temporal autocorrelation
    print("\n3. Computing temporal autocorrelation...")
    autocorr = analyzer.compute_temporal_autocorrelation(data, max_lag=10)
    
    print(f"   Mean lag-1 autocorr: {autocorr[:, 1].mean():.4f}")
    print(f"   Mean lag-5 autocorr: {autocorr[:, 5].mean():.4f}")
    
    analyzer.visualize_autocorrelation(
        autocorr,
        node_indices=range(min(5, num_nodes)),
        save_path='example_autocorrelation.png'
    )
    
    print(f"\n✓ Example 2 complete!")
    print(f"  Saved correlation visualizations")


def example_multi_step_prediction():
    """
    Example 3: Multi-step ahead prediction
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 3: MULTI-STEP PREDICTION")
    print("=" * 70)
    
    # Create data
    num_nodes = 10
    num_timesteps = 100
    data, adj_matrix = create_synthetic_traffic_data(num_nodes, num_timesteps)
    
    print(f"\nData created:")
    print(f"  - Nodes: {num_nodes}")
    print(f"  - Timesteps: {num_timesteps}")
    
    # Create model
    model = TGCN(
        num_nodes=num_nodes,
        input_dim=1,
        hidden_dim=32,
        output_dim=1
    )
    
    # Prepare input
    seq_len = 12
    x = torch.FloatTensor(data[-seq_len:]).unsqueeze(0).unsqueeze(-1)
    adj_normalized = normalize_adjacency_matrix(adj_matrix)
    adj_tensor = torch.FloatTensor(adj_normalized)
    
    # Multi-step prediction
    horizon = 6
    print(f"\nPredicting {horizon} steps ahead...")
    
    predictions = model.predict(x, adj_tensor, horizon=horizon)
    predictions = predictions.squeeze().numpy()
    
    print(f"  Prediction shape: {predictions.shape}")
    
    # Visualize predictions for one node
    node_idx = 0
    plt.figure(figsize=(12, 5))
    
    # Plot historical data
    plt.plot(range(seq_len), data[-seq_len:, node_idx], 
             'b-o', label='Historical', linewidth=2, markersize=6)
    
    # Plot predictions
    pred_range = range(seq_len, seq_len + horizon)
    plt.plot(pred_range, predictions[:, node_idx, 0], 
             'r--s', label='Prediction', linewidth=2, markersize=6)
    
    plt.xlabel('Timestep')
    plt.ylabel('Traffic Value')
    plt.title(f'Multi-step Prediction for Node {node_idx}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('example_multistep_prediction.png', dpi=300, bbox_inches='tight')
    
    print("\n✓ Example 3 complete!")
    print(f"  Saved prediction visualization: example_multistep_prediction.png")


def main():
    """
    Run all examples
    """
    print("\n" + "=" * 70)
    print("T-GCN EXAMPLES - QUICK START GUIDE")
    print("=" * 70)
    print("\nThis script demonstrates:")
    print("  1. Model training on synthetic data")
    print("  2. Comprehensive correlation analysis")
    print("  3. Multi-step ahead prediction")
    print("\n" + "=" * 70)
    
    # Run examples
    model, data, adj_matrix, predictions, targets = example_model_training()
    example_correlation_analysis()
    example_multi_step_prediction()
    
    print("\n" + "=" * 70)
    print("ALL EXAMPLES COMPLETE!")
    print("=" * 70)
    print("\nGenerated files:")
    print("  - example_training_loss.png")
    print("  - example_node_correlation.png")
    print("  - example_spatial_correlation.png")
    print("  - example_autocorrelation.png")
    print("  - example_multistep_prediction.png")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    # Set random seed
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Run examples
    main()