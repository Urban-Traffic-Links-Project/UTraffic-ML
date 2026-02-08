"""
Correlation Analysis for T-GCN Model
Computes various correlation matrices as described in TH-Hierarchical paper:
- Spatial correlation between nodes
- Temporal correlation (autocorrelation)
- Prediction-truth correlation
- Regional correlation (if hierarchical structure provided)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import torch
from t_gcn_model import TGCN, normalize_adjacency_matrix


class CorrelationAnalyzer:
    """
    Analyzer for computing and visualizing various correlation matrices
    """
    def __init__(self, model=None, device='cuda'):
        """
        Args:
            model: Trained T-GCN model (optional)
            device: Device to use
        """
        self.model = model
        self.device = device
        
    def compute_node_correlation_matrix(self, data):
        """
        Compute correlation matrix between all nodes
        
        Args:
            data: Time series data (num_timesteps, num_nodes) or (num_timesteps, num_nodes, num_features)
            
        Returns:
            correlation_matrix: Correlation matrix (num_nodes, num_nodes)
        """
        if len(data.shape) == 3:
            # Average over features
            data = data.mean(axis=2)
        
        num_nodes = data.shape[1]
        correlation_matrix = np.zeros((num_nodes, num_nodes))
        p_values = np.zeros((num_nodes, num_nodes))
        
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i == j:
                    correlation_matrix[i, j] = 1.0
                    p_values[i, j] = 0.0
                else:
                    corr, p_val = pearsonr(data[:, i], data[:, j])
                    correlation_matrix[i, j] = corr
                    p_values[i, j] = p_val
        
        return correlation_matrix, p_values
    
    def compute_spatial_correlation(self, data, adj_matrix, threshold=0.0):
        """
        Compute spatial correlation weighted by adjacency matrix
        
        Args:
            data: Time series data (num_timesteps, num_nodes, num_features)
            adj_matrix: Adjacency matrix (num_nodes, num_nodes)
            threshold: Threshold for adjacency (default: 0.0)
            
        Returns:
            spatial_corr: Spatial correlation matrix (num_nodes, num_nodes)
        """
        if len(data.shape) == 3:
            # Flatten features
            num_timesteps, num_nodes, num_features = data.shape
            data_flat = data.reshape(num_timesteps, num_nodes, -1)
        else:
            data_flat = data[:, :, np.newaxis]
            num_nodes = data.shape[1]
        
        spatial_corr = np.zeros((num_nodes, num_nodes))
        
        for i in range(num_nodes):
            for j in range(num_nodes):
                if adj_matrix[i, j] > threshold:
                    feat_i = data_flat[:, i, :].flatten()
                    feat_j = data_flat[:, j, :].flatten()
                    
                    if np.std(feat_i) > 0 and np.std(feat_j) > 0:
                        corr, _ = pearsonr(feat_i, feat_j)
                        spatial_corr[i, j] = corr * adj_matrix[i, j]
        
        return spatial_corr
    
    def compute_temporal_autocorrelation(self, data, max_lag=10):
        """
        Compute temporal autocorrelation for each node
        
        Args:
            data: Time series data (num_timesteps, num_nodes) or (num_timesteps, num_nodes, num_features)
            max_lag: Maximum lag to compute
            
        Returns:
            autocorr: Autocorrelation values (num_nodes, max_lag)
        """
        if len(data.shape) == 3:
            data = data.mean(axis=2)
        
        num_timesteps, num_nodes = data.shape
        autocorr = np.zeros((num_nodes, max_lag))
        
        for i in range(num_nodes):
            series = data[:, i]
            series_mean = np.mean(series)
            series_std = np.std(series)
            
            if series_std > 0:
                for lag in range(max_lag):
                    if lag < num_timesteps:
                        # Compute autocorrelation at lag
                        c0 = np.sum((series[:-lag if lag > 0 else None] - series_mean) * 
                                   (series[lag:] - series_mean))
                        autocorr[i, lag] = c0 / ((num_timesteps - lag) * series_std ** 2)
        
        return autocorr
    
    def compute_prediction_correlation(self, predictions, ground_truth):
        """
        Compute correlation between predictions and ground truth
        
        Args:
            predictions: Predicted values (num_samples, num_nodes) or (num_samples, num_nodes, horizon)
            ground_truth: Ground truth values (num_samples, num_nodes) or (num_samples, num_nodes, horizon)
            
        Returns:
            corr_matrix: Correlation matrix (num_nodes, num_nodes)
            node_correlations: Per-node correlation (num_nodes,)
        """
        if len(predictions.shape) == 3:
            # Average over horizon
            predictions = predictions.mean(axis=2)
            ground_truth = ground_truth.mean(axis=2)
        
        num_nodes = predictions.shape[1]
        
        # Node-wise correlation (diagonal)
        node_correlations = np.zeros(num_nodes)
        for i in range(num_nodes):
            if np.std(predictions[:, i]) > 0 and np.std(ground_truth[:, i]) > 0:
                corr, _ = pearsonr(predictions[:, i], ground_truth[:, i])
                node_correlations[i] = corr
        
        # Cross-node correlation matrix
        corr_matrix = np.zeros((num_nodes, num_nodes))
        for i in range(num_nodes):
            for j in range(num_nodes):
                if np.std(predictions[:, i]) > 0 and np.std(ground_truth[:, j]) > 0:
                    corr, _ = pearsonr(predictions[:, i], ground_truth[:, j])
                    corr_matrix[i, j] = corr
        
        return corr_matrix, node_correlations
    
    def compute_regional_correlation(self, data, region_labels):
        """
        Compute correlation between regions (hierarchical structure)
        
        Args:
            data: Time series data (num_timesteps, num_nodes)
            region_labels: Region assignment for each node (num_nodes,)
            
        Returns:
            regional_corr: Regional correlation matrix (num_regions, num_regions)
        """
        if len(data.shape) == 3:
            data = data.mean(axis=2)
        
        unique_regions = np.unique(region_labels)
        num_regions = len(unique_regions)
        
        # Aggregate data by region (mean)
        regional_data = np.zeros((data.shape[0], num_regions))
        for i, region in enumerate(unique_regions):
            mask = region_labels == region
            regional_data[:, i] = data[:, mask].mean(axis=1)
        
        # Compute correlation between regions
        regional_corr = np.corrcoef(regional_data.T)
        
        return regional_corr
    
    def visualize_correlation_matrix(self, corr_matrix, title="Correlation Matrix", 
                                     save_path=None, figsize=(10, 8), annot=False):
        """
        Visualize correlation matrix as heatmap
        
        Args:
            corr_matrix: Correlation matrix to visualize
            title: Title of the plot
            save_path: Path to save the figure (optional)
            figsize: Figure size
            annot: Whether to annotate cells with values
        """
        plt.figure(figsize=figsize)
        
        # Create mask for upper triangle (optional)
        # mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        sns.heatmap(corr_matrix, 
                   annot=annot,
                   fmt='.2f' if annot else None,
                   cmap='RdBu_r',
                   center=0,
                   vmin=-1, vmax=1,
                   square=True,
                   linewidths=0.5,
                   cbar_kws={"shrink": 0.8, "label": "Correlation"})
        
        plt.title(title, fontsize=14, fontweight='bold', pad=20)
        plt.xlabel('Node Index', fontsize=12)
        plt.ylabel('Node Index', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved correlation matrix to {save_path}")
        
        plt.show()
    
    def visualize_autocorrelation(self, autocorr, node_indices=None, 
                                  save_path=None, figsize=(12, 6)):
        """
        Visualize temporal autocorrelation
        
        Args:
            autocorr: Autocorrelation values (num_nodes, max_lag)
            node_indices: Indices of nodes to plot (optional, default: all)
            save_path: Path to save the figure
            figsize: Figure size
        """
        if node_indices is None:
            node_indices = range(min(10, autocorr.shape[0]))  # Plot first 10 nodes
        
        plt.figure(figsize=figsize)
        
        for idx in node_indices:
            plt.plot(autocorr[idx, :], marker='o', label=f'Node {idx}', alpha=0.7)
        
        plt.xlabel('Lag', fontsize=12)
        plt.ylabel('Autocorrelation', fontsize=12)
        plt.title('Temporal Autocorrelation by Node', fontsize=14, fontweight='bold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved autocorrelation plot to {save_path}")
        
        plt.show()
    
    def generate_correlation_report(self, data, adj_matrix, predictions=None, 
                                   ground_truth=None, region_labels=None,
                                   save_dir='./correlation_analysis'):
        """
        Generate comprehensive correlation analysis report
        
        Args:
            data: Time series data (num_timesteps, num_nodes, num_features)
            adj_matrix: Adjacency matrix
            predictions: Model predictions (optional)
            ground_truth: Ground truth for predictions (optional)
            region_labels: Regional structure (optional)
            save_dir: Directory to save results
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        print("=" * 70)
        print("CORRELATION ANALYSIS REPORT")
        print("=" * 70)
        
        # 1. Node-to-Node Correlation
        print("\n1. Computing Node-to-Node Correlation...")
        node_corr, p_values = self.compute_node_correlation_matrix(data)
        
        print(f"   Mean correlation: {node_corr[~np.eye(node_corr.shape[0], dtype=bool)].mean():.4f}")
        print(f"   Std correlation: {node_corr[~np.eye(node_corr.shape[0], dtype=bool)].std():.4f}")
        print(f"   Max correlation: {node_corr[~np.eye(node_corr.shape[0], dtype=bool)].max():.4f}")
        print(f"   Min correlation: {node_corr[~np.eye(node_corr.shape[0], dtype=bool)].min():.4f}")
        
        # Save correlation matrix
        np.save(os.path.join(save_dir, 'node_correlation_matrix.npy'), node_corr)
        self.visualize_correlation_matrix(
            node_corr, 
            title="Node-to-Node Correlation Matrix",
            save_path=os.path.join(save_dir, 'node_correlation_matrix.png'),
            annot=False
        )
        
        # 2. Spatial Correlation
        print("\n2. Computing Spatial Correlation (Weighted by Adjacency)...")
        spatial_corr = self.compute_spatial_correlation(data, adj_matrix)
        
        non_zero_spatial = spatial_corr[spatial_corr != 0]
        if len(non_zero_spatial) > 0:
            print(f"   Mean spatial correlation: {non_zero_spatial.mean():.4f}")
            print(f"   Std spatial correlation: {non_zero_spatial.std():.4f}")
        
        np.save(os.path.join(save_dir, 'spatial_correlation_matrix.npy'), spatial_corr)
        self.visualize_correlation_matrix(
            spatial_corr,
            title="Spatial Correlation Matrix (Weighted by Adjacency)",
            save_path=os.path.join(save_dir, 'spatial_correlation_matrix.png')
        )
        
        # 3. Temporal Autocorrelation
        print("\n3. Computing Temporal Autocorrelation...")
        autocorr = self.compute_temporal_autocorrelation(data, max_lag=10)
        
        print(f"   Mean lag-1 autocorrelation: {autocorr[:, 1].mean():.4f}")
        print(f"   Mean lag-5 autocorrelation: {autocorr[:, 5].mean():.4f}")
        
        np.save(os.path.join(save_dir, 'temporal_autocorrelation.npy'), autocorr)
        self.visualize_autocorrelation(
            autocorr,
            save_path=os.path.join(save_dir, 'temporal_autocorrelation.png')
        )
        
        # 4. Prediction Correlation (if available)
        if predictions is not None and ground_truth is not None:
            print("\n4. Computing Prediction-Truth Correlation...")
            pred_corr, node_pred_corr = self.compute_prediction_correlation(
                predictions, ground_truth)
            
            print(f"   Mean node-wise correlation: {node_pred_corr.mean():.4f}")
            print(f"   Std node-wise correlation: {node_pred_corr.std():.4f}")
            print(f"   Best node correlation: {node_pred_corr.max():.4f} (Node {node_pred_corr.argmax()})")
            print(f"   Worst node correlation: {node_pred_corr.min():.4f} (Node {node_pred_corr.argmin()})")
            
            np.save(os.path.join(save_dir, 'prediction_correlation_matrix.npy'), pred_corr)
            np.save(os.path.join(save_dir, 'node_prediction_correlations.npy'), node_pred_corr)
            
            self.visualize_correlation_matrix(
                pred_corr,
                title="Prediction-Truth Correlation Matrix",
                save_path=os.path.join(save_dir, 'prediction_correlation_matrix.png')
            )
        
        # 5. Regional Correlation (if available)
        if region_labels is not None:
            print("\n5. Computing Regional Correlation...")
            regional_corr = self.compute_regional_correlation(data, region_labels)
            
            print(f"   Number of regions: {regional_corr.shape[0]}")
            print(f"   Mean inter-regional correlation: {regional_corr[~np.eye(regional_corr.shape[0], dtype=bool)].mean():.4f}")
            
            np.save(os.path.join(save_dir, 'regional_correlation_matrix.npy'), regional_corr)
            self.visualize_correlation_matrix(
                regional_corr,
                title="Regional Correlation Matrix",
                save_path=os.path.join(save_dir, 'regional_correlation_matrix.png'),
                annot=True
            )
        
        # Create summary report
        summary_file = os.path.join(save_dir, 'correlation_summary.txt')
        with open(summary_file, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("CORRELATION ANALYSIS SUMMARY\n")
            f.write("=" * 70 + "\n\n")
            
            f.write("1. NODE-TO-NODE CORRELATION\n")
            f.write(f"   Mean: {node_corr[~np.eye(node_corr.shape[0], dtype=bool)].mean():.4f}\n")
            f.write(f"   Std: {node_corr[~np.eye(node_corr.shape[0], dtype=bool)].std():.4f}\n")
            f.write(f"   Range: [{node_corr[~np.eye(node_corr.shape[0], dtype=bool)].min():.4f}, "
                   f"{node_corr[~np.eye(node_corr.shape[0], dtype=bool)].max():.4f}]\n\n")
            
            if len(non_zero_spatial) > 0:
                f.write("2. SPATIAL CORRELATION\n")
                f.write(f"   Mean: {non_zero_spatial.mean():.4f}\n")
                f.write(f"   Std: {non_zero_spatial.std():.4f}\n\n")
            
            f.write("3. TEMPORAL AUTOCORRELATION\n")
            f.write(f"   Lag-1 Mean: {autocorr[:, 1].mean():.4f}\n")
            f.write(f"   Lag-5 Mean: {autocorr[:, 5].mean():.4f}\n\n")
            
            if predictions is not None and ground_truth is not None:
                f.write("4. PREDICTION CORRELATION\n")
                f.write(f"   Mean: {node_pred_corr.mean():.4f}\n")
                f.write(f"   Std: {node_pred_corr.std():.4f}\n")
                f.write(f"   Best: {node_pred_corr.max():.4f} (Node {node_pred_corr.argmax()})\n")
                f.write(f"   Worst: {node_pred_corr.min():.4f} (Node {node_pred_corr.argmin()})\n\n")
        
        print(f"\n✓ Correlation analysis complete! Results saved to {save_dir}")
        print(f"✓ Summary report: {summary_file}")
        
        return {
            'node_correlation': node_corr,
            'spatial_correlation': spatial_corr,
            'temporal_autocorrelation': autocorr,
            'prediction_correlation': pred_corr if predictions is not None else None,
            'node_prediction_correlations': node_pred_corr if predictions is not None else None,
            'regional_correlation': regional_corr if region_labels is not None else None
        }


if __name__ == "__main__":
    # Example usage
    print("Correlation Analysis Example")
    print("=" * 70)
    
    # Generate sample data
    num_timesteps = 100
    num_nodes = 20
    num_features = 1
    
    # Create synthetic data
    np.random.seed(42)
    data = np.random.randn(num_timesteps, num_nodes, num_features)
    
    # Add temporal correlation
    for i in range(1, num_timesteps):
        data[i] = 0.7 * data[i-1] + 0.3 * data[i]
    
    # Create adjacency matrix
    adj = np.random.rand(num_nodes, num_nodes)
    adj = (adj + adj.T) / 2  # Make symmetric
    adj = (adj > 0.7).astype(float)  # Threshold
    
    # Create analyzer
    analyzer = CorrelationAnalyzer()
    
    # Generate comprehensive report
    results = analyzer.generate_correlation_report(
        data=data,
        adj_matrix=adj,
        save_dir='./example_correlation_analysis'
    )
    
    print("\nExample analysis complete!")