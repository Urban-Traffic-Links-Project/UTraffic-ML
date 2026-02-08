"""
T-GCN: A Temporal Graph Convolutional Network for Traffic Prediction
Based on the paper: "T-GCN: A Temporal Graph Convolutional Network for Traffic Prediction"
by Ling Zhao et al.

This implementation combines Graph Convolutional Network (GCN) and Gated Recurrent Unit (GRU)
to capture both spatial and temporal dependencies in traffic data.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math


class GraphConvolution(nn.Module):
    """
    Graph Convolutional Layer
    Implements the graph convolution operation as described in the T-GCN paper.
    """
    def __init__(self, in_features, out_features, bias=True):
        """
        Args:
            in_features: Number of input features
            out_features: Number of output features
            bias: Whether to use bias term
        """
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Initialize weight matrix
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters using Xavier uniform initialization"""
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
    
    def forward(self, input, adj):
        """
        Forward pass of graph convolution
        
        Args:
            input: Input features (batch_size, num_nodes, in_features)
            adj: Normalized adjacency matrix (num_nodes, num_nodes)
            
        Returns:
            output: Output features (batch_size, num_nodes, out_features)
        """
        # Linear transformation: XW
        support = torch.matmul(input, self.weight)
        
        # Graph convolution: AXW
        output = torch.matmul(adj, support)
        
        if self.bias is not None:
            output = output + self.bias
            
        return output


class GCN(nn.Module):
    """
    2-layer Graph Convolutional Network
    As described in T-GCN paper: f(X,A) = σ(ÃReLU(ÃXW0)W1)
    """
    def __init__(self, in_features, hidden_features, out_features):
        """
        Args:
            in_features: Number of input features
            hidden_features: Number of hidden features
            out_features: Number of output features
        """
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(in_features, hidden_features)
        self.gc2 = GraphConvolution(hidden_features, out_features)
    
    def forward(self, x, adj):
        """
        Forward pass of 2-layer GCN
        
        Args:
            x: Input features (batch_size, num_nodes, in_features)
            adj: Normalized adjacency matrix (num_nodes, num_nodes)
            
        Returns:
            output: Output features (batch_size, num_nodes, out_features)
        """
        x = F.relu(self.gc1(x, adj))
        x = self.gc2(x, adj)
        return x


class TGCNCell(nn.Module):
    """
    T-GCN Cell: Combines GCN and GRU
    
    The T-GCN cell integrates graph convolution into GRU gates:
    - Update gate: ut = σ(Wu[f(A,Xt), ht-1] + bu)
    - Reset gate: rt = σ(Wr[f(A,Xt), ht-1] + br)
    - Candidate: ct = tanh(Wc[f(A,Xt), (rt ⊙ ht-1)] + bc)
    - Hidden state: ht = ut ⊙ ht-1 + (1-ut) ⊙ ct
    """
    def __init__(self, num_nodes, input_dim, hidden_dim, gcn_depth=2):
        """
        Args:
            num_nodes: Number of nodes in the graph
            input_dim: Dimension of input features
            hidden_dim: Dimension of hidden state
            gcn_depth: Depth of GCN (default: 2)
        """
        super(TGCNCell, self).__init__()
        
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # GCN for processing input
        if gcn_depth == 2:
            self.gcn = GCN(input_dim, hidden_dim, hidden_dim)
        else:
            self.gcn = GraphConvolution(input_dim, hidden_dim)
        
        # GRU gates
        # Update gate
        self.weight_u = Parameter(torch.FloatTensor(hidden_dim * 2, hidden_dim))
        self.bias_u = Parameter(torch.FloatTensor(hidden_dim))
        
        # Reset gate
        self.weight_r = Parameter(torch.FloatTensor(hidden_dim * 2, hidden_dim))
        self.bias_r = Parameter(torch.FloatTensor(hidden_dim))
        
        # Candidate hidden state
        self.weight_c = Parameter(torch.FloatTensor(hidden_dim * 2, hidden_dim))
        self.bias_c = Parameter(torch.FloatTensor(hidden_dim))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters"""
        stdv = 1.0 / math.sqrt(self.hidden_dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
    
    def forward(self, x, adj, h):
        """
        Forward pass of T-GCN cell
        
        Args:
            x: Input at current timestep (batch_size, num_nodes, input_dim)
            adj: Normalized adjacency matrix (num_nodes, num_nodes)
            h: Hidden state from previous timestep (batch_size, num_nodes, hidden_dim)
            
        Returns:
            new_h: New hidden state (batch_size, num_nodes, hidden_dim)
        """
        # Apply GCN to input: f(A, Xt)
        gcn_out = self.gcn(x, adj)  # (batch_size, num_nodes, hidden_dim)
        
        # Concatenate GCN output and previous hidden state
        combined = torch.cat([gcn_out, h], dim=-1)  # (batch_size, num_nodes, hidden_dim*2)
        
        # Update gate: ut = σ(Wu[f(A,Xt), ht-1] + bu)
        u = torch.sigmoid(torch.matmul(combined, self.weight_u) + self.bias_u)
        
        # Reset gate: rt = σ(Wr[f(A,Xt), ht-1] + br)
        r = torch.sigmoid(torch.matmul(combined, self.weight_r) + self.bias_r)
        
        # Candidate hidden state: ct = tanh(Wc[f(A,Xt), (rt ⊙ ht-1)] + bc)
        combined_c = torch.cat([gcn_out, r * h], dim=-1)
        c = torch.tanh(torch.matmul(combined_c, self.weight_c) + self.bias_c)
        
        # New hidden state: ht = ut ⊙ ht-1 + (1-ut) ⊙ ct
        new_h = u * h + (1 - u) * c
        
        return new_h


class TGCN(nn.Module):
    """
    T-GCN: Temporal Graph Convolutional Network
    
    Full T-GCN model that processes sequential data through T-GCN cells
    and produces predictions for future timesteps.
    """
    def __init__(self, num_nodes, input_dim, hidden_dim, output_dim, 
                 num_layers=1, gcn_depth=2):
        """
        Args:
            num_nodes: Number of nodes in the graph
            input_dim: Dimension of input features
            hidden_dim: Dimension of hidden state
            output_dim: Dimension of output (prediction horizon)
            num_layers: Number of T-GCN layers (default: 1)
            gcn_depth: Depth of GCN in each cell (default: 2)
        """
        super(TGCN, self).__init__()
        
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        # Create T-GCN cells
        self.tgcn_cells = nn.ModuleList([
            TGCNCell(num_nodes, input_dim if i == 0 else hidden_dim, 
                    hidden_dim, gcn_depth)
            for i in range(num_layers)
        ])
        
        # Fully connected layer for output
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x, adj):
        """
        Forward pass of T-GCN
        
        Args:
            x: Input sequence (batch_size, seq_len, num_nodes, input_dim)
            adj: Normalized adjacency matrix (num_nodes, num_nodes)
            
        Returns:
            output: Predictions (batch_size, num_nodes, output_dim)
        """
        batch_size, seq_len, num_nodes, _ = x.size()
        
        # Initialize hidden states for all layers
        h = [torch.zeros(batch_size, num_nodes, self.hidden_dim).to(x.device)
             for _ in range(self.num_layers)]
        
        # Process sequence
        for t in range(seq_len):
            x_t = x[:, t, :, :]  # (batch_size, num_nodes, input_dim)
            
            # Pass through all T-GCN layers
            for layer in range(self.num_layers):
                h[layer] = self.tgcn_cells[layer](x_t if layer == 0 else h[layer-1], 
                                                   adj, h[layer])
        
        # Use final hidden state for prediction
        output = self.fc(h[-1])  # (batch_size, num_nodes, output_dim)
        
        return output
    
    def predict(self, x, adj, horizon):
        """
        Multi-step ahead prediction
        
        Args:
            x: Input sequence (batch_size, seq_len, num_nodes, input_dim)
            adj: Normalized adjacency matrix (num_nodes, num_nodes)
            horizon: Number of timesteps to predict
            
        Returns:
            predictions: Multi-step predictions (batch_size, horizon, num_nodes, 1)
        """
        self.eval()
        with torch.no_grad():
            batch_size, seq_len, num_nodes, _ = x.size()
            
            # Get initial hidden state by processing input sequence
            h = [torch.zeros(batch_size, num_nodes, self.hidden_dim).to(x.device)
                 for _ in range(self.num_layers)]
            
            for t in range(seq_len):
                x_t = x[:, t, :, :]
                for layer in range(self.num_layers):
                    h[layer] = self.tgcn_cells[layer](
                        x_t if layer == 0 else h[layer-1], adj, h[layer])
            
            # Multi-step prediction
            predictions = []
            current_input = x[:, -1, :, :]  # Last timestep as initial input
            
            for _ in range(horizon):
                # Predict next step
                for layer in range(self.num_layers):
                    h[layer] = self.tgcn_cells[layer](
                        current_input if layer == 0 else h[layer-1], adj, h[layer])
                
                pred = self.fc(h[-1])  # (batch_size, num_nodes, 1)
                predictions.append(pred.unsqueeze(1))
                
                # Use prediction as input for next step
                current_input = pred
            
            predictions = torch.cat(predictions, dim=1)  # (batch_size, horizon, num_nodes, 1)
            
        return predictions


def normalize_adjacency_matrix(adj):
    """
    Normalize adjacency matrix: Ã = D^(-1/2) * (A + I) * D^(-1/2)
    
    Args:
        adj: Adjacency matrix (num_nodes, num_nodes)
        
    Returns:
        normalized_adj: Normalized adjacency matrix
    """
    # Add self-connections
    adj = adj + np.eye(adj.shape[0])
    
    # Compute degree matrix
    d = np.array(adj.sum(1))
    
    # Compute D^(-1/2)
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    
    # Normalize: D^(-1/2) * A * D^(-1/2)
    normalized_adj = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
    
    return normalized_adj


def compute_correlation_matrix(predictions, ground_truth):
    """
    Compute correlation matrix between predictions and ground truth
    Similar to the analysis in TH-Hierarchical paper
    
    Args:
        predictions: Predicted values (num_samples, num_nodes)
        ground_truth: Ground truth values (num_samples, num_nodes)
        
    Returns:
        correlation_matrix: Correlation matrix (num_nodes, num_nodes)
    """
    num_nodes = predictions.shape[1]
    correlation_matrix = np.zeros((num_nodes, num_nodes))
    
    for i in range(num_nodes):
        for j in range(num_nodes):
            # Compute Pearson correlation coefficient
            pred_i = predictions[:, i]
            gt_j = ground_truth[:, j]
            
            corr = np.corrcoef(pred_i, gt_j)[0, 1]
            correlation_matrix[i, j] = corr
    
    return correlation_matrix


def compute_spatial_correlation(features, adj):
    """
    Compute spatial correlation based on graph structure
    Useful for analyzing spatial dependencies like in TH-Hierarchical paper
    
    Args:
        features: Node features (num_samples, num_nodes, num_features)
        adj: Adjacency matrix (num_nodes, num_nodes)
        
    Returns:
        spatial_corr: Spatial correlation matrix (num_nodes, num_nodes)
    """
    num_nodes = features.shape[1]
    spatial_corr = np.zeros((num_nodes, num_nodes))
    
    # Compute correlation weighted by adjacency
    for i in range(num_nodes):
        for j in range(num_nodes):
            if adj[i, j] > 0:  # Only for connected nodes
                feat_i = features[:, i, :].flatten()
                feat_j = features[:, j, :].flatten()
                
                corr = np.corrcoef(feat_i, feat_j)[0, 1]
                spatial_corr[i, j] = corr * adj[i, j]
    
    return spatial_corr


def compute_temporal_correlation(features, lag=1):
    """
    Compute temporal correlation (autocorrelation) for each node
    
    Args:
        features: Time series features (num_timesteps, num_nodes, num_features)
        lag: Time lag for correlation computation
        
    Returns:
        temporal_corr: Temporal correlation for each node (num_nodes,)
    """
    num_timesteps, num_nodes, num_features = features.shape
    temporal_corr = np.zeros(num_nodes)
    
    for i in range(num_nodes):
        node_series = features[:, i, :].flatten()
        
        # Compute autocorrelation at given lag
        if len(node_series) > lag:
            series_1 = node_series[:-lag]
            series_2 = node_series[lag:]
            
            if len(series_1) > 0 and len(series_2) > 0:
                corr = np.corrcoef(series_1, series_2)[0, 1]
                temporal_corr[i] = corr
    
    return temporal_corr


class TGCNTrainer:
    """
    Trainer class for T-GCN model
    """
    def __init__(self, model, optimizer, criterion, device='cuda'):
        """
        Args:
            model: T-GCN model
            optimizer: Optimizer
            criterion: Loss function
            device: Device to use ('cuda' or 'cpu')
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        
        self.model.to(device)
    
    def train_epoch(self, train_loader, adj):
        """
        Train for one epoch
        
        Args:
            train_loader: Training data loader
            adj: Normalized adjacency matrix
            
        Returns:
            avg_loss: Average loss for the epoch
        """
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        adj = torch.FloatTensor(adj).to(self.device)
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(batch_x, adj)
            
            # Compute loss
            loss = self.criterion(output, batch_y)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def evaluate(self, test_loader, adj):
        """
        Evaluate model on test data
        
        Args:
            test_loader: Test data loader
            adj: Normalized adjacency matrix
            
        Returns:
            avg_loss: Average loss
            predictions: All predictions
            targets: All targets
        """
        self.model.eval()
        total_loss = 0
        num_batches = 0
        all_predictions = []
        all_targets = []
        
        adj = torch.FloatTensor(adj).to(self.device)
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                # Forward pass
                output = self.model(batch_x, adj)
                
                # Compute loss
                loss = self.criterion(output, batch_y)
                
                total_loss += loss.item()
                num_batches += 1
                
                all_predictions.append(output.cpu().numpy())
                all_targets.append(batch_y.cpu().numpy())
        
        avg_loss = total_loss / num_batches
        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        
        return avg_loss, predictions, targets


if __name__ == "__main__":
    # Example usage
    print("T-GCN Model Implementation")
    print("=" * 50)
    
    # Model parameters
    num_nodes = 10
    input_dim = 1
    hidden_dim = 64
    output_dim = 1
    seq_len = 12
    batch_size = 32
    
    # Create random data for testing
    x = torch.randn(batch_size, seq_len, num_nodes, input_dim)
    adj = np.random.rand(num_nodes, num_nodes)
    adj = (adj + adj.T) / 2  # Make symmetric
    adj = normalize_adjacency_matrix(adj)
    adj_tensor = torch.FloatTensor(adj)
    
    # Create model
    model = TGCN(num_nodes, input_dim, hidden_dim, output_dim)
    
    print(f"Model created with:")
    print(f"  - Number of nodes: {num_nodes}")
    print(f"  - Input dimension: {input_dim}")
    print(f"  - Hidden dimension: {hidden_dim}")
    print(f"  - Output dimension: {output_dim}")
    print(f"  - Sequence length: {seq_len}")
    
    # Forward pass
    output = model(x, adj_tensor)
    print(f"\nOutput shape: {output.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")