"""

Dynamic Adjacency Matrix Builder for DTC-STGCN

Based on: Xu et al. "Dynamic Traffic Correlations based Spatio-Temporal GCN" (2023)

Information Sciences 621 (2023) 580-595



Three methods for computing dynamic adjacency matrix (Equations 2-6):

    Method 1 (TN): Number of vehicles transferring between roads  [Eq. 2]

    Method 2 (FR): Feature ratio                                   [Eq. 3]

    Method 3 (FD): Absolute feature difference                     [Eq. 4]



Final dynamic adjacency: At = St ⊙ A  (Hadamard product with fixed connectivity)

"""

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
class DynamicAdjacencyBuilder(nn.Module):
    """
    Compute dynamic adjacency matrix At from traffic features.
    
    At = St ⊙ A  (Eq. 5)
    where A is the fixed connectivity matrix (0/1), ⊙ is Hadamard product.
    
    St is computed by one of three methods:
    - FR (Feature Ratio):    St_ij = F_i^t / (F_i^t + F_j^t)   [Eq. 3]
    - FD (Feature Diff):     St_ij = |F_i^t - F_j^t|             [Eq. 4]
    - TN (Transfer Number):  St_ij = N_ij^t  (if trajectory data available) [Eq. 2]
    """
    def __init__(self, fixed_adj, method="FD", normalize=True, eps=1e-8):
        """
        Args:
            fixed_adj: (num_nodes, num_nodes) fixed binary connectivity matrix
            method: "FR" | "FD" | "TN"
            normalize: Whether to normalize the dynamic adjacency
            eps: Small value to prevent division by zero
        """
        super(DynamicAdjacencyBuilder, self).__init__()
        self.method = method.upper()
        self.normalize = normalize
        self.eps = eps
        
        # Register fixed adjacency (binary connectivity)
        fixed_adj_tensor = torch.FloatTensor(fixed_adj)
        self.register_buffer("fixed_adj", fixed_adj_tensor)
        self.num_nodes = fixed_adj.shape[0]

    def forward(self, F_t):
        """
        Compute dynamic adjacency At for current timestep features.
        
        Args:
            F_t: traffic feature at timestep t.
                 Supported shapes:
                   - (batch_size, num_nodes)
                   - (batch_size, num_nodes, 1)
                   - (batch_size, num_nodes, num_features)
        Returns:
            At: (batch_size, num_nodes, num_nodes) dynamic adjacency matrix
        """
        # DTC-STGCN paper defines St based on a *scalar* traffic feature per node (e.g., speed).
        # Our pipeline often uses multi-feature inputs; default to the first channel as "speed".
        if F_t.dim() == 3:
            if F_t.shape[-1] == 1:
                F_t = F_t.squeeze(-1)
            else:
                F_t = F_t[..., 0]
        elif F_t.dim() != 2:
            raise ValueError(f"F_t must have 2 or 3 dims, got shape={tuple(F_t.shape)}")
        
        batch_size, N = F_t.shape

        # IMPORTANT NUMERICAL STABILITY NOTE:
        # In this repo, inputs are often normalized (e.g., z-score) => values can be negative and near 0.
        # The FR formula St_ij = Fi / (Fi + Fj) becomes unstable when (Fi + Fj) ≈ 0 or changes sign,
        # producing huge magnitudes/NaN/Inf and exploding training loss.
        # To keep FR well-behaved, map the scalar feature to (0, 1) before building St.
        if self.method == "FR":
            F_t = torch.sigmoid(F_t)

        # 1. Tính ma trận tương quan St (chọn FD hoặc FR)
        St = self._compute_correlation(F_t, N)  # (batch, N, N)

        # Apply Hadamard product with fixed adjacency: At = St ⊙ A
        At = St * self.fixed_adj.unsqueeze(0)  # (batch, N, N)

        # Clean up any potential NaN/Inf from upstream numeric issues
        At = torch.nan_to_num(At, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Normalize if requested
        if self.normalize:
            At = self._normalize(At)

        return At
    
    def _compute_correlation(self, F_t, N):
        """Compute spatial correlation matrix St"""
        if self.method == "FR":
            return self._feature_ratio(F_t, N)
        elif self.method == "FD":
            return self._feature_difference(F_t, N)
        elif self.method == "TN":
            return self._transfer_number(F_t, N)
        else:
            raise ValueError(f"Unknown method: {self.method}. Choose from FR, FD, TN.")

    def _feature_ratio(self, F_t, N):
        """
        Equation 3: St_ij = F_i^t / (F_i^t + F_j^t)
        Captures relative proportion of feature between road segments.
        """
        Fi = F_t.unsqueeze(2).expand(-1, -1, N)  # (batch, N, N) - row broadcast
        Fj = F_t.unsqueeze(1).expand(-1, N, -1)  # (batch, N, N) - col broadcast

        St = Fi / (Fi + Fj + self.eps)
        return St

    def _feature_difference(self, F_t, N):
        """
        Equation 4: St_ij = |F_i^t - F_j^t|
        Captures absolute influence/difference between road segments.
        """
        Fi = F_t.unsqueeze(2).expand(-1, -1, N)
        Fj = F_t.unsqueeze(1).expand(-1, N, -1)

        St = torch.abs(Fi - Fj)
        return St

    def _transfer_number(self, F_t, N):
        """
        Equation 2: St_ij = N_ij^t (vehicle transfer count)
        Used when trajectory data is available.
        Falls back to FR method if trajectory data unavailable.
        """
        # Without trajectory data, use flow-based approximation:
        # Vehicles tend to transfer from higher-flow to lower-flow roads
        Fi = F_t.unsqueeze(2).expand(-1, -1, N)
        Fj = F_t.unsqueeze(1).expand(-1, N, -1)
        St = F.relu(Fi - Fj)  # Transfer from i to j when Fi > Fj
        return St


    def _normalize(self, At):
        """
        Symmetric normalization: D^(-1/2) * A * D^(-1/2)
        Applied per batch sample.
        """
        # Row sum (degree)
        deg = At.sum(dim=-1, keepdim=True).clamp(min=self.eps)  # (batch, N, 1)
        d_inv_sqrt = deg.pow(-0.5)
        At_norm = At * d_inv_sqrt * d_inv_sqrt.transpose(1, 2)
        return At_norm

if __name__ == "__main__":
    batch_size = 4
    N = 10
    fixed_adj = (np.random.rand(N, N) > 0.5).astype(np.float32)
    fixed_adj = np.clip(fixed_adj + fixed_adj.T, 0, 1)
    np.fill_diagonal(fixed_adj, 0)

    F_t = torch.rand(batch_size, N, 1)

    for method in ["FR", "FD"]:
        builder = DynamicAdjacencyBuilder(fixed_adj, method=method)
        At = builder(F_t)
        print(f"Method {method}: At shape={At.shape}, min={At.min():.3f}, max={At.max():.3f}")

    print("✓ DynamicAdjacencyBuilder OK")