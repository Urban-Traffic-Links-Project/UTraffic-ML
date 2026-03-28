"""
Graph Convolutional Network (GCN) Module
Based on T-GCN paper (Zhao et al., 2019)

Fixes vs original:
    - normalize_adj: thêm self-loop TRƯỚC khi tính degree (đúng Kipf 2017)
    - GraphConvolution: hỗ trợ batch matmul an toàn hơn
    - GCN: thêm residual projection khi in_features != out_features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math


class GraphConvolution(nn.Module):
    """
    Single GCN layer: (batch_size, num_nodes, in_features) -> (batch_size, num_nodes, out_features)
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain("relu"))
        if self.bias is not None:
            stdv = 1.0 / math.sqrt(self.out_features)
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x  : (batch, nodes, in_features)  OR  (nodes, in_features)
            adj: (nodes, nodes)  — already normalized
        Returns:
            (batch, nodes, out_features)  OR  (nodes, out_features)
        """
        # Linear transform: (..., nodes, in) @ (in, out) -> (..., nodes, out)
        support = x @ self.weight
        # Graph aggregation: (nodes, nodes) @ (..., nodes, out)
        # Use torch.matmul which broadcasts correctly for batch dims.
        output = torch.matmul(adj, support)
        if self.bias is not None:
            output = output + self.bias
        return output


class GCN(nn.Module):
    """
    2-layer GCN with optional residual connection.
    f(X, A) = σ( Â · ReLU( Â · X · W0 ) · W1 )
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        dropout: float = 0.0,
        residual: bool = True,
    ):
        super().__init__()
        self.gc1 = GraphConvolution(in_features, hidden_features)
        self.gc2 = GraphConvolution(hidden_features, out_features)
        self.dropout = dropout

        # Residual projection khi chiều không khớp
        self.residual = residual
        if residual and in_features != out_features:
            self.res_proj = nn.Linear(in_features, out_features, bias=False)
        else:
            self.res_proj = None

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x  : (batch, nodes, in_features)
            adj: (nodes, nodes)
        Returns:
            (batch, nodes, out_features)
        """
        identity = x

        out = F.relu(self.gc1(x, adj))
        out = F.dropout(out, self.dropout, training=self.training)
        out = self.gc2(out, adj)

        # Residual
        if self.residual:
            if self.res_proj is not None:
                identity = self.res_proj(identity)
            out = out + identity

        return out


# ---------------------------------------------------------------------------
# Adjacency normalization
# ---------------------------------------------------------------------------

def normalize_adj(adj) -> torch.Tensor:
    """
    Symmetric normalization: D^{-1/2} (A + I) D^{-1/2}

    BUG FIX vs original: self-loop phải được thêm TRƯỚC khi tính rowsum/degree,
    không phải sau — đúng theo Kipf & Welling (2017).

    Args:
        adj: (num_nodes, num_nodes) — numpy array hoặc torch.Tensor
    Returns:
        Normalized adjacency matrix as torch.FloatTensor
    """
    is_numpy = not torch.is_tensor(adj)
    if is_numpy:
        adj = torch.FloatTensor(adj)
    else:
        adj = adj.float()

    device = adj.device

    # 1. Thêm self-loop TRƯỚC (A_hat = A + I)
    adj = adj + torch.eye(adj.size(0), device=device)

    # 2. Tính degree
    rowsum = adj.sum(dim=1)                              # [N]
    d_inv_sqrt = rowsum.pow(-0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    D = torch.diag(d_inv_sqrt)                          # [N, N]

    # 3. D^{-1/2} A_hat D^{-1/2}
    adj_norm = D @ adj @ D

    if is_numpy:
        return adj_norm.cpu()
    return adj_norm


# ---------------------------------------------------------------------------
# Smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    batch_size = 32
    num_nodes = 114
    in_features = 1
    hidden_features = 64
    out_features = 64

    model = GCN(in_features, hidden_features, out_features, residual=True)

    x = torch.randn(batch_size, num_nodes, in_features)
    adj_raw = torch.rand(num_nodes, num_nodes)
    adj_raw = (adj_raw + adj_raw.T) / 2
    adj = normalize_adj(adj_raw)

    out = model(x, adj)
    print(f"Input : {x.shape}")
    print(f"Adj   : {adj.shape}")
    print(f"Output: {out.shape}")   # expect [32, 114, 64]