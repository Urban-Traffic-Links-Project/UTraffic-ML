"""
DTC-STGCN: Dynamic Traffic Correlation-based Spatio-Temporal Graph Convolutional Network
Based on: Xu et al. "Dynamic traffic correlations based spatio-temporal GCN" (2023)
Information Sciences 621 (2023) 580-595

Architecture (Figure 3 in paper):
    For each timestep t=1..n:
        1. Compute dynamic adjacency At via DynamicAdjacencyBuilder
        2. Compute spatial attention Pt (Eq. 11-12)
        3. Apply 3 GCN layers → spatial features Ht ∈ R^{N×e}
    
    4. Concatenate spatial features: Hs = H1 ⊕ H2 ⊕ ... ⊕ Hn ∈ R^{N×(e·n)}
    5. Feed historical observations On into LSTM → temporal features Ft ∈ R^{N×b}
    
    6. Concatenate road info features Fr ∈ R^{N×c} (lanes, direction, type)
    7. Combine: Fo = Fs ⊕ Ft ⊕ Fr (or Fs ⊕ Ft if no road info)
    8. Spatio-temporal attention Ro (Eq. 25)
    9. 2 more GCN layers with Fo → Ho
    10. FC → Output
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .graph.graph_builder import DynamicAdjacencyBuilder


class SpatialAttention(nn.Module):
    """
    Spatial Attention mechanism (Equations 11-12 in DTC-STGCN paper).

    Pt = Vs · sigma((Ft-1 · Ws) · Wp · (Wa · At-1) + bs)
    Pt_ij = softmax(Pt_ij)

    Paper assumes scalar input per node (speed only, in_features=1).
    When in_features > 1, feat_proj reduces F to (batch, N, 1) first so that
    Ws can remain (1, N) and term1 = F @ Ws gives (batch, N, N) as expected.
    """

    def __init__(self, num_nodes, in_features, hidden_size=None):
        super(SpatialAttention, self).__init__()
        self.num_nodes = num_nodes
        self.in_features = in_features

        # Project multi-feature input to 1 scalar per node before attention.
        # If in_features==1 this is skipped and F_prev is used directly.
        self.feat_proj = nn.Linear(in_features, 1, bias=False) if in_features > 1 else None

        # All weight matrices are N x N as in the paper.
        # Ws shape is (1, N): maps (batch,N,1) -> (batch,N,N) via matmul.
        self.Vs = nn.Parameter(torch.FloatTensor(num_nodes, num_nodes))
        self.Ws = nn.Parameter(torch.FloatTensor(1, num_nodes))   # (1, N)
        self.Wp = nn.Parameter(torch.FloatTensor(num_nodes, num_nodes))
        self.Wa = nn.Parameter(torch.FloatTensor(num_nodes, num_nodes))
        self.bs = nn.Parameter(torch.FloatTensor(num_nodes, num_nodes))

        nn.init.xavier_uniform_(self.Vs)
        nn.init.xavier_uniform_(self.Ws)
        nn.init.xavier_uniform_(self.Wp)
        nn.init.xavier_uniform_(self.Wa)
        nn.init.zeros_(self.bs)

    def forward(self, F_prev, A_prev):
        """
        Args:
            F_prev: (batch, N, in_features)
            A_prev: (batch, N, N)
        Returns:
            Pt: (batch, N, N)
        """
        # Step 1: reduce to scalar per node -> (batch, N, 1)
        if self.feat_proj is not None:
            F_scalar = self.feat_proj(F_prev)   # (batch, N, 1)
        else:
            F_scalar = F_prev                    # already (batch, N, 1)

        # Step 2: Pt = Vs * sigma(F*Ws * Wp * (Wa*A) + bs)
        # (batch,N,1) @ (1,N) -> (batch,N,N)
        term1 = torch.matmul(F_scalar, self.Ws)      # (batch, N, N)
        term2 = torch.matmul(term1, self.Wp)          # (batch, N, N)
        term3 = torch.matmul(self.Wa, A_prev)         # (batch, N, N)
        pre_sig = torch.matmul(term2, term3) + self.bs
        Pt = torch.matmul(torch.sigmoid(pre_sig), self.Vs)  # (batch, N, N)

        # Step 3: row-wise softmax (Eq. 12)
        Pt = F.softmax(Pt, dim=-1)
        return Pt


class GCNLayer(nn.Module):
    """
    Single GCN layer with dynamic adjacency support.
    
    Equation (10): f(H^i_t, At) = ReLU(D̃^{-1/2} · Ã_t · D̃^{-1/2} · H^{i-1}_t · W^i_t)
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GCNLayer, self).__init__()
        self.W = nn.Linear(in_features, out_features, bias=bias)
        nn.init.xavier_uniform_(self.W.weight)

    def forward(self, H, A):
        """
        Args:
            H: (batch, N, in_features)
            A: (batch, N, N) dynamic adjacency
        Returns:
            (batch, N, out_features)
        """
        # Graph convolution: A · H · W
        support = self.W(H)           # (batch, N, out_features)
        output = torch.bmm(A, support)  # (batch, N, out_features)
        return F.leaky_relu(output, negative_slope=0.01)


class AttentionGCNBlock(nn.Module):
    """
    Attention-based GCN block with 3 layers (as in DTC-STGCN Figure 3).
    
    Process:
        1. Compute spatial attention Pt from previous features and adjacency
        2. Adjust first GCN layer with attention: A_attended = At * Pt
        3. Pass through 3 GCN layers to get spatial features Ht ∈ R^{N×e}
    """

    def __init__(self, num_nodes, in_features, gcn_hidden, out_features):
        """
        Args:
            num_nodes: Number of graph nodes
            in_features: Input feature dimension (e.g., 1 for speed)
            gcn_hidden: Hidden dimension of GCN layers (e.g., 32)
            out_features: Output dimension e per timestep
        """
        super(AttentionGCNBlock, self).__init__()

        self.spatial_attn = SpatialAttention(num_nodes, in_features)

        # 3 GCN layers (as in paper)
        self.gcn1 = GCNLayer(in_features, gcn_hidden)
        self.gcn2 = GCNLayer(gcn_hidden, gcn_hidden)
        self.gcn3 = GCNLayer(gcn_hidden, out_features)

    def forward(self, F_t, At, F_prev=None, A_prev=None):
        """
        Args:
            F_t: (batch, N, in_features) current timestep features
            At: (batch, N, N) dynamic adjacency at t
            F_prev: (batch, N, in_features) previous features for attention (optional)
            A_prev: (batch, N, N) previous adjacency (optional)
        Returns:
            Ht: (batch, N, out_features) spatial features
        """
        # Spatial attention: Pt from previous timestep info
        if F_prev is not None and A_prev is not None:
            Pt = self.spatial_attn(F_prev, A_prev)  # (batch, N, N)
            # Combine dynamic adjacency with spatial attention
            A_attended = At * Pt
        else:
            A_attended = At

        # 3 GCN layers
        H1 = self.gcn1(F_t, A_attended)  # With attention on first layer
        H2 = self.gcn2(H1, At)           # Standard on remaining layers
        H3 = self.gcn3(H2, At)

        return H3


class SpatioTemporalAttention(nn.Module):
    """
    Spatio-temporal attention Ro (Equation 25).
    
    Ro = softmax(tanh(wo · Fo + bo))
    
    Adaptively adjusts contribution of spatial, temporal, and road features.
    """

    def __init__(self, in_features):
        super(SpatioTemporalAttention, self).__init__()
        self.attn = nn.Sequential(
            nn.Linear(in_features, in_features, bias=True),
            nn.Tanh(),
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, Fo):
        """
        Args:
            Fo: (batch, N, a+b+c) concatenated features
        Returns:
            Ro: (batch, N, a+b+c) attention weights
        """
        Ro = self.softmax(self.attn(Fo))
        return Ro


class DTCSTGCN(nn.Module):
    """
    DTC-STGCN: Dynamic Traffic Correlations based Spatio-Temporal GCN
    
    Four main components:
    1. Dynamic adjacency matrix (At) via DynamicAdjacencyBuilder
    2. Attention-based GCN → dynamic spatial features Hs
    3. LSTM → dynamic temporal features Ft
    4. Hybrid attention-based GCN (Hs + Ft + Fr) → prediction
    """

    def __init__(
        self,
        num_nodes,
        input_dim,
        hidden_dim,
        output_dim,
        seq_len,
        pred_len,
        adj,
        gcn_hidden=32,
        gcn_out=64,         # spatial feature dim 'a' per timestep
        lstm_hidden=64,     # temporal feature dim 'b' (same as hidden_dim usually)
        hybrid_hidden1=32,
        hybrid_hidden2=16,
        dynamic_method="FD",
        dropout=0.0,
    ):
        """
        Args:
            num_nodes: Number of road segments
            input_dim: Input feature dimension (1 for speed)
            hidden_dim: General hidden dimension
            output_dim: Output dimension (1)
            seq_len: Historical observation horizon n
            pred_len: Prediction horizon k (1, 2, 3, or 4 steps)
            adj: (num_nodes, num_nodes) fixed binary adjacency matrix
            gcn_hidden: GCN intermediate hidden units
            gcn_out: Per-timestep spatial feature dimension 'a'
            lstm_hidden: LSTM hidden units = temporal feature dim 'b'
            hybrid_hidden1: First hybrid GCN layer units
            hybrid_hidden2: Second hybrid GCN layer units
            dynamic_method: "FR" | "FD" | "TN"
            dropout: LSTM dropout
        """
        super(DTCSTGCN, self).__init__()

        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.gcn_out = gcn_out
        self.lstm_hidden = lstm_hidden

        # ── Component 1: Dynamic Adjacency ──
        self.dyn_adj = DynamicAdjacencyBuilder(adj, method=dynamic_method)

        # ── Component 2: Attention-based GCN (per timestep) ──
        self.spatial_gcn = AttentionGCNBlock(
            num_nodes=num_nodes,
            in_features=input_dim,
            gcn_hidden=gcn_hidden,
            out_features=gcn_out,
        )
        # Spatial feature: dùng mean pooling qua seq_len thay vì concat.
        # Paper (Eq.13) dùng concat → spatial_feat_dim = gcn_out * seq_len,
        # nhưng với seq_len=12 và gcn_out=64 → 768-dim combined feature → quá lớn
        # so với dataset nhỏ (504 train, 92 nodes).
        # Mean pooling giữ nguyên ý nghĩa semantic và giảm combined_feat_dim
        # từ 832 xuống 128, phù hợp với quy mô dữ liệu thực tế.
        spatial_feat_dim = gcn_out  # sau mean pooling: (batch, N, gcn_out)

        # ── Component 3: LSTM for Temporal Features ──
        # Input: historical observations On ∈ R^{N × n}
        # We feed each node's time series through LSTM
        # LSTM chỉ nhận speed feature (dim 0) làm temporal signal,
        # đúng với paper: "feed historical observations On" (traffic speed/flow scalar).
        # Với input_dim=40, chỉ feature đầu (speed) có ý nghĩa temporal rõ ràng.
        self.lstm = nn.LSTM(
            input_size=1,          # chỉ speed scalar (feature dim 0)
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            dropout=dropout,
        )
        # Use last hidden state: temporal feature Ft ∈ R^{N × lstm_hidden}
        temporal_feat_dim = lstm_hidden  # 'b'

        # ── Component 4: Hybrid Attention-based GCN ──
        # Combined features: Fo = Fs ⊕ Ft (omitting road info since not in our dataset)
        # Fo ∈ R^{N × (a*n + b)}
        combined_feat_dim = spatial_feat_dim + temporal_feat_dim

        # Spatio-temporal attention
        self.st_attention = SpatioTemporalAttention(combined_feat_dim)

        # 2 GCN layers on combined features
        self.hybrid_gcn1 = GCNLayer(combined_feat_dim, hybrid_hidden1)
        self.hybrid_gcn2 = GCNLayer(hybrid_hidden1, hybrid_hidden2)

        # Final prediction: FC layer
        # output_dim nên = 1 (predict speed scalar) để metrics có nghĩa.
        # Khi caller truyền output_dim=40, model predict 40 features nhưng
        # chỉ feature 0 (speed) được học từ LSTM và GCN signal.
        self.fc = nn.Linear(hybrid_hidden2, output_dim * pred_len)

        # Fixed adjacency for hybrid GCN (normalized)
        adj_norm = self._normalize_adj(adj)
        adj_tensor = torch.FloatTensor(adj_norm)
        self.register_buffer("adj_norm", adj_tensor)

    def _normalize_adj(self, adj):
        """Symmetric normalization D^(-1/2) * A * D^(-1/2)"""
        A = adj + np.eye(adj.shape[0])
        D = np.array(A.sum(axis=1))
        D_inv_sqrt = np.power(D, -0.5).flatten()
        D_inv_sqrt[np.isinf(D_inv_sqrt)] = 0.0
        D_mat = np.diag(D_inv_sqrt)
        return D_mat @ A @ D_mat

    def forward(self, x, adj=None):
        """
        Args:
            x: (batch_size, seq_len, num_nodes, input_dim)
            adj: Not used (internally uses dynamic adjacency), kept for API compatibility
        Returns:
            output: (batch_size, pred_len, num_nodes, output_dim)
        """
        batch_size, seq_len, num_nodes, input_dim = x.size()

        # ── Compute dynamic adjacencies for all timesteps ──
        dynamic_adjs = []
        for t in range(seq_len):
            F_t = x[:, t, :, :]         # (batch, N, input_dim)
            At = self.dyn_adj(F_t)      # (batch, N, N)
            dynamic_adjs.append(At)

        # ── Spatial Features (GCN per timestep) ──
        spatial_feats = []
        F_prev, A_prev = None, None
        for t in range(seq_len):
            F_t = x[:, t, :, :]         # (batch, N, input_dim)
            At = dynamic_adjs[t]         # (batch, N, N)

            Ht = self.spatial_gcn(F_t, At, F_prev, A_prev)  # (batch, N, gcn_out)
            spatial_feats.append(Ht)

            F_prev = F_t
            A_prev = At

        # Hs: mean pooling qua seq_len → (batch, N, gcn_out)
        # Thay concat (Eq.13) để tránh combined_feat_dim quá lớn.
        Hs = torch.stack(spatial_feats, dim=1).mean(dim=1)  # (batch, N, gcn_out)

        # ── Temporal Features (LSTM per node) ──
        # Chỉ dùng feature dim 0 (speed) cho LSTM, đúng với paper.
        # x: (batch, seq_len, N, input_dim) → lấy dim 0 → (batch, seq_len, N, 1)
        x_speed = x[..., :1]                               # (batch, seq_len, N, 1)
        x_permuted = x_speed.permute(0, 2, 1, 3)          # (batch, N, seq_len, 1)
        x_lstm = x_permuted.reshape(batch_size * num_nodes, seq_len, 1)

        _, (h_n, _) = self.lstm(x_lstm)
        # h_n: (1, batch*N, lstm_hidden) → last hidden state
        Ft_temporal = h_n.squeeze(0).reshape(batch_size, num_nodes, self.lstm_hidden)

        # ── Combined Features Fo ──
        Fo = torch.cat([Hs, Ft_temporal], dim=-1)  # (batch, N, a*n + b)

        # ── Spatio-temporal Attention (Eq. 25) ──
        Ro = self.st_attention(Fo)           # (batch, N, a*n+b)
        Fo_attended = Fo * Ro                # element-wise Hadamard

        # ── Hybrid GCN (2 layers with fixed adjacency) ──
        # Expand adj_norm to batch: (batch, N, N)
        adj_expanded = self.adj_norm.unsqueeze(0).expand(batch_size, -1, -1)

        Ho1 = self.hybrid_gcn1(Fo_attended, adj_expanded)  # (Eq. 26)
        Ho = self.hybrid_gcn2(Ho1, adj_expanded)            # (Eq. 27)

        # ── Prediction (Eq. 28) ──
        out = self.fc(Ho)  # (batch, N, output_dim * pred_len)
        out = out.reshape(batch_size, num_nodes, self.pred_len, -1)
        out = out.permute(0, 2, 1, 3)  # (batch, pred_len, N, output_dim)

        return out

    def get_dynamic_adjacency(self, x, timestep=0):
        """
        Get dynamic adjacency matrix for a specific timestep (for visualization).
        
        Args:
            x: (batch, seq_len, N, input_dim)
            timestep: Which timestep to compute adjacency for
        Returns:
            At: (batch, N, N)
        """
        F_t = x[:, timestep, :, :]
        return self.dyn_adj(F_t)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    num_nodes = 20
    seq_len = 8
    pred_len = 4
    batch_size = 4

    adj = np.random.randint(0, 2, (num_nodes, num_nodes)).astype(np.float32)
    adj = np.clip(adj + adj.T, 0, 1)
    np.fill_diagonal(adj, 0)

    for method in ["FR", "FD"]:
        model = DTCSTGCN(
            num_nodes=num_nodes,
            input_dim=1,
            hidden_dim=64,
            output_dim=1,
            seq_len=seq_len,
            pred_len=pred_len,
            adj=adj,
            dynamic_method=method,
        )
        x = torch.randn(batch_size, seq_len, num_nodes, 1)
        out = model(x)
        print(f"DTC-STGCN ({method}): {x.shape} → {out.shape}, params={count_parameters(model):,}")
        assert out.shape == (batch_size, pred_len, num_nodes, 1)

    print("✓ DTC-STGCN forward pass OK")