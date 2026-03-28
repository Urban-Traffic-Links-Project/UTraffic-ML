"""
T-GCN: Temporal Graph Convolutional Network
Based on: Zhao et al. "T-GCN: A Temporal Graph Convolutional Network
          for Traffic Prediction" (2019)

Fixes & improvements vs original:
    1. TGCNCell: thêm LayerNorm sau GRU để ổn định training
    2. TGCN decoder: dùng input projection thay vì truyền output (output_dim=1)
       thẳng vào TGCNCell nhận input_dim features — tránh shape mismatch khi
       input_dim > 1
    3. TGCN: lưu adj_norm một lần (lazy) để tránh re-normalize mỗi forward
    4. TGCN: hỗ trợ multi-feature input đúng chiều
    5. count_parameters helper giữ nguyên
"""

import torch
import torch.nn as nn
from .gcn import GCN, normalize_adj
from .gru import GRUCell


class TGCNCell(nn.Module):
    """
    T-GCN Cell = GCN (spatial) + GRU (temporal) + LayerNorm.

    Bước A: x_gcn  = GCN(x, adj)            — tổng hợp thông tin hàng xóm
    Bước B: h_new  = GRU(x_gcn, h_prev)     — cập nhật ký ức
    Bước C: h_norm = LayerNorm(h_new)        — ổn định gradient
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        gcn_hidden_dim: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.gcn = GCN(
            in_features=input_dim,
            hidden_features=gcn_hidden_dim,
            out_features=hidden_dim,
            dropout=dropout,
            residual=(input_dim == hidden_dim),  # residual chỉ khi chiều khớp
        )
        self.gru_cell = GRUCell(hidden_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        h: torch.Tensor,
        adj: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x  : (batch, nodes, input_dim)
            h  : (batch, nodes, hidden_dim)
            adj: (nodes, nodes) — normalized
        Returns:
            h_new: (batch, nodes, hidden_dim)
        """
        x_gcn = self.gcn(x, adj)           # (batch, nodes, hidden_dim)
        h_new = self.gru_cell(x_gcn, h)    # (batch, nodes, hidden_dim)
        h_new = self.layer_norm(h_new)      # stabilise
        return h_new


class TGCN(nn.Module):
    """
    T-GCN Model for Traffic Prediction.

    Architecture:
        Encoder : TGCNCell × seq_len   (encodes history into hidden state)
        Decoder : TGCNCell × pred_len  (auto-regressive multi-step prediction)
        Head    : Linear(hidden_dim → output_dim)

    BUG FIX (decoder input shape):
        Original code passed fc output (shape [B, N, output_dim=1]) directly
        back into TGCNCell which expects input_dim features.  When input_dim > 1
        this causes a matmul shape error.

        Fix: a separate `dec_proj` Linear maps output_dim → input_dim before
        feeding back into the cell, keeping enc_cell and dec_cell weight-tied
        on the same TGCNCell instance.
    """

    def __init__(
        self,
        num_nodes: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        seq_len: int,
        pred_len: int,
        gcn_hidden_dim: int = 64,
        dropout: float = 0.0,
    ):
        """
        Args:
            num_nodes    : Number of graph nodes (updated dynamically on forward)
            input_dim    : Feature dimension per node per timestep
            hidden_dim   : GRU hidden size
            output_dim   : Prediction target dimension (1 for speed)
            seq_len      : Input sequence length (history)
            pred_len     : Prediction horizon (multi-step)
            gcn_hidden_dim: GCN intermediate dimension
            dropout      : Dropout rate inside GCN layers
        """
        super().__init__()
        self.num_nodes   = num_nodes
        self.input_dim   = input_dim
        self.hidden_dim  = hidden_dim
        self.output_dim  = output_dim
        self.seq_len     = seq_len
        self.pred_len    = pred_len

        # Shared encoder/decoder cell
        self.tgcn_cell = TGCNCell(input_dim, hidden_dim, gcn_hidden_dim, dropout)

        # Prediction head: hidden → output
        self.fc = nn.Linear(hidden_dim, output_dim)

        # BUG FIX: project decoder output back to input_dim so TGCNCell
        # receives the correct number of features at each decoder step.
        # When output_dim == input_dim this is just identity (no param overhead).
        if output_dim != input_dim:
            self.dec_proj = nn.Linear(output_dim, input_dim, bias=False)
        else:
            self.dec_proj = None

        # Cache for normalized adj (avoid recompute every forward pass)
        self._adj_norm: torch.Tensor | None = None
        self._adj_raw_ptr: int | None = None   # id() of last seen raw adj

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_adj_norm(self, adj: torch.Tensor) -> torch.Tensor:
        """
        Normalize adj lazily and cache.  Re-normalizes if adj object changes.
        """
        ptr = id(adj)
        if self._adj_raw_ptr != ptr or self._adj_norm is None:
            self._adj_norm = normalize_adj(adj).to(adj.device)
            self._adj_raw_ptr = ptr
        return self._adj_norm

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x  : (batch, seq_len, num_nodes, input_dim)
            adj: (num_nodes, num_nodes) — raw binary / weighted adjacency
        Returns:
            output: (batch, pred_len, num_nodes, output_dim)
        """
        batch, seq_len, num_nodes, _ = x.size()

        # Normalize adjacency once per call (cached by ptr)
        adj_norm = self._get_adj_norm(adj)

        # ── Encoder ──────────────────────────────────────────────────
        h = torch.zeros(batch, num_nodes, self.hidden_dim, device=x.device)
        for t in range(seq_len):
            h = self.tgcn_cell(x[:, t], h, adj_norm)   # (B, N, hidden)

        # ── Decoder (auto-regressive) ─────────────────────────────────
        # Seed decoder với giá trị cuối của chuỗi đầu vào.
        dec_input = x[:, -1, :, :self.output_dim]       # (B, N, output_dim)

        predictions = []
        for _ in range(self.pred_len):
            # Project decoder token back to input_dim
            if self.dec_proj is not None:
                dec_feat = self.dec_proj(dec_input)     # (B, N, input_dim)
            else:
                dec_feat = dec_input                    # (B, N, input_dim)

            h = self.tgcn_cell(dec_feat, h, adj_norm)
            pred = self.fc(h)                           # (B, N, output_dim)
            predictions.append(pred)
            dec_input = pred                            # feed prediction forward

        # Stack: (batch, pred_len, nodes, output_dim)
        return torch.stack(predictions, dim=1)

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Convenience: zero hidden state."""
        return torch.zeros(batch_size, self.num_nodes, self.hidden_dim, device=device)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    num_nodes  = 114
    input_dim  = 5       # multi-feature test
    hidden_dim = 64
    output_dim = 1
    seq_len    = 12
    pred_len   = 12
    batch_size = 32

    model = TGCN(num_nodes, input_dim, hidden_dim, output_dim, seq_len, pred_len)

    x = torch.randn(batch_size, seq_len, num_nodes, input_dim)
    adj = torch.rand(num_nodes, num_nodes)
    adj = (adj + adj.T) / 2   # symmetric

    output = model(x, adj)

    print(f"Input  shape: {x.shape}")
    print(f"Output shape: {output.shape}")   # expect [32, 12, 114, 1]
    print(f"Parameters  : {count_parameters(model):,}")