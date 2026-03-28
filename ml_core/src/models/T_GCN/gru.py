"""
Gated Recurrent Unit (GRU) Module
Based on T-GCN paper (Zhao et al., 2019)

Fixes vs original:
    - GRUCell.forward: guard khi bias=None (original crash với `+ None`)
    - GRU: hidden_list dùng list để tránh in-place autograd issue (giữ nguyên)
    - Minor: type annotations, cleaner variable names
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class GRUCell(nn.Module):
    """
    Custom GRU Cell — graph-aware (hoạt động trên tensor 3-D).

    Equations:
        r = σ(x @ W_ir + h @ W_hr + b_r)
        z = σ(x @ W_iz + h @ W_hz + b_z)
        n = tanh(x @ W_in + b_in + r * (h @ W_hn + b_hn))
        h' = (1 - z) * n + z * h
    """

    def __init__(self, input_size: int, hidden_size: int, bias: bool = True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_bias = bias

        # Reset gate
        self.W_ir = Parameter(torch.FloatTensor(input_size,  hidden_size))
        self.W_hr = Parameter(torch.FloatTensor(hidden_size, hidden_size))

        # Update gate
        self.W_iz = Parameter(torch.FloatTensor(input_size,  hidden_size))
        self.W_hz = Parameter(torch.FloatTensor(hidden_size, hidden_size))

        # Candidate hidden state
        self.W_in = Parameter(torch.FloatTensor(input_size,  hidden_size))
        self.W_hn = Parameter(torch.FloatTensor(hidden_size, hidden_size))

        if bias:
            self.b_ir = Parameter(torch.FloatTensor(hidden_size))
            self.b_hr = Parameter(torch.FloatTensor(hidden_size))
            self.b_iz = Parameter(torch.FloatTensor(hidden_size))
            self.b_hz = Parameter(torch.FloatTensor(hidden_size))
            self.b_in = Parameter(torch.FloatTensor(hidden_size))
            self.b_hn = Parameter(torch.FloatTensor(hidden_size))
        else:
            for name in ("b_ir", "b_hr", "b_iz", "b_hz", "b_in", "b_hn"):
                self.register_parameter(name, None)

        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    # ------------------------------------------------------------------
    # Helper: safe bias addition (returns 0 when bias is disabled)
    # ------------------------------------------------------------------
    @staticmethod
    def _add(tensor: torch.Tensor, bias) -> torch.Tensor:
        return tensor + bias if bias is not None else tensor

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, nodes, input_size)
            h: (batch, nodes, hidden_size)
        Returns:
            h_new: (batch, nodes, hidden_size)
        """
        # Reset gate
        r = torch.sigmoid(
            self._add(x @ self.W_ir, self.b_ir)
            + self._add(h @ self.W_hr, self.b_hr)
        )

        # Update gate
        z = torch.sigmoid(
            self._add(x @ self.W_iz, self.b_iz)
            + self._add(h @ self.W_hz, self.b_hz)
        )

        # Candidate
        n = torch.tanh(
            self._add(x @ self.W_in, self.b_in)
            + r * self._add(h @ self.W_hn, self.b_hn)
        )

        # New hidden state
        h_new = (1.0 - z) * n + z * h
        return h_new


class GRU(nn.Module):
    """Multi-layer GRU for graph-structured inputs (4-D tensors)."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.batch_first = batch_first
        self.dropout_p   = dropout

        self.cells = nn.ModuleList()
        for layer in range(num_layers):
            in_dim = input_size if layer == 0 else hidden_size
            self.cells.append(GRUCell(in_dim, hidden_size, bias))

    def forward(self, x: torch.Tensor, hidden=None):
        """
        Args:
            x     : (batch, seq, nodes, feat) if batch_first else (seq, batch, nodes, feat)
            hidden: (num_layers, batch, nodes, hidden_size) or None
        Returns:
            output: same spatial dims as x but feat dim = hidden_size
            h_out : (num_layers, batch, nodes, hidden_size)
        """
        if self.batch_first:
            x = x.transpose(0, 1)          # -> (seq, batch, nodes, feat)

        seq_len, batch, num_nodes, _ = x.size()

        if hidden is None:
            h_list = [
                torch.zeros(batch, num_nodes, self.hidden_size, device=x.device)
                for _ in range(self.num_layers)
            ]
        else:
            h_list = list(hidden.unbind(0))

        outputs = []
        for t in range(seq_len):
            inp = x[t]
            for layer, cell in enumerate(self.cells):
                h_new = cell(inp, h_list[layer])
                h_list[layer] = h_new
                if layer < self.num_layers - 1 and self.dropout_p > 0:
                    inp = F.dropout(h_new, p=self.dropout_p, training=self.training)
                else:
                    inp = h_new
            outputs.append(inp)

        output = torch.stack(outputs)               # (seq, batch, nodes, hidden)
        if self.batch_first:
            output = output.transpose(0, 1)         # (batch, seq, nodes, hidden)

        h_out = torch.stack(h_list, dim=0)          # (layers, batch, nodes, hidden)
        return output, h_out


# ---------------------------------------------------------------------------
# Smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    batch, seq_len, nodes, feat = 32, 12, 100, 64
    hidden = 32

    model = GRU(feat, hidden, num_layers=1, batch_first=True)
    x = torch.randn(batch, seq_len, nodes, feat)
    output, h_last = model(x)
    print("Input :", x.shape)
    print("Output:", output.shape)   # expect [32, 12, 100, 32]
    print("Hidden:", h_last.shape)   # expect [1, 32, 100, 32]