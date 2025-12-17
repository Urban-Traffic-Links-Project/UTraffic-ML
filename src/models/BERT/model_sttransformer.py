# src/model_sttransformer.py
from __future__ import annotations
from dataclasses import dataclass
import math
import torch
import torch.nn as nn


@dataclass
class ModelParams:
    d_in: int = 1
    d_model: int = 128
    nhead: int = 4
    num_layers: int = 4
    dropout: float = 0.1
    d_spa: int = 16
    L_max: int = 64  # maximum L for positional embedding


class TemporalEncoding(nn.Module):
    """
    temp_pe(B,L,d_model) = pos_emb(L) + Linear([sin/cos(tod), sin/cos(dow)]).
    """
    def __init__(self, d_model: int, L_max: int = 256):
        super().__init__()
        self.pos_emb = nn.Embedding(L_max, d_model)
        self.feat_proj = nn.Linear(4, d_model)

    def forward(self, tod: torch.Tensor, dow: torch.Tensor) -> torch.Tensor:
        # tod: (B,L) float in [0,1]
        # dow: (B,L) int 0..6
        B, L = tod.shape
        device = tod.device

        pos = torch.arange(L, device=device).unsqueeze(0).expand(B, L)  # (B,L)
        pe_pos = self.pos_emb(pos)  # (B,L,d)

        # sin/cos
        # tod angle: 2pi * tod
        ang_tod = 2 * math.pi * tod
        sin_tod = torch.sin(ang_tod)
        cos_tod = torch.cos(ang_tod)

        # dow: map 0..6 to 0..1 cycle
        dow_f = dow.to(torch.float32) / 7.0
        ang_dow = 2 * math.pi * dow_f
        sin_dow = torch.sin(ang_dow)
        cos_dow = torch.cos(ang_dow)

        feats = torch.stack([sin_tod, cos_tod, sin_dow, cos_dow], dim=-1)  # (B,L,4)
        pe_time = self.feat_proj(feats)  # (B,L,d)

        return pe_pos + pe_time


class SpatialEncoding(nn.Module):
    def __init__(self, d_spa: int, d_model: int):
        super().__init__()
        self.proj = nn.Linear(d_spa, d_model) if d_spa > 0 else None

    def forward(self, lap: torch.Tensor) -> torch.Tensor:
        # lap: (B,N,d_spa)
        if self.proj is None:
            B, N, _ = lap.shape
            return torch.zeros((B, N, 0), device=lap.device, dtype=lap.dtype)
        return self.proj(lap)  # (B,N,d_model)


class STEncoderOnly(nn.Module):
    def __init__(self, mp: ModelParams):
        super().__init__()
        self.mp = mp
        self.x_proj = nn.Linear(mp.d_in, mp.d_model)
        self.temporal = TemporalEncoding(mp.d_model, L_max=mp.L_max)
        self.spatial = SpatialEncoding(mp.d_spa, mp.d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=mp.d_model,
            nhead=mp.nhead,
            dim_feedforward=4 * mp.d_model,
            dropout=mp.dropout,
            batch_first=True,   # (B,S,d)
            activation="gelu",
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=mp.num_layers)

        self.head = nn.Sequential(
            nn.Linear(mp.d_model, mp.d_model),
            nn.GELU(),
            nn.Dropout(mp.dropout),
            nn.Linear(mp.d_model, 1),
        )

    def forward(self, x: torch.Tensor, tod: torch.Tensor, dow: torch.Tensor,
                lap: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
        """
        x: (B,L,N,d_in)
        tod: (B,L)
        dow: (B,L)
        lap: (B,N,d_spa)
        node_mask: (B,N) 1 valid, 0 pad

        return logits: (B,N)
        """
        B, L, N, _ = x.shape

        x0 = self.x_proj(x)  # (B,L,N,d)

        temp_pe = self.temporal(tod, dow)            # (B,L,d)
        spat_pe = self.spatial(lap)                  # (B,N,d)

        # broadcast sum
        x0 = x0 + temp_pe.unsqueeze(2) + spat_pe.unsqueeze(1)  # (B,L,N,d)

        # flatten tokens S=L*N
        S = L * N
        tokens = x0.reshape(B, S, self.mp.d_model)

        # build padding mask for transformer: True means "mask out"
        # if a node is padding => all its L tokens are padding
        pad_nodes = (node_mask == 0)  # (B,N) bool
        pad_tokens = pad_nodes.unsqueeze(1).expand(B, L, N).reshape(B, S)  # (B,S)

        h = self.encoder(tokens, src_key_padding_mask=pad_tokens)  # (B,S,d)
        h = h.reshape(B, L, N, self.mp.d_model)

        h_last = h[:, -1, :, :]  # (B,N,d)
        logits = self.head(h_last).squeeze(-1)  # (B,N)
        return logits
