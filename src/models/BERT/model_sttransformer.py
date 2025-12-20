# src/model_sttransformer.py
from __future__ import annotations
from dataclasses import dataclass
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


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

        # enc_layer = nn.TransformerEncoderLayer(
        #     d_model=mp.d_model,
        #     nhead=mp.nhead,
        #     dim_feedforward=4 * mp.d_model,
        #     dropout=mp.dropout,
        #     batch_first=True,   # (B,S,d)
        #     activation="gelu",
        #     norm_first=True
        # )
        # self.encoder = nn.TransformerEncoder(enc_layer, num_layers=mp.num_layers)

        self.alpha = nn.Parameter(torch.tensor(1.0))
        dim_ff = 4 * mp.d_model
        self.layers = nn.ModuleList([
            STEncoderLayerWithBias(mp.d_model, mp.nhead, mp.dropout, dim_ff)
            for _ in range(mp.num_layers)
        ])

        self.head = nn.Sequential(
            nn.Linear(mp.d_model, mp.d_model),
            nn.GELU(),
            nn.Dropout(mp.dropout),
            nn.Linear(mp.d_model, 1),
        )

    def forward(self, x, tod, dow, lap, node_mask, attn_bias=None):
        B, L, N, _ = x.shape

        x0 = self.x_proj(x)
        temp_pe = self.temporal(tod, dow)
        spat_pe = self.spatial(lap)
        x0 = x0 + temp_pe.unsqueeze(2) + spat_pe.unsqueeze(1)

        S = L * N
        tokens = x0.reshape(B, S, self.mp.d_model)

        pad_nodes = (node_mask == 0)
        key_padding_mask = pad_nodes.unsqueeze(1).expand(B, L, N).reshape(B, S).to(torch.bool)

        attn_bias_flat = None
        if attn_bias is not None:
            bias5 = attn_bias[:, None, :, None, :].expand(B, L, N, L, N)
            attn_bias_flat = (self.alpha * bias5.reshape(B, S, S)).to(tokens.dtype)

        h = tokens
        for layer in self.layers:
            h = layer(h, attn_bias=attn_bias_flat, key_padding_mask=key_padding_mask)

        h = h.reshape(B, L, N, self.mp.d_model)
        h_last = h[:, -1, :, :]
        logits = self.head(h_last).squeeze(-1)
        return logits


class STEncoderLayerWithBias(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1, dim_ff: int = 2048):
        super().__init__()
        assert d_model % nhead == 0
        self.d_model = d_model
        self.nhead = nhead
        self.d_head = d_model // nhead

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=True)
        self.out = nn.Linear(d_model, d_model, bias=True)

        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, attn_bias: torch.Tensor, key_padding_mask: torch.Tensor):
        """
        x: (B,S,D)
        attn_bias: (B,S,S) float, additive bias (pre-softmax)
        key_padding_mask: (B,S) bool, True = PAD token (mask out)
        """
        B, S, D = x.shape
        h = self.norm1(x)

        qkv = self.qkv(h)  # (B,S,3D)
        q, k, v = qkv.chunk(3, dim=-1)

        # (B, nhead, S, d_head)
        q = q.view(B, S, self.nhead, self.d_head).transpose(1, 2)
        k = k.view(B, S, self.nhead, self.d_head).transpose(1, 2)
        v = v.view(B, S, self.nhead, self.d_head).transpose(1, 2)

        # Build attention mask = attn_bias + padding mask
        # scaled_dot_product_attention expects attn_mask broadcastable to (B,nhead,S,S)
        attn_mask = attn_bias[:, None, :, :]  # (B,1,S,S)

        # key padding: mask keys by adding -inf on columns
        if key_padding_mask is not None:
            # key_padding_mask: True=PAD -> should be masked
            neg_inf = torch.finfo(attn_mask.dtype).min
            key_mask = key_padding_mask[:, None, None, :].to(torch.bool)  # (B,1,1,S)
            attn_mask = attn_mask.masked_fill(key_mask, neg_inf)

        # Attention
        attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)  # (B,nhead,S,d_head)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, D)  # (B,S,D)
        attn_out = self.out(attn_out)
        x = x + self.dropout(attn_out)

        # FFN
        h2 = self.norm2(x)
        x = x + self.ffn(h2)
        return x