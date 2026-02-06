from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

class GATLayer(nn.Module):
    """
        GAT layer (multi-head) cho graph tĩnh edge_index.
        Trả về:
          - out: [N, heads*out_dim]
          - attn: [E, heads] (attention weights per edge)
        """
    def __init__(self, in_dim:int,out_dim:int, heads:int,dropout: float=0.1):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.heads = heads
        self.dropout = dropout
        self.W = nn.Linear(in_dim, heads * out_dim,bias=False)

        # attention vector cho mỗi head (a^T [Wh_i || Wh_j])
        self.attn_l = nn.Parameter(torch.empty(heads, out_dim))
        self.attn_r = nn.Parameter(torch.empty(heads, out_dim))
        nn.init.xavier_uniform_(self.attn_l)
        nn.init.xavier_uniform_(self.attn_r)

        self.leaky = nn.LeakyReLU(0.2)
    @staticmethod
    def _edge_softmax(scores: torch.Tensor, dst: torch.Tensor, num_nodes:int) -> torch.Tensor:
        """
                scores: [E, H]
                dst:    [E]
                return: [E, H] softmax theo nhóm dst
                """
        # ổn định số học
        max_per_dst = torch.full ((num_nodes, scores.size(1)), -1e15, device=scores.device)
        max_per_dst.index_reduce_(0,dst, scores, reduce='amax')
        scores = scores - max_per_dst[dst]

        exp = torch.exp(scores)
        denom = torch.zeros((num_nodes, exp.size(1)), device=exp.device)
        denom.index_add_(0, dst, exp)
        return exp / (denom[dst] + 1e-12)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
                x: [N, in_dim]
                edge_index: [2, E] (src, dst)
                """
        N = x.size(0)
        src,dst = edge_index[0], edge_index[1]

        h = self.W(x)  # [N, H*out_dim]
        h = h.view(N, self.heads, self.out_dim)  # [N, H, D]

        # e_ij = LeakyReLU( a_l^T h_i + a_r^T h_j )
        e_src = (h[src] * self.attn_l.unsqueeze(0)).sum(-1)  # [E, H]
        e_dst = (h[dst] * self.attn_r.unsqueeze(0)).sum(-1)  # [E, H]
        e = self.leaky(e_src + e_dst)

        alpha = self._edge_softmax(e, dst, N)  # [E, H]
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        out = torch.zeros((N, self.heads, self.out_dim), device=x.device)
        out.index_add_(0, dst, alpha.unsqueeze(-1) * h[src])  # [N, H, D]

        out = out.reshape(N, self.heads * self.out_dim)  # [N, H*D]
        return out, alpha

class RouteAttentionPooling(nn.Module):
    """
    Attention pooling trên tập segments của một route.
    Input:  route_nodes: [L] indices
            h: [N, D] segment embeddings
    Output: r: [D] route embedding, weights beta: [L]
    """
    def __init__(self, dim: int, hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1)
        )

    def forward(self, h: torch.Tensor, route_nodes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # h_route: [L, D]
        h_route = h.index_select(0, route_nodes)
        scores = self.mlp(h_route).squeeze(-1)           # [L]
        beta = torch.softmax(scores, dim=0)              # [L]
        r = (beta.unsqueeze(-1) * h_route).sum(dim=0)    # [D]
        return r, beta

@dataclass
class ForwardOutputs:
    y_hat: torch.Tensor                    # [B, H, N]
    gat_attn: Optional[List[torch.Tensor]] # list of [B, E, heads]
    route_pool_attn: Optional[torch.Tensor]# [B, routes_per_batch, Lmax] padded, -1 where invalid

class GATGRU(nn.Module):
    """
    Full model:
      - GAT (spatial) applied at each time step
      - GRU (temporal) per node
      - Forecast head -> H steps
    """
    def __init__(
        self,
        num_nodes: int,
        edge_index: torch.Tensor,
        in_dim: int = 1,
        gat_hidden: int = 32,
        gat_heads: int = 4,
        gat_layers: int = 2,
        gru_hidden: int = 64,
        horizon: int = 6,
        dropout: float = 0.1,
        route_pool_hidden: int = 64,
        routes_per_batch: int = 8,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.edge_index = edge_index
        self.horizon = horizon
        self.routes_per_batch = routes_per_batch

        layers = []
        d_in = in_dim
        for _ in range(gat_layers):
            layers.append(GATLayer(d_in, gat_hidden, heads=gat_heads, dropout=dropout))
            d_in = gat_hidden * gat_heads
        self.gat_layers = nn.ModuleList(layers)

        self.gru = nn.GRU(input_size=d_in, hidden_size=gru_hidden, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(gru_hidden, horizon)  # dự báo H bước cho từng node

        # route pooling dùng trên embedding cuối (gru_hidden)
        self.route_pool = RouteAttentionPooling(gru_hidden, hidden=route_pool_hidden, dropout=dropout)
        # (NEW) route -> segment feedback
        self.route_fb_proj = nn.Linear(gru_hidden, gru_hidden, bias=True)
        self.route_fb_gate = nn.Linear(gru_hidden, gru_hidden, bias=True)  # cho kiểu gating
        self.route_fb_mode = "add"  # "add" hoặc "gate"
    def forward(
            self,
            x: torch.Tensor,
            routes: Optional[torch.Tensor] = None,
            return_attn: bool = False,
    ) -> ForwardOutputs:
        """
        x: [B, P, N, C]
        routes: [B, R, L] indices (padded -1) hoặc None
        B: batch size (số mẫu)

        P: số bước thời gian quá khứ (lookback)

        N: số node / segment

        C: số feature mỗi node (≥1, thường là speed)
        """
        B, P, N, C = x.shape
        assert N == self.num_nodes, "num_nodes mismatch"
        assert C >= 1, "need at least 1 feature (speed)"

        gat_attn_all: Optional[List[torch.Tensor]] = [] if return_attn else None
        route_pool_attn = None

        # Apply GAT per time step
        # We'll process each t, flatten batch by looping (P nhỏ nên loop ok)
        Z_seq = []
        for t in range(P):
            xt = x[:, t]  # [B, N, C]
            # run GAT for each batch element (graph same, so loop B)
            zt_list = []
            attn_layers_bt: Optional[List[torch.Tensor]] = [] if return_attn else None

            for b in range(B):
                hb = xt[b]  # [N, C]
                attn_each_layer = []
                for layer in self.gat_layers:
                    hb, alpha = layer(hb, self.edge_index)  # hb:[N,D], alpha:[E,H]
                    hb = F.elu(hb)
                    hb = self.dropout(hb)
                    if return_attn:
                        attn_each_layer.append(alpha.unsqueeze(0))  # [1,E,H]
                zt_list.append(hb.unsqueeze(0))  # [1,N,D]

                if return_attn:
                    # stack layers: [L, 1, E, H]
                    attn_layers_bt.append(torch.stack(attn_each_layer, dim=0))  # [L,1,E,H]

            zt = torch.cat(zt_list, dim=0)  # [B,N,D]
            Z_seq.append(zt.unsqueeze(1))  # [B,1,N,D]

            if return_attn:
                # attn_layers_bt: list length B of [L,1,E,H] -> [B,L,E,H]
                attn_layers_bt = torch.cat(attn_layers_bt, dim=1).permute(1, 0, 2, 3)  # [B,L,E,H]
                # save per time step (optional). To keep lighter, save only last t later
                # Here we keep last time step only to export
                last_attn = attn_layers_bt
                # store only last time step's attn once outside loop:
                last_attn_for_epoch = last_attn

        Z = torch.cat(Z_seq, dim=1)  # [B,P,N,D]
        # GRU per node: reshape to [B*N, P, D]
        Z_bn = Z.permute(0, 2, 1, 3).contiguous().view(B * N, P, Z.size(-1))
        out, h_last = self.gru(Z_bn)  # h_last: [1, B*N, Hh]
        h_last = h_last.squeeze(0)  # [B*N, Hh]
        h_last = self.dropout(h_last)
        h_nodes = h_last.view(B, N, -1)  # [B,N,gru_hidden]
        # (NEW) Route -> Segment Feedback (apply only if routes is provided)
        if routes is not None:
            B2, R, L = routes.shape
            assert B2 == B

            # accumulators in case a segment appears in multiple routes
            delta = torch.zeros_like(h_nodes)  # [B,N,D]
            cnt = torch.zeros((B, N, 1), device=x.device)
            route_pool_attn = torch.full((B, R, L), -1.0, device=x.device)

            for b in range(B):
                for r in range(R):
                    nodes = routes[b, r]
                    valid = nodes[nodes >= 0].to(torch.long)
                    if valid.numel() == 0:
                        continue

                    # route embedding r_vec and pooling weights beta
                    r_vec, beta = self.route_pool(h_nodes[b], valid)  # r_vec:[D], beta:[Lv]
                    route_pool_attn[b, r, :beta.numel()] = beta

                    if self.route_fb_mode == "add":
                        fb = self.route_fb_proj(r_vec)  # phi(r): [D]
                        delta[b, valid] += fb
                    else:
                        gate = torch.sigmoid(self.route_fb_gate(r_vec))  # [D]
                        # convert gating to additive delta so we can average when overlap
                        delta[b, valid] += h_nodes[b, valid] * (gate - 1.0)

                    cnt[b, valid] += 1.0

            # average if overlaps
            cnt = torch.clamp(cnt, min=1.0)
            h_nodes = h_nodes + delta / cnt

        # after feedback -> flatten back
        h_last_fb = h_nodes.view(B * N, -1)  # [B*N, D]

        # predict: [B*N, H] -> [B,H,N]
        pred = self.head(h_last_fb)  # [B*N, H]
        pred = pred.view(B, N, self.horizon).permute(0, 2, 1).contiguous()


        if return_attn:
            # Only export attention from the last GAT time step (lightweight)
            # last_attn_for_epoch: [B, L, E, H]
            # Convert to list per layer: [B,E,H]
            gat_attn_all = [last_attn_for_epoch[:, li] for li in range(last_attn_for_epoch.size(1))]

        return ForwardOutputs(
            y_hat=pred,
            gat_attn=gat_attn_all,
            route_pool_attn=route_pool_attn
        )