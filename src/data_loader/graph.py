# File: src/data/graph.py
from __future__ import annotations
import pandas as pd
import torch
from typing import Dict, Optional

def load_edge_index(edges_csv: str, num_nodes: int, segid_to_idx: Optional[Dict[int,int]] = None, add_self_loops: bool=True) -> torch.Tensor:
    df = pd.read_csv(edges_csv)

    if {"src","dst"}.issubset(df.columns):
        src = df["src"].astype(int).to_numpy()
        dst = df["dst"].astype(int).to_numpy()
    elif {"segment_u","segment_v"}.issubset(df.columns):
        if segid_to_idx is None:
            raise ValueError("edges.csv uses segment_id columns but segid_to_idx is None")
        src = df["segment_u"].map(segid_to_idx).to_numpy()
        dst = df["segment_v"].map(segid_to_idx).to_numpy()
        if pd.isna(src).any() or pd.isna(dst).any():
            raise ValueError("Some segment ids in edges.csv not found in segment_index.csv")
        src = src.astype(int)
        dst = dst.astype(int)
    elif {"node_u","node_v"}.issubset(df.columns):
        src = df["node_u"].astype(int).to_numpy()
        dst = df["node_v"].astype(int).to_numpy()
    else:
        raise ValueError(f"Unknown edges schema: {df.columns.tolist()}")

    src_t = torch.tensor(src, dtype=torch.long)
    dst_t = torch.tensor(dst, dtype=torch.long)

    if add_self_loops:
        self_idx = torch.arange(num_nodes, dtype=torch.long)
        src_t = torch.cat([src_t, self_idx], dim=0)
        dst_t = torch.cat([dst_t, self_idx], dim=0)

    return torch.stack([src_t, dst_t], dim=0)  # [2,E]
