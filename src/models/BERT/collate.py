# src/collate.py
from __future__ import annotations
from typing import List, Dict, Any
import torch


def pad_collate_zones(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Pad variable N_zone to N_max within batch.

    Output:
      x: (B, L, N_max, d_in)
      y: (B, N_max)
      tod: (B, L)
      dow: (B, L)
      lap: (B, N_max, d_spa)
      attn_bias: (B, N_max, N_max)
      node_mask: (B, N_max)
      target_mask: (B, N_max)
      final_mask: (B, N_max)
    """
    B = len(batch)
    L = batch[0]["x"].shape[0]
    d_in = batch[0]["x"].shape[-1]
    d_spa = batch[0]["lap"].shape[-1] if batch[0]["lap"].ndim == 2 else 0

    Nz_list = [item["x"].shape[1] for item in batch]
    N_max = max(Nz_list)

    # ---------- allocate ----------
    x = torch.zeros((B, L, N_max, d_in), dtype=batch[0]["x"].dtype)
    y = torch.zeros((B, N_max), dtype=batch[0]["y"].dtype)
    lap = torch.zeros((B, N_max, d_spa), dtype=batch[0]["lap"].dtype) if d_spa > 0 else torch.zeros((B, N_max, 0))

    # NEW: attention bias
    attn_bias = torch.zeros((B, N_max, N_max), dtype=torch.float32)

    tod = torch.stack([item["tod"] for item in batch], dim=0)  # (B,L)
    dow = torch.stack([item["dow"] for item in batch], dim=0)  # (B,L)

    node_mask = torch.zeros((B, N_max), dtype=torch.int64)
    target_mask = torch.zeros((B, N_max), dtype=torch.int64)

    metas = [item.get("meta", None) for item in batch]

    # ---------- fill ----------
    for b, item in enumerate(batch):
        Nz = item["x"].shape[1]

        x[b, :, :Nz, :] = item["x"]
        y[b, :Nz] = item["y"]

        if d_spa > 0:
            lap[b, :Nz, :] = item["lap"]

        node_mask[b, :Nz] = item["node_mask"].to(torch.int64)
        target_mask[b, :Nz] = item["target_mask"].to(torch.int64)

        # NEW: copy attention bias
        if "attn_bias" in item and item["attn_bias"] is not None:
            attn_bias[b, :Nz, :Nz] = item["attn_bias"]

    final_mask = node_mask * target_mask

    return {
        "x": x,
        "tod": tod,
        "dow": dow,
        "lap": lap,
        "attn_bias": attn_bias,
        "y": y,
        "node_mask": node_mask,
        "target_mask": target_mask,
        "final_mask": final_mask,
        "meta": metas,
    }
