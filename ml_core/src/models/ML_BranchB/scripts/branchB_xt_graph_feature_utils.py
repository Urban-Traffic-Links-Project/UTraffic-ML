# ml_core/src/models/ML_BranchB/scripts/branchB_xt_graph_feature_utils.py
"""
Utilities for Branch B XT graph features.

Updated version:
- Default Top-K is 20.
- Gamma is removed.
- Graph signal is:
      gx = TopK(G) @ x_t
  optionally row-L1-normalized and with self-loop removed.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-8


def topk_filter_G(
    G: np.ndarray,
    k: int = 20,
    normalize: bool = True,
    remove_self_loop: bool = True,
) -> np.ndarray:
    """
    Keep top-k strongest edges per row.

    G shape convention:
        G[target, source]

    Parameters
    ----------
    G:
        Rt/Gt matrix with shape (N, N).
    k:
        Number of strongest source edges retained for each target row.
        k <= 0 or k >= N means keep full graph.
    normalize:
        If True, row-normalize by sum(abs(weights)).
    remove_self_loop:
        If True, set diagonal to zero before Top-K selection.
    """
    G = np.asarray(G, dtype=np.float32)
    if G.ndim != 2 or G.shape[0] != G.shape[1]:
        raise ValueError(f"G must be square, got shape={G.shape}")

    N = int(G.shape[0])
    k = int(k)

    G_work = G.astype(np.float32, copy=True)
    if remove_self_loop:
        diag = np.arange(N)
        G_work[diag, diag] = 0.0

    if k <= 0 or k >= N:
        G_new = G_work
    else:
        G_new = np.zeros_like(G_work, dtype=np.float32)
        absG = np.abs(G_work)
        idx = np.argpartition(absG, -k, axis=1)[:, -k:]
        rows = np.arange(N)[:, None]
        G_new[rows, idx] = G_work[rows, idx]

    if normalize:
        denom = np.sum(np.abs(G_new), axis=1, keepdims=True) + EPS
        G_new = G_new / denom

    return np.nan_to_num(G_new, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def build_graph_signal_topk(
    G_used: np.ndarray,
    x_t: np.ndarray,
    k: int = 20,
    normalize: bool = True,
    remove_self_loop: bool = True,
) -> np.ndarray:
    """
    Build graph signal without gamma.

    Formula:
        gx = TopK(G_used) @ x_t
    """
    G_filtered = topk_filter_G(
        G_used,
        k=k,
        normalize=normalize,
        remove_self_loop=remove_self_loop,
    )
    gx = G_filtered @ np.asarray(x_t, dtype=np.float32)
    return np.nan_to_num(gx, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
