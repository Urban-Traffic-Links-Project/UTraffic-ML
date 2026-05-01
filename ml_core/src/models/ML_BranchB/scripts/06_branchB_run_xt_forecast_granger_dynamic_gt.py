"""
Branch B method module: Granger-Dynamic-GT.

This file is loaded by 06B_branchB_run_xt_forecast_topk_gt.py.
It expects prepared data from:

    ml_core/src/data_processing/outputs/branchB/osm_edge_granger_dynamic_like_branchA

Graph format:
    graphs/G_bucket_hXXX.npy[bucket, target, source]
    split/origin_bucket_ids.npy[t]

At forecast origin index t and horizon h:
    G_used = G_bucket_hXXX[origin_bucket_ids[t]]

The downstream forecast remains:
    X_hat[t+h] = A_h X_t + B_h(TopK(G_used) X_t)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

ALPHA = 0.001
L1_RATIO = 0.5
MAX_ITER = 500
TOL = 1e-2
SELECTION = "random"
RANDOM_STATE = 42


def check_branchB_common_dir_ready(common_dir: Path) -> None:
    common_dir = Path(common_dir)
    required = [
        common_dir / "graphs" / "bucket_table.csv",
        common_dir / "graphs" / "available_horizons.npy",
    ]
    for split in ["train", "val", "test"]:
        d = common_dir / split
        for name in [
            "z.npy",
            "segment_ids.npy",
            "timestamps.npy",
            "G_series_meta.csv",
            "origin_bucket_ids.npy",
        ]:
            required.append(d / name)
    missing = [p for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing dynamic Granger prepared files:\n" + "\n".join(map(str, missing)))


def load_gt_split(common_dir: Path, split: str) -> Dict[str, Any]:
    common_dir = Path(common_dir)
    d = common_dir / split
    check_branchB_common_dir_ready(common_dir)

    z = np.load(d / "z.npy", mmap_mode="r")
    segment_ids = np.asarray(np.load(d / "segment_ids.npy"), dtype=np.int64)
    timestamps = pd.to_datetime(np.load(d / "timestamps.npy"))
    meta = pd.read_csv(d / "G_series_meta.csv")
    if "timestamp_local" in meta.columns:
        meta["timestamp_local"] = pd.to_datetime(meta["timestamp_local"])
    bucket_ids = np.asarray(np.load(d / "origin_bucket_ids.npy"), dtype=np.int16)

    return {
        "z": z,
        "segment_ids": segment_ids,
        "timestamps": timestamps,
        "meta": meta,
        "origin_bucket_ids": bucket_ids,
        "common_dir": common_dir,
        "split_name": split,
    }


def _session_groups(meta: pd.DataFrame) -> List[np.ndarray]:
    if "session_id" in meta.columns:
        groups = []
        for _, sub in meta.groupby("session_id", sort=False):
            idx = sub.index.to_numpy(dtype=np.int64)
            if len(idx):
                groups.append(idx)
        return groups
    return [np.arange(len(meta), dtype=np.int64)]


def iter_eval_pairs(meta: pd.DataFrame, horizon: int):
    h = int(horizon)
    for idx in _session_groups(meta):
        if len(idx) <= h:
            continue
        for pos in range(0, len(idx) - h):
            yield int(idx[pos]), int(idx[pos + h])


def batch_vector_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    diff = y_pred - y_true
    mse = float(np.mean(diff ** 2))
    return {
        "mae": float(np.mean(np.abs(diff))),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
    }


def build_g_model(method_name: str, train: Dict[str, Any], val: Dict[str, Any], test: Dict[str, Any]) -> Dict[str, Any]:
    common_dir = Path(train["common_dir"])
    graphs_dir = common_dir / "graphs"
    horizons = np.asarray(np.load(graphs_dir / "available_horizons.npy"), dtype=np.int16)
    bucket_table = pd.read_csv(graphs_dir / "bucket_table.csv")

    graph_paths: Dict[int, Path] = {}
    for h in horizons.tolist():
        p = graphs_dir / f"G_bucket_h{int(h):03d}.npy"
        if not p.exists():
            raise FileNotFoundError(f"Missing graph bank for horizon {h}: {p}")
        graph_paths[int(h)] = p

    return {
        "method": method_name,
        "common_dir": common_dir,
        "graphs_dir": graphs_dir,
        "available_horizons": set(map(int, horizons.tolist())),
        "bucket_table": bucket_table,
        "graph_paths": graph_paths,
        "graph_cache": {},
        "semantics": "bucket-dynamic train-only Granger graph: G_bucket_h[h][bucket,target,source]",
    }


def _load_graph_bank(g_model: Dict[str, Any], horizon: int) -> np.ndarray:
    h = int(horizon)
    cache: Dict[int, np.ndarray] = g_model.setdefault("graph_cache", {})
    if h not in cache:
        if h not in g_model["graph_paths"]:
            raise IndexError(
                f"Horizon {h} is not available in dynamic Granger graph. "
                f"Available={sorted(g_model['available_horizons'])}"
            )
        cache[h] = np.load(g_model["graph_paths"][h], mmap_mode="r")
    return cache[h]


def predict_G_method(
    method_name: str,
    g_model: Dict[str, Any],
    split_name: str,
    split_data: Dict[str, Any],
    origin_idx: int,
    target_idx: int,
    horizon: int,
) -> np.ndarray:
    bank = _load_graph_bank(g_model, int(horizon))
    bucket_ids = np.asarray(split_data["origin_bucket_ids"], dtype=np.int16)
    b = int(bucket_ids[int(origin_idx)])
    if b < 0 or b >= bank.shape[0]:
        raise IndexError(f"bucket_id {b} out of range for graph bank shape={bank.shape}")
    G = bank[b]
    if "_node_idx" in split_data:
        idx = np.asarray(split_data["_node_idx"], dtype=np.int64)
        G = G[idx, :][:, idx]
    return np.asarray(G, dtype=np.float32)
