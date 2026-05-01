"""
Branch B method module: Granger-GT.

This file is loaded by 06B_branchB_run_xt_forecast_topk_gt.py.
It expects prepared data from:

    ml_core/src/data_processing/outputs/branchB/osm_edge_granger_like_branchA

where G_weight_series.npy is NOT time-indexed correlation Rt. Instead:

    G_weight_series[h, target, source]

stores a static train-only Granger-style predictive influence graph for forecast
horizon h.

The downstream forecast remains:

    X_hat[t+h] = A_h X_t + B_h(TopK(G_granger[h]) X_t)

where A_h and B_h are learned by the wrapper's MultiTaskElasticNet.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

# Hyperparameters used by the wrapper's MultiTaskElasticNet.
ALPHA = 0.001
L1_RATIO = 0.5
MAX_ITER = 500
TOL = 1e-2
SELECTION = "random"
RANDOM_STATE = 42


# -----------------------------------------------------------------------------
# Basic utilities expected by 06B wrapper
# -----------------------------------------------------------------------------
def check_branchB_common_dir_ready(common_dir: Path) -> None:
    required = []
    for split in ["train", "val", "test"]:
        d = Path(common_dir) / split
        for name in [
            "G_weight_series.npy",
            "z.npy",
            "segment_ids.npy",
            "timestamps.npy",
            "G_series_meta.csv",
        ]:
            required.append(d / name)
    missing = [p for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing Granger prepared Branch-B files:\n" + "\n".join(map(str, missing)))


def load_gt_split(common_dir: Path, split: str) -> Dict[str, Any]:
    d = Path(common_dir) / split
    check_files = [
        d / "G_weight_series.npy",
        d / "z.npy",
        d / "segment_ids.npy",
        d / "timestamps.npy",
        d / "G_series_meta.csv",
    ]
    missing = [p for p in check_files if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing files for split={}:\n{}".format(split, "\n".join(map(str, missing))))

    G = np.load(d / "G_weight_series.npy", mmap_mode="r")
    L = None
    if (d / "G_best_lag_series.npy").exists():
        L = np.load(d / "G_best_lag_series.npy", mmap_mode="r")

    z = np.load(d / "z.npy", mmap_mode="r")
    segment_ids = np.asarray(np.load(d / "segment_ids.npy"), dtype=np.int64)
    timestamps = pd.to_datetime(np.load(d / "timestamps.npy"))
    meta = pd.read_csv(d / "G_series_meta.csv")
    if "timestamp_local" in meta.columns:
        meta["timestamp_local"] = pd.to_datetime(meta["timestamp_local"])

    out: Dict[str, Any] = {
        "G_weight_series": G,
        "z": z,
        "segment_ids": segment_ids,
        "timestamps": timestamps,
        "meta": meta,
    }
    if L is not None:
        out["G_best_lag_series"] = L
    return out


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
    mae = float(np.mean(np.abs(diff)))
    mse = float(np.mean(diff ** 2))
    rmse = float(np.sqrt(mse))
    return {"mae": mae, "mse": mse, "rmse": rmse}


# -----------------------------------------------------------------------------
# Granger graph model interface expected by 06B wrapper
# -----------------------------------------------------------------------------
def build_g_model(method_name: str, train: Dict[str, Any], val: Dict[str, Any], test: Dict[str, Any]) -> Dict[str, Any]:
    G = train["G_weight_series"]
    if G.ndim != 3 or G.shape[1] != G.shape[2]:
        raise ValueError(f"Expected G_weight_series shape [H+1,N,N], got {G.shape}")
    return {
        "method": method_name,
        "max_horizon_index": int(G.shape[0] - 1),
        "n_segments": int(G.shape[1]),
        "semantics": "static train-only Granger graph indexed by horizon: G[h,target,source]",
    }


def predict_G_method(
    method_name: str,
    g_model: Dict[str, Any],
    split_name: str,
    split_data: Dict[str, Any],
    origin_idx: int,
    target_idx: int,
    horizon: int,
) -> np.ndarray:
    G_series = split_data["G_weight_series"]
    h = int(horizon)
    if h < 0 or h >= G_series.shape[0]:
        raise IndexError(
            f"Horizon {h} is outside Granger G horizon axis with shape={G_series.shape}. "
            "Re-run prepare_branchB_osm_edge_granger_like_branchA.py with larger --horizons."
        )
    return np.asarray(G_series[h], dtype=np.float32)
