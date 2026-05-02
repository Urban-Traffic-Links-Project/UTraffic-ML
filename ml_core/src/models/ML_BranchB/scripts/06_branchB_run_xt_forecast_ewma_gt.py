"""
Branch B method module: EWMA-Gt.
Builds an exponentially weighted moving average of Granger-Gt series per split.
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
EPS = 1e-8


def check_branchB_common_dir_ready(common_dir: Path) -> None:
    common_dir = Path(common_dir)
    required = []
    for split in ["train", "val", "test"]:
        d = common_dir / split
        for name in ["z.npy", "segment_ids.npy", "timestamps.npy", "G_series_meta.csv", "G_weight_series.npy", "G_best_lag_series.npy"]:
            required.append(d / name)
    missing = [p for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing standard Gt prepared files:\n" + "\n".join(map(str, missing)))


def _safe_datetime(arr):
    arr = np.asarray(arr).astype(str)
    s = pd.Series(arr).str.replace("__", " ", regex=False).str.replace("Slot_", "", regex=False)
    # convert YYYY-MM-DD 0600 -> YYYY-MM-DD 06:00:00
    s = s.str.replace(r"(\d{4}-\d{2}-\d{2})\s+(\d{2})(\d{2})", r"\1 \2:\3:00", regex=True)
    return pd.to_datetime(s, errors="coerce")


def load_gt_split(common_dir: Path, split: str) -> Dict[str, Any]:
    common_dir = Path(common_dir)
    d = common_dir / split
    check_branchB_common_dir_ready(common_dir)
    z = np.load(d / "z.npy", mmap_mode="r")
    G = np.load(d / "G_weight_series.npy", mmap_mode="r")
    L = np.load(d / "G_best_lag_series.npy", mmap_mode="r")
    segment_ids = np.asarray(np.load(d / "segment_ids.npy"), dtype=np.int64)
    timestamps_raw = np.asarray(np.load(d / "timestamps.npy")).astype(str)
    meta = pd.read_csv(d / "G_series_meta.csv")
    # Critical: keep meta/z/G aligned. Some earlier prepared outputs may have mismatched lengths.
    m = int(min(len(meta), z.shape[0], G.shape[0], L.shape[0], len(timestamps_raw)))
    if m <= 0:
        raise ValueError(f"Empty split after alignment: {split}")
    if len(meta) != m or z.shape[0] != m or G.shape[0] != m:
        print(f"[WARN] Aligning split={split}: meta={len(meta)}, z={z.shape[0]}, G={G.shape[0]} -> {m}", flush=True)
    meta = meta.iloc[:m].reset_index(drop=True)
    if "timestamp_local" in meta.columns:
        meta["timestamp_local"] = pd.to_datetime(meta["timestamp_local"], errors="coerce")
    if "session_id" not in meta.columns:
        if "date_key" in meta.columns:
            meta["session_id"] = meta["date_key"].astype(str)
        else:
            meta["session_id"] = "session_0"
    return {
        "z": z[:m],
        "G_weight_series": G[:m],
        "G_best_lag_series": L[:m],
        "segment_ids": segment_ids,
        "timestamps": _safe_datetime(timestamps_raw[:m]),
        "meta": meta,
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
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(mse))
    return {"mae": mae, "mse": mse, "rmse": rmse}


def _safe_idx(i: int, length: int) -> int:
    if length <= 0:
        return 0
    return int(max(0, min(int(i), int(length) - 1)))

EWMA_ALPHA = 0.30


def _build_ewma(G, alpha: float = EWMA_ALPHA) -> np.ndarray:
    G = np.asarray(G, dtype=np.float32)
    T = G.shape[0]
    out = np.zeros_like(G, dtype=np.float32)
    if T == 0:
        return out
    out[0] = G[0]
    a = float(alpha)
    for t in range(1, T):
        out[t] = a * G[t] + (1.0 - a) * out[t - 1]
    return out


def build_g_model(method_name: str, train: Dict[str, Any], val: Dict[str, Any], test: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "method": "ewma_gt",
        "alpha": float(EWMA_ALPHA),
        "ewma_by_split": {
            "train": _build_ewma(train["G_weight_series"]),
            "val": _build_ewma(val["G_weight_series"]),
            "test": _build_ewma(test["G_weight_series"]),
        },
    }


def predict_G_method(method_name: str, g_model: Dict[str, Any], split_name: str, split_data: Dict[str, Any], origin_idx: int, target_idx: int, horizon: int) -> np.ndarray:
    E = g_model["ewma_by_split"][split_name]
    return np.asarray(E[_safe_idx(origin_idx, E.shape[0])], dtype=np.float32)

# ---------------------------------------------------------------------------
# OVERRIDE: fair session-aware evaluation pairs for Granger-Gt timeline.
# A prediction origin is allowed only if prepare step marked can_predict_granger=True,
# meaning it has enough previous p history in the same day. Target must stay in same day.
# ---------------------------------------------------------------------------
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
            origin = int(idx[pos])
            target = int(idx[pos + h])
            if "can_predict_granger" in meta.columns:
                if not bool(meta.loc[origin, "can_predict_granger"]):
                    continue
            elif "can_predict" in meta.columns:
                if not bool(meta.loc[origin, "can_predict"]):
                    continue
            yield origin, target
