"""
Branch B method module: Sparse VAR-Gt.

This file is loaded by 06B_branchB_run_xt_forecast_topk_gt.py.
It expects standard Branch-B prepared data where G_weight_series.npy is a
Granger-Gt series, not lagged-correlation Rt.

Method idea
-----------
Fit a practical low-dimensional Sparse VAR on flattened Gt matrices:
    G_t -> factor score f_t by PCA/method-of-snapshots
    f_{t+h} = f_t B_h + e
    G_hat_{t+h|t} = mean_G + f_hat_{t+h} components

Then downstream forecast remains:
    X_hat[t+h] = A_h X_t + B_h(TopK(G_hat[t,h]) X_t)

This is a practical Sparse VAR-Gt baseline. For very large full-N, use a small rank and
expect high disk/RAM usage because G_t has N x N entries.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    from sklearn.linear_model import MultiTaskElasticNet
except Exception:
    MultiTaskElasticNet = None

ALPHA = 0.001
L1_RATIO = 0.5
MAX_ITER = 500
TOL = 1e-2
SELECTION = "random"
RANDOM_STATE = 42

SPARSE_SPARSE_VAR_RANK = 16
SPARSE_VAR_ALPHA = 0.001
SPARSE_VAR_L1_RATIO = 0.5
SPARSE_VAR_MAX_ITER = 1000
MAX_TRAIN_MATS = 0  # 0 = use all train matrices
EPS = 1e-8


def check_branchB_common_dir_ready(common_dir: Path) -> None:
    common_dir = Path(common_dir)
    required = []
    for split in ["train", "val", "test"]:
        d = common_dir / split
        for name in [
            "z.npy",
            "segment_ids.npy",
            "timestamps.npy",
            "G_series_meta.csv",
            "G_weight_series.npy",
            "G_best_lag_series.npy",
        ]:
            required.append(d / name)
    missing = [p for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing standard Gt prepared files:\n" + "\n".join(map(str, missing)))


def load_gt_split(common_dir: Path, split: str) -> Dict[str, Any]:
    common_dir = Path(common_dir)
    d = common_dir / split
    check_branchB_common_dir_ready(common_dir)

    z = np.load(d / "z.npy", mmap_mode="r")
    G = np.load(d / "G_weight_series.npy", mmap_mode="r")
    L = np.load(d / "G_best_lag_series.npy", mmap_mode="r")
    segment_ids = np.asarray(np.load(d / "segment_ids.npy"), dtype=np.int64)
    timestamps = pd.to_datetime(np.load(d / "timestamps.npy"), errors="coerce")
    meta = pd.read_csv(d / "G_series_meta.csv")
    if "timestamp_local" in meta.columns:
        meta["timestamp_local"] = pd.to_datetime(meta["timestamp_local"], errors="coerce")

    return {
        "z": z,
        "G_weight_series": G,
        "G_best_lag_series": L,
        "segment_ids": segment_ids,
        "timestamps": timestamps,
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


def _flatten_G_series(G: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Load selected G matrices into a 2D float32 array [M, N*N]."""
    mats = []
    for t in indices.tolist():
        mats.append(np.asarray(G[int(t)], dtype=np.float32).reshape(-1))
    return np.stack(mats, axis=0).astype(np.float32)


def _fit_pca_snapshot(X: np.ndarray, rank: int) -> Dict[str, np.ndarray]:
    """Method-of-snapshots PCA for X [T,D]."""
    X = np.asarray(X, dtype=np.float32)
    T, D = X.shape
    r = min(int(rank), T - 1, D)
    if r <= 0:
        raise ValueError(f"Invalid PCA rank={rank} for X shape={X.shape}")

    mean = X.mean(axis=0).astype(np.float32)
    Xc = X - mean[None, :]
    # Snapshot Gram is T x T, much smaller than D x D.
    K = (Xc @ Xc.T).astype(np.float64) / max(1, T - 1)
    evals, evecs = np.linalg.eigh(K)
    order = np.argsort(evals)[::-1]
    evals = np.maximum(evals[order][:r], EPS)
    evecs = evecs[:, order][:, :r]
    # components[r,D]
    components = (evecs.T @ Xc.astype(np.float64)) / np.sqrt(evals[:, None] * max(1, T - 1))
    components = components.astype(np.float32)
    scores = (Xc @ components.T).astype(np.float32)
    return {"mean": mean, "components": components, "scores": scores, "evals": evals.astype(np.float32)}


def _fit_sparse_var_by_horizon(
    scores: np.ndarray,
    meta: pd.DataFrame,
    horizons: List[int],
    alpha: float,
    l1_ratio: float,
    max_iter: int,
) -> Dict[int, Dict[str, np.ndarray]]:
    """
    Fit direct sparse VAR by horizon in the low-dimensional PCA score space.

    For each horizon h:
        score_{t+h} = intercept_h + score_t B_h + error

    B_h is estimated with MultiTaskElasticNet, so many coefficients are shrunk
    to zero. This is the sparse VAR part. If sklearn is unavailable or fitting
    fails, it falls back to a tiny ridge-like least-squares solution.
    """
    scores = np.asarray(scores, dtype=np.float32)
    models: Dict[int, Dict[str, np.ndarray]] = {}
    r = scores.shape[1]
    for h in horizons:
        X_rows = []
        Y_rows = []
        for origin_idx, target_idx in iter_eval_pairs(meta, int(h)):
            if origin_idx < len(scores) and target_idx < len(scores):
                X_rows.append(scores[origin_idx])
                Y_rows.append(scores[target_idx])
        if not X_rows:
            models[int(h)] = {
                "coef": np.eye(r, dtype=np.float32),
                "intercept": np.zeros(r, dtype=np.float32),
                "solver": np.array(["identity_fallback"]),
            }
            continue

        X = np.stack(X_rows, axis=0).astype(np.float32)
        Y = np.stack(Y_rows, axis=0).astype(np.float32)

        if MultiTaskElasticNet is not None:
            try:
                model = MultiTaskElasticNet(
                    alpha=float(alpha),
                    l1_ratio=float(l1_ratio),
                    fit_intercept=True,
                    max_iter=int(max_iter),
                    tol=1e-3,
                    selection="random",
                    random_state=42,
                )
                model.fit(X, Y)
                # sklearn coef_: [target_dim, feature_dim]. We use f @ coef + intercept.
                coef = np.asarray(model.coef_.T, dtype=np.float32)
                intercept = np.asarray(model.intercept_, dtype=np.float32)
                models[int(h)] = {
                    "coef": coef,
                    "intercept": intercept,
                    "solver": np.array(["multitask_elasticnet"]),
                    "alpha": np.array([float(alpha)], dtype=np.float32),
                    "l1_ratio": np.array([float(l1_ratio)], dtype=np.float32),
                }
                continue
            except Exception as exc:
                print(f"[Sparse VAR-Gt][WARN] ElasticNet failed for h={h}: {exc}. Fallback to ridge.", flush=True)

        # Fallback: ridge-like direct VAR.
        X64 = X.astype(np.float64)
        Y64 = Y.astype(np.float64)
        ridge = max(float(alpha), 1e-6)
        A = X64.T @ X64 + ridge * np.eye(r, dtype=np.float64)
        B = X64.T @ Y64
        try:
            coef = np.linalg.solve(A, B)
        except np.linalg.LinAlgError:
            coef = np.linalg.pinv(A) @ B
        intercept = Y64.mean(axis=0) - X64.mean(axis=0) @ coef
        models[int(h)] = {
            "coef": coef.astype(np.float32),
            "intercept": intercept.astype(np.float32),
            "solver": np.array(["ridge_fallback"]),
        }
    return models


def build_g_model(method_name: str, train: Dict[str, Any], val: Dict[str, Any], test: Dict[str, Any]) -> Dict[str, Any]:
    G_train = train["G_weight_series"]
    T, N, _ = G_train.shape
    rank = min(int(SPARSE_SPARSE_VAR_RANK), max(1, T - 2))
    max_mats = int(MAX_TRAIN_MATS)
    if max_mats > 0 and T > max_mats:
        # Use evenly spaced train matrices to keep memory bounded.
        indices = np.linspace(0, T - 1, max_mats).round().astype(np.int64)
        indices = np.unique(indices)
    else:
        indices = np.arange(T, dtype=np.int64)

    print(f"[Sparse VAR-Gt] loading train G matrices: selected={len(indices)}/{T}, N={N}, rank={rank}", flush=True)
    X = _flatten_G_series(G_train, indices)
    pca = _fit_pca_snapshot(X, rank=rank)

    # Need scores aligned to full train time indices. Project all train matrices if sampled.
    if len(indices) == T and np.array_equal(indices, np.arange(T)):
        full_scores = pca["scores"]
    else:
        mean = pca["mean"]
        comps = pca["components"]
        Xfull = _flatten_G_series(G_train, np.arange(T, dtype=np.int64))
        full_scores = ((Xfull - mean[None, :]) @ comps.T).astype(np.float32)

    horizons = list(map(int, globals().get("HORIZONS", list(range(1, 10)))))
    coef_by_h = _fit_sparse_var_by_horizon(
        full_scores,
        train["meta"],
        horizons=horizons,
        alpha=float(SPARSE_VAR_ALPHA),
        l1_ratio=float(SPARSE_VAR_L1_RATIO),
        max_iter=int(SPARSE_VAR_MAX_ITER),
    )

    return {
        "mean": pca["mean"],
        "components": pca["components"],
        "coef_by_h": coef_by_h,
        "rank": int(rank),
        "n_segments": int(N),
        "horizons": horizons,
        "method": "pca_sparse_var_gt",
        "alpha": float(SPARSE_VAR_ALPHA),
        "l1_ratio": float(SPARSE_VAR_L1_RATIO),
    }


def _project_one(G: np.ndarray, mean: np.ndarray, components: np.ndarray) -> np.ndarray:
    x = np.asarray(G, dtype=np.float32).reshape(-1)
    return ((x - mean) @ components.T).astype(np.float32)


def _reconstruct_one(score: np.ndarray, mean: np.ndarray, components: np.ndarray, N: int) -> np.ndarray:
    x = mean + np.asarray(score, dtype=np.float32) @ components
    G = x.reshape(N, N).astype(np.float32)
    np.fill_diagonal(G, 0.0)
    return np.nan_to_num(G, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def predict_G_method(
    method_name: str,
    g_model: Dict[str, Any],
    split_name: str,
    split_data: Dict[str, Any],
    origin_idx: int,
    target_idx: int,
    horizon: int,
) -> np.ndarray:
    G_origin = split_data["G_weight_series"][int(origin_idx)]
    N = int(G_origin.shape[0])
    mean = g_model["mean"]
    components = g_model["components"]
    f = _project_one(G_origin, mean, components)
    model_h = g_model["coef_by_h"].get(int(horizon))
    if model_h is None:
        coef = np.eye(len(f), dtype=np.float32)
        intercept = np.zeros(len(f), dtype=np.float32)
    elif isinstance(model_h, dict):
        coef = np.asarray(model_h.get("coef"), dtype=np.float32)
        intercept = np.asarray(model_h.get("intercept", np.zeros(len(f), dtype=np.float32)), dtype=np.float32)
    else:
        # Backward compatible with old plain coefficient matrix.
        coef = np.asarray(model_h, dtype=np.float32)
        intercept = np.zeros(len(f), dtype=np.float32)
    f_hat = f @ coef + intercept
    return _reconstruct_one(f_hat, mean, components, N=N)
