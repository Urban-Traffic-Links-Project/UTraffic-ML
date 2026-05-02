"""
Branch B method module: Sparse TVP-VAR-Gt.
Practical TVP-like sparse VAR on PCA scores of Granger-Gt with recency-weighted fitting.
Includes safe split alignment to avoid index mismatch.
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

try:
    from sklearn.linear_model import MultiTaskElasticNet
except Exception:
    MultiTaskElasticNet = None

SPARSE_VAR_RANK = 8
SPARSE_VAR_ALPHA = 0.002
SPARSE_VAR_L1_RATIO = 0.5
SPARSE_VAR_MAX_ITER = 3000
MAX_TRAIN_MATS = 0  # 0 = all


def _flatten_G_series(G: np.ndarray, indices: np.ndarray) -> np.ndarray:
    mats = []
    T = int(G.shape[0])
    for t in indices.tolist():
        tt = _safe_idx(int(t), T)
        mats.append(np.asarray(G[tt], dtype=np.float32).reshape(-1))
    return np.stack(mats, axis=0).astype(np.float32)


def _fit_pca_snapshot(X: np.ndarray, rank: int) -> Dict[str, np.ndarray]:
    X = np.asarray(X, dtype=np.float32)
    T, D = X.shape
    r = min(int(rank), max(1, T - 1), D)
    mean = X.mean(axis=0).astype(np.float32)
    Xc = X - mean[None, :]
    if T <= 1:
        comps = np.zeros((1, D), dtype=np.float32)
        scores = np.zeros((T, 1), dtype=np.float32)
        return {"mean": mean, "components": comps, "scores": scores}
    K = (Xc @ Xc.T).astype(np.float64) / max(1, T - 1)
    evals, evecs = np.linalg.eigh(K)
    order = np.argsort(evals)[::-1]
    evals = np.maximum(evals[order][:r], EPS)
    evecs = evecs[:, order][:, :r]
    comps = (evecs.T @ Xc.astype(np.float64)) / np.sqrt(evals[:, None] * max(1, T - 1))
    comps = comps.astype(np.float32)
    scores = (Xc @ comps.T).astype(np.float32)
    return {"mean": mean, "components": comps, "scores": scores}


def _fit_sparse_var_by_horizon(scores: np.ndarray, meta: pd.DataFrame, horizons: List[int], alpha: float, l1_ratio: float, max_iter: int, weighted: bool=False) -> Dict[int, Dict[str, np.ndarray]]:
    scores = np.asarray(scores, dtype=np.float32)
    r = int(scores.shape[1])
    models = {}
    for h in horizons:
        X_rows, Y_rows, w_rows = [], [], []
        pairs = list(iter_eval_pairs(meta, int(h)))
        n_pairs = len(pairs)
        for k, (origin_idx, target_idx) in enumerate(pairs):
            if origin_idx < len(scores) and target_idx < len(scores):
                X_rows.append(scores[origin_idx])
                Y_rows.append(scores[target_idx])
                # More recent observations get larger weights for TVP-like variant.
                w_rows.append(0.98 ** max(0, n_pairs - 1 - k) if weighted else 1.0)
        if len(X_rows) < 2:
            models[int(h)] = {"coef": np.eye(r, dtype=np.float32), "intercept": np.zeros(r, dtype=np.float32), "solver": np.array(["identity"])}
            continue
        X = np.stack(X_rows).astype(np.float32)
        Y = np.stack(Y_rows).astype(np.float32)
        sw = np.asarray(w_rows, dtype=np.float32)
        if MultiTaskElasticNet is not None:
            try:
                model = MultiTaskElasticNet(
                    alpha=float(alpha), l1_ratio=float(l1_ratio), fit_intercept=True,
                    max_iter=int(max_iter), tol=1e-3, selection="random", random_state=42
                )
                # sklearn supports sample_weight for multi-task elasticnet in recent versions.
                try:
                    model.fit(X, Y, sample_weight=sw)
                except TypeError:
                    model.fit(X, Y)
                coef = np.asarray(model.coef_.T, dtype=np.float32)
                intercept = np.asarray(model.intercept_, dtype=np.float32)
                models[int(h)] = {"coef": coef, "intercept": intercept, "solver": np.array(["multitask_elasticnet"])}
                continue
            except Exception as exc:
                print(f"[Sparse VAR-Gt][WARN] ElasticNet failed h={h}: {exc}; fallback ridge", flush=True)
        # Ridge fallback, weighted if requested.
        X64, Y64 = X.astype(np.float64), Y.astype(np.float64)
        sw64 = sw.astype(np.float64)
        Xw = X64 * np.sqrt(sw64[:, None])
        Yw = Y64 * np.sqrt(sw64[:, None])
        ridge = max(float(alpha), 1e-6)
        A = Xw.T @ Xw + ridge * np.eye(r, dtype=np.float64)
        B = Xw.T @ Yw
        try:
            coef = np.linalg.solve(A, B)
        except np.linalg.LinAlgError:
            coef = np.linalg.pinv(A) @ B
        intercept = Y64.mean(axis=0) - X64.mean(axis=0) @ coef
        models[int(h)] = {"coef": coef.astype(np.float32), "intercept": intercept.astype(np.float32), "solver": np.array(["ridge"])}
    return models


def build_g_model(method_name: str, train: Dict[str, Any], val: Dict[str, Any], test: Dict[str, Any]) -> Dict[str, Any]:
    G_train = train["G_weight_series"]
    T, N, _ = G_train.shape
    rank = min(int(SPARSE_VAR_RANK), max(1, T - 1))
    max_mats = int(MAX_TRAIN_MATS)
    if max_mats > 0 and T > max_mats:
        indices = np.unique(np.linspace(0, T - 1, max_mats).round().astype(np.int64))
    else:
        indices = np.arange(T, dtype=np.int64)
    print(f"[{method_name}] loading train G matrices: selected={len(indices)}/{T}, N={N}, rank={rank}", flush=True)
    X = _flatten_G_series(G_train, indices)
    pca = _fit_pca_snapshot(X, rank=rank)
    if len(indices) == T and np.array_equal(indices, np.arange(T)):
        full_scores = pca["scores"]
    else:
        Xfull = _flatten_G_series(G_train, np.arange(T, dtype=np.int64))
        full_scores = ((Xfull - pca["mean"][None, :]) @ pca["components"].T).astype(np.float32)
    horizons = list(map(int, globals().get("HORIZONS", list(range(1, 10)))))
    weighted = (method_name == "sparse_tvpvar_gt")
    coef_by_h = _fit_sparse_var_by_horizon(full_scores, train["meta"], horizons, SPARSE_VAR_ALPHA, SPARSE_VAR_L1_RATIO, SPARSE_VAR_MAX_ITER, weighted=weighted)
    return {"mean": pca["mean"], "components": pca["components"], "coef_by_h": coef_by_h, "n_segments": int(N), "method": method_name}


def _project_one(G: np.ndarray, mean: np.ndarray, components: np.ndarray) -> np.ndarray:
    x = np.asarray(G, dtype=np.float32).reshape(-1)
    return ((x - mean) @ components.T).astype(np.float32)


def _reconstruct_one(score: np.ndarray, mean: np.ndarray, components: np.ndarray, N: int) -> np.ndarray:
    x = mean + np.asarray(score, dtype=np.float32) @ components
    G = x.reshape(N, N).astype(np.float32)
    np.fill_diagonal(G, 0.0)
    return np.nan_to_num(G, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def predict_G_method(method_name: str, g_model: Dict[str, Any], split_name: str, split_data: Dict[str, Any], origin_idx: int, target_idx: int, horizon: int) -> np.ndarray:
    G_series = split_data["G_weight_series"]
    G_origin = np.asarray(G_series[_safe_idx(origin_idx, G_series.shape[0])], dtype=np.float32)
    N = int(G_origin.shape[0])
    mean = g_model["mean"]
    comps = g_model["components"]
    f = _project_one(G_origin, mean, comps)
    model_h = g_model["coef_by_h"].get(int(horizon))
    if model_h is None:
        coef = np.eye(len(f), dtype=np.float32)
        intercept = np.zeros(len(f), dtype=np.float32)
    else:
        coef = np.asarray(model_h.get("coef"), dtype=np.float32)
        intercept = np.asarray(model_h.get("intercept", np.zeros(len(f), dtype=np.float32)), dtype=np.float32)
    f_hat = f @ coef + intercept
    return _reconstruct_one(f_hat, mean, comps, N=N)
