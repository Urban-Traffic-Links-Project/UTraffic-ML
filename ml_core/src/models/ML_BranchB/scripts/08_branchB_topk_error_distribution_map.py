
from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from scipy.stats import wasserstein_distance, ks_2samp, spearmanr
except Exception:
    wasserstein_distance = None
    ks_2samp = None
    spearmanr = None

try:
    import folium
except Exception:
    folium = None


EPS = 1e-8
DEFAULT_LAGS = [1, 3, 6, 9]
DEFAULT_SPLITS = ["val", "test"]
DEFAULT_TOPK = [5, 10, 50, 100, 500]
DEFAULT_METHODS = [
    "true_gt",
    "persistence_gt",
    "ewma_gt",
    "factorized_var_gt",
    "factorized_mar_gt",
    "factorized_tvpvar_gt",
]
DEFAULT_EWMA_ALPHA = 0.30
DEFAULT_RANK = 12
DEFAULT_VAR_RIDGE = 1e-2
DEFAULT_MAR_RIDGE = 1e-2
DEFAULT_TVPVAR_RIDGE = 5e-2
DEFAULT_TVPVAR_FORGETTING = 0.98
DEFAULT_STABILITY_TARGET = 0.98
SEED = 42

METHOD_LABELS = {
    "true_gt": "True-G",
    "persistence_gt": "Persistence-G",
    "ewma_gt": "EWMA-G",
    "no_gt": "No-G / Zero-G",
    "factorized_var_gt": "Factorized VAR-G",
    "factorized_mar_gt": "Factorized MAR-G",
    "factorized_tvpvar_gt": "Factorized TVP-VAR-G",
}


def now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def print_stage(title: str) -> None:
    print("\n" + "=" * 96)
    print(f"{now_str()} | {title}")
    print("=" * 96)


def parse_int_list(s: str) -> List[int]:
    out = []
    for part in str(s).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def parse_str_list(s: str) -> List[str]:
    return [x.strip() for x in str(s).split(",") if x.strip()]


def find_project_root() -> Path:
    cwd = Path.cwd().resolve()
    candidates = [cwd, *cwd.parents, Path("/kaggle/working/UTraffic-ML"), Path("/kaggle/working")]
    for p in candidates:
        if (p / "ml_core").exists() and (p / "dataset").exists():
            return p
        if p.name == "UTraffic-ML":
            return p
        if (p / "UTraffic-ML").exists():
            pp = p / "UTraffic-ML"
            if (pp / "ml_core").exists():
                return pp
    return cwd


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def check_branchB_common_dir_ready(common_dir: Path) -> None:
    required = []
    for split in ["train", "val", "test"]:
        for name in [
            "G_weight_series.npy",
            "G_best_lag_series.npy",
            "z.npy",
            "segment_ids.npy",
            "timestamps.npy",
            "G_series_meta.csv",
            "raw_meta.csv",
        ]:
            required.append(common_dir / split / name)
    missing = [p for p in required if not p.exists()]
    if missing:
        msg = "\n".join(str(p) for p in missing[:30])
        raise FileNotFoundError(
            "Branch B prepared data is incomplete.\n"
            f"Expected COMMON_DIR: {common_dir}\n\n"
            f"Missing files:\n{msg}\n\n"
            "Run:\n"
            "  python ml_core/src/data_processing/prepare_branchB_osm_edge_gt_like_branchA.py --overwrite\n"
        )


def load_gt_split(common_dir: Path, split_name: str, mmap_mode: str = "r") -> Dict[str, Any]:
    split_dir = common_dir / split_name
    G_weight_series = np.load(split_dir / "G_weight_series.npy", mmap_mode=mmap_mode)
    G_best_lag_series = np.load(split_dir / "G_best_lag_series.npy", mmap_mode=mmap_mode)
    z = np.load(split_dir / "z.npy", mmap_mode=mmap_mode)
    segment_ids = np.load(split_dir / "segment_ids.npy").astype(np.int64)
    timestamps = pd.to_datetime(np.load(split_dir / "timestamps.npy"))
    meta = pd.read_csv(split_dir / "G_series_meta.csv")
    if "timestamp_local" in meta.columns:
        meta["timestamp_local"] = pd.to_datetime(meta["timestamp_local"])
    raw_meta = None
    raw_meta_path = split_dir / "raw_meta.csv"
    if raw_meta_path.exists():
        raw_meta = pd.read_csv(raw_meta_path)
        if "timestamp_local" in raw_meta.columns:
            raw_meta["timestamp_local"] = pd.to_datetime(raw_meta["timestamp_local"])
    return {
        "G_weight_series": G_weight_series,
        "G_best_lag_series": G_best_lag_series,
        "z": z,
        "segment_ids": segment_ids,
        "timestamps": timestamps,
        "meta": meta,
        "raw_meta": raw_meta,
    }


def subset_split(data: Dict[str, Any], node_idx: Optional[np.ndarray]) -> Dict[str, Any]:
    if node_idx is None:
        return data
    node_idx = np.asarray(node_idx, dtype=np.int64)
    out = dict(data)
    out["node_idx"] = node_idx
    out["z"] = np.asarray(data["z"][:, node_idx], dtype=np.float32)
    out["segment_ids"] = np.asarray(data["segment_ids"])[node_idx].astype(np.int64)
    return out


def get_G(data: Dict[str, Any], t: int) -> np.ndarray:
    G_full = data["G_weight_series"][int(t)]
    node_idx = data.get("node_idx", None)
    if node_idx is None:
        return np.asarray(G_full, dtype=np.float32)
    return np.asarray(G_full[np.ix_(node_idx, node_idx)], dtype=np.float32)


def _session_index_groups(meta: pd.DataFrame) -> List[np.ndarray]:
    if "session_id" not in meta.columns:
        return [np.arange(len(meta), dtype=np.int64)]
    groups = []
    for _, sub in meta.groupby("session_id", sort=False):
        idx = sub.index.to_numpy(dtype=np.int64)
        if len(idx):
            groups.append(idx)
    return groups or [np.arange(len(meta), dtype=np.int64)]


def iter_eval_pairs(meta: pd.DataFrame, lag: int):
    if "session_id" in meta.columns:
        for idx in _session_index_groups(meta):
            if len(idx) <= lag:
                continue
            for k in range(0, len(idx) - lag):
                yield int(idx[k]), int(idx[k + lag])
    else:
        for origin_idx in range(len(meta) - lag):
            yield int(origin_idx), int(origin_idx + lag)


def sample_eval_pairs(meta: pd.DataFrame, lag: int, max_samples: int, rng: np.random.Generator) -> List[Tuple[int, int]]:
    pairs = list(iter_eval_pairs(meta, lag))
    if max_samples <= 0 or len(pairs) <= max_samples:
        return pairs
    idx = rng.choice(len(pairs), size=max_samples, replace=False)
    idx = np.sort(idx)
    return [pairs[int(i)] for i in idx]


def history_indices_for_origin(meta: pd.DataFrame, origin_idx: int) -> np.ndarray:
    if "session_id" not in meta.columns:
        return np.arange(origin_idx + 1, dtype=np.int64)
    sess = meta["session_id"].to_numpy()
    mask = (sess == sess[origin_idx]) & (np.arange(len(meta)) <= origin_idx)
    return np.where(mask)[0].astype(np.int64)


def clip_G(G: np.ndarray) -> np.ndarray:
    G = np.asarray(G, dtype=np.float32)
    np.nan_to_num(G, copy=False, nan=0.0, posinf=1.0, neginf=-1.0)
    np.clip(G, -1.0, 1.0, out=G)
    return G


def predict_G_basic(method_name: str, split_data: Dict[str, Any], origin_idx: int, target_idx: int, horizon: int,
                    alpha_ewma: float = DEFAULT_EWMA_ALPHA) -> np.ndarray:
    if method_name == "true_gt":
        return clip_G(get_G(split_data, target_idx))
    if method_name == "persistence_gt":
        return clip_G(get_G(split_data, origin_idx))
    if method_name == "ewma_gt":
        meta = split_data["meta"]
        hist_idx = history_indices_for_origin(meta, origin_idx)
        hist = np.stack([get_G(split_data, int(t)) for t in hist_idx], axis=0).astype(np.float32)
        weights = np.array([(1.0 - alpha_ewma) ** (len(hist) - 1 - k) for k in range(len(hist))], dtype=np.float64)
        weights = weights / max(weights.sum(), EPS)
        return clip_G(np.tensordot(weights, hist, axes=(0, 0)).astype(np.float32))
    if method_name == "no_gt":
        N = len(split_data["segment_ids"])
        return np.zeros((N, N), dtype=np.float32)
    raise ValueError(f"Unsupported basic method: {method_name}")


def spectral_radius(M: np.ndarray) -> float:
    arr = np.asarray(M, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    vals = np.linalg.eigvals(arr)
    return float(np.max(np.abs(vals))) if vals.size else 0.0


def stabilize_transition(Phi: np.ndarray, target: float = DEFAULT_STABILITY_TARGET) -> np.ndarray:
    Phi = np.asarray(Phi, dtype=np.float32)
    rho = spectral_radius(Phi)
    if not np.isfinite(rho) or rho <= 0.0 or rho < target:
        return Phi
    return (Phi * (target / max(rho, EPS))).astype(np.float32)


def stabilize_mar_params(A: np.ndarray, B: np.ndarray, target: float = DEFAULT_STABILITY_TARGET) -> Tuple[np.ndarray, np.ndarray]:
    A = np.asarray(A, dtype=np.float32)
    B = np.asarray(B, dtype=np.float32)
    rho_prod = spectral_radius(A) * spectral_radius(B)
    if not np.isfinite(rho_prod) or rho_prod <= 0.0 or rho_prod < target:
        return A, B
    shrink = float(np.sqrt(target / max(rho_prod, EPS)))
    return (A * shrink).astype(np.float32), (B * shrink).astype(np.float32)


def fit_fulltrain_factor_bases_streaming(train_data: Dict[str, Any], rank: int = DEFAULT_RANK) -> Dict[str, np.ndarray]:
    G_train = train_data["G_weight_series"]
    T = int(G_train.shape[0])
    N = len(train_data["segment_ids"])
    print(f"[FACTOR BASES] streaming fit: T={T}, N={N}, rank={rank}")
    progress_every = max(1, T // 10)

    mean_acc = np.zeros((N, N), dtype=np.float64)
    for t in range(T):
        mean_acc += get_G(train_data, t)
        if (t + 1) % progress_every == 0 or (t + 1) == T:
            print(f"  mean_G: {t+1}/{T}")
    mean_G = (mean_acc / max(T, 1)).astype(np.float32)
    del mean_acc

    row_cov = np.zeros((N, N), dtype=np.float64)
    col_cov = np.zeros((N, N), dtype=np.float64)
    for t in range(T):
        C = get_G(train_data, t) - mean_G
        row_cov += C @ C.T
        col_cov += C.T @ C
        if (t + 1) % progress_every == 0 or (t + 1) == T:
            print(f"  cov: {t+1}/{T}")
    row_cov = 0.5 * (row_cov + row_cov.T) / max(T, 1)
    col_cov = 0.5 * (col_cov + col_cov.T) / max(T, 1)

    print("  eigen row/col covariance ...")
    evals_u, evecs_u = np.linalg.eigh(row_cov)
    evals_v, evecs_v = np.linalg.eigh(col_cov)
    r = int(min(rank, N))
    U = evecs_u[:, np.argsort(evals_u)[::-1][:r]].astype(np.float32)
    V = evecs_v[:, np.argsort(evals_v)[::-1][:r]].astype(np.float32)
    return {"U": U, "V": V, "mean_G": mean_G, "rank": np.array([r], dtype=np.int64)}


def compress_series(data: Dict[str, Any], U: np.ndarray, V: np.ndarray, mean_G: np.ndarray) -> np.ndarray:
    T = int(data["G_weight_series"].shape[0])
    r = U.shape[1]
    X = np.empty((T, r, r), dtype=np.float32)
    for t in range(T):
        X[t] = U.T @ (get_G(data, t) - mean_G) @ V
        if (t + 1) % max(1, T // 10) == 0 or (t + 1) == T:
            print(f"  compress: {t+1}/{T}")
    return X


def fit_latent_var(X: np.ndarray, ridge: float = DEFAULT_VAR_RIDGE) -> Dict[str, np.ndarray]:
    Z = X.reshape(X.shape[0], -1).astype(np.float32)
    if len(Z) < 2:
        d = Z.shape[1]
        return {"type": "var", "c": np.zeros(d, dtype=np.float32), "Phi": np.eye(d, dtype=np.float32)}
    X0, X1 = Z[:-1], Z[1:]
    D = np.concatenate([np.ones((len(X0), 1), dtype=np.float32), X0], axis=1)
    reg = ridge * np.eye(D.shape[1], dtype=np.float32)
    reg[0, 0] = 0.0
    B = np.linalg.solve(D.T @ D + reg, D.T @ X1)
    c = B[0].astype(np.float32)
    Phi = stabilize_transition(B[1:].T.astype(np.float32))
    return {"type": "var", "c": c, "Phi": Phi}


def fit_latent_tvpvar(X: np.ndarray, ridge: float = DEFAULT_TVPVAR_RIDGE, forgetting: float = DEFAULT_TVPVAR_FORGETTING) -> Dict[str, np.ndarray]:
    Z = X.reshape(X.shape[0], -1).astype(np.float32)
    d = Z.shape[1]
    if len(Z) < 2:
        return {"type": "tvpvar", "c": np.zeros(d, dtype=np.float32), "Phi": np.eye(d, dtype=np.float32)}
    # Recursive least squares for z_next = c + Phi z.
    p = d + 1
    theta = np.zeros((p, d), dtype=np.float64)
    theta[1:, :] = np.eye(d, dtype=np.float64)
    P = (1.0 / max(ridge, EPS)) * np.eye(p, dtype=np.float64)
    lam = float(forgetting)

    for t in range(len(Z) - 1):
        x = np.concatenate([[1.0], Z[t].astype(np.float64)])
        y = Z[t + 1].astype(np.float64)
        Px = P @ x
        denom = lam + x @ Px
        K = Px / max(denom, EPS)
        err = y - x @ theta
        theta = theta + np.outer(K, err)
        P = (P - np.outer(K, x @ P)) / lam

    c = theta[0].astype(np.float32)
    Phi = stabilize_transition(theta[1:].T.astype(np.float32))
    return {"type": "tvpvar", "c": c, "Phi": Phi}


def fit_latent_mar(X: np.ndarray, ridge: float = DEFAULT_MAR_RIDGE, n_iter: int = 4) -> Dict[str, np.ndarray]:
    T, r, _ = X.shape
    if T < 2:
        return {"type": "mar", "A": np.eye(r, dtype=np.float32), "B": np.eye(r, dtype=np.float32)}
    X0 = X[:-1].astype(np.float64)
    X1 = X[1:].astype(np.float64)

    A = np.eye(r, dtype=np.float64)
    B = np.eye(r, dtype=np.float64)

    for _ in range(n_iter):
        # Solve A with B fixed: Y ≈ A X B^T
        left = np.zeros((r, r), dtype=np.float64)
        right = np.zeros((r, r), dtype=np.float64)
        BtB = B.T @ B
        for t in range(T - 1):
            left += X1[t] @ B @ X0[t].T
            right += X0[t] @ BtB @ X0[t].T
        A = left @ np.linalg.pinv(right + ridge * np.eye(r))

        # Solve B with A fixed.
        left = np.zeros((r, r), dtype=np.float64)
        right = np.zeros((r, r), dtype=np.float64)
        AtA = A.T @ A
        for t in range(T - 1):
            left += X1[t].T @ A @ X0[t]
            right += X0[t].T @ AtA @ X0[t]
        B = left @ np.linalg.pinv(right + ridge * np.eye(r))

    A, B = stabilize_mar_params(A.astype(np.float32), B.astype(np.float32))
    return {"type": "mar", "A": A, "B": B}


def fit_factorized_models(methods: List[str], train_data: Dict[str, Any], args) -> Dict[str, Any]:
    needs_factor = any(m in methods for m in ["factorized_var_gt", "factorized_mar_gt", "factorized_tvpvar_gt"])
    if not needs_factor:
        return {}

    print_stage("FIT BRANCH B FACTORIZED G MODELS")
    bases = fit_fulltrain_factor_bases_streaming(train_data, rank=args.rank)
    U, V, mean_G = bases["U"], bases["V"], bases["mean_G"]
    X_train = compress_series(train_data, U, V, mean_G)

    models = {"bases": bases}
    if "factorized_var_gt" in methods:
        print("Fitting latent VAR ...")
        models["factorized_var_gt"] = fit_latent_var(X_train, ridge=args.var_ridge)
    if "factorized_mar_gt" in methods:
        print("Fitting latent MAR ...")
        models["factorized_mar_gt"] = fit_latent_mar(X_train, ridge=args.mar_ridge)
    if "factorized_tvpvar_gt" in methods:
        print("Fitting latent TVP-VAR/RLS ...")
        models["factorized_tvpvar_gt"] = fit_latent_tvpvar(X_train, ridge=args.tvpvar_ridge, forgetting=args.tvpvar_forgetting)
    return models


def predict_factorized(method: str, G_origin: np.ndarray, models: Dict[str, Any], horizon: int) -> np.ndarray:
    bases = models["bases"]
    U, V, mean_G = bases["U"], bases["V"], bases["mean_G"]
    X = U.T @ (G_origin - mean_G) @ V
    model = models[method]

    if model["type"] in ["var", "tvpvar"]:
        z = X.reshape(-1).astype(np.float32)
        c, Phi = model["c"], model["Phi"]
        for _ in range(int(horizon)):
            z = c + Phi @ z
        Xh = z.reshape(X.shape).astype(np.float32)
    elif model["type"] == "mar":
        Xh = X.astype(np.float32)
        A, B = model["A"], model["B"]
        for _ in range(int(horizon)):
            Xh = A @ Xh @ B.T
    else:
        raise ValueError(f"Unknown factorized model type: {model['type']}")

    G_hat = mean_G + U @ Xh @ V.T
    return clip_G(G_hat)


def predict_G(method: str, train_data: Dict[str, Any], split_data: Dict[str, Any],
              origin_idx: int, target_idx: int, lag: int, models: Dict[str, Any], args) -> np.ndarray:
    if method in ["true_gt", "persistence_gt", "ewma_gt", "no_gt"]:
        return predict_G_basic(method, split_data, origin_idx, target_idx, lag, alpha_ewma=args.ewma_alpha)
    if method in ["factorized_var_gt", "factorized_mar_gt", "factorized_tvpvar_gt"]:
        return predict_factorized(method, get_G(split_data, origin_idx), models, lag)
    raise ValueError(f"Unsupported method: {method}")


def topk_indices_abs_directed(G: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    A = np.abs(np.asarray(G, dtype=np.float32)).copy()
    n = A.shape[0]
    np.fill_diagonal(A, -np.inf)
    flat = A.ravel()
    K = min(int(k), n * n - n)
    if K <= 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    idx = np.argpartition(flat, -K)[-K:]
    idx = idx[np.argsort(flat[idx])[::-1]]
    i = (idx // n).astype(np.int64)
    j = (idx % n).astype(np.int64)
    return i, j


def topk_overlap_metrics(G_true: np.ndarray, G_pred: np.ndarray, topk_values: List[int]) -> List[Dict[str, Any]]:
    rows = []
    for k in topk_values:
        ti, tj = topk_indices_abs_directed(G_true, k)
        pi, pj = topk_indices_abs_directed(G_pred, k)
        true_set = set(zip(ti.tolist(), tj.tolist()))
        pred_set = set(zip(pi.tolist(), pj.tolist()))
        inter = len(true_set & pred_set)
        rows.append({
            "topk": int(k),
            "true_topk_size": int(len(true_set)),
            "pred_topk_size": int(len(pred_set)),
            "overlap": int(inter),
            "precision_at_k": float(inter / max(1, len(pred_set))),
            "recall_at_k": float(inter / max(1, len(true_set))),
            "overlap_ratio": float(inter / max(1, min(len(true_set), len(pred_set)))),
        })
    return rows


def sample_directed_offdiag_pairs(n: int, n_pairs: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    if n <= 1:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    i = rng.integers(0, n, size=n_pairs, dtype=np.int64)
    j = rng.integers(0, n - 1, size=n_pairs, dtype=np.int64)
    j = j + (j >= i)
    return i, j


def distribution_metrics(true_vals: np.ndarray, pred_vals: np.ndarray) -> Dict[str, float]:
    true_vals = np.asarray(true_vals, dtype=np.float32)
    pred_vals = np.asarray(pred_vals, dtype=np.float32)
    mask = np.isfinite(true_vals) & np.isfinite(pred_vals)
    true_vals = true_vals[mask]
    pred_vals = pred_vals[mask]
    if len(true_vals) == 0:
        return {}
    diff = pred_vals - true_vals
    out = {
        "n_pairs": int(len(true_vals)),
        "mae": float(np.mean(np.abs(diff))),
        "mse": float(np.mean(diff ** 2)),
        "rmse": float(np.sqrt(np.mean(diff ** 2))),
        "true_mean": float(np.mean(true_vals)),
        "pred_mean": float(np.mean(pred_vals)),
        "true_std": float(np.std(true_vals)),
        "pred_std": float(np.std(pred_vals)),
        "true_q05": float(np.quantile(true_vals, 0.05)),
        "pred_q05": float(np.quantile(pred_vals, 0.05)),
        "true_q50": float(np.quantile(true_vals, 0.50)),
        "pred_q50": float(np.quantile(pred_vals, 0.50)),
        "true_q95": float(np.quantile(true_vals, 0.95)),
        "pred_q95": float(np.quantile(pred_vals, 0.95)),
    }
    if wasserstein_distance is not None:
        out["wasserstein"] = float(wasserstein_distance(true_vals, pred_vals))
    else:
        qs = np.linspace(0.01, 0.99, 99)
        out["wasserstein"] = float(np.mean(np.abs(np.quantile(true_vals, qs) - np.quantile(pred_vals, qs))))
    if ks_2samp is not None:
        out["ks_stat"] = float(ks_2samp(true_vals, pred_vals).statistic)
    else:
        out["ks_stat"] = np.nan
    if np.std(true_vals) > EPS and np.std(pred_vals) > EPS:
        out["pearson"] = float(np.corrcoef(true_vals, pred_vals)[0, 1])
    else:
        out["pearson"] = np.nan
    if spearmanr is not None and len(true_vals) > 3:
        out["spearman"] = float(spearmanr(true_vals, pred_vals).correlation)
    else:
        out["spearman"] = np.nan
    return out


def method_order(existing: List[str], preferred: List[str]) -> List[str]:
    ordered = [m for m in preferred if m in existing]
    ordered += [m for m in sorted(existing) if m not in ordered]
    return ordered


def load_edge_metadata(project_root: Path, segment_ids: np.ndarray) -> Optional[pd.DataFrame]:
    candidates = [
        project_root / "ml_core" / "src" / "data_processing" / "outputs" / "branchA" / "match_summary" / "matched_osm_edge_metadata.csv",
        project_root / "ml_core" / "src" / "data_processing" / "outputs" / "branchA" / "osm_edge_forecasting_dataset" / "tables" / "node_quality.csv",
        project_root / "ml_core" / "src" / "data_processing" / "outputs" / "branchA" / "osm_edge_forecasting_dataset" / "tables" / "segment_to_model_node_mapping.csv",
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        print("[WARN] Cannot find OSM edge metadata. Map/topology spatial metadata may be skipped.")
        return None

    df = pd.read_csv(path)
    if "model_node_id" not in df.columns:
        print("[WARN] edge metadata has no model_node_id column.")
        return None

    df = df.drop_duplicates("model_node_id").copy()
    df["model_node_id"] = df["model_node_id"].astype(np.int64)
    df = df.set_index("model_node_id", drop=False)
    rows = []
    for pos, mid in enumerate(segment_ids.astype(np.int64)):
        if int(mid) in df.index:
            row = df.loc[int(mid)].copy()
            row["position"] = pos
            rows.append(row)
    if not rows:
        print("[WARN] No matching rows between segment_ids and edge metadata.")
        return None
    out = pd.DataFrame(rows).reset_index(drop=True)
    out["position"] = out["position"].astype(np.int64)
    return out


def build_edge_line_graph_neighbors(edge_meta: Optional[pd.DataFrame], n: int) -> Optional[List[List[int]]]:
    if edge_meta is None:
        return None
    required = {"osm_u_id", "osm_v_id", "position"}
    if not required.issubset(edge_meta.columns):
        print("[WARN] Missing OSM endpoint columns; skip hop topology.")
        return None
    endpoint_to_edges = defaultdict(list)
    for _, row in edge_meta.iterrows():
        pos = int(row["position"])
        endpoint_to_edges[int(row["osm_u_id"])].append(pos)
        endpoint_to_edges[int(row["osm_v_id"])].append(pos)
    neigh_sets = [set() for _ in range(n)]
    for positions in endpoint_to_edges.values():
        if len(positions) <= 1:
            continue
        for a in positions:
            neigh_sets[a].update(positions)
    for i in range(n):
        neigh_sets[i].discard(i)
    return [sorted(s) for s in neigh_sets]


def bfs_limited(neighbors: List[List[int]], source: int, max_hop: int) -> Dict[int, int]:
    dist = {int(source): 0}
    q = deque([int(source)])
    while q:
        u = q.popleft()
        if dist[u] >= max_hop:
            continue
        for v in neighbors[u]:
            if v not in dist:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist


def compute_pair_hop_bins(neighbors: Optional[List[List[int]]], pair_i: np.ndarray, pair_j: np.ndarray, max_hop: int) -> np.ndarray:
    if neighbors is None:
        return np.array(["unknown"] * len(pair_i), dtype=object)
    out = np.empty(len(pair_i), dtype=object)
    by_source = defaultdict(list)
    for idx, s in enumerate(pair_i.tolist()):
        by_source[int(s)].append(idx)
    for s, indices in by_source.items():
        dist = bfs_limited(neighbors, s, max_hop=max_hop)
        for idx in indices:
            d = dist.get(int(pair_j[idx]), None)
            out[idx] = f">{max_hop}/unreachable" if d is None else f"{d}-hop"
    return out


def haversine_m(lat1, lon1, lat2, lon2) -> np.ndarray:
    R = 6371000.0
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def compute_geo_bins(edge_meta: Optional[pd.DataFrame], pair_i: np.ndarray, pair_j: np.ndarray) -> np.ndarray:
    if edge_meta is None or not {"mid_lat", "mid_lon"}.issubset(edge_meta.columns):
        return np.array(["unknown"] * len(pair_i), dtype=object)
    meta = edge_meta.sort_values("position")
    lat = meta["mid_lat"].to_numpy(dtype=float)
    lon = meta["mid_lon"].to_numpy(dtype=float)
    d = haversine_m(lat[pair_i], lon[pair_i], lat[pair_j], lon[pair_j])
    bins = []
    for x in d:
        if x <= 250:
            bins.append("0-250m")
        elif x <= 500:
            bins.append("250-500m")
        elif x <= 1000:
            bins.append("500m-1km")
        elif x <= 2000:
            bins.append("1-2km")
        elif x <= 5000:
            bins.append("2-5km")
        else:
            bins.append(">5km")
    return np.array(bins, dtype=object)


def rank_color(rank: int) -> str:
    if rank <= 10:
        return "red"
    if rank <= 20:
        return "orange"
    return "green"


def edge_popup(row: pd.Series, extra: str = "") -> str:
    street = row.get("street_names", "")
    eid = row.get("osm_edge_id", "")
    mid = row.get("model_node_id", "")
    return f"""
    <b>model_node_id:</b> {mid}<br>
    <b>osm_edge_id:</b> {eid}<br>
    <b>street:</b> {street}<br>
    {extra}
    """


def add_edge_polyline(m, row: pd.Series, color: str, weight: int, opacity: float, popup_html: str):
    if not {"u_lat", "u_lon", "v_lat", "v_lon"}.issubset(row.index):
        return
    coords = [[float(row["u_lat"]), float(row["u_lon"])], [float(row["v_lat"]), float(row["v_lon"])]]
    folium.PolyLine(coords, color=color, weight=weight, opacity=opacity,
                    popup=folium.Popup(popup_html, max_width=500)).add_to(m)

def plot_abs_error_boxplot(df: pd.DataFrame, out_dir: Path, split: str, lag: int, topk: int, methods: List[str]):
    sub = df[df["topk"] == topk].copy()
    if sub.empty:
        return
    ordered = method_order(list(sub["method"].unique()), methods)
    data = [sub[sub["method"] == m]["abs_error"].to_numpy(dtype=float) for m in ordered]
    labels = [METHOD_LABELS.get(m, m) for m in ordered]
    plt.figure(figsize=(11, 6))
    bp = plt.boxplot(data, labels=labels, showmeans=True, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_alpha(0.45)
    plt.title(f"Branch B Top-{topk} | Absolute G error | {split.upper()} | lag={lag}")
    plt.xlabel("Method")
    plt.ylabel("|G_pred - G_true|")
    plt.xticks(rotation=25, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = out_dir / f"branchB_topk_abs_error_boxplot_{split}_lag{lag}_top{topk}.png"
    plt.savefig(path, dpi=180)
    plt.close()
    print("Saved:", path)


def plot_abs_error_hist(df: pd.DataFrame, out_dir: Path, split: str, lag: int, topk: int, methods: List[str]):
    sub = df[df["topk"] == topk].copy()
    if sub.empty:
        return
    plt.figure(figsize=(11, 6))
    for method in method_order(list(sub["method"].unique()), methods):
        vals = sub[sub["method"] == method]["abs_error"].to_numpy(dtype=float)
        if len(vals):
            plt.hist(vals, bins=40, density=True, alpha=0.35, label=METHOD_LABELS.get(method, method))
    plt.title(f"Branch B Top-{topk} | Absolute G error distribution | {split.upper()} | lag={lag}")
    plt.xlabel("|G_pred - G_true|")
    plt.ylabel("Density")
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    path = out_dir / f"branchB_topk_abs_error_hist_{split}_lag{lag}_top{topk}.png"
    plt.savefig(path, dpi=180)
    plt.close()
    print("Saved:", path)


def plot_signed_error_hist(df: pd.DataFrame, out_dir: Path, split: str, lag: int, topk: int, methods: List[str]):
    sub = df[df["topk"] == topk].copy()
    if sub.empty:
        return
    plt.figure(figsize=(11, 6))
    for method in method_order(list(sub["method"].unique()), methods):
        vals = sub[sub["method"] == method]["signed_error"].to_numpy(dtype=float)
        if len(vals):
            plt.hist(vals, bins=50, density=True, alpha=0.35, label=METHOD_LABELS.get(method, method))
    plt.axvline(0, color="black", linewidth=1.2)
    plt.title(f"Branch B Top-{topk} | Signed G error distribution | {split.upper()} | lag={lag}")
    plt.xlabel("G_pred - G_true")
    plt.ylabel("Density")
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    path = out_dir / f"branchB_topk_signed_error_hist_{split}_lag{lag}_top{topk}.png"
    plt.savefig(path, dpi=180)
    plt.close()
    print("Saved:", path)


def plot_topk_overlap_summary(overlap_df: pd.DataFrame, out_dir: Path, split: str, topk: int, methods: List[str]):
    if overlap_df.empty:
        return
    summary = overlap_df.groupby(["method", "split", "lag", "topk"], as_index=False).agg(overlap_ratio=("overlap_ratio", "mean"))
    sub = summary[(summary["split"] == split) & (summary["topk"] == topk)].copy()
    if sub.empty:
        return
    plt.figure(figsize=(10, 5))
    for method in method_order(list(sub["method"].unique()), methods):
        g = sub[sub["method"] == method].sort_values("lag")
        if not g.empty:
            plt.plot(g["lag"], g["overlap_ratio"], marker="o", linewidth=2, label=METHOD_LABELS.get(method, method))
    plt.title(f"Branch B Top-{topk} directed strong-link overlap by lag | {split.upper()}")
    plt.xlabel("Lag")
    plt.ylabel("Overlap ratio")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    path = out_dir / f"branchB_topk_overlap_top{topk}_{split}.png"
    plt.savefig(path, dpi=180)
    plt.close()
    print("Saved:", path)


def collect_topk_errors_and_overlap(train_data, split_data, methods, topk_values, lag, sample_pairs, models, args, split_name):
    error_rows, overlap_rows = [], []
    for sample_id, (origin_idx, target_idx) in enumerate(sample_pairs):
        G_true = get_G(split_data, target_idx)
        pred_cache = {}
        for method in methods:
            pred_cache[method] = predict_G(method, train_data, split_data, origin_idx, target_idx, lag, models, args)

        for topk in topk_values:
            ti, tj = topk_indices_abs_directed(G_true, topk)
            true_vals = G_true[ti, tj].astype(np.float32)
            for method in methods:
                pred_vals = pred_cache[method][ti, tj].astype(np.float32)
                signed_err = pred_vals - true_vals
                abs_err = np.abs(signed_err)
                for idx in range(len(ti)):
                    error_rows.append({
                        "sample_id": int(sample_id),
                        "origin_idx": int(origin_idx),
                        "target_idx": int(target_idx),
                        "topk": int(topk),
                        "method": method,
                        "i": int(ti[idx]),
                        "j": int(tj[idx]),
                        "g_true": float(true_vals[idx]),
                        "g_pred": float(pred_vals[idx]),
                        "signed_error": float(signed_err[idx]),
                        "abs_error": float(abs_err[idx]),
                    })

        for method in methods:
            for row in topk_overlap_metrics(G_true, pred_cache[method], topk_values):
                row.update({
                    "sample_id": int(sample_id),
                    "origin_idx": int(origin_idx),
                    "target_idx": int(target_idx),
                    "split": split_name,
                    "lag": int(lag),
                    "method": method,
                })
                overlap_rows.append(row)

        print(f"  sample {sample_id+1}/{len(sample_pairs)} done: origin={origin_idx}, target={target_idx}")
    return pd.DataFrame(error_rows), pd.DataFrame(overlap_rows)


def make_fixed_source_method_map(edge_meta, true_mean, pred_means, out_path, methods, n_sources=5, top_targets=30):
    if folium is None:
        print("[WARN] folium not installed; skip map.")
        return
    if edge_meta is None:
        print("[WARN] no edge metadata; skip map.")
        return
    required = {"position", "u_lat", "u_lon", "v_lat", "v_lon", "mid_lat", "mid_lon"}
    if not required.issubset(edge_meta.columns):
        print("[WARN] metadata lacks coordinates; skip map.")
        return
    meta = edge_meta.sort_values("position").reset_index(drop=True)
    center = [float(meta["mid_lat"].mean()), float(meta["mid_lon"].mean())]
    fmap = folium.Map(location=center, zoom_start=13, tiles="OpenStreetMap")

    abs_true = np.abs(true_mean.copy())
    np.fill_diagonal(abs_true, 0.0)
    source_score = abs_true.mean(axis=1)
    source_idx = np.argsort(source_score)[::-1][:n_sources]
    print("[MAP] fixed source node ids:", [int(meta.loc[s, "model_node_id"]) for s in source_idx])

    for method in methods:
        G_for_rank = true_mean if method == "true_gt" else pred_means.get(method)
        if G_for_rank is None:
            continue
        for source_rank, src in enumerate(source_idx, start=1):
            source_node_id = int(meta.loc[src, "model_node_id"]) if "model_node_id" in meta.columns else int(src)
            fg = folium.FeatureGroup(name=f"{METHOD_LABELS.get(method, method)} | SOURCE {source_rank} | node {source_node_id}",
                                     show=(method == "true_gt" and source_rank == 1))
            src_row = meta.loc[src]
            scores = np.abs(G_for_rank[src]).copy()
            scores[src] = -np.inf
            top_idx = np.argsort(scores)[::-1][:top_targets]

            for rank, tgt in enumerate(top_idx, start=1):
                row = meta.loc[tgt]
                true_g = float(true_mean[src, tgt])
                pred_g = float(G_for_rank[src, tgt])
                extra = (
                    f"<b>method:</b> {METHOD_LABELS.get(method, method)}<br>"
                    f"<b>fixed_source_node_id:</b> {source_node_id}<br>"
                    f"<b>rank_in_method:</b> {rank}<br>"
                    f"<b>G_true_mean:</b> {true_g:.5f}<br>"
                    f"<b>G_{method}_mean:</b> {pred_g:.5f}<br>"
                    f"<b>abs_error:</b> {abs(pred_g - true_g):.5f}<br>"
                )
                color = rank_color(rank)
                add_edge_polyline(fg, row, color=color, weight=5 if rank <= 10 else 4, opacity=0.85,
                                  popup_html=edge_popup(row, extra=extra))
                folium.PolyLine(
                    [[float(src_row["mid_lat"]), float(src_row["mid_lon"])],
                     [float(row["mid_lat"]), float(row["mid_lon"])]],
                    color=color, weight=1, opacity=0.22,
                ).add_to(fg)

            source_extra = (
                f"<b>THIS IS FIXED SOURCE EDGE</b><br>"
                f"<b>source_rank:</b> {source_rank}<br>"
                f"<b>fixed_source_node_id:</b> {source_node_id}<br>"
                f"<b>method layer:</b> {METHOD_LABELS.get(method, method)}<br>"
                f"<b>Color:</b> BLACK<br>"
            )
            add_edge_polyline(fg, src_row, color="#000000", weight=16, opacity=1.0,
                              popup_html=edge_popup(src_row, extra=source_extra))
            folium.CircleMarker(
                location=[float(src_row["mid_lat"]), float(src_row["mid_lon"])],
                radius=9, color="#000000", fill=True, fill_color="#000000", fill_opacity=1.0,
                tooltip=f"SOURCE {source_rank} | node {source_node_id}",
                popup=folium.Popup(edge_popup(src_row, extra=source_extra), max_width=500),
            ).add_to(fg)
            folium.Marker(
                location=[float(src_row["mid_lat"]), float(src_row["mid_lon"])],
                icon=folium.DivIcon(html=f"""
                    <div style="font-size:13px;color:white;background:black;padding:2px 6px;
                                border-radius:4px;border:1px solid white;white-space:nowrap;">
                    SOURCE {source_rank}<br>node {source_node_id}</div>"""),
            ).add_to(fg)
            fg.add_to(fmap)

    legend_html = """
    <div style="position: fixed; bottom: 30px; left: 30px; width: 340px; z-index: 9999;
                background-color: white; border: 2px solid grey; border-radius: 6px;
                padding: 10px; font-size: 13px;">
        <b>Branch B fixed-source directed G map</b><br>
        <span style="color:black;font-weight:bold;">━━</span> Fixed source edge<br>
        <span style="color:red;font-weight:bold;">━━</span> Top 1-10 directed related edges<br>
        <span style="color:orange;font-weight:bold;">━━</span> Top 11-20 directed related edges<br>
        <span style="color:green;font-weight:bold;">━━</span> Top 21-30 directed related edges<br>
        Toggle method/source layers on the right.
    </div>
    """
    fmap.get_root().html.add_child(folium.Element(legend_html))
    folium.LayerControl(collapsed=False).add_to(fmap)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fmap.save(str(out_path))
    print("Saved map:", out_path)


def main():
    project_root = find_project_root()
    branch_root = project_root / "ml_core" / "src" / "models" / "ML_BranchB"
    default_common = project_root / "ml_core" / "src" / "data_processing" / "outputs" / "branchB" / "osm_edge_gt_like_branchA"
    default_out = branch_root / "results" / "08_topk_error_distribution_map"

    ap = argparse.ArgumentParser()
    ap.add_argument("--common-dir", type=str, default=str(default_common))
    ap.add_argument("--output-dir", type=str, default=str(default_out))
    ap.add_argument("--clean-output", action="store_true", default=True)
    ap.add_argument("--no-clean-output", dest="clean_output", action="store_false")
    ap.add_argument("--methods", type=parse_str_list, default=DEFAULT_METHODS)
    ap.add_argument("--splits", type=parse_str_list, default=DEFAULT_SPLITS)
    ap.add_argument("--lags", type=parse_int_list, default=DEFAULT_LAGS)
    ap.add_argument("--topk-values", type=parse_int_list, default=DEFAULT_TOPK)
    ap.add_argument("--samples-per-split-lag", type=int, default=6)
    ap.add_argument("--max-nodes", type=int, default=0)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--rank", type=int, default=DEFAULT_RANK)
    ap.add_argument("--ewma-alpha", type=float, default=DEFAULT_EWMA_ALPHA)
    ap.add_argument("--var-ridge", type=float, default=DEFAULT_VAR_RIDGE)
    ap.add_argument("--mar-ridge", type=float, default=DEFAULT_MAR_RIDGE)
    ap.add_argument("--tvpvar-ridge", type=float, default=DEFAULT_TVPVAR_RIDGE)
    ap.add_argument("--tvpvar-forgetting", type=float, default=DEFAULT_TVPVAR_FORGETTING)
    ap.add_argument("--map-split", type=str, default="test")
    ap.add_argument("--map-lag", type=int, default=1)
    ap.add_argument("--map-snapshots", type=int, default=5)
    ap.add_argument("--map-sources", type=int, default=5)
    ap.add_argument("--map-top-targets", type=int, default=30)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    common_dir = Path(args.common_dir).resolve()
    out_dir = Path(args.output_dir).resolve()
    if args.clean_output and out_dir.exists():
        print("[CLEAN] Removing:", out_dir)
        shutil.rmtree(out_dir)
    plots_dir = ensure_dir(out_dir / "plots")
    maps_dir = ensure_dir(out_dir / "maps")

    print_stage("BRANCH B — TOP-K ERROR DISTRIBUTION + TOP-K OVERLAP + FIXED-SOURCE MAP")
    print("PROJECT_ROOT:", project_root)
    print("COMMON_DIR  :", common_dir)
    print("OUT_DIR     :", out_dir)
    print("methods     :", args.methods)
    print("lags        :", args.lags)
    print("topk_values :", args.topk_values)
    check_branchB_common_dir_ready(common_dir)

    all_data = {"train": load_gt_split(common_dir, "train")}
    needed_splits = sorted(set(args.splits + [args.map_split]))
    for split in needed_splits:
        if split != "train":
            all_data[split] = load_gt_split(common_dir, split)

    node_idx = None
    n_full = len(all_data["train"]["segment_ids"])
    if args.max_nodes and args.max_nodes > 0 and args.max_nodes < n_full:
        node_idx = np.linspace(0, n_full - 1, args.max_nodes).round().astype(np.int64)
        node_idx = np.unique(node_idx)
        print(f"[NODE SUBSET] Using {len(node_idx)}/{n_full} nodes.")

    for split in list(all_data.keys()):
        all_data[split] = subset_split(all_data[split], node_idx)

    train = all_data["train"]
    N = len(train["segment_ids"])
    print("N analysis nodes:", N)
    edge_meta = load_edge_metadata(project_root, train["segment_ids"])

    models = fit_factorized_models(args.methods, train, args)

    print_stage("TOP-K ERROR + OVERLAP PLOTS")
    for split in args.splits:
        split_data = all_data[split]
        overlap_blocks = []
        for lag in args.lags:
            sample_pairs = sample_eval_pairs(split_data["meta"], lag, args.samples_per_split_lag, rng)
            print(f"\n[EVAL] split={split}, lag={lag}, n_snapshots={len(sample_pairs)}")
            err_df, overlap_df = collect_topk_errors_and_overlap(
                train, split_data, args.methods, args.topk_values, lag, sample_pairs, models, args, split
            )
            for topk in args.topk_values:
                plot_abs_error_boxplot(err_df, plots_dir, split, lag, topk, args.methods)
                plot_abs_error_hist(err_df, plots_dir, split, lag, topk, args.methods)
                plot_signed_error_hist(err_df, plots_dir, split, lag, topk, args.methods)
            if not overlap_df.empty:
                overlap_blocks.append(overlap_df)
            del err_df, overlap_df

        if overlap_blocks:
            overlap_all = pd.concat(overlap_blocks, ignore_index=True)
            for topk in args.topk_values:
                plot_topk_overlap_summary(overlap_all, plots_dir, split, topk, args.methods)
            del overlap_all

    print_stage("FIXED-SOURCE OSM MAP")
    if args.map_split in all_data:
        map_data = all_data[args.map_split]
        map_pairs = sample_eval_pairs(map_data["meta"], args.map_lag, args.map_snapshots, rng)
        if map_pairs:
            G_true_acc = np.zeros((N, N), dtype=np.float64)
            pred_acc = {m: np.zeros((N, N), dtype=np.float64) for m in args.methods}
            for pair_id, (origin_idx, target_idx) in enumerate(map_pairs, start=1):
                print(f"[MAP] pair {pair_id}/{len(map_pairs)}: origin={origin_idx}, target={target_idx}")
                G_true = get_G(map_data, target_idx)
                G_true_acc += G_true
                for method in args.methods:
                    pred_acc[method] += predict_G(method, train, map_data, origin_idx, target_idx, args.map_lag, models, args)
            G_true_mean = (G_true_acc / len(map_pairs)).astype(np.float32)
            pred_means = {m: (mat / len(map_pairs)).astype(np.float32) for m, mat in pred_acc.items()}
            map_path = maps_dir / f"branchB_fixed_sources_method_G_map_{args.map_split}_lag{args.map_lag}.html"
            make_fixed_source_method_map(edge_meta, G_true_mean, pred_means, map_path, args.methods,
                                         n_sources=args.map_sources, top_targets=args.map_top_targets)

    print_stage("DONE")
    print("Plots:", plots_dir)
    print("Maps :", maps_dir)


if __name__ == "__main__":
    main()
