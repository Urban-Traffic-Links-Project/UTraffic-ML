# Auto-converted from notebook for Branch B OSM-edge workflow.
# Fixed syntax version.
# Memory-safe update: fit_fulltrain_factor_bases() now streams G_train instead of loading full tensor.
# Method: factorized_var_gt
# Reads prepared GT from:
#   ml_core/src/data_processing/outputs/branchB/osm_edge_gt_like_branchA
# Saves results to:
#   ml_core/src/models/ML_BranchB/results/06_branchB_run_xt_forecast/factorized_var_gt

# ============================================================
# Branch B: Xt Forecast using Gt relation matrices (standalone)
# Goal: predict X_{t+h}=z_{t+h} using either no Gt or Gt from a method.
# Only saves per_lag_metrics CSV.
# ============================================================

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import MultiTaskElasticNet

# -------------------------
# Config
# -------------------------
HORIZONS = list(range(1, 10))
EPS = 1e-8

# Downstream Xt model: X_{t+h} = A_h X_t + B_h(G_hat_{t,h} X_t)
ALPHA = 0.001
L1_RATIO = 0.5
MAX_ITER = 500          # faster than 2000
TOL = 1e-2              # faster convergence tolerance
SELECTION = 'random'    # often faster than cyclic for large ElasticNet
RANDOM_STATE = 42

# Branch B relation methods
DEFAULT_EWMA_ALPHA = 0.30
DEFAULT_RANK = 16
DEFAULT_VAR_RIDGE = 1e-2
DEFAULT_MAR_RIDGE = 1e-2
DEFAULT_TVPVAR_RIDGE = 5e-2
DEFAULT_TVPVAR_FORGETTING = 0.98
DEFAULT_STABILITY_TARGET = 0.98
DEFAULT_SPARSE_TVPVAR_RIDGE = 1e-2
DEFAULT_SPARSE_TVPVAR_FORGETTING = 0.98
DEFAULT_SPARSE_TVPVAR_TOPK = 4
MAX_DENSE_TVPVAR_DIM = 6000  # safety guard for vec(G) dimension; dense TVP-VAR is infeasible for large N.

# -------------------------
# Paths / loading
# -------------------------
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



def check_branchB_common_dir_ready(common_dir: Path) -> None:
    """
    Verify Branch B prepared GT files exist.
    This script does NOT auto-run prepare because Branch B prepare can create very large files.
    """
    required = [
        common_dir / "train" / "G_weight_series.npy",
        common_dir / "train" / "G_best_lag_series.npy",
        common_dir / "train" / "z.npy",
        common_dir / "val" / "G_weight_series.npy",
        common_dir / "val" / "G_best_lag_series.npy",
        common_dir / "val" / "z.npy",
        common_dir / "test" / "G_weight_series.npy",
        common_dir / "test" / "G_best_lag_series.npy",
        common_dir / "test" / "z.npy",
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        msg = "\n".join(str(p) for p in missing[:20])
        raise FileNotFoundError(
            "Branch B prepared GT data is missing.\n"
            f"Expected COMMON_DIR: {common_dir}\n\n"
            "Missing files:\n"
            f"{msg}\n\n"
            "Run this first from project root:\n"
            "  python ml_core/src/data_processing/prepare_branchB_osm_edge_gt_like_branchA.py --overwrite\n\n"
            "For a lightweight test:\n"
            "  python ml_core/src/data_processing/prepare_branchB_osm_edge_gt_like_branchA.py --max-nodes 512 --overwrite\n"
        )


def load_gt_split(common_dir: Path, split_name: str, mmap_mode: str = 'r') -> Dict[str, object]:
    split_dir = common_dir / split_name
    G_weight_series = np.load(split_dir / 'G_weight_series.npy', mmap_mode=mmap_mode)
    G_best_lag_series = np.load(split_dir / 'G_best_lag_series.npy', mmap_mode=mmap_mode)
    z = np.load(split_dir / 'z.npy', mmap_mode=mmap_mode)
    segment_ids = np.load(split_dir / 'segment_ids.npy')
    timestamps = pd.to_datetime(np.load(split_dir / 'timestamps.npy'))
    meta = pd.read_csv(split_dir / 'G_series_meta.csv')
    if 'timestamp_local' in meta.columns:
        meta['timestamp_local'] = pd.to_datetime(meta['timestamp_local'])
    raw_meta = None
    raw_meta_path = split_dir / 'raw_meta.csv'
    if raw_meta_path.exists():
        raw_meta = pd.read_csv(raw_meta_path)
        if 'timestamp_local' in raw_meta.columns:
            raw_meta['timestamp_local'] = pd.to_datetime(raw_meta['timestamp_local'])
    return {
        'G_weight_series': G_weight_series,
        'G_best_lag_series': G_best_lag_series,
        'z': z,
        'segment_ids': segment_ids.astype(np.int64),
        'timestamps': timestamps,
        'meta': meta,
        'raw_meta': raw_meta,
    }


def _session_index_groups(meta: pd.DataFrame) -> List[np.ndarray]:
    if 'session_id' not in meta.columns:
        return [np.arange(len(meta), dtype=np.int64)]
    groups = []
    for _, sub in meta.groupby('session_id', sort=False):
        idx = sub.index.to_numpy(dtype=np.int64)
        if len(idx) > 0:
            groups.append(idx)
    return groups if groups else [np.arange(len(meta), dtype=np.int64)]


def iter_eval_pairs(meta: pd.DataFrame, horizon: int):
    T = len(meta)
    sess = meta['session_id'].to_numpy() if 'session_id' in meta.columns else None
    for origin_idx in range(T - horizon):
        target_idx = origin_idx + horizon
        if sess is not None and sess[origin_idx] != sess[target_idx]:
            continue
        yield origin_idx, target_idx


def _history_indices_for_origin(meta: pd.DataFrame, origin_idx: int) -> np.ndarray:
    if 'session_id' not in meta.columns:
        return np.arange(origin_idx + 1, dtype=np.int64)
    sess = meta['session_id'].to_numpy()
    mask = (sess == sess[origin_idx]) & (np.arange(len(meta)) <= origin_idx)
    return np.where(mask)[0].astype(np.int64)

# -------------------------
# Metrics
# -------------------------
def batch_vector_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    yt = np.asarray(y_true, dtype=np.float32)
    yp = np.asarray(y_pred, dtype=np.float32)
    diff = yp - yt
    mse = float(np.mean(diff ** 2))
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(mse))
    return {'mae': mae, 'mse': mse, 'rmse': rmse}

# -------------------------
# Basic G predictors
# -------------------------
def predict_G_basic(method_name: str, split_data: Dict[str, object], origin_idx: int, target_idx: int, horizon: int,
                    alpha_ewma: float = DEFAULT_EWMA_ALPHA) -> np.ndarray:
    G = split_data['G_weight_series']
    meta = split_data['meta']
    if method_name == 'true_gt':
        return np.asarray(G[target_idx], dtype=np.float32)
    if method_name == 'persistence_gt':
        return np.asarray(G[origin_idx], dtype=np.float32)
    if method_name == 'ewma_gt':
        hist_idx = _history_indices_for_origin(meta, origin_idx)
        hist = np.asarray(G[hist_idx], dtype=np.float32)
        weights = np.array([(1.0 - alpha_ewma) ** (len(hist) - 1 - k) for k in range(len(hist))], dtype=np.float64)
        weights = weights / max(weights.sum(), EPS)
        return np.tensordot(weights, hist, axes=(0, 0)).astype(np.float32)
    raise ValueError(f'Unsupported basic method: {method_name}')

# -------------------------
# Factorized G predictors: G_t ≈ mean_G + U X_t V^T
# -------------------------
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


def fit_fulltrain_factor_bases(G_train: np.ndarray, rank: int = DEFAULT_RANK) -> Dict[str, np.ndarray]:
    """
    Memory-safe streaming version.

    Old version:
        G = np.asarray(G_train, dtype=np.float32)
        mean_G = np.mean(G, axis=0)
        centered = G - mean_G[None, :, :]

    That loads the whole [T, N, N] tensor and creates another centered copy,
    which can easily explode RAM for full OSM-edge Branch B.

    New version:
        Pass 1: stream G_t from disk/memmap to compute mean_G.
        Pass 2: stream G_t again to accumulate row_cov and col_cov.

    This keeps the same mathematical objective as the old implementation,
    but avoids materializing the full G_train and centered tensors in RAM.
    """
    T, N, M = G_train.shape
    print(f"[FACTOR BASES STREAMING] T={T}, N={N}, M={M}, rank={rank}")
    print("[FACTOR BASES STREAMING] Pass 1/2: computing mean_G without loading full G_train ...")

    progress_every = max(1, T // 10)

    # Pass 1: mean_G
    mean_acc = np.zeros((N, M), dtype=np.float64)
    for t in range(T):
        Gt = np.asarray(G_train[t], dtype=np.float32)
        mean_acc += Gt
        if (t + 1) % progress_every == 0 or (t + 1) == T:
            print(f"  mean_G progress: {t + 1}/{T}")

    mean_G = (mean_acc / max(T, 1)).astype(np.float32)
    del mean_acc

    print("[FACTOR BASES STREAMING] Pass 2/2: accumulating row_cov and col_cov ...")

    row_cov = np.zeros((N, N), dtype=np.float64)
    col_cov = np.zeros((M, M), dtype=np.float64)

    for t in range(T):
        Gt = np.asarray(G_train[t], dtype=np.float32)
        centered_t = (Gt - mean_G).astype(np.float32, copy=False)

        row_cov += centered_t @ centered_t.T
        col_cov += centered_t.T @ centered_t

        if (t + 1) % progress_every == 0 or (t + 1) == T:
            print(f"  covariance progress: {t + 1}/{T}")

    row_cov /= max(T, 1)
    col_cov /= max(T, 1)

    # Symmetrize to reduce numerical asymmetry.
    row_cov = 0.5 * (row_cov + row_cov.T)
    col_cov = 0.5 * (col_cov + col_cov.T)

    print("[FACTOR BASES STREAMING] Eigen decomposition for U and V ...")
    evals_u, evecs_u = np.linalg.eigh(row_cov)
    evals_v, evecs_v = np.linalg.eigh(col_cov)

    ru = int(min(rank, evecs_u.shape[1]))
    rv = int(min(rank, evecs_v.shape[1]))

    U = evecs_u[:, np.argsort(evals_u)[::-1][:ru]].astype(np.float32)
    V = evecs_v[:, np.argsort(evals_v)[::-1][:rv]].astype(np.float32)

    print(f"[FACTOR BASES STREAMING] Done. U={U.shape}, V={V.shape}, mean_G={mean_G.shape}")

    return {'U': U, 'V': V, 'mean_G': mean_G}

def compress_series(G_series: np.ndarray, U: np.ndarray, V: np.ndarray, mean_G: Optional[np.ndarray] = None) -> np.ndarray:
    T = len(G_series)
    X = np.empty((T, U.shape[1], V.shape[1]), dtype=np.float32)
    for t in range(T):
        Gt = np.asarray(G_series[t], dtype=np.float32)
        if mean_G is not None:
            Gt = Gt - mean_G
        X[t] = U.T @ Gt @ V
    return X


def reconstruct_from_latent(X_t: np.ndarray, U: np.ndarray, V: np.ndarray, mean_G: Optional[np.ndarray] = None) -> np.ndarray:
    Ghat = (U @ np.asarray(X_t, dtype=np.float32) @ V.T).astype(np.float32)
    if mean_G is not None:
        Ghat = Ghat + np.asarray(mean_G, dtype=np.float32)
    return Ghat.astype(np.float32)


def _vec(X: np.ndarray) -> np.ndarray:
    return np.asarray(X, dtype=np.float32).reshape(-1, order='F')


def flatten_latent_series(X_series: np.ndarray) -> np.ndarray:
    return np.stack([_vec(X_series[t]) for t in range(X_series.shape[0])], axis=0).astype(np.float32)


def build_lagged_pairs_vector(Z_series: np.ndarray, meta: Optional[pd.DataFrame] = None) -> Tuple[np.ndarray, np.ndarray]:
    Z = np.asarray(Z_series, dtype=np.float32)
    X_parts, Y_parts = [], []
    groups = _session_index_groups(meta) if meta is not None else [np.arange(len(Z), dtype=np.int64)]
    for idx in groups:
        if len(idx) < 2:
            continue
        X_parts.append(Z[idx[:-1]])
        Y_parts.append(Z[idx[1:]])
    if not X_parts:
        d = Z.shape[1]
        return np.empty((0, d), dtype=np.float32), np.empty((0, d), dtype=np.float32)
    return np.concatenate(X_parts, axis=0).astype(np.float32), np.concatenate(Y_parts, axis=0).astype(np.float32)


def build_lagged_pairs_matrix(X_series: np.ndarray, meta: Optional[pd.DataFrame] = None) -> Tuple[np.ndarray, np.ndarray]:
    Xs = np.asarray(X_series, dtype=np.float32)
    A_parts, B_parts = [], []
    groups = _session_index_groups(meta) if meta is not None else [np.arange(len(Xs), dtype=np.int64)]
    for idx in groups:
        if len(idx) < 2:
            continue
        A_parts.append(Xs[idx[:-1]])
        B_parts.append(Xs[idx[1:]])
    if not A_parts:
        shape = Xs.shape[1:]
        return np.empty((0, *shape), dtype=np.float32), np.empty((0, *shape), dtype=np.float32)
    return np.concatenate(A_parts, axis=0).astype(np.float32), np.concatenate(B_parts, axis=0).astype(np.float32)


def fit_feature_standardizer(Z_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = np.mean(Z_train, axis=0, keepdims=True).astype(np.float32)
    std = np.std(Z_train - mu, axis=0, ddof=1, keepdims=True)
    std = np.where(np.isfinite(std) & (std > 1e-6), std, 1.0).astype(np.float32)
    return mu, std


def standardize_features(Z: np.ndarray, mu: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((np.asarray(Z, dtype=np.float32) - mu) / std).astype(np.float32)


def unstandardize_features(Z: np.ndarray, mu: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (np.asarray(Z, dtype=np.float32) * std + mu).astype(np.float32)


def fit_ridge_var1(Z_series: np.ndarray, train_meta: Optional[pd.DataFrame] = None, ridge: float = DEFAULT_VAR_RIDGE) -> Tuple[np.ndarray, np.ndarray]:
    X, Y = build_lagged_pairs_vector(Z_series, train_meta)
    if len(X) == 0:
        d = Z_series.shape[1]
        return np.zeros(d, dtype=np.float32), np.zeros((d, d), dtype=np.float32)
    n, d = X.shape
    X_design = np.concatenate([np.ones((n, 1), dtype=np.float32), X], axis=1)
    reg = ridge * np.eye(d + 1, dtype=np.float32)
    reg[0, 0] = 0.0
    B = np.linalg.solve(X_design.T @ X_design + reg, X_design.T @ Y)
    return B[0].astype(np.float32), B[1:].T.astype(np.float32)


def var1_predict_h(c: np.ndarray, Phi: np.ndarray, z_origin: np.ndarray, horizon: int) -> np.ndarray:
    z = np.asarray(z_origin, dtype=np.float32).reshape(-1)
    for _ in range(horizon):
        z = c + Phi @ z
    return z.astype(np.float32)

# MAR latent

def fit_var1_ols_on_latent(X_series: np.ndarray, train_meta: Optional[pd.DataFrame] = None, ridge: float = DEFAULT_MAR_RIDGE) -> np.ndarray:
    _, m, n = X_series.shape
    d = m * n
    X_prev, X_next = build_lagged_pairs_matrix(X_series, train_meta)
    if len(X_prev) == 0:
        return np.zeros((d, d), dtype=np.float32)
    Y = np.stack([_vec(X_next[t]) for t in range(len(X_next))], axis=1)
    X = np.stack([_vec(X_prev[t]) for t in range(len(X_prev))], axis=1)
    XXt = X @ X.T
    return ((Y @ X.T) @ np.linalg.inv(XXt + ridge * np.eye(d, dtype=np.float32))).astype(np.float32)


def rearrange_for_nkp(Phi: np.ndarray, m: int, n: int) -> np.ndarray:
    out = np.empty((m * m, n * n), dtype=np.float32)
    row = 0
    for i in range(m):
        for j in range(m):
            block = Phi[i + np.arange(n) * m][:, j + np.arange(n) * m]
            out[row] = block.reshape(-1, order='F')
            row += 1
    return out


def proj_mar1_init(X_train: np.ndarray, train_meta: Optional[pd.DataFrame] = None, ridge: float = DEFAULT_MAR_RIDGE) -> Tuple[np.ndarray, np.ndarray]:
    _, m, n = X_train.shape
    Phi = fit_var1_ols_on_latent(X_train, train_meta=train_meta, ridge=ridge)
    Phi_tilde = rearrange_for_nkp(Phi, m, n)
    U_svd, s, Vt = np.linalg.svd(Phi_tilde, full_matrices=False)
    a = np.sqrt(s[0]) * U_svd[:, 0]
    b = np.sqrt(s[0]) * Vt[0]
    A = a.reshape(m, m, order='F').astype(np.float32)
    B = b.reshape(n, n, order='F').astype(np.float32)
    normA = max(np.linalg.norm(A, ord='fro'), EPS)
    return (A / normA).astype(np.float32), (B * normA).astype(np.float32)


def lse_mar1(X_train: np.ndarray, A_init: np.ndarray, B_init: np.ndarray, train_meta: Optional[pd.DataFrame] = None,
             n_iter: int = 8, ridge: float = DEFAULT_MAR_RIDGE) -> Tuple[np.ndarray, np.ndarray]:
    A = np.asarray(A_init, dtype=np.float32).copy()
    B = np.asarray(B_init, dtype=np.float32).copy()
    _, m, n = X_train.shape
    X_prev_all, X_next_all = build_lagged_pairs_matrix(X_train, train_meta)
    if len(X_prev_all) == 0:
        return A, B
    for _ in range(n_iter):
        num_B = np.zeros((n, n), dtype=np.float64)
        den_B = np.zeros((n, n), dtype=np.float64)
        AtA = A.T @ A
        for Xprev, Xt in zip(X_prev_all, X_next_all):
            num_B += Xt.T @ A @ Xprev
            den_B += Xprev.T @ AtA @ Xprev
        B = (num_B @ np.linalg.inv(den_B + ridge * np.eye(n))).astype(np.float32)
        num_A = np.zeros((m, m), dtype=np.float64)
        den_A = np.zeros((m, m), dtype=np.float64)
        BtB = B.T @ B
        for Xprev, Xt in zip(X_prev_all, X_next_all):
            num_A += Xt @ B @ Xprev.T
            den_A += Xprev @ BtB @ Xprev.T
        A = (num_A @ np.linalg.inv(den_A + ridge * np.eye(m))).astype(np.float32)
        normA = max(np.linalg.norm(A, ord='fro'), EPS)
        A = A / normA
        B = B * normA
    return A.astype(np.float32), B.astype(np.float32)


def mar1_predict_h(A: np.ndarray, B: np.ndarray, X_origin: np.ndarray, horizon: int) -> np.ndarray:
    Xhat = np.asarray(X_origin, dtype=np.float32)
    for _ in range(horizon):
        Xhat = A @ Xhat @ B.T
    return Xhat.astype(np.float32)

# TVP VAR latent/dense

def _c_phi_to_theta(c: np.ndarray, Phi: np.ndarray) -> np.ndarray:
    return np.concatenate([np.asarray(c, dtype=np.float32).reshape(1, -1), np.asarray(Phi, dtype=np.float32).T], axis=0).astype(np.float32)


def _theta_to_c_phi(theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return theta[0].astype(np.float32), theta[1:].T.astype(np.float32)


def rls_update(theta: np.ndarray, P: np.ndarray, x: np.ndarray, y: np.ndarray, forgetting: float = DEFAULT_TVPVAR_FORGETTING) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    Px = P @ x
    denom = float(forgetting + x @ Px)
    if not np.isfinite(denom) or denom <= EPS:
        denom = float(forgetting + EPS)
    gain = Px / denom
    err = y - (x @ theta)
    theta_new = theta + np.outer(gain, err).astype(np.float32)
    P_new = (P - np.outer(gain, x) @ P) / max(forgetting, EPS)
    P_new = 0.5 * (P_new + P_new.T)
    return theta_new.astype(np.float32), P_new.astype(np.float32)


def build_tvpvar_states(Z_train_std: np.ndarray, Z_split_std: np.ndarray, train_meta: Optional[pd.DataFrame], split_meta: pd.DataFrame,
                        ridge: float = DEFAULT_TVPVAR_RIDGE, forgetting: float = DEFAULT_TVPVAR_FORGETTING) -> List[Tuple[np.ndarray, np.ndarray]]:
    d = Z_train_std.shape[1]
    c0, Phi0 = fit_ridge_var1(Z_train_std, train_meta=train_meta, ridge=ridge)
    Phi0 = stabilize_transition(Phi0)
    states = [(c0.astype(np.float32), Phi0.astype(np.float32)) for _ in range(len(Z_split_std))]
    for idx in _session_index_groups(split_meta):
        if len(idx) == 0:
            continue
        theta = _c_phi_to_theta(c0, Phi0)
        P = (np.eye(d + 1, dtype=np.float32) / max(ridge, 1e-4)).astype(np.float32)
        states[int(idx[0])] = (c0.astype(np.float32), Phi0.astype(np.float32))
        for k in range(1, len(idx)):
            prev_i = int(idx[k - 1])
            cur_i = int(idx[k])
            x = np.concatenate([[1.0], Z_split_std[prev_i]]).astype(np.float32)
            y = Z_split_std[cur_i].astype(np.float32)
            theta, P = rls_update(theta, P, x, y, forgetting=forgetting)
            c_t, Phi_t = _theta_to_c_phi(theta)
            Phi_t = stabilize_transition(Phi_t)
            theta = _c_phi_to_theta(c_t, Phi_t)
            states[cur_i] = (c_t.astype(np.float32), Phi_t.astype(np.float32))
    return states

# Sparse TVP-VAR target-node

def _graph_summaries(G_prev: np.ndarray):
    G = np.asarray(G_prev, dtype=np.float32)
    return np.mean(G, axis=1).astype(np.float32), np.mean(np.abs(G), axis=1).astype(np.float32), np.mean(G, axis=0).astype(np.float32), np.mean(np.abs(G), axis=0).astype(np.float32)


def _targetnode_feature_matrix(G_prev: np.ndarray, row_mean: np.ndarray, row_abs_mean: np.ndarray, col_mean: np.ndarray, col_abs_mean: np.ndarray, target_j: int) -> np.ndarray:
    G = np.asarray(G_prev, dtype=np.float32)
    N = G.shape[0]
    own = G[:, target_j]
    recip = G[target_j, :]
    X = np.empty((N, 7), dtype=np.float32)
    X[:, 0] = 1.0
    X[:, 1] = own
    X[:, 2] = recip
    X[:, 3] = row_mean
    X[:, 4] = row_abs_mean
    X[:, 5] = np.float32(col_mean[target_j])
    X[:, 6] = np.float32(col_abs_mean[target_j])
    return X


def _ridge_mask_diag(p: int, ridge: float) -> np.ndarray:
    reg = ridge * np.eye(p, dtype=np.float32)
    reg[0, 0] = 0.0
    return reg


def _sparsify_beta(beta: np.ndarray, topk_nonintercept: int) -> np.ndarray:
    b = np.asarray(beta, dtype=np.float32).copy()
    if topk_nonintercept is None or topk_nonintercept <= 0 or len(b) <= 1:
        return b
    keep = min(int(topk_nonintercept), len(b) - 1)
    if keep >= len(b) - 1:
        return b
    idx = np.argsort(np.abs(b[1:]))[::-1]
    mask = np.zeros(len(b) - 1, dtype=bool)
    mask[idx[:keep]] = True
    b[1:][~mask] = 0.0
    return b


def fit_sparse_targetnode_prior(G_train: np.ndarray, meta_train: pd.DataFrame, ridge: float = DEFAULT_SPARSE_TVPVAR_RIDGE, topk_nonintercept: int = DEFAULT_SPARSE_TVPVAR_TOPK) -> Dict[str, np.ndarray]:
    G = np.asarray(G_train, dtype=np.float32)
    _, N, _ = G.shape
    p = 7
    Sxx = np.zeros((N, p, p), dtype=np.float64)
    Sxy = np.zeros((N, p), dtype=np.float64)
    for idxs in _session_index_groups(meta_train):
        if len(idxs) < 2:
            continue
        for prev_idx, curr_idx in zip(idxs[:-1], idxs[1:]):
            G_prev = G[prev_idx]
            G_curr = G[curr_idx]
            row_mean, row_abs_mean, col_mean, col_abs_mean = _graph_summaries(G_prev)
            for j in range(N):
                X = _targetnode_feature_matrix(G_prev, row_mean, row_abs_mean, col_mean, col_abs_mean, j)
                y = np.asarray(G_curr[:, j], dtype=np.float32)
                Sxx[j] += X.T @ X
                Sxy[j] += X.T @ y
    reg = _ridge_mask_diag(p, ridge).astype(np.float64)
    beta = np.zeros((N, p), dtype=np.float32)
    for j in range(N):
        bj = np.linalg.solve(Sxx[j] + reg, Sxy[j]).astype(np.float32)
        beta[j] = _sparsify_beta(bj, topk_nonintercept)
    return {'Sxx': Sxx, 'Sxy': Sxy, 'beta': beta}


def predict_next_G_sparse_targetnode(G_prev: np.ndarray, beta: np.ndarray) -> np.ndarray:
    G_prev = np.asarray(G_prev, dtype=np.float32)
    beta = np.asarray(beta, dtype=np.float32)
    N = G_prev.shape[0]
    G_next = np.empty_like(G_prev, dtype=np.float32)
    row_mean, row_abs_mean, col_mean, col_abs_mean = _graph_summaries(G_prev)
    for j in range(N):
        X = _targetnode_feature_matrix(G_prev, row_mean, row_abs_mean, col_mean, col_abs_mean, j)
        G_next[:, j] = (X @ beta[j]).astype(np.float32)
    return G_next.astype(np.float32)


def sparse_targetnode_predict_h(G_origin: np.ndarray, beta: np.ndarray, horizon: int) -> np.ndarray:
    Ghat = np.asarray(G_origin, dtype=np.float32)
    for _ in range(horizon):
        Ghat = predict_next_G_sparse_targetnode(Ghat, beta)
    return Ghat.astype(np.float32)

# -------------------------
# Build G predictor models once
# -------------------------
def build_g_model(method_name: str, train: Dict[str, object], val: Dict[str, object], test: Dict[str, object]) -> Dict[str, object]:
    G_train = np.asarray(train['G_weight_series'], dtype=np.float32)
    meta_train = train['meta']
    model = {'method': method_name}

    if method_name in {'no_gt', 'true_gt', 'persistence_gt', 'ewma_gt'}:
        return model

    if method_name in {'factorized_var_gt', 'factorized_mar_gt', 'factorized_tvpvar_gt'}:
        factors = fit_fulltrain_factor_bases(G_train, rank=DEFAULT_RANK)
        U, V, mean_G = factors['U'], factors['V'], factors['mean_G']
        X_train = compress_series(G_train, U, V, mean_G=mean_G)
        Z_train = flatten_latent_series(X_train)
        mu_z, std_z = fit_feature_standardizer(Z_train)
        Z_train_std = standardize_features(Z_train, mu_z, std_z)
        model.update({'U': U, 'V': V, 'mean_G': mean_G, 'mu_z': mu_z, 'std_z': std_z})
        if method_name == 'factorized_var_gt':
            c, Phi = fit_ridge_var1(Z_train_std, train_meta=meta_train, ridge=DEFAULT_VAR_RIDGE)
            Phi = stabilize_transition(Phi)
            model.update({'c': c, 'Phi': Phi})
        elif method_name == 'factorized_mar_gt':
            A0, B0 = proj_mar1_init(X_train, train_meta=meta_train, ridge=DEFAULT_MAR_RIDGE)
            A, B = lse_mar1(X_train, A0, B0, train_meta=meta_train, n_iter=8, ridge=DEFAULT_MAR_RIDGE)
            A, B = stabilize_mar_params(A, B)
            model.update({'A': A, 'B': B})
        elif method_name == 'factorized_tvpvar_gt':
            # Precompute split latent states for online TVP-VAR
            split_states = {}
            split_latent_std = {}
            split_shapes = {}
            for split_name, data in [('train', train), ('val', val), ('test', test)]:
                X_split = compress_series(data['G_weight_series'], U, V, mean_G=mean_G)
                Z_split = flatten_latent_series(X_split)
                Z_split_std = standardize_features(Z_split, mu_z, std_z)
                states = build_tvpvar_states(Z_train_std, Z_split_std, train_meta=meta_train, split_meta=data['meta'], ridge=DEFAULT_TVPVAR_RIDGE, forgetting=DEFAULT_TVPVAR_FORGETTING)
                split_states[split_name] = states
                split_latent_std[split_name] = Z_split_std
                split_shapes[split_name] = (X_split.shape[1], X_split.shape[2])
            model.update({'states': split_states, 'Z_split_std': split_latent_std, 'split_shapes': split_shapes})
        return model

    if method_name == 'sparse_tvpvar_gt':
        prior = fit_sparse_targetnode_prior(G_train, meta_train, ridge=DEFAULT_SPARSE_TVPVAR_RIDGE, topk_nonintercept=DEFAULT_SPARSE_TVPVAR_TOPK)
        model.update({'beta': prior['beta']})
        return model

    if method_name == 'dense_tvpvar_gt':
        N = G_train.shape[1]
        d = N * N
        if d > MAX_DENSE_TVPVAR_DIM:
            raise RuntimeError(f'Dense TVP-VAR uses vec(G) dimension {d}, which is too large. Increase MAX_DENSE_TVPVAR_DIM only if you really want to run it.')
        Z_train = np.stack([_vec(G_train[t]) for t in range(len(G_train))], axis=0).astype(np.float32)
        mu_z, std_z = fit_feature_standardizer(Z_train)
        Z_train_std = standardize_features(Z_train, mu_z, std_z)
        split_states, split_zstd = {}, {}
        for split_name, data in [('train', train), ('val', val), ('test', test)]:
            G_split = np.asarray(data['G_weight_series'], dtype=np.float32)
            Z_split = np.stack([_vec(G_split[t]) for t in range(len(G_split))], axis=0).astype(np.float32)
            Z_split_std = standardize_features(Z_split, mu_z, std_z)
            states = build_tvpvar_states(Z_train_std, Z_split_std, train_meta=meta_train, split_meta=data['meta'], ridge=DEFAULT_TVPVAR_RIDGE, forgetting=DEFAULT_TVPVAR_FORGETTING)
            split_states[split_name] = states
            split_zstd[split_name] = Z_split_std
        model.update({'N': N, 'mu_z': mu_z, 'std_z': std_z, 'states': split_states, 'Z_split_std': split_zstd})
        return model

    raise ValueError(f'Unsupported method_name={method_name}')


def predict_G_method(method_name: str, g_model: Dict[str, object], split_name: str, split_data: Dict[str, object], origin_idx: int, target_idx: int, horizon: int) -> np.ndarray:
    if method_name in {'true_gt', 'persistence_gt', 'ewma_gt'}:
        return predict_G_basic(method_name, split_data, origin_idx, target_idx, horizon)

    if method_name == 'factorized_var_gt':
        U, V, mean_G = g_model['U'], g_model['V'], g_model['mean_G']
        G_origin = np.asarray(split_data['G_weight_series'][origin_idx], dtype=np.float32)
        X_origin = U.T @ (G_origin - mean_G) @ V
        z_origin = _vec(X_origin)[None, :]
        z_origin_std = standardize_features(z_origin, g_model['mu_z'], g_model['std_z']).reshape(-1)
        z_pred_std = var1_predict_h(g_model['c'], g_model['Phi'], z_origin_std, horizon)
        z_pred = unstandardize_features(z_pred_std[None, :], g_model['mu_z'], g_model['std_z']).reshape(-1)
        X_pred = z_pred.reshape(U.shape[1], V.shape[1], order='F')
        return reconstruct_from_latent(X_pred, U, V, mean_G=mean_G)

    if method_name == 'factorized_mar_gt':
        U, V, mean_G = g_model['U'], g_model['V'], g_model['mean_G']
        G_origin = np.asarray(split_data['G_weight_series'][origin_idx], dtype=np.float32)
        X_origin = U.T @ (G_origin - mean_G) @ V
        X_pred = mar1_predict_h(g_model['A'], g_model['B'], X_origin, horizon)
        return reconstruct_from_latent(X_pred, U, V, mean_G=mean_G)

    if method_name == 'factorized_tvpvar_gt':
        U, V, mean_G = g_model['U'], g_model['V'], g_model['mean_G']
        c_t, Phi_t = g_model['states'][split_name][origin_idx]
        z_origin_std = g_model['Z_split_std'][split_name][origin_idx]
        z_pred_std = var1_predict_h(c_t, Phi_t, z_origin_std, horizon)
        z_pred = unstandardize_features(z_pred_std[None, :], g_model['mu_z'], g_model['std_z']).reshape(-1)
        r1, r2 = U.shape[1], V.shape[1]
        X_pred = z_pred.reshape(r1, r2, order='F')
        return reconstruct_from_latent(X_pred, U, V, mean_G=mean_G)

    if method_name == 'sparse_tvpvar_gt':
        G_origin = np.asarray(split_data['G_weight_series'][origin_idx], dtype=np.float32)
        return sparse_targetnode_predict_h(G_origin, g_model['beta'], horizon)

    if method_name == 'dense_tvpvar_gt':
        N = g_model['N']
        c_t, Phi_t = g_model['states'][split_name][origin_idx]
        z_origin_std = g_model['Z_split_std'][split_name][origin_idx]
        z_pred_std = var1_predict_h(c_t, Phi_t, z_origin_std, horizon)
        z_pred = unstandardize_features(z_pred_std[None, :], g_model['mu_z'], g_model['std_z']).reshape(-1)
        return z_pred.reshape(N, N, order='F').astype(np.float32)

    raise ValueError(f'Unsupported method_name={method_name}')

# -------------------------
# Xt downstream dataset/model
# -------------------------
def build_xt_dataset_for_horizon(method_name: str, g_model: Dict[str, object], train_data: Dict[str, object], split_name: str, split_data: Dict[str, object], horizon: int, use_gt: bool):
    z = np.asarray(split_data['z'], dtype=np.float32)
    meta = split_data['meta']
    X_rows, Y_rows = [], []
    for origin_idx, target_idx in iter_eval_pairs(meta, horizon):
        x_t = np.asarray(z[origin_idx], dtype=np.float32)
        y_true = np.asarray(z[target_idx], dtype=np.float32)
        if use_gt:
            G_used = predict_G_method(method_name, g_model, split_name, split_data, origin_idx, target_idx, horizon)
            gx = np.asarray(G_used @ x_t, dtype=np.float32)
            feat = np.concatenate([x_t, gx], axis=0)
        else:
            feat = x_t
        X_rows.append(feat.astype(np.float32))
        Y_rows.append(y_true.astype(np.float32))
    if not X_rows:
        n = z.shape[1]
        return np.empty((0, n * (2 if use_gt else 1)), dtype=np.float32), np.empty((0, n), dtype=np.float32)
    return np.stack(X_rows, axis=0).astype(np.float32), np.stack(Y_rows, axis=0).astype(np.float32)


def fit_direct_xt_model(X_train: np.ndarray, Y_train: np.ndarray) -> MultiTaskElasticNet:
    model = MultiTaskElasticNet(
        alpha=float(ALPHA),
        l1_ratio=float(L1_RATIO),
        fit_intercept=True,
        max_iter=int(MAX_ITER),
        tol=float(TOL),
        selection=SELECTION,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, Y_train)
    return model


def run_branchB_xt_forecast(method_name: str, common_dir: Path, out_dir: Path, use_gt: bool):
    out_dir.mkdir(parents=True, exist_ok=True)
    train = load_gt_split(common_dir, 'train')
    val = load_gt_split(common_dir, 'val')
    test = load_gt_split(common_dir, 'test')

    assert np.array_equal(train['segment_ids'], val['segment_ids'])
    assert np.array_equal(train['segment_ids'], test['segment_ids'])

    n_segments = int(len(train['segment_ids']))
    print('n_segments:', n_segments)
    print('method:', method_name, '| use_gt:', use_gt)

    g_model = build_g_model(method_name, train, val, test)

    rows = []
    for horizon in HORIZONS:
        print(f'\n[HORIZON {horizon}] building train features...')
        X_train, Y_train = build_xt_dataset_for_horizon(method_name, g_model, train, 'train', train, horizon, use_gt=use_gt)
        if len(X_train) == 0:
            print('No train samples; skip horizon', horizon)
            continue
        print('X_train:', X_train.shape, 'Y_train:', Y_train.shape)
        model = fit_direct_xt_model(X_train, Y_train)

        for split_name, split_data in [('val', val), ('test', test)]:
            print(f'[HORIZON {horizon}] evaluating {split_name}...')
            X_split, Y_split = build_xt_dataset_for_horizon(method_name, g_model, train, split_name, split_data, horizon, use_gt=use_gt)
            if len(X_split) == 0:
                continue
            Y_pred = model.predict(X_split).astype(np.float32)
            metrics = batch_vector_metrics(Y_split, Y_pred)
            rows.append({
                'method': method_name,
                'split': split_name,
                'lag': int(horizon),
                'n_samples': int(len(X_split)),
                'n_segments': int(n_segments),
                **metrics,
            })
            print(split_name, metrics)

        # release per-horizon memory
        del X_train, Y_train, model

    per_lag = pd.DataFrame(rows)
    per_lag = per_lag.sort_values(['method', 'split', 'lag']).reset_index(drop=True) if not per_lag.empty else per_lag
    out_path = out_dir / f'{method_name}_xt_per_lag_metrics.csv'
    per_lag.to_csv(out_path, index=False)
    print('\n[DONE]', method_name)
    print('saved:', out_path)
    print(per_lag)


# -------------------------
# Run
# -------------------------
METHOD_NAME = "factorized_var_gt"
USE_GT = True

PROJECT_ROOT = find_project_root()
BRANCHB_ROOT = PROJECT_ROOT / 'ml_core' / 'src' / 'models' / 'ML_BranchB'
COMMON_DIR = PROJECT_ROOT / 'ml_core' / 'src' / 'data_processing' / 'outputs' / 'branchB' / 'osm_edge_gt_like_branchA'
OUT_DIR = BRANCHB_ROOT / 'results' / '06_branchB_run_xt_forecast' / METHOD_NAME
OUT_DIR.mkdir(parents=True, exist_ok=True)

print('PROJECT_ROOT:', PROJECT_ROOT)
print('COMMON_DIR  :', COMMON_DIR)
print('OUT_DIR     :', OUT_DIR)

check_branchB_common_dir_ready(COMMON_DIR)

run_branchB_xt_forecast(
    method_name=METHOD_NAME,
    common_dir=COMMON_DIR,
    out_dir=OUT_DIR,
    use_gt=USE_GT,
)
