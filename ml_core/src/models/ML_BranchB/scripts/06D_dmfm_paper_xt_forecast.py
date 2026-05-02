# ml_core/src/models/ML_BranchB/scripts/06D_dmfm_paper_xt_forecast.py
"""
Run the DMFM method from "Dynamic Matrix Factor Models for High Dimensional Time Series"
for OSM-edge traffic speed forecasting.

This script intentionally DOES NOT use Branch-B graph methods such as:
- lagged-correlation Rt
- Granger Gt
- Persistence/EWMA/Sparse TVP-VAR/Factorized VAR/MAR/TVP-VAR on graph matrices

Instead, it follows the paper's idea:
    X_t = U1 F_t U2^T + E_t
    F_t = A1 F_{t-1} A2^T + xi_t

Workflow:
1. Read train/val/test tensors from train_val_test_split.npz.
2. Select one traffic feature, usually average_speed.
3. Reshape each traffic vector x_t [N] into matrix X_t [d1,d2]. If needed, pad zeros.
4. Estimate matrix factor loadings U1, U2 from train matrices by covariance PCA.
5. Estimate latent factor series F_t = U1^T X_t U2.
6. Fit MAR(1) dynamics in factor space.
7. Forecast X_{t+h} from the latent factor dynamics.
8. Save MAE/MSE/RMSE per split, horizon, method, and plots.

Implemented paper-style prediction variants:
- dmfm_lse       : MAR-LSE plug-in prediction, like Eq. (17).
- dmfm_l2e       : MAR-L2E plug-in prediction using lag-2 moment + nearest Kronecker projection.
- dmfm_l2e_kf    : MAR-L2E + Kalman filter factor estimate, like Eq. (18), practical implementation.
- dmfm_l2e_plus  : use dmfm_l2e_kf if estimated measurement-noise covariance is positive definite, else dmfm_lse.
- dmfm_vlse      : vector VAR-LSE on vec(F_t), included because paper compares V.LSE.
- dmfm_vl2e      : vector VAR lag-2 estimator on vec(F_t), included because paper compares V.L2E.

Notes:
- The paper uses iTIPUP/iTOPUP for matrix factor loading estimation. This script uses a
  covariance-PCA estimator as a practical replacement for your current pipeline.
- For OSM edges, the matrix shape is an index-based reshape of nodes. Use --matrix-shape
  to control it. For full N=3696, auto gives 48x77 exactly.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


# =============================================================================
# Paths / logging
# =============================================================================

THIS_FILE = Path(__file__).resolve()
DATA_PROCESSING_DIR = THIS_FILE.parents[2] / "data_processing"
PROJECT_ROOT = THIS_FILE.parents[4]
DEFAULT_SOURCE_NPZ = DATA_PROCESSING_DIR / "outputs" / "branchA" / "osm_edge_forecasting_dataset" / "train_val_test_split.npz"
DEFAULT_OUTPUT_DIR = THIS_FILE.parents[1] / "results" / "06D_dmfm_paper_xt_forecast"


def now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    print(f"[{now_str()}] {msg}", flush=True)


def print_stage(title: str) -> None:
    print("\n" + "=" * 88, flush=True)
    print(title, flush=True)
    print("=" * 88, flush=True)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


class NumpyJsonEncoder(json.JSONEncoder):
    def default(self, obj: Any):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, Path):
            return str(obj)
        return super().default(obj)


def save_json(obj: Dict[str, Any], path: Path) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, cls=NumpyJsonEncoder)


def parse_int_list(s: str) -> List[int]:
    out: List[int] = []
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


def fmt_shape(x: Any) -> str:
    return "x".join(map(str, np.asarray(x).shape))


# =============================================================================
# Data loading / node reshape
# =============================================================================


def load_npz_arrays(source_npz: Path, feature: str, prefer: str) -> Dict[str, Any]:
    if not source_npz.exists():
        raise FileNotFoundError(f"Missing source NPZ: {source_npz}")
    data = np.load(str(source_npz), allow_pickle=True)
    keys = set(data.files)

    feature_names = [str(x) for x in data["feature_names"].tolist()] if "feature_names" in keys else []
    if feature_names and feature in feature_names:
        f_idx = feature_names.index(feature)
    else:
        f_idx = 0
        if feature_names:
            log(f"[WARN] feature={feature} not found. Use first feature: {feature_names[0]}")
        else:
            log("[WARN] feature_names not found. Use feature index 0.")

    def pick(split: str) -> np.ndarray:
        candidates: List[str]
        if prefer == "norm":
            candidates = [f"X_{split}", f"X_{split}_norm", f"X_{split}_filled"]
        elif prefer == "filled":
            candidates = [f"X_{split}_filled", f"X_{split}", f"X_{split}_norm"]
        else:
            candidates = [f"X_{split}", f"X_{split}_norm", f"X_{split}_filled"]
        key = next((k for k in candidates if k in keys), None)
        if key is None:
            raise KeyError(f"Cannot find X for split={split}; tried={candidates}; keys={sorted(keys)}")
        X = np.asarray(data[key])
        if X.ndim == 3:
            return X[:, :, f_idx].astype(np.float32)
        if X.ndim == 2:
            return X.astype(np.float32)
        raise ValueError(f"{key} must be 2D or 3D, got shape={X.shape}")

    out = {
        "X_train": pick("train"),
        "X_val": pick("val"),
        "X_test": pick("test"),
        "feature_names": feature_names,
        "feature_used": feature_names[f_idx] if feature_names else f"feature_{f_idx}",
        "feature_index": int(f_idx),
    }
    if "model_node_ids" in keys:
        out["segment_ids"] = np.asarray(data["model_node_ids"], dtype=np.int64)
    elif "segment_ids" in keys:
        out["segment_ids"] = np.asarray(data["segment_ids"], dtype=np.int64)
    else:
        out["segment_ids"] = np.arange(out["X_train"].shape[1], dtype=np.int64)

    for split in ["train", "val", "test"]:
        tkey = f"timestamps_{split}"
        if tkey in keys:
            out[f"timestamps_{split}"] = np.asarray(data[tkey]).astype(str)
        else:
            out[f"timestamps_{split}"] = np.asarray([f"{split}_{i}" for i in range(out[f"X_{split}"].shape[0])]).astype(str)
    return out


def resolve_node_indices(
    segment_ids: np.ndarray,
    max_nodes: int,
    node_indices: Optional[str],
    node_ids: Optional[str],
    node_sample: str,
    seed: int,
) -> Optional[np.ndarray]:
    N = len(segment_ids)
    idx: Optional[np.ndarray] = None

    if node_indices:
        parsed = np.asarray(parse_int_list(node_indices), dtype=np.int64)
        if parsed.size == 0:
            raise ValueError("--node-indices parsed empty.")
        if parsed.min() < 0 or parsed.max() >= N:
            raise ValueError(f"node index out of range for N={N}: min={parsed.min()}, max={parsed.max()}")
        idx = parsed

    if node_ids:
        req = np.asarray(parse_int_list(node_ids), dtype=np.int64)
        pos = {int(v): i for i, v in enumerate(segment_ids)}
        miss = [int(v) for v in req if int(v) not in pos]
        if miss:
            raise ValueError(f"node ids missing from segment_ids: {miss[:20]}")
        idx2 = np.asarray([pos[int(v)] for v in req], dtype=np.int64)
        idx = idx2 if idx is None else np.intersect1d(idx, idx2)

    if idx is None and int(max_nodes) > 0 and int(max_nodes) < N:
        if node_sample == "first":
            idx = np.arange(int(max_nodes), dtype=np.int64)
        elif node_sample == "random":
            rng = np.random.default_rng(int(seed))
            idx = np.sort(rng.choice(N, size=int(max_nodes), replace=False).astype(np.int64))
        else:
            raise ValueError("--node-sample must be first or random")

    if idx is None:
        return None
    idx = np.asarray(sorted(set(map(int, idx.tolist()))), dtype=np.int64)
    if idx.size == 0:
        raise ValueError("Selected node set is empty.")
    return idx


def auto_matrix_shape(n: int) -> Tuple[int, int]:
    # Prefer exact factors closest to square. If no exact factor near, pad to next rectangle.
    root = int(math.sqrt(n))
    best = (1, n)
    best_gap = n - 1
    for a in range(1, root + 1):
        if n % a == 0:
            b = n // a
            gap = abs(b - a)
            if gap < best_gap:
                best = (a, b)
                best_gap = gap
    return best


def parse_matrix_shape(arg: str, n: int) -> Tuple[int, int, int]:
    if str(arg).lower() == "auto":
        d1, d2 = auto_matrix_shape(n)
        return d1, d2, d1 * d2
    m = re.match(r"^\s*(\d+)\s*[,xX]\s*(\d+)\s*$", str(arg))
    if not m:
        raise ValueError("--matrix-shape must be 'auto' or like '48,77' / '48x77'")
    d1, d2 = int(m.group(1)), int(m.group(2))
    if d1 * d2 < n:
        raise ValueError(f"matrix-shape {d1}x{d2} has capacity {d1*d2} < N={n}")
    return d1, d2, d1 * d2


def vectors_to_matrices(X: np.ndarray, d1: int, d2: int) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    T, N = X.shape
    cap = d1 * d2
    if cap < N:
        raise ValueError(f"matrix capacity {cap} < N={N}")
    if cap == N:
        return X.reshape(T, d1, d2).astype(np.float32)
    padded = np.zeros((T, cap), dtype=np.float32)
    padded[:, :N] = X
    return padded.reshape(T, d1, d2).astype(np.float32)


def matrices_to_vectors(Xm: np.ndarray, n_original: int) -> np.ndarray:
    T = Xm.shape[0]
    return Xm.reshape(T, -1)[:, :n_original].astype(np.float32)


# =============================================================================
# Linear algebra helpers
# =============================================================================


def top_eigenvectors_symmetric(M: np.ndarray, rank: int) -> np.ndarray:
    M = np.asarray(M, dtype=np.float64)
    M = 0.5 * (M + M.T)
    vals, vecs = np.linalg.eigh(M)
    order = np.argsort(vals)[::-1]
    rank = min(int(rank), vecs.shape[1])
    U = vecs[:, order[:rank]]
    # deterministic sign
    for k in range(U.shape[1]):
        imax = int(np.argmax(np.abs(U[:, k])))
        if U[imax, k] < 0:
            U[:, k] *= -1
    return U.astype(np.float32)


def estimate_loadings_cov_pca(X_train_centered: np.ndarray, r1: int, r2: int) -> Tuple[np.ndarray, np.ndarray]:
    T, d1, d2 = X_train_centered.shape
    row_cov = np.zeros((d1, d1), dtype=np.float64)
    col_cov = np.zeros((d2, d2), dtype=np.float64)
    for X in X_train_centered:
        X64 = X.astype(np.float64, copy=False)
        row_cov += X64 @ X64.T / max(1, d2)
        col_cov += X64.T @ X64 / max(1, d1)
    row_cov /= max(1, T)
    col_cov /= max(1, T)
    U1 = top_eigenvectors_symmetric(row_cov, r1)
    U2 = top_eigenvectors_symmetric(col_cov, r2)
    return U1, U2


def project_factors(Xm_centered: np.ndarray, U1: np.ndarray, U2: np.ndarray) -> np.ndarray:
    # F_t = U1^T X_t U2
    return np.einsum("ia,tij,jb->tab", U1, Xm_centered, U2, optimize=True).astype(np.float32)


def reconstruct_from_factors(F: np.ndarray, U1: np.ndarray, U2: np.ndarray, mean_matrix: np.ndarray) -> np.ndarray:
    Xc = np.einsum("ia,tab,jb->tij", U1, F, U2, optimize=True).astype(np.float32)
    return Xc + mean_matrix[None, :, :].astype(np.float32)


def vecF(F: np.ndarray) -> np.ndarray:
    # column-major vec to match vec(A1 F A2^T) = (A2 kron A1) vec(F)
    return np.asarray([f.reshape(-1, order="F") for f in F], dtype=np.float64)


def unvecF(v: np.ndarray, r1: int, r2: int) -> np.ndarray:
    return np.asarray(v, dtype=np.float64).reshape(r1, r2, order="F")


def kron_phi(A1: np.ndarray, A2: np.ndarray) -> np.ndarray:
    return np.kron(A2, A1).astype(np.float64)


def project_psd(M: np.ndarray, eps: float = 0.0) -> np.ndarray:
    M = 0.5 * (np.asarray(M, dtype=np.float64) + np.asarray(M, dtype=np.float64).T)
    vals, vecs = np.linalg.eigh(M)
    vals = np.maximum(vals, eps)
    return (vecs * vals) @ vecs.T


def nearest_kron_A2_A1(Phi: np.ndarray, r1: int, r2: int) -> Tuple[np.ndarray, np.ndarray]:
    """Approximate Phi ~= A2 kron A1 via SVD rearrangement."""
    Phi = np.asarray(Phi, dtype=np.float64)
    q = r1 * r2
    if Phi.shape != (q, q):
        raise ValueError(f"Phi shape must be {(q,q)}, got {Phi.shape}")
    R = np.zeros((r2 * r2, r1 * r1), dtype=np.float64)
    row = 0
    for j_col in range(r2):
        for i_row in range(r2):
            # block corresponding to A2[i_row, j_col] * A1
            block = Phi[i_row * r1:(i_row + 1) * r1, j_col * r1:(j_col + 1) * r1]
            R[row, :] = block.reshape(-1, order="F")
            row += 1
    U, S, Vt = np.linalg.svd(R, full_matrices=False)
    s = math.sqrt(max(float(S[0]), 1e-12))
    a2_vec = U[:, 0] * s
    a1_vec = Vt[0, :] * s
    A2 = a2_vec.reshape(r2, r2, order="F")
    A1 = a1_vec.reshape(r1, r1, order="F")

    # Stabilize scale ambiguity while preserving A2 kron A1 approximately.
    n1 = np.linalg.norm(A1, "fro")
    if n1 > 1e-12:
        A1 = A1 / n1
        A2 = A2 * n1
    return A1.astype(np.float64), A2.astype(np.float64)


# =============================================================================
# Dynamics estimators
# =============================================================================


def fit_var_lse(F_train: np.ndarray, ridge: float) -> np.ndarray:
    V = vecF(F_train)
    X = V[:-1]
    Y = V[1:]
    q = X.shape[1]
    XtX = X.T @ X + float(ridge) * np.eye(q)
    B = np.linalg.solve(XtX, X.T @ Y)  # q x q, y = x @ B
    return B.T.astype(np.float64)      # column vector transition: y_col = Phi @ x_col


def fit_var_l2e(F_train: np.ndarray, ridge: float) -> np.ndarray:
    V = vecF(F_train)
    if V.shape[0] < 4:
        return fit_var_lse(F_train, ridge)
    X0 = V[:-2]
    X1 = V[1:-1]
    X2 = V[2:]
    q = V.shape[1]
    Gamma1 = (X1.T @ X0) / max(1, X0.shape[0])
    Gamma2 = (X2.T @ X0) / max(1, X0.shape[0])
    Phi = Gamma2 @ np.linalg.pinv(Gamma1 + float(ridge) * np.eye(q))
    return Phi.astype(np.float64)


def fit_mar_lse(F_train: np.ndarray, ridge: float, max_iter: int = 100, tol: float = 1e-7) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    F = np.asarray(F_train, dtype=np.float64)
    T, r1, r2 = F.shape
    Phi0 = fit_var_lse(F, ridge=ridge)
    A1, A2 = nearest_kron_A2_A1(Phi0, r1, r2)

    prev_loss = np.inf
    losses = []
    I1 = np.eye(r1)
    I2 = np.eye(r2)

    for it in range(int(max_iter)):
        # Update A1 given A2.
        num1 = np.zeros((r1, r1), dtype=np.float64)
        den1 = float(ridge) * I1.copy()
        A2TA2 = A2.T @ A2
        for t in range(1, T):
            Y = F[t]
            X = F[t - 1]
            Xtilde = X @ A2.T
            num1 += Y @ Xtilde.T
            den1 += Xtilde @ Xtilde.T
        A1 = num1 @ np.linalg.pinv(den1)

        # Update A2 given A1.
        num2 = np.zeros((r2, r2), dtype=np.float64)
        den2 = float(ridge) * I2.copy()
        A1TA1 = A1.T @ A1
        for t in range(1, T):
            Y = F[t]
            X = F[t - 1]
            Xtilde_t = X.T @ A1.T  # r2 x r1; Y.T = A2 @ Xtilde_t
            num2 += Y.T @ Xtilde_t.T
            den2 += Xtilde_t @ Xtilde_t.T
        A2 = num2 @ np.linalg.pinv(den2)

        # Scale normalization: A1*c and A2/c are equivalent in prediction.
        n1 = np.linalg.norm(A1, "fro")
        if n1 > 1e-12:
            A1 = A1 / n1
            A2 = A2 * n1

        loss = 0.0
        denom = 0.0
        for t in range(1, T):
            pred = A1 @ F[t - 1] @ A2.T
            err = F[t] - pred
            loss += float(np.sum(err * err))
            denom += float(np.sum(F[t] * F[t]))
        rel = loss / max(denom, 1e-12)
        losses.append(rel)
        if abs(prev_loss - rel) < float(tol):
            break
        prev_loss = rel

    info = {"n_iter": int(len(losses)), "final_relative_loss": float(losses[-1] if losses else np.nan)}
    return A1.astype(np.float64), A2.astype(np.float64), info


def fit_mar_l2e(F_train: np.ndarray, ridge: float) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    F = np.asarray(F_train, dtype=np.float64)
    T, r1, r2 = F.shape
    Phi_l2e = fit_var_l2e(F, ridge=ridge)
    A1, A2 = nearest_kron_A2_A1(Phi_l2e, r1, r2)
    info = {"estimator": "nearest_kron_projection_of_lag2_VAR", "phi_norm": float(np.linalg.norm(Phi_l2e, "fro"))}
    return A1, A2, info


def estimate_kalman_covariances(F_train: np.ndarray, Phi: np.ndarray, eps: float = 1e-7) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    V = vecF(F_train)
    q = V.shape[1]
    if V.shape[0] < 4:
        Sz = eps * np.eye(q)
        Sx = eps * np.eye(q)
        return Sz, Sx, {"kalman_cov_fallback": True, "min_eig_sigma_zeta": float(eps)}

    pred = (Phi @ V[:-1].T).T
    W = V[1:] - pred
    Wc = W - W.mean(axis=0, keepdims=True)
    G0 = (Wc.T @ Wc) / max(1, Wc.shape[0] - 1)
    if Wc.shape[0] >= 2:
        G1 = (Wc[1:].T @ Wc[:-1]) / max(1, Wc.shape[0] - 1)
    else:
        G1 = np.zeros_like(G0)

    Phi_inv = np.linalg.pinv(Phi)
    Sigma_zeta_raw = -0.5 * (Phi_inv @ G1 + G1.T @ Phi_inv.T)
    Sigma_zeta = project_psd(Sigma_zeta_raw, eps=0.0)
    Sigma_xi_raw = G0 - Sigma_zeta - Phi @ Sigma_zeta @ Phi.T
    Sigma_xi = project_psd(Sigma_xi_raw, eps=eps)

    eig_z = np.linalg.eigvalsh(0.5 * (Sigma_zeta + Sigma_zeta.T))
    info = {
        "kalman_cov_fallback": False,
        "min_eig_sigma_zeta": float(np.min(eig_z)),
        "max_eig_sigma_zeta": float(np.max(eig_z)),
        "trace_sigma_zeta": float(np.trace(Sigma_zeta)),
        "trace_sigma_xi": float(np.trace(Sigma_xi)),
    }
    return Sigma_zeta.astype(np.float64), Sigma_xi.astype(np.float64), info


def kalman_filter_factor_observations(F_obs: np.ndarray, Phi: np.ndarray, Sigma_zeta: np.ndarray, Sigma_xi: np.ndarray) -> np.ndarray:
    """Return filtered state mean f_{t|t} for observed factor vectors."""
    V = vecF(F_obs)
    T, q = V.shape
    I = np.eye(q)
    R = project_psd(Sigma_zeta, eps=1e-7)
    Q = project_psd(Sigma_xi, eps=1e-7)

    means = np.zeros_like(V, dtype=np.float64)
    m = V[0].astype(np.float64)
    P = R.copy() + 1e-6 * I

    for t in range(T):
        if t > 0:
            m = Phi @ m
            P = Phi @ P @ Phi.T + Q
            P = 0.5 * (P + P.T)

        y = V[t]
        S = P + R + 1e-7 * I
        K = P @ np.linalg.pinv(S)
        m = m + K @ (y - m)
        P = (I - K) @ P
        P = 0.5 * (P + P.T)
        means[t] = m
    return means.astype(np.float64)


# =============================================================================
# Forecast / metrics
# =============================================================================


def predict_factor_mar(F_origin: np.ndarray, A1: np.ndarray, A2: np.ndarray, horizon: int) -> np.ndarray:
    F = np.asarray(F_origin, dtype=np.float64)
    for _ in range(int(horizon)):
        F = A1 @ F @ A2.T
    return F.astype(np.float32)


def predict_factor_phi(f_origin_vec: np.ndarray, Phi: np.ndarray, r1: int, r2: int, horizon: int) -> np.ndarray:
    f = np.asarray(f_origin_vec, dtype=np.float64)
    for _ in range(int(horizon)):
        f = Phi @ f
    return unvecF(f, r1, r2).astype(np.float32)


def evaluate_method_on_split(
    method: str,
    X_split_matrix_centered: np.ndarray,
    X_split_vector_true: np.ndarray,
    U1: np.ndarray,
    U2: np.ndarray,
    mean_matrix: np.ndarray,
    n_original: int,
    horizons: Sequence[int],
    params: Dict[str, Any],
) -> pd.DataFrame:
    F_obs = project_factors(X_split_matrix_centered, U1, U2)
    T, r1, r2 = F_obs.shape

    # Precompute filtered factors for KF methods.
    filtered_vecs: Optional[np.ndarray] = None
    if method in {"dmfm_l2e_kf", "dmfm_l2e_plus"}:
        filtered_vecs = kalman_filter_factor_observations(
            F_obs=F_obs,
            Phi=params["Phi_l2e"],
            Sigma_zeta=params["Sigma_zeta"],
            Sigma_xi=params["Sigma_xi"],
        )

    rows = []
    for h in horizons:
        h = int(h)
        if T <= h:
            continue
        preds: List[np.ndarray] = []
        trues: List[np.ndarray] = []
        for t in range(0, T - h):
            if method == "dmfm_lse":
                Fp = predict_factor_mar(F_obs[t], params["A1_lse"], params["A2_lse"], h)
            elif method == "dmfm_l2e":
                Fp = predict_factor_mar(F_obs[t], params["A1_l2e"], params["A2_l2e"], h)
            elif method == "dmfm_l2e_kf":
                assert filtered_vecs is not None
                Fp = predict_factor_phi(filtered_vecs[t], params["Phi_l2e"], r1, r2, h)
            elif method == "dmfm_l2e_plus":
                if params.get("use_kalman_for_l2e_plus", False):
                    assert filtered_vecs is not None
                    Fp = predict_factor_phi(filtered_vecs[t], params["Phi_l2e"], r1, r2, h)
                else:
                    Fp = predict_factor_mar(F_obs[t], params["A1_lse"], params["A2_lse"], h)
            elif method == "dmfm_vlse":
                f = F_obs[t].reshape(-1, order="F")
                Fp = predict_factor_phi(f, params["Phi_vlse"], r1, r2, h)
            elif method == "dmfm_vl2e":
                f = F_obs[t].reshape(-1, order="F")
                Fp = predict_factor_phi(f, params["Phi_l2e"], r1, r2, h)
            else:
                raise ValueError(f"Unknown method: {method}")

            Xhat_m = reconstruct_from_factors(Fp[None, :, :], U1, U2, mean_matrix)[0]
            xhat = Xhat_m.reshape(-1)[:n_original].astype(np.float32)
            preds.append(xhat)
            trues.append(X_split_vector_true[t + h].astype(np.float32))

        P = np.stack(preds, axis=0)
        Y = np.stack(trues, axis=0)
        err = P - Y
        mae = float(np.mean(np.abs(err)))
        mse = float(np.mean(err ** 2))
        rmse = float(math.sqrt(mse))
        rows.append({
            "method": method,
            "horizon": int(h),
            "n_samples": int(P.shape[0]),
            "n_nodes": int(P.shape[1]),
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
        })
    return pd.DataFrame(rows)


def plot_metrics(metrics: pd.DataFrame, output_dir: Path) -> None:
    if plt is None:
        log("matplotlib not installed; skip plots")
        return
    plot_dir = ensure_dir(output_dir / "plots")
    for split in sorted(metrics["split"].unique()):
        df_s = metrics[metrics["split"] == split]
        for metric in ["mae", "rmse", "mse"]:
            plt.figure(figsize=(12, 5))
            for method, g in df_s.groupby("method"):
                g = g.sort_values("horizon")
                plt.plot(g["horizon"], g[metric], marker="o", label=method)
            plt.title(f"DMFM paper forecast - {metric.upper()} by horizon ({split})")
            plt.xlabel("Forecast horizon")
            plt.ylabel(metric.upper())
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            path = plot_dir / f"dmfm_{metric}_by_horizon_{split}.png"
            plt.savefig(path, dpi=160, bbox_inches="tight")
            plt.close()
            log(f"Saved plot: {path}")


def make_summary_text(output_dir: Path, config: Dict[str, Any], fit_info: Dict[str, Any]) -> None:
    lines = []
    lines.append("DMFM paper-based XT forecast")
    lines.append("=" * 80)
    lines.append("")
    lines.append("This run does not use lagged-correlation Rt, Granger Gt, or graph-forecasting pipelines.")
    lines.append("It uses Dynamic Matrix Factor Model: X_t = U1 F_t U2^T + E_t; F_t = A1 F_{t-1} A2^T + xi_t.")
    lines.append("")
    lines.append("Config:")
    lines.append(json.dumps(config, ensure_ascii=False, indent=2, cls=NumpyJsonEncoder))
    lines.append("")
    lines.append("Fit info:")
    lines.append(json.dumps(fit_info, ensure_ascii=False, indent=2, cls=NumpyJsonEncoder))
    (output_dir / "README_DMFM_PAPER_RUN.txt").write_text("\n".join(lines), encoding="utf-8")


# =============================================================================
# CLI
# =============================================================================


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run DMFM paper method for XT traffic forecasting.")
    p.add_argument("--source-npz", type=str, default=str(DEFAULT_SOURCE_NPZ))
    p.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    p.add_argument("--feature", type=str, default="average_speed")
    p.add_argument("--prefer", type=str, default="norm", choices=["norm", "filled"])
    p.add_argument("--matrix-shape", type=str, default="auto", help="auto or e.g. 48,77 / 16x32")
    p.add_argument("--rank", type=str, default="8,8", help="latent matrix rank r1,r2")
    p.add_argument("--horizons", type=str, default="1-9")
    p.add_argument("--methods", type=str, default="all", help="all or comma list: dmfm_lse,dmfm_l2e,dmfm_l2e_kf,dmfm_l2e_plus,dmfm_vlse,dmfm_vl2e")
    p.add_argument("--max-nodes", type=int, default=0)
    p.add_argument("--node-indices", type=str, default=None)
    p.add_argument("--node-ids", type=str, default=None)
    p.add_argument("--node-sample", type=str, choices=["first", "random"], default="first")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ridge", type=float, default=1e-4)
    p.add_argument("--max-iter", type=int, default=100)
    p.add_argument("--tol", type=float, default=1e-7)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    np.random.seed(int(args.seed))

    source_npz = Path(args.source_npz)
    output_dir = Path(args.output_dir)
    if not source_npz.is_absolute():
        source_npz = PROJECT_ROOT / source_npz
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)
    ensure_dir(output_dir)

    r1, r2 = [int(x) for x in re.split(r"[,xX]", args.rank.strip())]
    horizons = parse_int_list(args.horizons)
    if args.methods == "all":
        methods = ["dmfm_lse", "dmfm_l2e", "dmfm_l2e_kf", "dmfm_l2e_plus", "dmfm_vlse", "dmfm_vl2e"]
    else:
        methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    print_stage("DMFM PAPER METHOD - LOAD DATA")
    log(f"PROJECT_ROOT: {PROJECT_ROOT}")
    log(f"SOURCE_NPZ  : {source_npz}")
    log(f"OUTPUT_DIR  : {output_dir}")
    log(f"FEATURE     : {args.feature}")
    log(f"RANK        : {(r1, r2)}")
    log(f"HORIZONS    : {horizons}")
    log(f"METHODS     : {methods}")

    data = load_npz_arrays(source_npz, feature=args.feature, prefer=args.prefer)
    seg_ids = np.asarray(data["segment_ids"], dtype=np.int64)
    node_idx = resolve_node_indices(seg_ids, args.max_nodes, args.node_indices, args.node_ids, args.node_sample, args.seed)

    X_train = data["X_train"]
    X_val = data["X_val"]
    X_test = data["X_test"]
    if node_idx is not None:
        X_train = X_train[:, node_idx]
        X_val = X_val[:, node_idx]
        X_test = X_test[:, node_idx]
        seg_ids = seg_ids[node_idx]

    N = int(X_train.shape[1])
    d1, d2, cap = parse_matrix_shape(args.matrix_shape, N)
    if r1 > d1 or r2 > d2:
        raise ValueError(f"rank {(r1,r2)} must be <= matrix shape {(d1,d2)}")

    log(f"X_train shape: {X_train.shape}")
    log(f"X_val shape  : {X_val.shape}")
    log(f"X_test shape : {X_test.shape}")
    log(f"node mode    : {'full' if node_idx is None else f'subset n={len(node_idx)}'}")
    log(f"matrix shape : {d1} x {d2}, capacity={cap}, padded_nodes={cap - N}")
    if args.dry_run:
        return

    print_stage("RESHAPE VECTOR TIME SERIES TO MATRIX TIME SERIES")
    Xtr_m = vectors_to_matrices(X_train, d1, d2)
    Xva_m = vectors_to_matrices(X_val, d1, d2)
    Xte_m = vectors_to_matrices(X_test, d1, d2)

    mean_matrix = Xtr_m.mean(axis=0).astype(np.float32)
    Xtr_c = Xtr_m - mean_matrix[None, :, :]
    Xva_c = Xva_m - mean_matrix[None, :, :]
    Xte_c = Xte_m - mean_matrix[None, :, :]

    print_stage("STAGE 1 - MATRIX FACTOR ESTIMATION")
    U1, U2 = estimate_loadings_cov_pca(Xtr_c, r1=r1, r2=r2)
    F_train = project_factors(Xtr_c, U1, U2)
    log(f"U1 shape={U1.shape}, U2 shape={U2.shape}, F_train shape={F_train.shape}")

    print_stage("STAGE 2 - LATENT MAR / VAR ESTIMATION")
    A1_lse, A2_lse, info_lse = fit_mar_lse(F_train, ridge=args.ridge, max_iter=args.max_iter, tol=args.tol)
    A1_l2e, A2_l2e, info_l2e = fit_mar_l2e(F_train, ridge=args.ridge)
    Phi_vlse = fit_var_lse(F_train, ridge=args.ridge)
    Phi_l2e = fit_var_l2e(F_train, ridge=args.ridge)
    Phi_mar_l2e = kron_phi(A1_l2e, A2_l2e)
    Sigma_zeta, Sigma_xi, info_kf = estimate_kalman_covariances(F_train, Phi_mar_l2e)
    use_kf = bool(info_kf.get("min_eig_sigma_zeta", 0.0) > 1e-10)
    log(f"MAR-LSE info: {info_lse}")
    log(f"MAR-L2E info: {info_l2e}")
    log(f"Kalman covariance info: {info_kf}; L2E+ use_kalman={use_kf}")

    params = {
        "A1_lse": A1_lse,
        "A2_lse": A2_lse,
        "A1_l2e": A1_l2e,
        "A2_l2e": A2_l2e,
        "Phi_vlse": Phi_vlse,
        "Phi_l2e": Phi_mar_l2e,  # L2E/KF uses MAR-L2E Kronecker transition.
        "Sigma_zeta": Sigma_zeta,
        "Sigma_xi": Sigma_xi,
        "use_kalman_for_l2e_plus": use_kf,
    }

    print_stage("EVALUATE VAL / TEST FORECASTS")
    all_metrics = []
    for split_name, Xm_c, X_true in [
        ("val", Xva_c, X_val),
        ("test", Xte_c, X_test),
    ]:
        for method in methods:
            log(f"Evaluating split={split_name}, method={method}")
            df = evaluate_method_on_split(
                method=method,
                X_split_matrix_centered=Xm_c,
                X_split_vector_true=X_true,
                U1=U1,
                U2=U2,
                mean_matrix=mean_matrix,
                n_original=N,
                horizons=horizons,
                params=params,
            )
            df.insert(0, "split", split_name)
            all_metrics.append(df)
    metrics = pd.concat(all_metrics, ignore_index=True)
    metrics_path = output_dir / "dmfm_paper_metrics.csv"
    metrics.to_csv(metrics_path, index=False)
    log(f"Saved metrics: {metrics_path}")

    # Save compact artifacts.
    np.savez_compressed(
        output_dir / "dmfm_paper_model_compact.npz",
        U1=U1.astype(np.float32),
        U2=U2.astype(np.float32),
        mean_matrix=mean_matrix.astype(np.float32),
        A1_lse=A1_lse.astype(np.float32),
        A2_lse=A2_lse.astype(np.float32),
        A1_l2e=A1_l2e.astype(np.float32),
        A2_l2e=A2_l2e.astype(np.float32),
        Phi_vlse=Phi_vlse.astype(np.float32),
        Phi_l2e=Phi_mar_l2e.astype(np.float32),
        Sigma_zeta=Sigma_zeta.astype(np.float32),
        Sigma_xi=Sigma_xi.astype(np.float32),
        segment_ids=seg_ids.astype(np.int64),
        matrix_shape=np.asarray([d1, d2], dtype=np.int64),
        rank=np.asarray([r1, r2], dtype=np.int64),
    )

    config = vars(args).copy()
    config.update({
        "source_npz": str(source_npz),
        "output_dir": str(output_dir),
        "feature_used": data["feature_used"],
        "n_nodes": N,
        "matrix_shape": [d1, d2],
        "rank": [r1, r2],
        "methods": methods,
        "horizons": horizons,
        "node_mode": "full" if node_idx is None else f"subset n={len(node_idx)}",
    })
    fit_info = {"mar_lse": info_lse, "mar_l2e": info_l2e, "kalman": info_kf, "l2e_plus_use_kalman": use_kf}
    save_json({"config": config, "fit_info": fit_info}, output_dir / "dmfm_paper_run_summary.json")
    make_summary_text(output_dir, config, fit_info)

    if not args.no_plots:
        plot_metrics(metrics, output_dir)

    print_stage("DONE")
    log(f"Metrics: {metrics_path}")
    log(f"Plots  : {output_dir / 'plots'}")


if __name__ == "__main__":
    main()
