# Auto-converted from: 06_branchA_run_xt_forecast_factorized_uut_standalone_fast_per_lag(3).ipynb
# Folder target: ml_core/src/models/ML_BranchA/scripts
# Results are saved under: ml_core/src/models/ML_BranchA/results
# Generated for OSM-edge Branch A workflow.

# 06_branchA_run_xt_forecast_factorized_uut_standalone_fast_per_lag.ipynb
# Branch A replacement for old DMFM/PCA-upper-triangle.
# Memory-safe factorized correlation model:
#   R_t ~= mean_R + U X_t U^T
#   X_t = U^T (R_t - mean_R) U
#   vec(X_{t+1}) = c + Phi vec(X_t)
# For downstream X forecast, we never reconstruct full R_hat.
# We only compute R_hat @ x = mean_R @ x + U @ (X_hat @ (U.T @ x)).

import json
import gc
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import MultiTaskElasticNet

# ===================== CONFIG =====================
FORECAST_HORIZONS = list(range(1, 10))
RANK = 12
RIDGE = 1e-2
STABILITY_TARGET = 0.98

ALPHA = 0.001
L1_RATIO = 0.5
MAX_ITER = 1000
TOL = 1e-3

METHOD_NAME = "factorized_uut"
SAVE_ONLY_PER_LAG = True

# ===================== PATHS =====================
def find_project_root() -> Path:
    cwd = Path.cwd().resolve()
    candidates: List[Path] = []
    for p in [cwd, *cwd.parents, Path("/kaggle/working/Correlation_Urban_Traffic"), Path("/kaggle/working")]:
        if p not in candidates:
            candidates.append(p)
    for p in candidates:
        if (p / "ml_core").exists() and (p / "dataset").exists():
            return p
        if (p / "UTraffic-ML").exists():
            pp = p / "UTraffic-ML"
            if (pp / "ml_core").exists():
                return pp
        if (p / "data").exists():
            return p
        if (p / "Correlation_Urban_Traffic" / "data").exists():
            return p / "Correlation_Urban_Traffic"
    return cwd



def ensure_branchA_common_data_ready():
    """
    Auto-create Branch A common data if the user runs a 06_* model script directly.

    Required expected format:
        ML_BranchA/data/05_branchA_prepare_segment_segment_rt/train/R_series.npy
        ML_BranchA/data/05_branchA_prepare_segment_segment_rt/val/R_series.npy
        ML_BranchA/data/05_branchA_prepare_segment_segment_rt/test/R_series.npy
    """
    required = [
        COMMON_DIR / "train" / "R_series.npy",
        COMMON_DIR / "val" / "R_series.npy",
        COMMON_DIR / "test" / "R_series.npy",
    ]
    if all(p.exists() for p in required):
        return

    import subprocess
    import sys

    prep_script = Path(__file__).resolve().parent / "00_prepare_branchA_common_from_osm.py"
    if not prep_script.exists():
        raise FileNotFoundError(
            "Missing Branch A common data and cannot find prepare script: "
            f"{prep_script}"
        )

    print("\n" + "=" * 90)
    print("[AUTO-PREPARE] Missing Branch A common data:")
    for p in required:
        print(f"  {p} => {p.exists()}")
    print("[AUTO-PREPARE] Running:")
    print(f"  {sys.executable} {prep_script} --overwrite")
    print("=" * 90)

    subprocess.run([sys.executable, str(prep_script), "--overwrite"], check=True)

    missing_after = [p for p in required if not p.exists()]
    if missing_after:
        raise FileNotFoundError(
            "Auto-prepare finished but required files are still missing: "
            + ", ".join(str(p) for p in missing_after)
        )


PROJECT_ROOT = find_project_root()
BRANCHA_ROOT = PROJECT_ROOT / "ml_core" / "src" / "models" / "ML_BranchA"
COMMON_DIR = BRANCHA_ROOT / "data" / "05_branchA_prepare_segment_segment_rt"
OUT_DIR = BRANCHA_ROOT / "results" / "06_branchA_run_xt_forecast" / METHOD_NAME
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("PROJECT_ROOT:", PROJECT_ROOT)
print("COMMON_DIR  :", COMMON_DIR)
print("OUT_DIR     :", OUT_DIR)

# ===================== DATA LOADING =====================
def load_rt_split(common_dir: Path, split_name: str, mmap_mode: str = "r") -> Dict[str, object]:
    split_dir = common_dir / split_name
    R_series = np.load(split_dir / "R_series.npy", mmap_mode=mmap_mode)
    z = np.load(split_dir / "z.npy", mmap_mode=mmap_mode)
    segment_ids = np.load(split_dir / "segment_ids.npy")
    timestamps = pd.to_datetime(np.load(split_dir / "timestamps.npy"))
    meta = pd.read_csv(split_dir / "R_series_meta.csv")
    if "timestamp_local" in meta.columns:
        meta["timestamp_local"] = pd.to_datetime(meta["timestamp_local"])
    raw_meta = None
    raw_meta_path = split_dir / "raw_meta.csv"
    if raw_meta_path.exists():
        raw_meta = pd.read_csv(raw_meta_path)
        if "timestamp_local" in raw_meta.columns:
            raw_meta["timestamp_local"] = pd.to_datetime(raw_meta["timestamp_local"])
    return {
        "R_series": R_series,
        "z": z,
        "segment_ids": segment_ids.astype(np.int64),
        "timestamps": timestamps,
        "meta": meta,
        "raw_meta": raw_meta,
    }

# ===================== BASIC UTILITIES =====================
def _same_session_pairs(meta: pd.DataFrame, lag: int):
    T = len(meta)
    sess = meta["session_id"].to_numpy() if "session_id" in meta.columns else None
    for origin_idx in range(T - lag):
        target_idx = origin_idx + lag
        if sess is not None and sess[origin_idx] != sess[target_idx]:
            continue
        yield int(origin_idx), int(target_idx)


def vector_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    yt = np.asarray(y_true, dtype=np.float32)
    yp = np.asarray(y_pred, dtype=np.float32)
    diff = yp - yt
    mse = float(np.mean(diff ** 2))
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(mse))
    return {"mae": mae, "mse": mse, "rmse": rmse}


def spectral_radius(M: np.ndarray) -> float:
    vals = np.linalg.eigvals(np.asarray(M, dtype=np.float64))
    return float(np.max(np.abs(vals))) if vals.size else 0.0


def stabilize_transition(Phi: np.ndarray, target: float = STABILITY_TARGET) -> np.ndarray:
    rho = spectral_radius(Phi)
    if not np.isfinite(rho) or rho <= 0.0 or rho < target:
        return Phi.astype(np.float32)
    return (Phi * (target / rho)).astype(np.float32)

# ===================== FACTORIZED UUt MODEL =====================
def compute_mean_R_stream(R_series: np.ndarray) -> np.ndarray:
    """Compute mean_R without copying the whole R_series to float32."""
    T, N, _ = R_series.shape
    acc = np.zeros((N, N), dtype=np.float64)
    for i in range(T):
        Ri = np.asarray(R_series[i], dtype=np.float32)
        acc += Ri
        if (i + 1) % 20 == 0:
            print(f"  mean_R: processed {i+1}/{T}")
    mean_R = (acc / max(T, 1)).astype(np.float32)
    mean_R = 0.5 * (mean_R + mean_R.T)
    np.fill_diagonal(mean_R, 1.0)
    return mean_R


def fit_uut_basis_from_mean(mean_R: np.ndarray, rank: int = RANK) -> np.ndarray:
    """Top eigenvectors of mean correlation. mean_R is symmetric."""
    print("[UUt] eigendecomposition of mean_R ...")
    A = np.asarray(mean_R, dtype=np.float64)
    A = 0.5 * (A + A.T)
    vals, vecs = np.linalg.eigh(A)
    idx = np.argsort(vals)[::-1]
    r = int(min(rank, vecs.shape[1]))
    U = vecs[:, idx[:r]].astype(np.float32)
    print("[UUt] top eigenvalues:", vals[idx[:min(r, 10)]])
    return U


def compress_R_to_X(R: np.ndarray, U: np.ndarray, mean_R: np.ndarray) -> np.ndarray:
    """X_t = U^T (R_t - mean_R) U, rank x rank."""
    Rt = np.asarray(R, dtype=np.float32)
    return (U.T @ (Rt - mean_R) @ U).astype(np.float32)


def compress_series_stream(R_series: np.ndarray, U: np.ndarray, mean_R: np.ndarray) -> np.ndarray:
    T = R_series.shape[0]
    r = U.shape[1]
    Xs = np.empty((T, r, r), dtype=np.float32)
    for i in range(T):
        Xs[i] = compress_R_to_X(R_series[i], U, mean_R)
        if (i + 1) % 50 == 0:
            print(f"  compress: processed {i+1}/{T}")
    return Xs


def _vec(X: np.ndarray) -> np.ndarray:
    return np.asarray(X, dtype=np.float32).reshape(-1, order="F")


def _unvec(z: np.ndarray, r: int) -> np.ndarray:
    return np.asarray(z, dtype=np.float32).reshape(r, r, order="F")


def build_lagged_pairs_latent(Z: np.ndarray, meta: Optional[pd.DataFrame] = None) -> Tuple[np.ndarray, np.ndarray]:
    X_parts, Y_parts = [], []
    if meta is not None and "session_id" in meta.columns:
        for _, sub in meta.groupby("session_id", sort=False):
            idx = sub.index.to_numpy(dtype=np.int64)
            if len(idx) < 2:
                continue
            X_parts.append(Z[idx[:-1]])
            Y_parts.append(Z[idx[1:]])
    else:
        if len(Z) >= 2:
            X_parts.append(Z[:-1])
            Y_parts.append(Z[1:])
    if not X_parts:
        d = Z.shape[1]
        return np.empty((0, d), dtype=np.float32), np.empty((0, d), dtype=np.float32)
    return np.concatenate(X_parts, axis=0).astype(np.float32), np.concatenate(Y_parts, axis=0).astype(np.float32)


def fit_ridge_var1(Z_train: np.ndarray, meta: Optional[pd.DataFrame], ridge: float = RIDGE):
    X, Y = build_lagged_pairs_latent(Z_train, meta)
    if len(X) == 0:
        d = Z_train.shape[1]
        return np.zeros(d, dtype=np.float32), np.eye(d, dtype=np.float32)
    n, d = X.shape
    X_design = np.concatenate([np.ones((n, 1), dtype=np.float32), X], axis=1)
    reg = ridge * np.eye(d + 1, dtype=np.float32)
    reg[0, 0] = 0.0
    B = np.linalg.solve(X_design.T @ X_design + reg, X_design.T @ Y)
    c = B[0].astype(np.float32)
    Phi = B[1:].T.astype(np.float32)
    Phi = stabilize_transition(Phi)
    return c, Phi


def fit_feature_standardizer(Z: np.ndarray):
    mu = np.mean(Z, axis=0, keepdims=True).astype(np.float32)
    std = np.std(Z - mu, axis=0, ddof=1, keepdims=True)
    std = np.where(np.isfinite(std) & (std > 1e-6), std, 1.0).astype(np.float32)
    return mu, std


def standardize(Z: np.ndarray, mu: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((np.asarray(Z, dtype=np.float32) - mu) / std).astype(np.float32)


def unstandardize(Z: np.ndarray, mu: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (np.asarray(Z, dtype=np.float32) * std + mu).astype(np.float32)


def var_predict_h(c: np.ndarray, Phi: np.ndarray, z0: np.ndarray, h: int) -> np.ndarray:
    z = np.asarray(z0, dtype=np.float32).reshape(-1)
    for _ in range(int(h)):
        z = c + Phi @ z
    return z.astype(np.float32)


def fit_factorized_uut_model(train_data: Dict[str, object], rank: int = RANK) -> Dict[str, object]:
    R_train = train_data["R_series"]
    meta_train = train_data["meta"]
    print("[UUt] fitting mean_R ...")
    mean_R = compute_mean_R_stream(R_train)
    print("[UUt] fitting U basis ...")
    U = fit_uut_basis_from_mean(mean_R, rank=rank)
    print("[UUt] compressing train R_series ...")
    X_train = compress_series_stream(R_train, U, mean_R)
    Z_train = np.stack([_vec(X_train[t]) for t in range(len(X_train))], axis=0).astype(np.float32)
    mu, std = fit_feature_standardizer(Z_train)
    Z_train_std = standardize(Z_train, mu, std)
    print("[UUt] fitting ridge VAR(1) on latent ...", Z_train_std.shape)
    c, Phi = fit_ridge_var1(Z_train_std, meta_train, ridge=RIDGE)
    return {"mean_R": mean_R, "U": U, "mu": mu, "std": std, "c": c, "Phi": Phi, "rank": int(U.shape[1])}


def predict_relation_action_uut(model: Dict[str, object], R_origin: np.ndarray, x_t: np.ndarray, h: int) -> np.ndarray:
    """Compute R_hat_{t+h} @ x_t without reconstructing R_hat.

    R_hat = mean_R + U X_hat U^T
    R_hat @ x = mean_R @ x + U @ (X_hat @ (U.T @ x))
    """
    mean_R = model["mean_R"]
    U = model["U"]
    mu = model["mu"]
    std = model["std"]
    c = model["c"]
    Phi = model["Phi"]
    r = model["rank"]

    X0 = compress_R_to_X(R_origin, U, mean_R)
    z0 = _vec(X0)[None, :]
    z0s = standardize(z0, mu, std).reshape(-1)
    zhs = var_predict_h(c, Phi, z0s, h)
    zh = unstandardize(zhs[None, :], mu, std).reshape(-1)
    Xh = _unvec(zh, r)

    x = np.asarray(x_t, dtype=np.float32)
    base = mean_R @ x
    low = U @ (Xh @ (U.T @ x))
    return (base + low).astype(np.float32)

# ===================== DOWNSTREAM X FORECAST =====================
def _build_dataset_for_horizon(train_data: Dict[str, object], split_data: Dict[str, object], lag: int, model_uut: Dict[str, object]):
    z = np.asarray(split_data["z"], dtype=np.float32)
    R_series = split_data["R_series"]
    meta = split_data["meta"]
    X_rows, Y_rows = [], []
    for origin_idx, target_idx in _same_session_pairs(meta, lag):
        x_t = np.asarray(z[origin_idx], dtype=np.float32)
        y_t = np.asarray(z[target_idx], dtype=np.float32)
        rx = predict_relation_action_uut(model_uut, R_series[origin_idx], x_t, h=lag)
        feat = np.concatenate([x_t, rx], axis=0).astype(np.float32)
        X_rows.append(feat)
        Y_rows.append(y_t.astype(np.float32))
    if not X_rows:
        N = z.shape[1]
        return np.zeros((0, 2 * N), dtype=np.float32), np.zeros((0, N), dtype=np.float32)
    return np.stack(X_rows, axis=0), np.stack(Y_rows, axis=0)


def fit_direct_model(X_train: np.ndarray, Y_train: np.ndarray) -> MultiTaskElasticNet:
    model = MultiTaskElasticNet(
        alpha=float(ALPHA),
        l1_ratio=float(L1_RATIO),
        fit_intercept=True,
        max_iter=int(MAX_ITER),
        tol=float(TOL),
        selection="random",
        random_state=42,
    )
    model.fit(X_train, Y_train)
    return model


def run_factorized_uut_xt_forecast():
    train = load_rt_split(COMMON_DIR, "train")
    val = load_rt_split(COMMON_DIR, "val")
    test = load_rt_split(COMMON_DIR, "test")

    assert np.array_equal(train["segment_ids"], val["segment_ids"])
    assert np.array_equal(train["segment_ids"], test["segment_ids"])
    n_segments = int(len(train["segment_ids"]))
    print("n_segments:", n_segments)
    print("method:", METHOD_NAME)

    model_uut = fit_factorized_uut_model(train, rank=RANK)
    gc.collect()

    rows = []
    for lag in FORECAST_HORIZONS:
        print(f"\n[HORIZON {lag}] building train features...")
        X_train, Y_train = _build_dataset_for_horizon(train, train, lag, model_uut)
        print("X_train:", X_train.shape, "Y_train:", Y_train.shape)
        if len(X_train) == 0:
            print("No train samples; skip horizon", lag)
            continue
        direct_model = fit_direct_model(X_train, Y_train)
        del X_train, Y_train
        gc.collect()

        for split_name, data in [("val", val), ("test", test)]:
            print(f"[HORIZON {lag}] evaluating {split_name}...")
            X_eval, Y_eval = _build_dataset_for_horizon(train, data, lag, model_uut)
            if len(X_eval) == 0:
                print("No eval samples", split_name, lag)
                continue
            Y_pred = direct_model.predict(X_eval).astype(np.float32)
            m = vector_metrics(Y_eval, Y_pred)
            row = {
                "method": METHOD_NAME,
                "split": split_name,
                "lag": int(lag),
                "n_samples": int(len(Y_eval)),
                "n_segments": int(n_segments),
                **m,
            }
            rows.append(row)
            print(split_name, m)
            del X_eval, Y_eval, Y_pred
            gc.collect()

    per_lag = pd.DataFrame(rows)
    out_path = OUT_DIR / f"{METHOD_NAME}_xt_per_lag_metrics.csv"
    per_lag.to_csv(out_path, index=False)
    print("\n[DONE]", METHOD_NAME)
    print("saved:", out_path)
    print(per_lag)

run_factorized_uut_xt_forecast()
