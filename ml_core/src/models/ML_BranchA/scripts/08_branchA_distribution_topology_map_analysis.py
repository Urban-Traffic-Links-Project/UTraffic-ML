# ml_core/src/models/ML_BranchA/scripts/08_branchA_distribution_topology_map_analysis.py
"""
Branch A advanced analysis:
1) So sánh phân phối giữa R_true và R_pred.
2) Đánh giá khả năng giữ cấu trúc tương quan mạnh.
3) Đánh giá lỗi theo topology OSM.
4) Vẽ map OSM thật:
   - Chọn 5 đoạn đường nguồn.
   - Với mỗi source, vẽ top 30 đoạn tương quan mạnh nhất.
   - Rank 1-10: đỏ, rank 11-20: vàng, rank 21-30: xanh.
   - Vẽ map sai số DMFM theo từng đoạn đường.

Input chính:
    ml_core/src/models/ML_BranchA/data/05_branchA_prepare_segment_segment_rt/
    ml_core/src/data_processing/outputs/branchA/match_summary/matched_osm_edge_metadata.csv

Output:
    ml_core/src/models/ML_BranchA/results/08_distribution_topology_map_analysis/

Chạy nhanh trên Kaggle:
    python -u ml_core/src/models/ML_BranchA/scripts/08_branchA_distribution_topology_map_analysis.py \
      --splits val,test --lags 1,3,6,9 --samples-per-split-lag 6 --pair-samples 80000 \
      2>&1 | tee logs_A_08_analysis.txt

Chạy nhẹ nếu full N quá nặng:
    python -u ml_core/src/models/ML_BranchA/scripts/08_branchA_distribution_topology_map_analysis.py \
      --max-nodes 1500 --splits val,test --lags 1,3,6,9 --samples-per-split-lag 6 --pair-samples 80000 \
      2>&1 | tee logs_A_08_analysis_1500.txt
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

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
    from sklearn.utils.extmath import randomized_svd
except Exception:
    randomized_svd = None

try:
    import folium
except Exception:
    folium = None


# ============================================================
# Config
# ============================================================

WINDOW = 10
DEFAULT_METHODS = ["true_rt", "persistence", "ewma", "dcc", "dmfm", "factorized_uut"]
DEFAULT_LAGS = [1, 3, 6, 9]
DEFAULT_SPLITS = ["val", "test"]

DEFAULT_EWMA_ALPHA = 0.30
DEFAULT_DCC_LAMBDA = 0.94
DEFAULT_DCC_DECAY = 0.97
DEFAULT_DMFM_FACTORS = 12
DEFAULT_UUT_RANK = 12
DEFAULT_UUT_RIDGE = 1e-2
DEFAULT_STABILITY_TARGET = 0.98

SEED = 42


# ============================================================
# Basic helpers
# ============================================================

def now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def print_stage(title: str) -> None:
    print("\n" + "=" * 96)
    print(f"{now_str()} | {title}")
    print("=" * 96)


class NumpyJsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def save_json(obj: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def parse_str_list(s: str) -> List[str]:
    return [x.strip() for x in str(s).split(",") if x.strip()]


def find_project_root() -> Path:
    cwd = Path.cwd().resolve()
    for p in [cwd, *cwd.parents, Path("/kaggle/working/UTraffic-ML")]:
        if (p / "ml_core").exists():
            return p
        if p.name == "UTraffic-ML":
            return p
    return cwd


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def sym_clip_diag(R: np.ndarray, diag_value: float = 1.0) -> np.ndarray:
    R = np.asarray(R, dtype=np.float32)
    R = 0.5 * (R + R.T)
    np.nan_to_num(R, copy=False, nan=0.0, posinf=1.0, neginf=-1.0)
    np.clip(R, -1.0, 1.0, out=R)
    np.fill_diagonal(R, diag_value)
    return R.astype(np.float32, copy=False)


def upper_values(R: np.ndarray) -> np.ndarray:
    iu = np.triu_indices(R.shape[0], k=1)
    return np.asarray(R, dtype=np.float32)[iu]


def sample_offdiag_pairs(n: int, n_pairs: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    if n <= 1:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    i = rng.integers(0, n, size=n_pairs, dtype=np.int64)
    j = rng.integers(0, n - 1, size=n_pairs, dtype=np.int64)
    j = j + (j >= i)
    return i.astype(np.int64), j.astype(np.int64)


def matrix_sample_values(R: np.ndarray, pair_i: np.ndarray, pair_j: np.ndarray) -> np.ndarray:
    return np.asarray(R, dtype=np.float32)[pair_i, pair_j].astype(np.float32, copy=False)


def fit_standardizer(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=np.float32)
    mu = X.mean(axis=0, keepdims=True)
    Xc = X - mu
    std = Xc.std(axis=0, ddof=1, keepdims=True)
    std = np.where(np.isfinite(std) & (std > 1e-6), std, 1.0).astype(np.float32)
    return mu.astype(np.float32), std


def standardize(X: np.ndarray, mu: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((np.asarray(X, dtype=np.float32) - mu) / std).astype(np.float32)


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


# ============================================================
# Data loading
# ============================================================

def load_rt_split(common_dir: Path, split_name: str, mmap_mode: str = "r") -> Dict[str, Any]:
    split_dir = common_dir / split_name
    required = [
        split_dir / "R_series.npy",
        split_dir / "z.npy",
        split_dir / "segment_ids.npy",
        split_dir / "timestamps.npy",
        split_dir / "R_series_meta.csv",
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing Branch A common files in {split_dir}. Missing:\n"
            + "\n".join(str(p) for p in missing)
            + "\nRun 00_prepare_branchA_common_from_osm.py first."
        )

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


def subset_split(data: Dict[str, Any], node_idx: Optional[np.ndarray]) -> Dict[str, Any]:
    if node_idx is None:
        return data
    node_idx = np.asarray(node_idx, dtype=np.int64)

    # R_series stays as memmap; we subset per access in get_R().
    out = dict(data)
    out["node_idx"] = node_idx
    out["z"] = np.asarray(data["z"][:, node_idx], dtype=np.float32)
    out["segment_ids"] = np.asarray(data["segment_ids"])[node_idx].astype(np.int64)
    return out


def get_R(data: Dict[str, Any], t: int) -> np.ndarray:
    R_full = data["R_series"][int(t)]
    node_idx = data.get("node_idx", None)
    if node_idx is None:
        return np.asarray(R_full, dtype=np.float32)
    return np.asarray(R_full[np.ix_(node_idx, node_idx)], dtype=np.float32)


def load_edge_metadata(project_root: Path, segment_ids: np.ndarray, node_idx: Optional[np.ndarray] = None) -> Optional[pd.DataFrame]:
    candidates = [
        project_root / "ml_core" / "src" / "data_processing" / "outputs" / "branchA" / "match_summary" / "matched_osm_edge_metadata.csv",
        project_root / "ml_core" / "src" / "data_processing" / "outputs" / "branchA" / "osm_edge_forecasting_dataset" / "tables" / "node_quality.csv",
        project_root / "ml_core" / "src" / "data_processing" / "outputs" / "branchA" / "osm_edge_forecasting_dataset" / "tables" / "segment_to_model_node_mapping.csv",
    ]

    path = None
    for p in candidates:
        if p.exists():
            path = p
            break
    if path is None:
        print("[WARN] Cannot find OSM edge metadata. Map/topology analysis will be skipped.")
        return None

    print("Using edge metadata:", path)
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

    needed = ["u_lat", "u_lon", "v_lat", "v_lon", "mid_lat", "mid_lon", "osm_u_id", "osm_v_id"]
    for c in needed:
        if c not in out.columns:
            print(f"[WARN] edge metadata missing column {c}. Some map/topology functions may be skipped.")

    return out


# ============================================================
# Evaluation pairs
# ============================================================

def iter_eval_pairs(meta: pd.DataFrame, lag: int):
    T = len(meta)
    sess = meta["session_id"].to_numpy() if "session_id" in meta.columns else None
    for origin_idx in range(T - lag):
        target_idx = origin_idx + lag
        if sess is not None and sess[origin_idx] != sess[target_idx]:
            continue
        yield origin_idx, target_idx


def sample_eval_pairs(meta: pd.DataFrame, lag: int, max_samples: int, rng: np.random.Generator) -> List[Tuple[int, int]]:
    pairs = list(iter_eval_pairs(meta, lag))
    if len(pairs) <= max_samples:
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


# ============================================================
# R predictors
# ============================================================

def compute_unconditional_corr(z: np.ndarray) -> np.ndarray:
    X = np.asarray(z, dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X = X - X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, ddof=1, keepdims=True)
    std = np.where(np.isfinite(std) & (std > 1e-8), std, 1.0)
    Z = X / std
    denom = max(Z.shape[0] - 1, 1)
    R = (Z.T @ Z) / float(denom)
    return sym_clip_diag(R)


def ewma_cov_corr(X: np.ndarray, lam: float = DEFAULT_DCC_LAMBDA) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    if len(X) < 2:
        return np.eye(X.shape[1], dtype=np.float32)

    X = X - X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, ddof=1, keepdims=True)
    std = np.where(np.isfinite(std) & (std > 1e-8), std, 1.0)
    X = X / std

    S = np.eye(X.shape[1], dtype=np.float64)
    for x in X:
        S = lam * S + (1.0 - lam) * np.outer(x, x)

    d = np.sqrt(np.clip(np.diag(S), 1e-8, None))
    R = S / np.outer(d, d)
    return sym_clip_diag(R)


def fit_dmfm_model(train_data: Dict[str, Any], max_factors: int, train_samples: int, rng: np.random.Generator) -> Dict[str, Any]:
    R_series = train_data["R_series"]
    T = int(R_series.shape[0])
    N = len(train_data["segment_ids"])

    if train_samples and train_samples > 0 and train_samples < T:
        idx = np.linspace(0, T - 1, train_samples).round().astype(int)
        idx = np.unique(idx)
    else:
        idx = np.arange(T, dtype=int)

    iu = np.triu_indices(N, k=1)
    P = len(iu[0])
    print(f"[DMFM] fitting sampled PCA: T_fit={len(idx)}, N={N}, P_upper={P:,}, k={max_factors}")

    X = np.empty((len(idx), P), dtype=np.float32)
    for row, t in enumerate(idx):
        R = get_R(train_data, int(t))
        X[row] = R[iu]
        if (row + 1) % 10 == 0 or row + 1 == len(idx):
            print(f"  [DMFM] loaded train R {row+1}/{len(idx)}")

    mean_vec = X.mean(axis=0, keepdims=True).astype(np.float32)
    Xc = X - mean_vec

    k = int(min(max_factors, max(1, Xc.shape[0] - 1), Xc.shape[1]))

    if randomized_svd is not None:
        print("[DMFM] randomized_svd ...")
        _, _, vt = randomized_svd(Xc, n_components=k, random_state=42)
    else:
        print("[DMFM] np.linalg.svd ...")
        _, _, vt = np.linalg.svd(Xc, full_matrices=False)
        vt = vt[:k]

    comp = vt.T.astype(np.float32)
    fac = (Xc @ comp).astype(np.float32)

    if len(fac) < 2:
        A = np.eye(k, dtype=np.float32)
    else:
        X_prev, X_next = fac[:-1], fac[1:]
        ridge = 1e-3 * np.eye(k, dtype=np.float32)
        A = np.linalg.solve(X_prev.T @ X_prev + ridge, X_prev.T @ X_next).astype(np.float32)

    A_pows = {
        h: np.linalg.matrix_power(A, int(h)).astype(np.float32)
        for h in range(1, 10)
    }

    return {
        "n": N,
        "iu": iu,
        "mean_vec": mean_vec,
        "components": comp,
        "A": A,
        "A_pows": A_pows,
    }


def predict_dmfm(model: Dict[str, Any], R_origin: np.ndarray, lag: int) -> np.ndarray:
    iu = model["iu"]
    vec = np.asarray(R_origin, dtype=np.float32)[iu][None, :]
    score = (vec - model["mean_vec"]) @ model["components"]
    A_pow = model["A_pows"].get(int(lag), np.linalg.matrix_power(model["A"], int(lag)))
    pred_score = score @ A_pow
    pred_vec = model["mean_vec"] + pred_score @ model["components"].T

    N = model["n"]
    R = np.eye(N, dtype=np.float32)
    R[iu] = pred_vec.ravel().astype(np.float32)
    R[(iu[1], iu[0])] = pred_vec.ravel().astype(np.float32)
    return sym_clip_diag(R)


def spectral_radius(M: np.ndarray) -> float:
    vals = np.linalg.eigvals(np.asarray(M, dtype=np.float64))
    return float(np.max(np.abs(vals))) if vals.size else 0.0


def stabilize_transition(Phi: np.ndarray, target: float = DEFAULT_STABILITY_TARGET) -> np.ndarray:
    Phi = np.asarray(Phi, dtype=np.float32)
    rho = spectral_radius(Phi)
    if not np.isfinite(rho) or rho <= 0 or rho < target:
        return Phi
    return (Phi * (target / max(rho, 1e-8))).astype(np.float32)


def compute_mean_R_stream(train_data: Dict[str, Any]) -> np.ndarray:
    T = int(train_data["R_series"].shape[0])
    N = len(train_data["segment_ids"])
    acc = np.zeros((N, N), dtype=np.float64)
    for t in range(T):
        acc += get_R(train_data, t)
        if (t + 1) % 20 == 0 or t + 1 == T:
            print(f"  [UUT] mean_R processed {t+1}/{T}")
    R = (acc / max(T, 1)).astype(np.float32)
    return sym_clip_diag(R)


def fit_factorized_uut_model(train_data: Dict[str, Any], rank: int, ridge: float) -> Dict[str, Any]:
    print("[UUT] fitting mean_R ...")
    mean_R = compute_mean_R_stream(train_data)

    print("[UUT] eigendecomposition of mean_R ...")
    vals, vecs = np.linalg.eigh(0.5 * (mean_R.astype(np.float64) + mean_R.T.astype(np.float64)))
    idx = np.argsort(vals)[::-1]
    r = int(min(rank, mean_R.shape[0]))
    U = vecs[:, idx[:r]].astype(np.float32)

    T = int(train_data["R_series"].shape[0])
    Z = np.empty((T, r * r), dtype=np.float32)

    for t in range(T):
        R = get_R(train_data, t)
        X = U.T @ (R - mean_R) @ U
        Z[t] = X.reshape(-1, order="F")
        if (t + 1) % 30 == 0 or t + 1 == T:
            print(f"  [UUT] compressed train R {t+1}/{T}")

    mu, std = fit_standardizer(Z)
    Zs = standardize(Z, mu, std)

    if len(Zs) < 2:
        c = np.zeros(r * r, dtype=np.float32)
        Phi = np.eye(r * r, dtype=np.float32)
    else:
        X_prev, X_next = Zs[:-1], Zs[1:]
        X_design = np.concatenate([np.ones((len(X_prev), 1), dtype=np.float32), X_prev], axis=1)
        reg = ridge * np.eye(X_design.shape[1], dtype=np.float32)
        reg[0, 0] = 0.0
        B = np.linalg.solve(X_design.T @ X_design + reg, X_design.T @ X_next)
        c = B[0].astype(np.float32)
        Phi = B[1:].T.astype(np.float32)
        Phi = stabilize_transition(Phi)

    return {"mean_R": mean_R, "U": U, "mu": mu, "std": std, "c": c, "Phi": Phi, "rank": r}


def predict_factorized_uut(model: Dict[str, Any], R_origin: np.ndarray, lag: int) -> np.ndarray:
    mean_R = model["mean_R"]
    U = model["U"]
    r = model["rank"]

    X0 = U.T @ (np.asarray(R_origin, dtype=np.float32) - mean_R) @ U
    z = X0.reshape(-1, order="F")[None, :]
    zs = standardize(z, model["mu"], model["std"]).reshape(-1)

    for _ in range(int(lag)):
        zs = model["c"] + model["Phi"] @ zs

    zh = (zs[None, :] * model["std"] + model["mu"]).reshape(-1)
    Xh = zh.reshape(r, r, order="F").astype(np.float32)

    R = mean_R + U @ Xh @ U.T
    return sym_clip_diag(R)


def fit_predictor_models(methods: List[str], train_data: Dict[str, Any], args, rng) -> Dict[str, Any]:
    models = {}
    if "dmfm" in methods:
        models["dmfm"] = fit_dmfm_model(
            train_data,
            max_factors=args.dmfm_factors,
            train_samples=args.dmfm_train_samples,
            rng=rng,
        )

    if "factorized_uut" in methods:
        models["factorized_uut"] = fit_factorized_uut_model(
            train_data,
            rank=args.uut_rank,
            ridge=args.uut_ridge,
        )

    if "dcc" in methods:
        print("[DCC] computing unconditional train correlation ...")
        models["dcc_unc"] = compute_unconditional_corr(np.asarray(train_data["z"], dtype=np.float32))

    return models


def predict_R(method: str, train_data: Dict[str, Any], split_data: Dict[str, Any],
              origin_idx: int, target_idx: int, lag: int, models: Dict[str, Any], args) -> np.ndarray:
    if method == "true_rt":
        return get_R(split_data, target_idx)

    if method == "persistence":
        return get_R(split_data, origin_idx)

    if method == "ewma":
        meta = split_data["meta"]
        idx = history_indices_for_origin(meta, origin_idx)
        hist = np.stack([get_R(split_data, int(t)) for t in idx], axis=0).astype(np.float32)
        weights = np.array([(1.0 - args.ewma_alpha) ** (len(hist) - 1 - k) for k in range(len(hist))], dtype=np.float64)
        weights = weights / max(weights.sum(), 1e-12)
        R = np.tensordot(weights, hist, axes=(0, 0)).astype(np.float32)
        return sym_clip_diag(R)

    if method == "dcc":
        meta = split_data["meta"]
        idx = history_indices_for_origin(meta, origin_idx)
        hist = np.asarray(split_data["z"][idx], dtype=np.float32)
        if len(hist) < WINDOW:
            base = get_R(split_data, origin_idx)
        else:
            base = ewma_cov_corr(hist, lam=args.dcc_lambda)
        unc = models.get("dcc_unc")
        if unc is None:
            unc = compute_unconditional_corr(np.asarray(train_data["z"], dtype=np.float32))
        decay = args.dcc_decay ** max(int(lag) - 1, 0)
        return sym_clip_diag(decay * base + (1.0 - decay) * unc)

    if method == "dmfm":
        return predict_dmfm(models["dmfm"], get_R(split_data, origin_idx), lag)

    if method == "factorized_uut":
        return predict_factorized_uut(models["factorized_uut"], get_R(split_data, origin_idx), lag)

    raise ValueError(f"Unsupported method: {method}")


# ============================================================
# Metrics
# ============================================================

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
        "true_mean": float(np.mean(true_vals)),
        "pred_mean": float(np.mean(pred_vals)),
        "true_std": float(np.std(true_vals)),
        "pred_std": float(np.std(pred_vals)),
        "mean_abs_diff": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff ** 2))),
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
        qt = np.linspace(0.01, 0.99, 99)
        out["wasserstein"] = float(np.mean(np.abs(np.quantile(true_vals, qt) - np.quantile(pred_vals, qt))))

    if ks_2samp is not None:
        out["ks_stat"] = float(ks_2samp(true_vals, pred_vals).statistic)
    else:
        out["ks_stat"] = np.nan

    if np.std(true_vals) > 1e-8 and np.std(pred_vals) > 1e-8:
        out["pearson"] = float(np.corrcoef(true_vals, pred_vals)[0, 1])
    else:
        out["pearson"] = np.nan

    if spearmanr is not None and len(true_vals) > 3:
        out["spearman"] = float(spearmanr(true_vals, pred_vals).correlation)
    else:
        out["spearman"] = np.nan

    return out


def topk_indices_abs_upper(R: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    N = R.shape[0]
    iu = np.triu_indices(N, k=1)
    vals = np.abs(R[iu])
    K = min(int(k), len(vals))
    if K <= 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    idx = np.argpartition(vals, -K)[-K:]
    idx = idx[np.argsort(vals[idx])[::-1]]
    return iu[0][idx].astype(np.int64), iu[1][idx].astype(np.int64)


def topk_overlap_metrics(R_true: np.ndarray, R_pred: np.ndarray, k_values: List[int]) -> List[Dict[str, Any]]:
    rows = []
    for k in k_values:
        ti, tj = topk_indices_abs_upper(R_true, k)
        pi, pj = topk_indices_abs_upper(R_pred, k)
        true_set = set(zip(ti.tolist(), tj.tolist()))
        pred_set = set(zip(pi.tolist(), pj.tolist()))
        inter = len(true_set & pred_set)
        denom = max(1, min(len(true_set), len(pred_set)))
        rows.append({
            "topk": int(k),
            "true_topk_size": int(len(true_set)),
            "pred_topk_size": int(len(pred_set)),
            "overlap": int(inter),
            "precision_at_k": float(inter / max(1, len(pred_set))),
            "recall_at_k": float(inter / max(1, len(true_set))),
            "overlap_ratio": float(inter / denom),
        })
    return rows


# ============================================================
# Topology
# ============================================================

def build_edge_line_graph_neighbors(edge_meta: Optional[pd.DataFrame], n: int) -> Optional[List[List[int]]]:
    if edge_meta is None:
        return None
    required = {"osm_u_id", "osm_v_id", "position"}
    if not required.issubset(edge_meta.columns):
        print("[WARN] Missing OSM endpoint columns; skip topology hop analysis.")
        return None

    endpoint_to_edges = defaultdict(list)
    for _, row in edge_meta.iterrows():
        pos = int(row["position"])
        endpoint_to_edges[int(row["osm_u_id"])].append(pos)
        endpoint_to_edges[int(row["osm_v_id"])].append(pos)

    neigh_sets = [set() for _ in range(n)]
    for _, positions in endpoint_to_edges.items():
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
            if d is None:
                out[idx] = f">{max_hop}/unreachable"
            else:
                out[idx] = f"{d}-hop"
    return out


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


# ============================================================
# Plotting
# ============================================================

def plot_distribution_overlay(values_df: pd.DataFrame, out_dir: Path, split: str, lag: int, method: str, max_points: int = 200000):
    sub = values_df[(values_df["split"] == split) & (values_df["lag"] == lag) & (values_df["method"] == method)]
    if sub.empty:
        return

    if len(sub) > max_points:
        sub = sub.sample(max_points, random_state=42)

    plt.figure(figsize=(9, 5))
    plt.hist(sub["true_value"], bins=80, alpha=0.55, density=True, label="R_true")
    plt.hist(sub["pred_value"], bins=80, alpha=0.55, density=True, label=f"R_pred {method}")
    plt.title(f"Distribution: R_true vs R_pred | {method} | {split} | lag={lag}")
    plt.xlabel("Correlation value")
    plt.ylabel("Density")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    path = out_dir / f"dist_overlay_{method}_{split}_lag{lag}.png"
    plt.savefig(path, dpi=170)
    plt.close()


def plot_metric_by_lag(summary_df: pd.DataFrame, out_dir: Path, metric: str, split: str):
    sub = summary_df[summary_df["split"] == split].copy()
    if sub.empty or metric not in sub.columns:
        return

    plt.figure(figsize=(10, 5))
    for method, g in sub.groupby("method"):
        g = g.sort_values("lag")
        plt.plot(g["lag"], g[metric], marker="o", linewidth=2, label=method)
    plt.title(f"Distribution metric: {metric} by lag | {split}")
    plt.xlabel("Lag")
    plt.ylabel(metric)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    path = out_dir / f"dist_metric_{metric}_{split}.png"
    plt.savefig(path, dpi=170)
    plt.close()


def plot_topk_overlap(topk_df: pd.DataFrame, out_dir: Path, split: str, topk: int):
    sub = topk_df[(topk_df["split"] == split) & (topk_df["topk"] == topk)].copy()
    if sub.empty:
        return
    plt.figure(figsize=(10, 5))
    for method, g in sub.groupby("method"):
        g = g.sort_values("lag")
        plt.plot(g["lag"], g["overlap_ratio"], marker="o", linewidth=2, label=method)
    plt.title(f"Top-{topk} strong-correlation overlap by lag | {split}")
    plt.xlabel("Lag")
    plt.ylabel("Overlap ratio")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    path = out_dir / f"topk_overlap_top{topk}_{split}.png"
    plt.savefig(path, dpi=170)
    plt.close()


def plot_topology_metric(topology_df: pd.DataFrame, out_dir: Path, split: str, lag: int, metric: str = "mae"):
    sub = topology_df[(topology_df["split"] == split) & (topology_df["lag"] == lag)].copy()
    if sub.empty:
        return

    bins = ["0-hop", "1-hop", "2-hop", "3-hop", "4-hop", ">4/unreachable", "unknown"]
    sub["hop_bin"] = pd.Categorical(sub["hop_bin"], categories=bins, ordered=True)
    sub = sub.sort_values("hop_bin")

    plt.figure(figsize=(11, 5))
    for method, g in sub.groupby("method", observed=False):
        g = g.sort_values("hop_bin")
        plt.plot(g["hop_bin"].astype(str), g[metric], marker="o", linewidth=2, label=method)
    plt.title(f"Topology OSM: {metric.upper()} by hop distance | {split} | lag={lag}")
    plt.xlabel("OSM line-graph hop distance")
    plt.ylabel(metric.upper())
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    path = out_dir / f"topology_{metric}_{split}_lag{lag}.png"
    plt.savefig(path, dpi=170)
    plt.close()


# ============================================================
# Maps
# ============================================================

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
    coords = [
        [float(row["u_lat"]), float(row["u_lon"])],
        [float(row["v_lat"]), float(row["v_lon"])],
    ]
    folium.PolyLine(
        coords,
        color=color,
        weight=weight,
        opacity=opacity,
        popup=folium.Popup(popup_html, max_width=450),
    ).add_to(m)


def make_source_top30_map(edge_meta: Optional[pd.DataFrame], true_mean: np.ndarray,
                          pred_means: Dict[str, np.ndarray], out_path: Path,
                          n_sources: int = 5, top_targets: int = 30):
    if folium is None:
        print("[WARN] folium not installed; skip source map.")
        return
    if edge_meta is None:
        print("[WARN] no edge metadata; skip source map.")
        return
    required = {"position", "u_lat", "u_lon", "v_lat", "v_lon", "mid_lat", "mid_lon"}
    if not required.issubset(edge_meta.columns):
        print("[WARN] edge metadata lacks coordinate columns; skip source map.")
        return

    meta = edge_meta.sort_values("position").reset_index(drop=True)
    center = [float(meta["mid_lat"].mean()), float(meta["mid_lon"].mean())]
    fmap = folium.Map(location=center, zoom_start=13, tiles="OpenStreetMap")

    N = true_mean.shape[0]
    abs_true = np.abs(true_mean.copy())
    np.fill_diagonal(abs_true, 0.0)

    source_score = abs_true.mean(axis=1)
    source_idx = np.argsort(source_score)[::-1][:n_sources]

    for source_rank, src in enumerate(source_idx, start=1):
        fg = folium.FeatureGroup(name=f"Source {source_rank}: node {int(meta.loc[src, 'model_node_id'])}", show=(source_rank == 1))

        src_row = meta.loc[src]
        add_edge_polyline(
            fg,
            src_row,
            color="black",
            weight=8,
            opacity=0.95,
            popup_html=edge_popup(src_row, extra=f"<b>Source rank:</b> {source_rank}<br><b>mean |corr|:</b> {source_score[src]:.4f}<br>"),
        )

        scores = abs_true[src].copy()
        scores[src] = -np.inf
        top_idx = np.argsort(scores)[::-1][:top_targets]

        for rank, tgt in enumerate(top_idx, start=1):
            row = meta.loc[tgt]
            true_corr = float(true_mean[src, tgt])
            extra = f"<b>source_model_node_id:</b> {int(meta.loc[src, 'model_node_id'])}<br>"
            extra += f"<b>rank:</b> {rank}<br>"
            extra += f"<b>R_true_mean:</b> {true_corr:.5f}<br>"
            for method, P in pred_means.items():
                pred = float(P[src, tgt])
                extra += f"<b>{method} pred:</b> {pred:.5f} | <b>abs_err:</b> {abs(pred - true_corr):.5f}<br>"

            add_edge_polyline(
                fg,
                row,
                color=rank_color(rank),
                weight=5 if rank <= 10 else 4,
                opacity=0.85,
                popup_html=edge_popup(row, extra=extra),
            )

            # line between midpoints for visual relation
            folium.PolyLine(
                [[float(src_row["mid_lat"]), float(src_row["mid_lon"])],
                 [float(row["mid_lat"]), float(row["mid_lon"])]],
                color=rank_color(rank),
                weight=1,
                opacity=0.25,
            ).add_to(fg)

        fg.add_to(fmap)

    folium.LayerControl(collapsed=False).add_to(fmap)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fmap.save(str(out_path))
    print("Saved map:", out_path)


def make_error_by_edge_map(edge_meta: Optional[pd.DataFrame], edge_error: np.ndarray, out_path: Path):
    if folium is None:
        print("[WARN] folium not installed; skip error map.")
        return
    if edge_meta is None:
        print("[WARN] no edge metadata; skip error map.")
        return
    required = {"position", "u_lat", "u_lon", "v_lat", "v_lon", "mid_lat", "mid_lon"}
    if not required.issubset(edge_meta.columns):
        print("[WARN] edge metadata lacks coordinate columns; skip error map.")
        return

    meta = edge_meta.sort_values("position").reset_index(drop=True).copy()
    meta["dmfm_edge_mean_abs_error"] = edge_error[:len(meta)]

    q80 = float(np.nanquantile(edge_error, 0.80))
    q50 = float(np.nanquantile(edge_error, 0.50))

    def color_err(x):
        if x >= q80:
            return "red"
        if x >= q50:
            return "orange"
        return "green"

    center = [float(meta["mid_lat"].mean()), float(meta["mid_lon"].mean())]
    fmap = folium.Map(location=center, zoom_start=13, tiles="OpenStreetMap")

    for _, row in meta.iterrows():
        err = float(row["dmfm_edge_mean_abs_error"])
        extra = f"<b>DMFM mean abs error:</b> {err:.6f}<br>"
        add_edge_polyline(
            fmap,
            row,
            color=color_err(err),
            weight=4 if err >= q80 else 3,
            opacity=0.82,
            popup_html=edge_popup(row, extra=extra),
        )

    folium.LayerControl(collapsed=False).add_to(fmap)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fmap.save(str(out_path))
    print("Saved map:", out_path)


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--common-dir", type=str, default=None)
    ap.add_argument("--output-dir", type=str, default=None)
    ap.add_argument("--methods", type=parse_str_list, default=DEFAULT_METHODS)
    ap.add_argument("--splits", type=parse_str_list, default=DEFAULT_SPLITS)
    ap.add_argument("--lags", type=parse_int_list, default=DEFAULT_LAGS)

    ap.add_argument("--samples-per-split-lag", type=int, default=6)
    ap.add_argument("--pair-samples", type=int, default=80000)
    ap.add_argument("--topk-values", type=parse_int_list, default=[100, 500, 1000])

    ap.add_argument("--max-nodes", type=int, default=0, help="0 = use all nodes. Use 1000/1500 for quick analysis.")
    ap.add_argument("--seed", type=int, default=SEED)

    ap.add_argument("--dmfm-factors", type=int, default=DEFAULT_DMFM_FACTORS)
    ap.add_argument("--dmfm-train-samples", type=int, default=80, help="0 = all train samples; default uses 80 for memory safety.")
    ap.add_argument("--uut-rank", type=int, default=DEFAULT_UUT_RANK)
    ap.add_argument("--uut-ridge", type=float, default=DEFAULT_UUT_RIDGE)
    ap.add_argument("--ewma-alpha", type=float, default=DEFAULT_EWMA_ALPHA)
    ap.add_argument("--dcc-lambda", type=float, default=DEFAULT_DCC_LAMBDA)
    ap.add_argument("--dcc-decay", type=float, default=DEFAULT_DCC_DECAY)

    ap.add_argument("--max-hop", type=int, default=4)
    ap.add_argument("--map-split", type=str, default="test")
    ap.add_argument("--map-lag", type=int, default=1)
    ap.add_argument("--map-snapshots", type=int, default=5)
    ap.add_argument("--map-sources", type=int, default=5)
    ap.add_argument("--map-top-targets", type=int, default=30)

    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    project_root = find_project_root()
    branchA_root = project_root / "ml_core" / "src" / "models" / "ML_BranchA"
    common_dir = Path(args.common_dir).resolve() if args.common_dir else branchA_root / "data" / "05_branchA_prepare_segment_segment_rt"
    out_dir = Path(args.output_dir).resolve() if args.output_dir else branchA_root / "results" / "08_distribution_topology_map_analysis"

    tables_dir = ensure_dir(out_dir / "tables")
    plots_dir = ensure_dir(out_dir / "plots")
    maps_dir = ensure_dir(out_dir / "maps")

    print_stage("BRANCH A — DISTRIBUTION / STRONG STRUCTURE / TOPOLOGY / OSM MAP ANALYSIS")
    print("PROJECT_ROOT:", project_root)
    print("COMMON_DIR  :", common_dir)
    print("OUT_DIR     :", out_dir)
    print("methods     :", args.methods)
    print("splits      :", args.splits)
    print("lags        :", args.lags)
    print("max_nodes   :", args.max_nodes if args.max_nodes else "all")

    print_stage("LOAD BRANCH A COMMON DATA")
    train = load_rt_split(common_dir, "train")
    all_data = {"train": train}
    for split in set(args.splits + [args.map_split]):
        if split != "train":
            all_data[split] = load_rt_split(common_dir, split)

    # Optional node subset for faster analysis.
    node_idx = None
    n_full = len(train["segment_ids"])
    if args.max_nodes and args.max_nodes > 0 and args.max_nodes < n_full:
        node_idx = np.linspace(0, n_full - 1, args.max_nodes).round().astype(np.int64)
        node_idx = np.unique(node_idx)
        print(f"[NODE SUBSET] Using {len(node_idx)}/{n_full} nodes for analysis.")

    train = subset_split(train, node_idx)
    for split in list(all_data.keys()):
        all_data[split] = subset_split(all_data[split], node_idx)
    train = all_data["train"]

    N = len(train["segment_ids"])
    print("N analysis nodes:", N)

    edge_meta = load_edge_metadata(project_root, train["segment_ids"], node_idx=node_idx)

    # Shared pair samples for distribution/topology.
    pair_i, pair_j = sample_offdiag_pairs(N, args.pair_samples, rng)
    neighbors = build_edge_line_graph_neighbors(edge_meta, N)
    hop_bins = compute_pair_hop_bins(neighbors, pair_i, pair_j, max_hop=args.max_hop)
    geo_bins = compute_geo_bins(edge_meta, pair_i, pair_j)

    print_stage("FIT PREDICTOR MODELS")
    models = fit_predictor_models(args.methods, train, args, rng)

    print_stage("DISTRIBUTION + STRONG STRUCTURE + TOPOLOGY EVALUATION")

    dist_rows = []
    topk_rows = []
    topology_rows = []
    sampled_value_blocks = []

    for split in args.splits:
        split_data = all_data[split]
        for lag in args.lags:
            pairs = sample_eval_pairs(split_data["meta"], lag, args.samples_per_split_lag, rng)
            print(f"\n[EVAL] split={split}, lag={lag}, n_snapshots={len(pairs)}")

            for sample_id, (origin_idx, target_idx) in enumerate(pairs):
                R_true = get_R(split_data, target_idx)
                true_vals = matrix_sample_values(R_true, pair_i, pair_j)

                for method in args.methods:
                    t0 = time.time()
                    R_pred = predict_R(method, train, split_data, origin_idx, target_idx, lag, models, args)
                    pred_vals = matrix_sample_values(R_pred, pair_i, pair_j)

                    m = distribution_metrics(true_vals, pred_vals)
                    m.update({
                        "method": method,
                        "split": split,
                        "lag": int(lag),
                        "sample_id": int(sample_id),
                        "origin_idx": int(origin_idx),
                        "target_idx": int(target_idx),
                        "elapsed_sec": float(time.time() - t0),
                    })
                    dist_rows.append(m)

                    # Store sampled values for a subset of snapshots to plot distributions.
                    if sample_id < 2 and method in args.methods:
                        sampled_value_blocks.append(pd.DataFrame({
                            "method": method,
                            "split": split,
                            "lag": int(lag),
                            "sample_id": int(sample_id),
                            "true_value": true_vals.astype(np.float32),
                            "pred_value": pred_vals.astype(np.float32),
                            "hop_bin": hop_bins,
                            "geo_bin": geo_bins,
                        }))

                    # Top-k strong correlation overlap.
                    for row in topk_overlap_metrics(R_true, R_pred, args.topk_values):
                        row.update({
                            "method": method,
                            "split": split,
                            "lag": int(lag),
                            "sample_id": int(sample_id),
                            "origin_idx": int(origin_idx),
                            "target_idx": int(target_idx),
                        })
                        topk_rows.append(row)

                    # Topology error on sampled pairs.
                    abs_diff = np.abs(pred_vals - true_vals)
                    sq_diff = (pred_vals - true_vals) ** 2
                    topo_df = pd.DataFrame({
                        "hop_bin": hop_bins,
                        "geo_bin": geo_bins,
                        "abs_diff": abs_diff,
                        "sq_diff": sq_diff,
                    })

                    by_hop = topo_df.groupby("hop_bin", as_index=False).agg(
                        n_pairs=("abs_diff", "size"),
                        mae=("abs_diff", "mean"),
                        rmse=("sq_diff", lambda x: float(np.sqrt(np.mean(x)))),
                    )
                    for _, r in by_hop.iterrows():
                        topology_rows.append({
                            "method": method,
                            "split": split,
                            "lag": int(lag),
                            "sample_id": int(sample_id),
                            "group_type": "hop_bin",
                            "hop_bin": r["hop_bin"],
                            "geo_bin": "",
                            "n_pairs": int(r["n_pairs"]),
                            "mae": float(r["mae"]),
                            "rmse": float(r["rmse"]),
                        })

                    by_geo = topo_df.groupby("geo_bin", as_index=False).agg(
                        n_pairs=("abs_diff", "size"),
                        mae=("abs_diff", "mean"),
                        rmse=("sq_diff", lambda x: float(np.sqrt(np.mean(x)))),
                    )
                    for _, r in by_geo.iterrows():
                        topology_rows.append({
                            "method": method,
                            "split": split,
                            "lag": int(lag),
                            "sample_id": int(sample_id),
                            "group_type": "geo_bin",
                            "hop_bin": "",
                            "geo_bin": r["geo_bin"],
                            "n_pairs": int(r["n_pairs"]),
                            "mae": float(r["mae"]),
                            "rmse": float(r["rmse"]),
                        })

                    print(f"  method={method:15s} sample={sample_id+1}/{len(pairs)} mean_abs_diff={m.get('mean_abs_diff', np.nan):.5f}")

    dist_df = pd.DataFrame(dist_rows)
    topk_df = pd.DataFrame(topk_rows)
    topology_df = pd.DataFrame(topology_rows)

    dist_path = tables_dir / "branchA_distribution_metrics_by_snapshot.csv"
    topk_path = tables_dir / "branchA_topk_strong_correlation_overlap.csv"
    topology_path = tables_dir / "branchA_topology_error_by_snapshot.csv"

    dist_df.to_csv(dist_path, index=False)
    topk_df.to_csv(topk_path, index=False)
    topology_df.to_csv(topology_path, index=False)

    print("Saved:", dist_path)
    print("Saved:", topk_path)
    print("Saved:", topology_path)

    dist_summary = (
        dist_df
        .groupby(["method", "split", "lag"], as_index=False)
        .mean(numeric_only=True)
        .sort_values(["split", "lag", "mean_abs_diff"])
    )
    dist_summary_path = tables_dir / "branchA_distribution_metrics_summary.csv"
    dist_summary.to_csv(dist_summary_path, index=False)
    print("Saved:", dist_summary_path)

    topk_summary = (
        topk_df
        .groupby(["method", "split", "lag", "topk"], as_index=False)
        .mean(numeric_only=True)
        .sort_values(["split", "lag", "topk", "overlap_ratio"], ascending=[True, True, True, False])
    )
    topk_summary_path = tables_dir / "branchA_topk_overlap_summary.csv"
    topk_summary.to_csv(topk_summary_path, index=False)
    print("Saved:", topk_summary_path)

    topology_summary = (
        topology_df
        .groupby(["method", "split", "lag", "group_type", "hop_bin", "geo_bin"], as_index=False)
        .mean(numeric_only=True)
        .sort_values(["split", "lag", "group_type", "method"])
    )
    topology_summary_path = tables_dir / "branchA_topology_error_summary.csv"
    topology_summary.to_csv(topology_summary_path, index=False)
    print("Saved:", topology_summary_path)

    if sampled_value_blocks:
        values_df = pd.concat(sampled_value_blocks, ignore_index=True)
        values_path = tables_dir / "branchA_sampled_true_pred_values.csv.gz"
        values_df.to_csv(values_path, index=False, compression="gzip")
        print("Saved:", values_path)

        # Distribution overlay plots for selected combinations.
        for split in args.splits:
            for lag in args.lags:
                for method in args.methods:
                    plot_distribution_overlay(values_df, plots_dir, split, lag, method)
    else:
        values_df = pd.DataFrame()

    # Summary plots.
    for split in args.splits:
        for metric in ["mean_abs_diff", "wasserstein", "ks_stat", "pearson"]:
            plot_metric_by_lag(dist_summary, plots_dir, metric, split)
        for topk in args.topk_values:
            plot_topk_overlap(topk_summary, plots_dir, split, topk)
        for lag in args.lags:
            plot_topology_metric(topology_summary[topology_summary["group_type"] == "hop_bin"], plots_dir, split, lag, metric="mae")
            plot_topology_metric(topology_summary[topology_summary["group_type"] == "hop_bin"], plots_dir, split, lag, metric="rmse")

    print_stage("OSM MAPS")

    # Build map mean matrices using map_split/map_lag.
    if args.map_split not in all_data:
        print(f"[WARN] map_split={args.map_split} not loaded; skip maps.")
    else:
        map_data = all_data[args.map_split]
        map_pairs = sample_eval_pairs(map_data["meta"], args.map_lag, args.map_snapshots, rng)

        if map_pairs:
            R_true_acc = np.zeros((N, N), dtype=np.float64)
            pred_acc: Dict[str, np.ndarray] = {
                method: np.zeros((N, N), dtype=np.float64)
                for method in args.methods
            }
            dmfm_edge_err_acc = np.zeros(N, dtype=np.float64)
            dmfm_edge_count = 0

            for origin_idx, target_idx in map_pairs:
                R_true = get_R(map_data, target_idx)
                R_true_acc += R_true

                for method in args.methods:
                    R_pred = predict_R(method, train, map_data, origin_idx, target_idx, args.map_lag, models, args)
                    pred_acc[method] += R_pred
                    if method == "dmfm":
                        err_by_edge = np.mean(np.abs(R_pred - R_true), axis=1)
                        dmfm_edge_err_acc += err_by_edge
                        dmfm_edge_count += 1

            R_true_mean = (R_true_acc / len(map_pairs)).astype(np.float32)
            pred_means = {
                method: (mat / len(map_pairs)).astype(np.float32)
                for method, mat in pred_acc.items()
            }

            source_map_path = maps_dir / f"branchA_source_top{args.map_top_targets}_correlation_map_{args.map_split}_lag{args.map_lag}.html"
            make_source_top30_map(
                edge_meta=edge_meta,
                true_mean=R_true_mean,
                pred_means=pred_means,
                out_path=source_map_path,
                n_sources=args.map_sources,
                top_targets=args.map_top_targets,
            )

            if dmfm_edge_count > 0:
                edge_error = (dmfm_edge_err_acc / dmfm_edge_count).astype(np.float32)
                error_map_path = maps_dir / f"branchA_dmfm_error_by_osm_edge_map_{args.map_split}_lag{args.map_lag}.html"
                make_error_by_edge_map(edge_meta, edge_error, error_map_path)

                err_df = pd.DataFrame({
                    "position": np.arange(N, dtype=np.int64),
                    "model_node_id": train["segment_ids"],
                    "dmfm_edge_mean_abs_error": edge_error,
                })
                if edge_meta is not None:
                    cols = [c for c in ["position", "osm_edge_id", "street_names", "mid_lat", "mid_lon"] if c in edge_meta.columns]
                    err_df = err_df.merge(edge_meta[cols], on="position", how="left")
                err_df.to_csv(tables_dir / "branchA_dmfm_error_by_edge_for_map.csv", index=False)

    metadata = {
        "project_root": str(project_root),
        "common_dir": str(common_dir),
        "out_dir": str(out_dir),
        "methods": args.methods,
        "splits": args.splits,
        "lags": args.lags,
        "N": int(N),
        "max_nodes": args.max_nodes,
        "samples_per_split_lag": args.samples_per_split_lag,
        "pair_samples": args.pair_samples,
        "dmfm_train_samples": args.dmfm_train_samples,
        "outputs": {
            "distribution_metrics": str(dist_summary_path),
            "topk_overlap": str(topk_summary_path),
            "topology_summary": str(topology_summary_path),
            "plots_dir": str(plots_dir),
            "maps_dir": str(maps_dir),
        },
    }
    save_json(metadata, out_dir / "analysis_metadata.json")

    print_stage("DONE")
    print("Tables:", tables_dir)
    print("Plots :", plots_dir)
    print("Maps  :", maps_dir)


if __name__ == "__main__":
    main()
