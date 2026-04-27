# Auto-converted from: 06_branchA_run_xt_forecast_ewma_standalone(13).ipynb
# Folder target: ml_core/src/models/ML_BranchA/scripts
# Results are saved under: ml_core/src/models/ML_BranchA/results
# Generated for OSM-edge Branch A workflow.

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

WINDOW = 10
HORIZONS = list(range(1, 10))
HORIZON_GROUPS = {}

TARGET_MIN_NODES = 2500
TARGET_MAX_NODES = 3000
DEFAULT_TARGET_NODES = 2800
MIN_STD = 1e-8
DIAG_VALUE = 1.0
DEFAULT_MAX_FACTORS = 12
ANCHOR_COUNT = 256


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



def normalize_scores(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    finite = np.isfinite(x)
    if not finite.any():
        return np.zeros_like(x, dtype=np.float32)
    lo = np.nanpercentile(x[finite], 1)
    hi = np.nanpercentile(x[finite], 99)
    if hi <= lo + 1e-12:
        out = np.zeros_like(x, dtype=np.float32)
        out[finite] = 1.0
        return out
    y = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    y[~finite] = 0.0
    return y.astype(np.float32)


def assign_split_from_times(unique_times: np.ndarray) -> Dict[pd.Timestamp, str]:
    n = len(unique_times)
    n_train = int(round(n * 0.70))
    n_val = int(round(n * 0.15))
    n_test = n - n_train - n_val
    if min(n_train, n_val, n_test) < 1:
        raise RuntimeError(f"Invalid split sizes: train={n_train}, val={n_val}, test={n_test}")

    train_times = set(pd.to_datetime(unique_times[:n_train]).tolist())
    val_times = set(pd.to_datetime(unique_times[n_train:n_train + n_val]).tolist())
    mapping: Dict[pd.Timestamp, str] = {}
    for ts in pd.to_datetime(unique_times):
        if ts in train_times:
            mapping[ts] = "train"
        elif ts in val_times:
            mapping[ts] = "val"
        else:
            mapping[ts] = "test"
    return mapping


def dense_speed_matrix(df: pd.DataFrame, times: np.ndarray, segment_ids: np.ndarray) -> np.ndarray:
    time_to_idx = {pd.Timestamp(t): i for i, t in enumerate(pd.to_datetime(times))}
    seg_to_idx = {int(s): i for i, s in enumerate(segment_ids)}
    T, N = len(times), len(segment_ids)
    out = np.full((T, N), np.nan, dtype=np.float32)

    use = df[["timestamp_local", "segment_id", "speed"]].copy()
    use["t_idx"] = pd.to_datetime(use["timestamp_local"]).map(time_to_idx)
    use["s_idx"] = use["segment_id"].map(seg_to_idx)
    use = use.dropna(subset=["t_idx", "s_idx", "speed"])
    if use.empty:
        return out
    use["t_idx"] = use["t_idx"].astype(np.int64)
    use["s_idx"] = use["s_idx"].astype(np.int64)
    for r in use.itertuples(index=False):
        out[r.t_idx, r.s_idx] = np.float32(r.speed)
    return out


def build_train_baseline(train_df: pd.DataFrame, segment_ids: np.ndarray) -> Tuple[pd.DataFrame, pd.Series]:
    tmp = train_df[["timestamp_local", "segment_id", "speed"]].copy()
    tmp["tod"] = pd.to_datetime(tmp["timestamp_local"]).dt.strftime("%H:%M:%S")
    tod_seg = tmp.pivot_table(index="tod", columns="segment_id", values="speed", aggfunc="mean")
    tod_seg = tod_seg.reindex(columns=segment_ids)
    seg_mean = tmp.groupby("segment_id")["speed"].mean().reindex(segment_ids)
    return tod_seg, seg_mean


def residualize_with_train_baseline(speed: np.ndarray, times: np.ndarray, tod_seg: pd.DataFrame, seg_mean: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    tod_index = pd.to_datetime(times).strftime("%H:%M:%S")
    baseline = tod_seg.reindex(index=tod_index).to_numpy(dtype=np.float32)
    seg_mean_arr = seg_mean.to_numpy(dtype=np.float32)
    if baseline.shape != speed.shape:
        baseline = np.broadcast_to(seg_mean_arr[None, :], speed.shape).astype(np.float32)
    missing_base = ~np.isfinite(baseline)
    if missing_base.any():
        baseline[missing_base] = np.broadcast_to(seg_mean_arr[None, :], speed.shape)[missing_base]
    resid = speed - baseline
    return resid.astype(np.float32), baseline.astype(np.float32)


def fit_standardizer(train_resid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = np.nanmean(train_resid, axis=0)
    mu = np.where(np.isfinite(mu), mu, 0.0).astype(np.float32)
    sigma = np.nanstd(train_resid - mu[None, :], axis=0, ddof=1)
    sigma = np.where(np.isfinite(sigma) & (sigma > MIN_STD), sigma, 1.0).astype(np.float32)
    return mu, sigma


def standardize_resid(resid: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    z = (resid - mu[None, :]) / sigma[None, :]
    return z.astype(np.float32)


def approximate_connectivity_score(z_train: np.ndarray, std_score: np.ndarray, anchor_count: int = ANCHOR_COUNT) -> np.ndarray:
    Z = np.array(z_train, dtype=np.float32, copy=True)
    Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)
    T, N = Z.shape
    if N == 0:
        return np.zeros(0, dtype=np.float32)
    order = np.argsort(-std_score)
    anchors = order[: min(anchor_count, N)]
    Za = Z[:, anchors]
    denom = max(T - 1, 1)
    corr = (Z.T @ Za) / denom
    corr = np.clip(corr, -1.0, 1.0)
    abs_corr = np.abs(corr)
    k = min(8, abs_corr.shape[1])
    if k == 0:
        return np.zeros(N, dtype=np.float32)
    topk = np.partition(abs_corr, kth=max(k - 1, 0), axis=1)[:, -k:]
    return topk.mean(axis=1).astype(np.float32)


def choose_segments(
    train_df: pd.DataFrame,
    all_segment_ids: np.ndarray,
    target_min: int = TARGET_MIN_NODES,
    target_max: int = TARGET_MAX_NODES,
    default_target: int = DEFAULT_TARGET_NODES,
) -> Tuple[np.ndarray, pd.DataFrame]:
    train_times = np.array(sorted(pd.to_datetime(train_df["timestamp_local"]).unique()))
    speed_train = dense_speed_matrix(train_df, train_times, all_segment_ids)
    observed = np.isfinite(speed_train)
    obs_ratio = observed.mean(axis=0).astype(np.float32)
    obs_count = observed.sum(axis=0).astype(np.int32)

    filled = np.where(observed, speed_train, np.nan)
    filled_df = pd.DataFrame(filled, columns=all_segment_ids)
    nunique = filled_df.nunique(dropna=True).to_numpy(dtype=np.float32)
    mode_share = []
    for seg in all_segment_ids:
        vc = filled_df[seg].value_counts(dropna=True, normalize=True)
        mode_share.append(float(vc.iloc[0]) if len(vc) else 1.0)
    mode_share = np.array(mode_share, dtype=np.float32)

    tod_seg, seg_mean = build_train_baseline(train_df, all_segment_ids)
    resid_train, _ = residualize_with_train_baseline(speed_train, train_times, tod_seg, seg_mean)
    resid_std = np.nanstd(resid_train, axis=0, ddof=1)
    resid_std = np.where(np.isfinite(resid_std), resid_std, 0.0).astype(np.float32)
    mu, sigma = fit_standardizer(resid_train)
    z_train = standardize_resid(resid_train, mu, sigma)
    conn_score = approximate_connectivity_score(z_train, resid_std)

    obs_s = normalize_scores(obs_ratio)
    var_s = normalize_scores(resid_std)
    conn_s = normalize_scores(conn_score)
    const_penalty = normalize_scores(mode_share)

    eligible = (obs_ratio >= 0.60) & (mode_share <= 0.995) & (nunique >= 4) & (resid_std > np.nanpercentile(resid_std, 20))
    base_score = 0.35 * obs_s + 0.30 * var_s + 0.30 * conn_s + 0.05 * (1.0 - const_penalty)
    base_score = base_score.astype(np.float32)

    score = base_score.copy()
    if not eligible.any() or int(eligible.sum()) < target_min:
        relaxed = (obs_ratio >= 0.45) & (mode_share <= 0.998) & (nunique >= 3)
        eligible = relaxed
        score = (0.30 * obs_s + 0.35 * var_s + 0.30 * conn_s + 0.05 * (1.0 - const_penalty)).astype(np.float32)

    stats = pd.DataFrame({
        "segment_id": all_segment_ids.astype(np.int64),
        "obs_ratio": obs_ratio,
        "obs_count": obs_count,
        "nunique": nunique,
        "dominant_value_share": mode_share,
        "resid_std": resid_std,
        "connectivity_score": conn_score,
        "score": score,
        "eligible": eligible.astype(np.int8),
    })

    meta_cols = [c for c in ["segment_id", "frc", "streetName", "newSegmentId"] if c in train_df.columns]
    if meta_cols:
        meta_seg = train_df[meta_cols].drop_duplicates(subset=["segment_id"])
        stats = stats.merge(meta_seg, on="segment_id", how="left")

    eligible_df = stats[stats["eligible"] == 1].copy().sort_values(["score", "resid_std", "connectivity_score"], ascending=False)

    if len(eligible_df) >= target_min:
        target_n = min(target_max, max(target_min, default_target))
    else:
        target_n = min(target_max, max(target_min, len(stats)))

    if len(eligible_df) > target_n and "frc" in eligible_df.columns and eligible_df["frc"].notna().any():
        chosen_parts = []
        used = set()
        frc_counts = eligible_df["frc"].value_counts(dropna=True)
        for frc_val, cnt in frc_counts.items():
            quota = max(20, int(round(target_n * cnt / len(eligible_df))))
            sub = eligible_df[eligible_df["frc"] == frc_val].head(quota)
            chosen_parts.append(sub)
            used.update(sub["segment_id"].tolist())
        chosen = pd.concat(chosen_parts, ignore_index=True).drop_duplicates(subset=["segment_id"])
        if len(chosen) < target_n:
            fill = eligible_df[~eligible_df["segment_id"].isin(chosen["segment_id"])].head(target_n - len(chosen))
            chosen = pd.concat([chosen, fill], ignore_index=True)
        chosen = chosen.head(target_n)
    else:
        chosen = eligible_df.head(target_n)
        if len(chosen) < target_min:
            fill = stats[~stats["segment_id"].isin(chosen["segment_id"])].sort_values(["score", "resid_std", "connectivity_score"], ascending=False).head(target_min - len(chosen))
            chosen = pd.concat([chosen, fill], ignore_index=True)

    chosen = chosen.drop_duplicates(subset=["segment_id"]).sort_values(["score", "resid_std", "connectivity_score"], ascending=False)
    chosen_ids = chosen["segment_id"].astype(np.int64).to_numpy()
    return chosen_ids, stats.sort_values(["score", "resid_std", "connectivity_score"], ascending=False).reset_index(drop=True)


def load_z_split(proc_root: Path, split_name: str) -> Dict[str, object]:
    split_dir = proc_root / split_name
    npz_path = split_dir / "traffic_tensor_resid.npz"
    meta_path = split_dir / "traffic_tensor_resid_meta.csv"
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing {npz_path}. Run file 02 first.")
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing {meta_path}. Run file 02 first.")

    obj = np.load(npz_path, allow_pickle=True)
    meta = pd.read_csv(meta_path)
    z = obj["z"].astype(np.float32)
    resid = obj["resid"].astype(np.float32) if "resid" in obj.files else None
    speed = obj["speed"].astype(np.float32) if "speed" in obj.files else None
    segment_ids = obj["segment_ids"].astype(np.int64)

    if "timestamp_local" in obj.files:
        timestamps = pd.to_datetime(obj["timestamp_local"])
    elif "timestamp" in obj.files:
        timestamps = pd.to_datetime(obj["timestamp"])
    elif "timestamp_local" in meta.columns:
        timestamps = pd.to_datetime(meta["timestamp_local"])
    else:
        raise RuntimeError("Cannot find timestamps in npz or meta.")

    if "timestamp_local" not in meta.columns:
        meta["timestamp_local"] = timestamps
    meta["timestamp_local"] = pd.to_datetime(meta["timestamp_local"])
    if "session_id" not in meta.columns:
        meta["session_id"] = meta["timestamp_local"].dt.date.astype(str)

    return {
        "z": z,
        "resid": resid,
        "speed": speed,
        "segment_ids": segment_ids,
        "timestamps": timestamps,
        "meta": meta,
    }


def save_z_split(out_dir: Path, split_name: str, arrays: Dict[str, np.ndarray], meta: pd.DataFrame):
    split_dir = out_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(split_dir / "traffic_tensor_resid.npz", **arrays)
    meta.to_csv(split_dir / "traffic_tensor_resid_meta.csv", index=False)


def count_rt_samples(meta: pd.DataFrame, window: int = WINDOW) -> int:
    total = 0
    if "session_id" not in meta.columns:
        return max(0, len(meta) - window + 1)
    for _, sub in meta.groupby("session_id", sort=False):
        total += max(0, len(sub) - window + 1)
    return int(total)


def _iter_session_index_groups(meta: pd.DataFrame) -> List[np.ndarray]:
    if "session_id" not in meta.columns:
        return [np.arange(len(meta), dtype=np.int64)]
    groups = []
    for _, sub in meta.groupby("session_id", sort=False):
        idx = sub.index.to_numpy(dtype=np.int64)
        if len(idx) > 0:
            groups.append(idx)
    return groups if groups else [np.arange(len(meta), dtype=np.int64)]


def nearest_corr(R: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    A = np.asarray(R, dtype=np.float64)
    A = 0.5 * (A + A.T)
    np.fill_diagonal(A, 1.0)
    vals, vecs = np.linalg.eigh(A)
    vals = np.clip(vals, eps, None)
    A_psd = (vecs * vals) @ vecs.T
    d = np.sqrt(np.clip(np.diag(A_psd), eps, None))
    A_corr = A_psd / np.outer(d, d)
    A_corr = np.clip(A_corr, -1.0, 1.0)
    A_corr = 0.5 * (A_corr + A_corr.T)
    np.fill_diagonal(A_corr, 1.0)
    return A_corr.astype(np.float32)


def build_rt_series_to_disk(out_dir: Path, split_name: str, z: np.ndarray, timestamps: Sequence, meta: pd.DataFrame, segment_ids: np.ndarray, raw_meta: Optional[pd.DataFrame] = None, window: int = WINDOW, dtype: str = "float16") -> Dict[str, object]:
    split_dir = out_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    n_samples = count_rt_samples(meta, window=window)
    N = int(len(segment_ids))
    arr_path = split_dir / "R_series.npy"
    R_mem = np.lib.format.open_memmap(arr_path, mode="w+", dtype=dtype, shape=(n_samples, N, N))

    rows: List[Dict[str, object]] = []
    cursor = 0
    for session_indices in _iter_session_index_groups(meta):
        if len(session_indices) < window:
            continue
        for local_end in range(window - 1, len(session_indices)):
            block_idx = session_indices[local_end - window + 1: local_end + 1]
            global_t = int(session_indices[local_end])
            block = np.nan_to_num(z[block_idx], nan=0.0, posinf=0.0, neginf=0.0)
            R = np.corrcoef(block, rowvar=False).astype(np.float32)
            R = np.nan_to_num(R, nan=0.0, posinf=0.0, neginf=0.0)
            R = nearest_corr(R)
            R_mem[cursor] = R.astype(dtype)

            src = meta.iloc[global_t]
            row = {
                "sample_id": int(cursor),
                "raw_row_idx": global_t,
                "window_end_idx": global_t,
                "session_step": int(local_end),
                "timestamp_local": pd.Timestamp(timestamps[global_t]),
            }
            for c in ["session_id", "date", "time_set_id", "slot_index", "tod_minutes"]:
                if c in meta.columns:
                    row[c] = src[c]
            rows.append(row)
            cursor += 1

    del R_mem
    meta_out = pd.DataFrame(rows)
    np.save(split_dir / "z.npy", z.astype(np.float32))
    np.save(split_dir / "segment_ids.npy", np.asarray(segment_ids, dtype=np.int64))
    np.save(split_dir / "timestamps.npy", np.asarray(pd.to_datetime(timestamps)).astype("datetime64[ns]"))
    meta_out.to_csv(split_dir / "R_series_meta.csv", index=False)
    if raw_meta is not None:
        raw_meta.to_csv(split_dir / "raw_meta.csv", index=False)
    return {
        "n_Rt": int(len(meta_out)),
        "n_segments": int(N),
        "shape": [int(len(meta_out)), int(N), int(N)],
        "path": str(arr_path),
    }


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


def validate_rt_split_meta(meta: pd.DataFrame, n_segments: int, shape: Sequence[int], split_name: str) -> Dict[str, object]:
    diag_ok = True
    sym_ok = True
    return {
        "split": split_name,
        "n_Rt": int(shape[0]),
        "n_segments": int(n_segments),
        "shape": list(map(int, shape)),
        "diag_all_one": diag_ok,
        "symmetric": sym_ok,
        "session_counts": {} if meta.empty or "session_id" not in meta.columns else {str(k): int(v) for k, v in meta["session_id"].value_counts(sort=False).to_dict().items()},
        "min_timestamp": None if meta.empty else str(pd.to_datetime(meta["timestamp_local"]).min()),
        "max_timestamp": None if meta.empty else str(pd.to_datetime(meta["timestamp_local"]).max()),
    }


def summarize_groups(df: pd.DataFrame, metric_cols: List[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=["method", "split", "horizon_group", *metric_cols])


def matrix_metrics(R_true: np.ndarray, R_pred: np.ndarray) -> Dict[str, float]:
    yt = np.asarray(R_true, dtype=np.float32)
    yp = np.asarray(R_pred, dtype=np.float32)
    diff = yp - yt
    mse = float(np.mean(diff ** 2))
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(mse))
    fro = float(np.linalg.norm(diff, ord="fro"))

    mask = ~np.eye(yt.shape[0], dtype=bool)
    off = diff[mask]
    mse_off = float(np.mean(off ** 2))
    mae_off = float(np.mean(np.abs(off)))
    rmse_off = float(np.sqrt(mse_off))
    return {
        "mae_matrix": mae,
        "mse_matrix": mse,
        "rmse_matrix": rmse,
        "fro_error": fro,
        "mae_offdiag": mae_off,
        "mse_offdiag": mse_off,
        "rmse_offdiag": rmse_off,
    }


def aggregate_metric_rows(rows: List[Dict[str, object]]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    detail = pd.DataFrame(rows)
    metric_cols = ["mae_matrix", "mse_matrix", "rmse_matrix", "fro_error", "mae_offdiag", "mse_offdiag", "rmse_offdiag"]
    if not detail.empty:
        per_lag = (
            detail.groupby(["method", "split", "lag"], as_index=False)[metric_cols]
            .mean()
            .sort_values(["method", "split", "lag"])
            .reset_index(drop=True)
        )
    else:
        per_lag = pd.DataFrame(columns=["method", "split", "lag", *metric_cols])

    by_group = pd.DataFrame(columns=["method", "split", "horizon_group", *metric_cols])
    return detail, per_lag, by_group

def evaluate_predictions(records: List[Dict[str, object]], n_segments: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: List[Dict[str, object]] = []
    for rec in records:
        metrics = matrix_metrics(rec["R_true"], rec["R_pred"])
        rows.append({
            "method": rec["method"],
            "split": rec["split"],
            "lag": int(rec["lag"]),
            "origin_idx": int(rec["origin_idx"]),
            "target_idx": int(rec["target_idx"]),
            "origin_timestamp": rec.get("origin_timestamp"),
            "target_timestamp": rec.get("target_timestamp"),
            **metrics,
        })
    return aggregate_metric_rows(rows)


def iter_eval_pairs(meta: pd.DataFrame, lag: int):
    T = len(meta)
    for origin_idx in range(T - lag):
        target_idx = origin_idx + lag
        origin_session = meta.iloc[origin_idx]["session_id"] if "session_id" in meta.columns else None
        target_session = meta.iloc[target_idx]["session_id"] if "session_id" in meta.columns else None
        if origin_session is not None and target_session is not None and origin_session != target_session:
            continue
        yield origin_idx, target_idx


def compute_unconditional_corr(z: np.ndarray) -> np.ndarray:
    zz = np.nan_to_num(np.asarray(z, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    R = np.corrcoef(zz, rowvar=False).astype(np.float32)
    R = np.nan_to_num(R, nan=0.0, posinf=0.0, neginf=0.0)
    return nearest_corr(R)


def ewma_cov(history: np.ndarray, lam: float = 0.94) -> np.ndarray:
    X = np.nan_to_num(np.asarray(history, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    n = X.shape[1]
    S = np.cov(X.T).astype(np.float64)
    if not np.isfinite(S).all():
        S = np.eye(n, dtype=np.float64)
    for x in X:
        S = lam * S + (1.0 - lam) * np.outer(x, x)
    d = np.sqrt(np.clip(np.diag(S), 1e-8, None))
    R = S / np.outer(d, d)
    return nearest_corr(R)


def _history_until_origin(z: np.ndarray, raw_meta: Optional[pd.DataFrame], origin_meta_row: pd.Series) -> np.ndarray:
    raw_row_idx = int(origin_meta_row.get("raw_row_idx", origin_meta_row.get("window_end_idx", 0)))
    if raw_meta is None or "session_id" not in raw_meta.columns or "session_id" not in origin_meta_row.index:
        return np.asarray(z[: raw_row_idx + 1], dtype=np.float32)
    session_id = origin_meta_row["session_id"]
    mask = (raw_meta["session_id"].to_numpy() == session_id) & (np.arange(len(raw_meta)) <= raw_row_idx)
    idx = np.where(mask)[0]
    return np.asarray(z[idx], dtype=np.float32) if len(idx) else np.asarray(z[: raw_row_idx + 1], dtype=np.float32)


def corr_to_vec(R: np.ndarray) -> np.ndarray:
    iu = np.triu_indices(R.shape[0], k=1)
    return np.asarray(R, dtype=np.float32)[iu]


def vec_to_corr(vec: np.ndarray, n_segments: int) -> np.ndarray:
    iu = np.triu_indices(n_segments, k=1)
    R = np.eye(n_segments, dtype=np.float32)
    R[iu] = vec.astype(np.float32)
    R[(iu[1], iu[0])] = vec.astype(np.float32)
    return nearest_corr(R)


def fit_pca_factor_model(R_train: np.ndarray, max_factors: int = DEFAULT_MAX_FACTORS):
    X = np.stack([corr_to_vec(np.asarray(R, dtype=np.float32)) for R in R_train], axis=0)
    mean_vec = X.mean(axis=0, keepdims=True)
    Xc = X - mean_vec
    _, _, vt = np.linalg.svd(Xc, full_matrices=False)
    k = int(min(max_factors, max(1, Xc.shape[0] - 1), Xc.shape[1]))
    comp = vt[:k].T.astype(np.float32)
    fac = (Xc @ comp).astype(np.float32)
    if len(fac) < 2:
        A = np.eye(k, dtype=np.float32)
    else:
        X_prev, X_next = fac[:-1], fac[1:]
        ridge = 1e-3 * np.eye(k, dtype=np.float32)
        A = np.linalg.solve(X_prev.T @ X_prev + ridge, X_prev.T @ X_next).astype(np.float32)
    return {"n_segments": int(R_train.shape[1]), "mean_vec": mean_vec.astype(np.float32), "components": comp, "A": A}


def dmfm_predict(model: Dict[str, np.ndarray], R_origin: np.ndarray, lag: int) -> np.ndarray:
    vec = corr_to_vec(np.asarray(R_origin, dtype=np.float32))[None, :]
    score = (vec - model["mean_vec"]) @ model["components"]
    A_pow = np.linalg.matrix_power(model["A"], lag)
    pred_score = score @ A_pow
    pred_vec = model["mean_vec"] + pred_score @ model["components"].T
    return vec_to_corr(pred_vec.ravel(), model["n_segments"])


def _metric_row(method_name: str, split_name: str, lag: int, origin_idx: int, target_idx: int, origin_ts, target_ts, R_true: np.ndarray, R_pred: np.ndarray) -> Dict[str, object]:
    row = {
        "method": method_name,
        "split": split_name,
        "lag": int(lag),
        "origin_idx": int(origin_idx),
        "target_idx": int(target_idx),
        "origin_timestamp": origin_ts,
        "target_timestamp": target_ts,
    }
    row.update(matrix_metrics(R_true, R_pred))
    return row


def run_persistence(split_name: str, R_series: np.ndarray, meta: pd.DataFrame, lags: List[int]) -> List[Dict[str, object]]:
    rows = []
    for lag in lags:
        for origin_idx, target_idx in iter_eval_pairs(meta, lag):
            rows.append(_metric_row("persistence", split_name, lag, origin_idx, target_idx, meta.iloc[origin_idx]["timestamp_local"], meta.iloc[target_idx]["timestamp_local"], np.asarray(R_series[target_idx], dtype=np.float32), np.asarray(R_series[origin_idx], dtype=np.float32)))
    return rows


def run_ewma(split_name: str, R_series: np.ndarray, meta: pd.DataFrame, lags: List[int], alpha: float = 0.30) -> List[Dict[str, object]]:
    rows = []
    idx_arr = np.arange(len(R_series))
    for lag in lags:
        for origin_idx, target_idx in iter_eval_pairs(meta, lag):
            same_session = np.ones(len(R_series), dtype=bool)
            if "session_id" in meta.columns:
                same_session = meta["session_id"].to_numpy() == meta.iloc[origin_idx]["session_id"]
            hist_idx = idx_arr[same_session & (idx_arr <= origin_idx)]
            hist = np.asarray(R_series[hist_idx], dtype=np.float32)
            weights = np.array([(1.0 - alpha) ** (len(hist) - 1 - k) for k in range(len(hist))], dtype=np.float64)
            weights = weights / max(weights.sum(), 1e-12)
            R_pred = nearest_corr(np.tensordot(weights, hist, axes=(0, 0)).astype(np.float32))
            rows.append(_metric_row("ewma", split_name, lag, origin_idx, target_idx, meta.iloc[origin_idx]["timestamp_local"], meta.iloc[target_idx]["timestamp_local"], np.asarray(R_series[target_idx], dtype=np.float32), R_pred))
    return rows


def run_dcc_like(method_name: str, split_name: str, z: np.ndarray, R_series: np.ndarray, meta: pd.DataFrame, lags: List[int], shrink: float = 0.0, ref_z: Optional[np.ndarray] = None, raw_meta: Optional[pd.DataFrame] = None, lam: float = 0.94) -> List[Dict[str, object]]:
    rows = []
    unc = compute_unconditional_corr(ref_z if ref_z is not None else z)
    for lag in lags:
        for origin_idx, target_idx in iter_eval_pairs(meta, lag):
            origin_row = meta.iloc[origin_idx]
            hist = _history_until_origin(z, raw_meta, origin_row)
            if len(hist) < WINDOW:
                continue
            R_now = ewma_cov(hist, lam=lam)
            if shrink > 0:
                R_now = nearest_corr((1.0 - shrink) * R_now + shrink * unc)
            decay = 0.97 ** max(lag - 1, 0)
            R_pred = nearest_corr(decay * R_now + (1.0 - decay) * unc)
            rows.append(_metric_row(method_name, split_name, lag, origin_idx, target_idx, meta.iloc[origin_idx]["timestamp_local"], meta.iloc[target_idx]["timestamp_local"], np.asarray(R_series[target_idx], dtype=np.float32), R_pred))
    return rows


def run_dmfm(split_name: str, R_train: np.ndarray, R_series: np.ndarray, meta: pd.DataFrame, lags: List[int], max_factors: int = DEFAULT_MAX_FACTORS) -> List[Dict[str, object]]:
    model = fit_pca_factor_model(np.asarray(R_train, dtype=np.float32), max_factors=max_factors)
    rows = []
    for lag in lags:
        for origin_idx, target_idx in iter_eval_pairs(meta, lag):
            R_pred = dmfm_predict(model, np.asarray(R_series[origin_idx], dtype=np.float32), lag)
            rows.append(_metric_row("dmfm", split_name, lag, origin_idx, target_idx, meta.iloc[origin_idx]["timestamp_local"], meta.iloc[target_idx]["timestamp_local"], np.asarray(R_series[target_idx], dtype=np.float32), R_pred))
    return rows


def save_results(out_dir: Path, method_name: str, detail: pd.DataFrame, per_lag: pd.DataFrame, by_group: pd.DataFrame):
    out_dir.mkdir(parents=True, exist_ok=True)
    detail.to_csv(out_dir / f"{method_name}_detail_metrics.csv", index=False)
    per_lag.to_csv(out_dir / f"{method_name}_per_lag_metrics.csv", index=False)
    by_group.to_csv(out_dir / f"{method_name}_by_horizon_group.csv", index=False)




from sklearn.linear_model import MultiTaskElasticNet

FORECAST_HORIZONS = list(range(1, 10))
ALPHA = 0.001
L1_RATIO = 0.5
MAX_ITER = 2000
TOL = 1e-3

def vector_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    yt = np.asarray(y_true, dtype=np.float32)
    yp = np.asarray(y_pred, dtype=np.float32)
    diff = yp - yt
    mse = float(np.mean(diff ** 2))
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(mse))
    return {"mae": mae, "mse": mse, "rmse": rmse}

def _same_session_pairs(meta: pd.DataFrame, lag: int):
    T = len(meta)
    sess = meta["session_id"].to_numpy() if "session_id" in meta.columns else None
    for origin_idx in range(T - lag):
        target_idx = origin_idx + lag
        if sess is not None and sess[origin_idx] != sess[target_idx]:
            continue
        yield origin_idx, target_idx

def _history_indices_for_origin(meta: pd.DataFrame, origin_idx: int) -> np.ndarray:
    if "session_id" not in meta.columns:
        return np.arange(origin_idx + 1, dtype=np.int64)
    sess = meta["session_id"].to_numpy()
    mask = (sess == sess[origin_idx])
    idx = np.where(mask & (np.arange(len(meta)) <= origin_idx))[0]
    return idx.astype(np.int64)

def _predict_R_for_pair(method_name: str, train_data: Dict[str, object], split_data: Dict[str, object], origin_idx: int, target_idx: int, lag: int,
                        alpha_ewma: float = 0.30, lam: float = 0.94, shrink: float = 0.0,
                        dmfm_model: Optional[Dict[str, np.ndarray]] = None) -> np.ndarray:
    R_series = split_data["R_series"]
    meta = split_data["meta"]
    z = np.asarray(split_data["z"], dtype=np.float32)
    if method_name == "true_rt":
        return np.asarray(R_series[target_idx], dtype=np.float32)
    if method_name == "persistence":
        return np.asarray(R_series[origin_idx], dtype=np.float32)
    if method_name == "ewma":
        idx_arr = _history_indices_for_origin(meta, origin_idx)
        hist = np.asarray(R_series[idx_arr], dtype=np.float32)
        weights = np.array([(1.0 - alpha_ewma) ** (len(hist) - 1 - k) for k in range(len(hist))], dtype=np.float64)
        weights = weights / max(weights.sum(), 1e-12)
        return nearest_corr(np.tensordot(weights, hist, axes=(0, 0)).astype(np.float32))
    unc = compute_unconditional_corr(np.asarray(train_data["z"], dtype=np.float32))
    if method_name in {"dcc", "prodcc"}:
        hist_idx = _history_indices_for_origin(meta, origin_idx)
        hist = np.asarray(z[hist_idx], dtype=np.float32)
        if len(hist) < WINDOW:
            base = np.asarray(R_series[origin_idx], dtype=np.float32)
        else:
            base = ewma_cov(hist, lam=lam)
        if method_name == "prodcc":
            base = nearest_corr((1.0 - shrink) * base + shrink * unc)
        decay = 0.97 ** max(lag - 1, 0)
        return nearest_corr(decay * base + (1.0 - decay) * unc)
    if method_name == "dmfm":
        if dmfm_model is None:
            raise RuntimeError("dmfm_model is required for method=dmfm")
        return dmfm_predict(dmfm_model, np.asarray(R_series[origin_idx], dtype=np.float32), lag)
    raise ValueError(f"Unsupported method_name={method_name}")

def _build_dataset_for_horizon(method_name: str, train_data: Dict[str, object], split_data: Dict[str, object], lag: int,
                               use_rt: bool = True, alpha_ewma: float = 0.30, lam: float = 0.94, shrink: float = 0.10,
                               dmfm_model: Optional[Dict[str, np.ndarray]] = None):
    z = np.asarray(split_data["z"], dtype=np.float32)
    meta = split_data["meta"]
    X_rows = []
    Y_rows = []
    keys = []
    for origin_idx, target_idx in _same_session_pairs(meta, lag):
        x_t = np.asarray(z[origin_idx], dtype=np.float32)
        y_t = np.asarray(z[target_idx], dtype=np.float32)
        if use_rt:
            R_used = _predict_R_for_pair(method_name, train_data, split_data, origin_idx, target_idx, lag,
                                         alpha_ewma=alpha_ewma, lam=lam, shrink=shrink, dmfm_model=dmfm_model)
            feat = np.concatenate([x_t, np.asarray(R_used @ x_t, dtype=np.float32)], axis=0)
        else:
            feat = x_t
        X_rows.append(feat.astype(np.float32))
        Y_rows.append(y_t.astype(np.float32))
        keys.append((origin_idx, target_idx))
    if not X_rows:
        return np.zeros((0, z.shape[1] * (2 if use_rt else 1)), dtype=np.float32), np.zeros((0, z.shape[1]), dtype=np.float32), keys
    return np.stack(X_rows, axis=0), np.stack(Y_rows, axis=0), keys

def fit_direct_model(X_train: np.ndarray, Y_train: np.ndarray,
                     alpha: float = ALPHA, l1_ratio: float = L1_RATIO,
                     max_iter: int = MAX_ITER, tol: float = TOL) -> MultiTaskElasticNet:
    model = MultiTaskElasticNet(
        alpha=float(alpha),
        l1_ratio=float(l1_ratio),
        fit_intercept=True,
        max_iter=int(max_iter),
        tol=float(tol),
        selection="cyclic",
        random_state=42,
    )
    model.fit(X_train, Y_train)
    return model

def coef_summary_rows(model: MultiTaskElasticNet, method_label: str, lag: int, n_segments: int, use_rt: bool) -> Dict[str, object]:
    coef = np.asarray(model.coef_, dtype=np.float32)  # shape [n_targets, n_features]
    if use_rt:
        A = coef[:, :n_segments]
        B = coef[:, n_segments:]
    else:
        A = coef[:, :n_segments]
        B = None
    row = {
        "method": method_label,
        "lag": int(lag),
        "n_targets": int(coef.shape[0]),
        "n_features": int(coef.shape[1]),
        "intercept_l2": float(np.linalg.norm(np.asarray(model.intercept_, dtype=np.float32))),
        "A_l1": float(np.abs(A).sum()),
        "A_l2": float(np.linalg.norm(A)),
        "A_nonzero": int(np.count_nonzero(np.abs(A) > 1e-8)),
    }
    if B is not None:
        row.update({
            "B_l1": float(np.abs(B).sum()),
            "B_l2": float(np.linalg.norm(B)),
            "B_nonzero": int(np.count_nonzero(np.abs(B) > 1e-8)),
        })
    else:
        row.update({"B_l1": 0.0, "B_l2": 0.0, "B_nonzero": 0})
    return row

def run_xt_direct_forecast(method_name: str, common_dir: Path, out_dir: Path, use_rt: bool,
                           alpha: float = ALPHA, l1_ratio: float = L1_RATIO,
                           dcc_shrink: float = 0.0):
    out_dir.mkdir(parents=True, exist_ok=True)
    train = load_rt_split(common_dir, "train")
    val = load_rt_split(common_dir, "val")
    test = load_rt_split(common_dir, "test")
    for k in ["segment_ids"]:
        assert np.array_equal(train[k], val[k])
        assert np.array_equal(train[k], test[k])

    n_segments = int(len(train["segment_ids"]))
    dmfm_model = None
    if method_name == "dmfm":
        dmfm_model = fit_pca_factor_model(np.asarray(train["R_series"], dtype=np.float32), max_factors=DEFAULT_MAX_FACTORS)

    detail_rows = []
    coef_rows = []
    pred_dir = out_dir / f"{method_name}_predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    for lag in FORECAST_HORIZONS:
        X_train, Y_train, train_keys = _build_dataset_for_horizon(
            method_name=method_name, train_data=train, split_data=train, lag=lag, use_rt=use_rt,
            shrink=dcc_shrink, dmfm_model=dmfm_model
        )
        if len(X_train) == 0:
            continue
        model = fit_direct_model(X_train, Y_train, alpha=alpha, l1_ratio=l1_ratio)
        coef_rows.append(coef_summary_rows(model, method_name, lag, n_segments, use_rt))
        np.savez_compressed(
            pred_dir / f"{method_name}_lag{lag}_model.npz",
            coef=model.coef_.astype(np.float32),
            intercept=np.asarray(model.intercept_, dtype=np.float32),
        )

        for split_name, data in [("val", val), ("test", test)]:
            X_split, Y_split, keys = _build_dataset_for_horizon(
                method_name=method_name, train_data=train, split_data=data, lag=lag, use_rt=use_rt,
                shrink=dcc_shrink, dmfm_model=dmfm_model
            )
            if len(X_split) == 0:
                continue
            Y_pred = model.predict(X_split).astype(np.float32)
            np.savez_compressed(
                pred_dir / f"{method_name}_{split_name}_lag{lag}_predictions.npz",
                y_true=Y_split.astype(np.float32),
                y_pred=Y_pred.astype(np.float32),
                keys=np.asarray(keys, dtype=np.int64),
            )
            for row_idx, (origin_idx, target_idx) in enumerate(keys):
                m = vector_metrics(Y_split[row_idx], Y_pred[row_idx])
                detail_rows.append({
                    "method": method_name,
                    "split": split_name,
                    "lag": int(lag),
                    "origin_idx": int(origin_idx),
                    "target_idx": int(target_idx),
                    "origin_timestamp": str(pd.Timestamp(data["meta"].iloc[origin_idx]["timestamp_local"])),
                    "target_timestamp": str(pd.Timestamp(data["meta"].iloc[target_idx]["timestamp_local"])),
                    **m,
                })

    detail = pd.DataFrame(detail_rows)
    if not detail.empty:
        per_lag = detail.groupby(["method", "split", "lag"], as_index=False)[["mae", "mse", "rmse"]].mean()
    else:
        per_lag = pd.DataFrame(columns=["method", "split", "lag", "mae", "mse", "rmse"])
    coef_df = pd.DataFrame(coef_rows)

    detail.to_csv(out_dir / f"{method_name}_xt_detail_metrics.csv", index=False)
    per_lag.to_csv(out_dir / f"{method_name}_xt_per_lag_metrics.csv", index=False)
    coef_df.to_csv(out_dir / f"{method_name}_xt_coef_summary.csv", index=False)

    summary = {
        "method": method_name,
        "use_rt": bool(use_rt),
        "alpha": float(alpha),
        "l1_ratio": float(l1_ratio),
        "horizons": FORECAST_HORIZONS,
        "splits": ["val", "test"],
        "n_segments": int(n_segments),
        "out_dir": str(out_dir),
        "pred_dir": str(pred_dir),
    }
    with open(out_dir / f"{method_name}_xt_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[DONE]", method_name)
    print(per_lag)




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
OUT_DIR = BRANCHA_ROOT / "results" / "06_branchA_run_xt_forecast"
METHOD_OUT_DIR = OUT_DIR / "ewma"
METHOD_OUT_DIR.mkdir(parents=True, exist_ok=True)

print("PROJECT_ROOT:", PROJECT_ROOT)
print("COMMON_DIR  :", COMMON_DIR)
print("OUT_DIR     :", METHOD_OUT_DIR)

ensure_branchA_common_data_ready()
run_xt_direct_forecast(
    method_name="ewma",
    common_dir=COMMON_DIR,
    out_dir=METHOD_OUT_DIR,
    use_rt=True,
    alpha=ALPHA,
    l1_ratio=L1_RATIO,
    dcc_shrink=0.0,
)
