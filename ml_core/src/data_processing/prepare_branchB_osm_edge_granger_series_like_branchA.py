"""
Prepare Branch B OSM-edge Granger-Gt TIMELINE series.

Purpose
-------
Create a fair Branch-B graph series G_t without using the old lagged-correlation Rt.

Input format, one folder per split:
    source_dir/{train,val,test}/z.npy
    source_dir/{train,val,test}/segment_ids.npy
    source_dir/{train,val,test}/timestamps.npy
    source_dir/{train,val,test}/G_series_meta.csv

Output format, one folder per split:
    output_dir/{train,val,test}/z.npy
    output_dir/{train,val,test}/segment_ids.npy
    output_dir/{train,val,test}/timestamps.npy
    output_dir/{train,val,test}/G_series_meta.csv
    output_dir/{train,val,test}/G_weight_series.npy       # [T, N, N]
    output_dir/{train,val,test}/G_best_lag_series.npy     # [T, N, N]

Key design choices
------------------
1) No old Rt / lagged correlation is used.
2) G_t is estimated by Granger-style predictive influence using TRAIN split only.
3) Day split is preserved and validated: no date may appear in more than one split.
4) For granger_p=p, a row at time t is valid only if it has p previous steps in the
   same day. Example p=3 and 15-minute data: graph at 10:00 uses 09:45, 09:30, 09:15.
   The history rows themselves are not used as prediction origins until enough
   previous history exists.
5) For stability, a separate graph is learned per time-of-day bucket from train days,
   then expanded to every timestamp row. For example bucket 10:00-11:00 uses only
   train samples whose origin time is in that bucket; every sample uses its own
   previous p steps inside the same day.

Graph convention:
    G[t, target, source]
means source road helps predict target road.

Weight:
    G[i,j] = sign(b_ij,r*) * max(0, log((MSE_R(i)+eps)/(MSE_F(i,j)+eps)))
where r* is the source lag with the largest absolute coefficient.
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
from typing import Any, Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None

EPS = 1e-8
THIS_FILE = Path(__file__).resolve()
DATA_PROCESSING_DIR = THIS_FILE.parent
PROJECT_ROOT = DATA_PROCESSING_DIR.parents[2]
DEFAULT_SOURCE_DIR = DATA_PROCESSING_DIR / "outputs" / "branchB" / "osm_edge_base_like_branchA"
DEFAULT_OUTPUT_DIR = DATA_PROCESSING_DIR / "outputs" / "branchB" / "osm_edge_granger_series_like_branchA"


def now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    print(f"[{now_str()}] {msg}", flush=True)


def print_stage(title: str) -> None:
    print("\n" + "=" * 96, flush=True)
    print(f"{now_str()} | {title}", flush=True)
    print("=" * 96, flush=True)


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


class NumpyJsonEncoder(json.JSONEncoder):
    def default(self, obj: Any):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.Timestamp):
            return str(obj)
        if isinstance(obj, Path):
            return str(obj)
        return super().default(obj)


def save_json(obj: Dict[str, Any], path: Path) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, cls=NumpyJsonEncoder)


def parse_int_list(s: Optional[str]) -> List[int]:
    if not s:
        return []
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


def maybe_iter(iterable, total: Optional[int] = None, desc: str = ""):
    if tqdm is not None:
        return tqdm(iterable, total=total, desc=desc)
    return iterable


_TIME_HHMM_COLON = re.compile(r"(?<!\d)([0-2]?\d):([0-5]\d)(?!\d)")
_TIME_HHMM_COMPACT = re.compile(r"(?<!\d)([0-2]\d)([0-5]\d)(?!\d)")


def parse_time_minutes(value: Any) -> int:
    s = str(value)
    m = _TIME_HHMM_COLON.search(s)
    if m:
        hh, mm = int(m.group(1)), int(m.group(2))
        if 0 <= hh <= 23:
            return hh * 60 + mm
    m = re.search(r"Slot[_-]?([0-2]\d)([0-5]\d)", s)
    if not m:
        m = _TIME_HHMM_COMPACT.search(s)
    if m:
        hh, mm = int(m.group(1)), int(m.group(2))
        if 0 <= hh <= 23:
            return hh * 60 + mm
    return -1


def normalize_timestamp(raw: Any) -> str:
    s = str(raw)
    date_part = s.split("__")[0].split()[0]
    tm = parse_time_minutes(s)
    if tm >= 0:
        return f"{date_part} {tm // 60:02d}:{tm % 60:02d}:00"
    parsed = pd.to_datetime(s.replace("__", " ").replace("Slot_", ""), errors="coerce")
    if pd.notna(parsed):
        return parsed.strftime("%Y-%m-%d %H:%M:%S")
    return s


def build_or_fix_meta(meta: pd.DataFrame, timestamps: np.ndarray) -> pd.DataFrame:
    meta = meta.copy().reset_index(drop=True)
    ts_raw = np.asarray(timestamps).astype(str)
    fixed = np.asarray([normalize_timestamp(x) for x in ts_raw], dtype=str)
    ts = pd.to_datetime(fixed, errors="coerce")

    meta["time_index"] = np.arange(len(meta), dtype=np.int64)
    meta["timestamp_raw"] = ts_raw[: len(meta)]
    meta["timestamp_local"] = ts[: len(meta)]
    meta["date_key"] = meta["timestamp_local"].dt.date.astype(str)
    meta["time_minutes"] = (meta["timestamp_local"].dt.hour * 60 + meta["timestamp_local"].dt.minute).astype("Int64")
    meta["time_set"] = meta["timestamp_local"].dt.strftime("%H:%M")
    meta["session_id"] = meta["date_key"].astype(str)

    # Fallback if timestamp parse failed for some rows.
    bad = meta["time_minutes"].isna()
    if bad.any():
        fallback_tm = [parse_time_minutes(x) for x in ts_raw[: len(meta)]]
        meta.loc[bad, "time_minutes"] = np.asarray(fallback_tm, dtype=np.int32)[bad.to_numpy()]
        if "date_key" not in meta.columns or meta["date_key"].isna().any():
            meta["date_key"] = [str(x).split("__")[0].split()[0] for x in ts_raw[: len(meta)]]
        meta["session_id"] = meta["date_key"].astype(str)

    meta["time_minutes"] = pd.to_numeric(meta["time_minutes"], errors="coerce").fillna(-1).astype(np.int32)
    return meta


def load_base_split(source_dir: Path, split: str, mmap: bool = False) -> Dict[str, Any]:
    d = source_dir / split
    required = ["z.npy", "segment_ids.npy", "timestamps.npy", "G_series_meta.csv"]
    missing = [d / name for name in required if not (d / name).exists()]
    if missing:
        raise FileNotFoundError("Missing source Branch-B base files:\n" + "\n".join(map(str, missing)))
    mmap_mode = "r" if mmap else None
    z = np.load(d / "z.npy", mmap_mode=mmap_mode)
    segment_ids = np.asarray(np.load(d / "segment_ids.npy"), dtype=np.int64)
    timestamps = np.asarray(np.load(d / "timestamps.npy", allow_pickle=True)).astype(str)
    meta = pd.read_csv(d / "G_series_meta.csv")
    meta = build_or_fix_meta(meta, timestamps)
    m = min(len(meta), z.shape[0], len(timestamps))
    return {
        "z": np.asarray(z[:m], dtype=np.float32),
        "segment_ids": segment_ids,
        "timestamps": np.asarray([normalize_timestamp(x) for x in timestamps[:m]], dtype=str),
        "meta": meta.iloc[:m].reset_index(drop=True),
    }


def sort_split_by_day_time(data: Dict[str, Any]) -> Dict[str, Any]:
    meta = data["meta"].copy().reset_index(drop=True)
    meta["_orig_pos"] = np.arange(len(meta), dtype=np.int64)
    meta["_date_sort"] = pd.to_datetime(meta["date_key"], errors="coerce")
    meta = meta.sort_values(["_date_sort", "date_key", "time_minutes", "_orig_pos"], kind="stable").reset_index(drop=True)
    order = meta["_orig_pos"].to_numpy(dtype=np.int64)
    meta = meta.drop(columns=["_orig_pos", "_date_sort"])
    meta["time_index"] = np.arange(len(meta), dtype=np.int64)
    out = dict(data)
    out["z"] = np.asarray(data["z"], dtype=np.float32)[order]
    out["timestamps"] = np.asarray(data["timestamps"]).astype(str)[order]
    out["meta"] = meta
    return out


def resolve_node_indices(
    segment_ids: np.ndarray,
    max_nodes: int = 0,
    node_indices_arg: Optional[str] = None,
    node_ids_arg: Optional[str] = None,
    node_sample: str = "first",
    seed: int = 42,
) -> Optional[np.ndarray]:
    seg = np.asarray(segment_ids, dtype=np.int64)
    N = len(seg)
    selected: Optional[np.ndarray] = None
    if node_indices_arg:
        idx = np.asarray(parse_int_list(node_indices_arg), dtype=np.int64)
        if idx.size == 0:
            raise ValueError("--node-indices was provided but parsed empty.")
        if idx.min() < 0 or idx.max() >= N:
            raise ValueError(f"node index out of range. N={N}, min={idx.min()}, max={idx.max()}")
        selected = idx
    if node_ids_arg:
        req = np.asarray(parse_int_list(node_ids_arg), dtype=np.int64)
        pos = {int(v): i for i, v in enumerate(seg)}
        missing = [int(x) for x in req if int(x) not in pos]
        if missing:
            raise ValueError(f"Some node ids are missing from segment_ids: {missing[:20]}")
        idx = np.asarray([pos[int(x)] for x in req], dtype=np.int64)
        selected = idx if selected is None else np.intersect1d(selected, idx)
    if selected is None and int(max_nodes) > 0 and int(max_nodes) < N:
        if node_sample == "first":
            selected = np.arange(int(max_nodes), dtype=np.int64)
        elif node_sample == "random":
            rng = np.random.default_rng(int(seed))
            selected = np.sort(rng.choice(N, size=int(max_nodes), replace=False).astype(np.int64))
        else:
            raise ValueError("--node-sample must be first or random")
    if selected is None:
        return None
    selected = np.asarray(sorted(set(map(int, selected.tolist()))), dtype=np.int64)
    if selected.size == 0:
        raise ValueError("Node selection is empty.")
    return selected


def subset_nodes(data: Dict[str, Any], node_idx: Optional[np.ndarray]) -> Dict[str, Any]:
    if node_idx is None:
        return data
    idx = np.asarray(node_idx, dtype=np.int64)
    out = dict(data)
    out["z"] = np.asarray(data["z"], dtype=np.float32)[:, idx]
    out["segment_ids"] = np.asarray(data["segment_ids"], dtype=np.int64)[idx]
    return out


def add_history_and_bucket_columns(data: Dict[str, Any], p: int, bucket_minutes: int) -> Dict[str, Any]:
    meta = data["meta"].copy().reset_index(drop=True)
    meta["pos_in_day"] = -1
    meta["can_predict_granger"] = False
    for _, sub in meta.groupby("session_id", sort=False):
        idx = sub.index.to_numpy(dtype=np.int64)
        meta.loc[idx, "pos_in_day"] = np.arange(len(idx), dtype=np.int32)
        # p previous steps excluding current: positions 0..p-1 are history warm-up, first origin is p.
        meta.loc[idx, "can_predict_granger"] = meta.loc[idx, "pos_in_day"].to_numpy(dtype=np.int32) >= int(p)
    tm = pd.to_numeric(meta["time_minutes"], errors="coerce").fillna(-1).astype(int)
    if bucket_minutes <= 0:
        # exact time slot bucket
        bucket_key = tm.to_numpy(dtype=np.int32)
    else:
        bucket_key = ((tm // int(bucket_minutes)) * int(bucket_minutes)).to_numpy(dtype=np.int32)
    # Stable bucket ids sorted by minute.
    unique_keys = sorted([int(x) for x in np.unique(bucket_key) if int(x) >= 0])
    key_to_bucket = {k: i for i, k in enumerate(unique_keys)}
    meta["bucket_key"] = bucket_key
    meta["bucket_id"] = [key_to_bucket.get(int(x), -1) for x in bucket_key]
    out = dict(data)
    out["meta"] = meta
    return out


def validate_day_split(splits: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    dates = {name: set(map(str, data["meta"]["date_key"].dropna().astype(str).unique())) for name, data in splits.items()}
    overlaps = {
        "train_val": sorted(dates["train"] & dates["val"]),
        "train_test": sorted(dates["train"] & dates["test"]),
        "val_test": sorted(dates["val"] & dates["test"]),
    }
    if any(overlaps[k] for k in overlaps):
        raise RuntimeError(f"Date leakage detected across splits: {overlaps}")
    return {
        "split_mode_verified": "by_whole_date_no_overlap",
        "n_train_dates": len(dates["train"]),
        "n_val_dates": len(dates["val"]),
        "n_test_dates": len(dates["test"]),
        "train_dates": sorted(dates["train"]),
        "val_dates": sorted(dates["val"]),
        "test_dates": sorted(dates["test"]),
    }


def build_training_samples_for_bucket(
    z: np.ndarray,
    meta: pd.DataFrame,
    p: int,
    bucket_id: Optional[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    z = np.asarray(z, dtype=np.float32)
    origins: List[int] = []
    for _, sub in meta.groupby("session_id", sort=False):
        idx = sub.index.to_numpy(dtype=np.int64)
        if len(idx) <= p:
            continue
        for local_pos in range(int(p), len(idx)):
            origin = int(idx[local_pos])
            if not bool(meta.loc[origin, "can_predict_granger"]):
                continue
            if bucket_id is not None and int(meta.loc[origin, "bucket_id"]) != int(bucket_id):
                continue
            origins.append(origin)
    if not origins:
        return np.empty((0, p, z.shape[1]), dtype=np.float32), np.empty((0, z.shape[1]), dtype=np.float32), np.empty(0, dtype=np.int64)
    origins_arr = np.asarray(origins, dtype=np.int64)
    lagged = []
    # Strictly previous steps, not current. r=0 -> t-1, r=1 -> t-2, ...
    for r in range(1, int(p) + 1):
        lagged.append(z[origins_arr - r])
    X_lags = np.stack(lagged, axis=1).astype(np.float32)
    Y = z[origins_arr].astype(np.float32)
    return X_lags, Y, origins_arr


def standardize_columns(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    mu = X.mean(axis=0, keepdims=True)
    Xc = X - mu
    std = Xc.std(axis=0, ddof=1, keepdims=True) if X.shape[0] > 1 else np.ones_like(mu)
    std = np.where(np.isfinite(std) & (std > 1e-8), std, 1.0).astype(np.float32)
    return np.nan_to_num(Xc / std, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def ridge_beta(X: np.ndarray, y: np.ndarray, ridge: float) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    p = X.shape[1]
    A = X.T @ X
    if ridge > 0:
        A = A + float(ridge) * np.eye(p, dtype=np.float64)
    b = X.T @ y
    try:
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(A, b, rcond=None)[0]


def fit_restricted_residuals(X_lags: np.ndarray, Y: np.ndarray, ridge: float, fit_intercept: bool) -> Tuple[np.ndarray, np.ndarray]:
    M, p, N = X_lags.shape
    residuals = np.empty((M, N), dtype=np.float32)
    mse_r = np.empty(N, dtype=np.float32)
    for i in maybe_iter(range(N), total=N, desc="restricted"):
        Xi = X_lags[:, :, i]
        if fit_intercept:
            Xi = np.concatenate([np.ones((M, 1), dtype=np.float32), Xi], axis=1)
        y = Y[:, i]
        beta = ridge_beta(Xi, y, ridge)
        pred = Xi @ beta
        res = y - pred
        residuals[:, i] = res.astype(np.float32)
        mse_r[i] = float(np.mean(res ** 2))
    return residuals, mse_r


def choose_candidates_by_partial_corr(X_lags: np.ndarray, residuals: np.ndarray, max_candidates: int, block_size: int) -> List[np.ndarray]:
    M, p, N = X_lags.shape
    K = max(1, min(int(max_candidates), max(N - 1, 1)))
    B = max(1, int(block_size))
    X_std_by_lag = [standardize_columns(X_lags[:, r, :]) for r in range(p)]
    R_std = standardize_columns(residuals)
    candidates: List[Optional[np.ndarray]] = [None] * N
    denom = float(max(M - 1, 1))
    for start in maybe_iter(range(0, N, B), total=math.ceil(N / B), desc="candidate blocks"):
        end = min(start + B, N)
        rb = R_std[:, start:end]
        score = np.zeros((N, end - start), dtype=np.float32)
        for Xs in X_std_by_lag:
            corr = (Xs.T @ rb).astype(np.float32) / denom
            np.maximum(score, np.abs(corr), out=score)
        for local, target in enumerate(range(start, end)):
            score[target, local] = -np.inf
        k_eff = min(K, N - 1)
        idx = np.argpartition(score, -k_eff, axis=0)[-k_eff:, :]
        sc = np.take_along_axis(score, idx, axis=0)
        order = np.argsort(-sc, axis=0)
        sorted_idx = np.take_along_axis(idx, order, axis=0)
        for local, target in enumerate(range(start, end)):
            candidates[target] = sorted_idx[:, local].astype(np.int64)
    return [np.asarray(c, dtype=np.int64) for c in candidates]


def compute_granger_graph(
    X_lags: np.ndarray,
    Y: np.ndarray,
    p: int,
    max_candidates: int,
    candidate_block_size: int,
    ridge: float,
    fit_intercept: bool,
    signed: bool,
    min_improvement: float,
    dtype: str,
    lag_dtype: str,
    label: str,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    t0 = time.time()
    X_lags = np.asarray(X_lags, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.float32)
    M, p_used, N = X_lags.shape
    if M <= max(2 * p_used + 2, 4):
        raise RuntimeError(f"Too few samples for Granger {label}: M={M}, p={p_used}")

    log(f"{label}: X_lags={X_lags.shape}, Y={Y.shape}")
    residuals, mse_r = fit_restricted_residuals(X_lags, Y, ridge=ridge, fit_intercept=fit_intercept)
    candidates = choose_candidates_by_partial_corr(
        X_lags=X_lags,
        residuals=residuals,
        max_candidates=max_candidates,
        block_size=candidate_block_size,
    )
    G = np.zeros((N, N), dtype=np.float32)
    L = np.zeros((N, N), dtype=np.int16)
    for target in maybe_iter(range(N), total=N, desc=f"granger {label}"):
        own = X_lags[:, :, target]
        y = Y[:, target]
        mse_base = float(mse_r[target])
        if not np.isfinite(mse_base) or mse_base <= EPS:
            continue
        for source in candidates[target]:
            source = int(source)
            if source == target:
                continue
            src = X_lags[:, :, source]
            Xf = np.concatenate([own, src], axis=1)
            if fit_intercept:
                Xf = np.concatenate([np.ones((M, 1), dtype=np.float32), Xf], axis=1)
            beta = ridge_beta(Xf, y, ridge=ridge)
            pred = Xf @ beta
            mse_f = float(np.mean((y - pred) ** 2))
            if not np.isfinite(mse_f) or mse_f <= 0:
                continue
            score = math.log((mse_base + EPS) / (mse_f + EPS))
            if score <= float(min_improvement):
                continue
            src_offset = (1 if fit_intercept else 0) + p_used
            src_coefs = np.asarray(beta[src_offset: src_offset + p_used], dtype=np.float64)
            if src_coefs.size == 0:
                continue
            best_lag_idx = int(np.argmax(np.abs(src_coefs)))
            coef = float(src_coefs[best_lag_idx])
            w = math.copysign(score, coef if coef != 0.0 else 1.0) if signed else score
            G[target, source] = float(w)
            # Store 1-based lag in time steps: 1 means t-1, 2 means t-2.
            L[target, source] = int(best_lag_idx + 1)
    np.fill_diagonal(G, 0.0)
    np.fill_diagonal(L, 0)
    elapsed = time.time() - t0
    summary = {
        "label": label,
        "n_samples": int(M),
        "n_segments": int(N),
        "granger_p_previous_steps": int(p),
        "max_candidates": int(max_candidates),
        "candidate_block_size": int(candidate_block_size),
        "ridge": float(ridge),
        "fit_intercept": bool(fit_intercept),
        "signed": bool(signed),
        "min_improvement": float(min_improvement),
        "n_nonzero_edges": int(np.count_nonzero(G)),
        "elapsed_seconds": float(elapsed),
    }
    log(f"{label} DONE | nonzero={summary['n_nonzero_edges']:,} | elapsed={elapsed/60:.2f} min")
    return G.astype(dtype), L.astype(lag_dtype), summary


def save_timeline_outputs(
    out_dir: Path,
    splits: Dict[str, Dict[str, Any]],
    bucket_graphs: Dict[int, Tuple[np.ndarray, np.ndarray]],
    global_graph: Tuple[np.ndarray, np.ndarray],
    args: argparse.Namespace,
    node_idx: Optional[np.ndarray],
    split_info: Dict[str, Any],
    bucket_summaries: List[Dict[str, Any]],
) -> None:
    if out_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output exists: {out_dir}. Use --overwrite.")
        log(f"[CLEAN] removing old output: {out_dir}")
        shutil.rmtree(out_dir)
    ensure_dir(out_dir)
    save_json({
        "graph_type": "granger_predictive_influence_timeline",
        "not_lagged_correlation": True,
        "g_shape_semantics": "G_weight_series[t,target,source]",
        "granger_history": "strict previous p steps; x_t is not used as lag input for G_t",
        "granger_p": int(args.granger_p),
        "bucket_minutes": int(args.bucket_minutes),
        "min_bucket_samples": int(args.min_bucket_samples),
        "max_candidates": int(args.max_candidates),
        "candidate_block_size": int(args.candidate_block_size),
        "node_subset": None if node_idx is None else node_idx.tolist(),
        "split_info": split_info,
        "bucket_summaries": bucket_summaries,
    }, out_dir / "branchB_granger_timeline_run_summary.json")

    for split_name, data in splits.items():
        split_out = ensure_dir(out_dir / split_name)
        z = np.asarray(data["z"], dtype=np.float32)
        meta = data["meta"].copy().reset_index(drop=True)
        T, N = z.shape
        np.save(split_out / "z.npy", z)
        np.save(split_out / "segment_ids.npy", np.asarray(data["segment_ids"], dtype=np.int64))
        np.save(split_out / "timestamps.npy", np.asarray(data["timestamps"]).astype(str))
        meta.to_csv(split_out / "G_series_meta.csv", index=False)

        G_mem = np.lib.format.open_memmap(split_out / "G_weight_series.npy", mode="w+", dtype=np.dtype(args.g_dtype), shape=(T, N, N))
        L_mem = np.lib.format.open_memmap(split_out / "G_best_lag_series.npy", mode="w+", dtype=np.dtype(args.lag_dtype), shape=(T, N, N))
        G_zero = np.zeros((N, N), dtype=np.dtype(args.g_dtype))
        L_zero = np.zeros((N, N), dtype=np.dtype(args.lag_dtype))
        G_global, L_global = global_graph
        for t in range(T):
            if not bool(meta.loc[t, "can_predict_granger"]):
                G_mem[t] = G_zero
                L_mem[t] = L_zero
                continue
            bid = int(meta.loc[t, "bucket_id"])
            G_base, L_base = bucket_graphs.get(bid, (G_global, L_global))
            G_mem[t] = G_base
            L_mem[t] = L_base
        G_mem.flush(); L_mem.flush()
        del G_mem, L_mem
        save_json({
            "split": split_name,
            "z_shape": list(map(int, z.shape)),
            "G_shape": [int(T), int(N), int(N)],
            "n_valid_origins_after_history": int(meta["can_predict_granger"].sum()),
            "n_dates": int(meta["date_key"].nunique()),
        }, split_out / "branchB_gt_split_summary.json")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--source-dir", type=str, default=str(DEFAULT_SOURCE_DIR))
    p.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    p.add_argument("--granger-p", type=int, default=3, help="Number of previous time steps used for G_t. p=3 uses t-1,t-2,t-3.")
    p.add_argument("--bucket-minutes", type=int, default=60, help="Time-of-day bucket size. 60 is stable; 15 gives exact slots.")
    p.add_argument("--min-bucket-samples", type=int, default=20)
    p.add_argument("--max-candidates", type=int, default=50)
    p.add_argument("--candidate-block-size", type=int, default=256)
    p.add_argument("--ridge", type=float, default=1e-3)
    p.add_argument("--min-improvement", type=float, default=0.0)
    p.add_argument("--unsigned", action="store_true")
    p.add_argument("--fit-intercept", action="store_true")
    p.add_argument("--g-dtype", type=str, default="float32", choices=["float16", "float32"])
    p.add_argument("--lag-dtype", type=str, default="int16")
    p.add_argument("--max-nodes", type=int, default=0)
    p.add_argument("--node-indices", type=str, default=None)
    p.add_argument("--node-ids", type=str, default=None)
    p.add_argument("--node-sample", type=str, default="first", choices=["first", "random"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    # Accepted for compatibility with previous commands; no longer used because G_t is a timeline state.
    p.add_argument("--horizons", type=str, default="1-9", help="Accepted for compatibility; G_t is horizon-independent timeline graph.")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    source_dir = Path(args.source_dir)
    if not source_dir.is_absolute():
        source_dir = PROJECT_ROOT / source_dir
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir

    print_stage("LOAD BASE DATA AND VALIDATE DAY SPLIT")
    log(f"PROJECT_ROOT: {PROJECT_ROOT}")
    log(f"SOURCE_DIR  : {source_dir}")
    log(f"OUTPUT_DIR  : {out_dir}")
    log(f"G_t history : previous p={args.granger_p} steps only")
    log(f"Bucket      : {args.bucket_minutes} minutes")

    splits = {name: sort_split_by_day_time(load_base_split(source_dir, name, mmap=False)) for name in ["train", "val", "test"]}
    split_info = validate_day_split(splits)
    log(f"Day split OK: train={split_info['n_train_dates']} days, val={split_info['n_val_dates']} days, test={split_info['n_test_dates']} days")

    node_idx = resolve_node_indices(
        splits["train"]["segment_ids"],
        max_nodes=args.max_nodes,
        node_indices_arg=args.node_indices,
        node_ids_arg=args.node_ids,
        node_sample=args.node_sample,
        seed=args.seed,
    )
    if node_idx is not None:
        log(f"Node subset: {len(node_idx)} / {len(splits['train']['segment_ids'])}")
        splits = {name: subset_nodes(data, node_idx) for name, data in splits.items()}
    else:
        log(f"Node mode: full N={len(splits['train']['segment_ids'])}")

    splits = {name: add_history_and_bucket_columns(data, p=args.granger_p, bucket_minutes=args.bucket_minutes) for name, data in splits.items()}
    for name, data in splits.items():
        log(f"{name}: z={data['z'].shape}, dates={data['meta']['date_key'].nunique()}, valid_origins={int(data['meta']['can_predict_granger'].sum())}")

    if args.dry_run:
        log("Dry run complete.")
        return

    print_stage("COMPUTE TRAIN-ONLY GRANGER G_t BY TIME-OF-DAY BUCKET")
    train = splits["train"]
    X_global, Y_global, origins_global = build_training_samples_for_bucket(train["z"], train["meta"], p=args.granger_p, bucket_id=None)
    log(f"Global train samples: {len(origins_global)}")
    G_global, L_global, global_summary = compute_granger_graph(
        X_lags=X_global,
        Y=Y_global,
        p=args.granger_p,
        max_candidates=args.max_candidates,
        candidate_block_size=args.candidate_block_size,
        ridge=args.ridge,
        fit_intercept=args.fit_intercept,
        signed=not args.unsigned,
        min_improvement=args.min_improvement,
        dtype=args.g_dtype,
        lag_dtype=args.lag_dtype,
        label="global_train",
    )

    bucket_graphs: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    bucket_summaries: List[Dict[str, Any]] = [dict(global_summary, bucket_id=-1, bucket_label="global")]
    bucket_ids = sorted([int(x) for x in train["meta"]["bucket_id"].unique() if int(x) >= 0])
    for bid in bucket_ids:
        Xb, Yb, origins = build_training_samples_for_bucket(train["z"], train["meta"], p=args.granger_p, bucket_id=bid)
        tm_vals = train["meta"].loc[train["meta"]["bucket_id"] == bid, "bucket_key"]
        bucket_key = int(tm_vals.iloc[0]) if len(tm_vals) else -1
        bucket_label = f"{bucket_key//60:02d}:{bucket_key%60:02d}"
        if len(origins) < int(args.min_bucket_samples):
            log(f"Bucket {bid} {bucket_label}: samples={len(origins)} < {args.min_bucket_samples}; fallback to global graph")
            bucket_graphs[bid] = (G_global, L_global)
            bucket_summaries.append({"bucket_id": int(bid), "bucket_label": bucket_label, "n_samples": int(len(origins)), "fallback": "global"})
            continue
        G, L, summary = compute_granger_graph(
            X_lags=Xb,
            Y=Yb,
            p=args.granger_p,
            max_candidates=args.max_candidates,
            candidate_block_size=args.candidate_block_size,
            ridge=args.ridge,
            fit_intercept=args.fit_intercept,
            signed=not args.unsigned,
            min_improvement=args.min_improvement,
            dtype=args.g_dtype,
            lag_dtype=args.lag_dtype,
            label=f"bucket_{bid}_{bucket_label}",
        )
        bucket_graphs[bid] = (G, L)
        summary.update({"bucket_id": int(bid), "bucket_label": bucket_label, "fallback": None})
        bucket_summaries.append(summary)

    print_stage("SAVE FULL TIMELINE G_t SERIES")
    save_timeline_outputs(
        out_dir=out_dir,
        splits=splits,
        bucket_graphs=bucket_graphs,
        global_graph=(G_global, L_global),
        args=args,
        node_idx=node_idx,
        split_info=split_info,
        bucket_summaries=bucket_summaries,
    )
    log(f"DONE: {out_dir}")


if __name__ == "__main__":
    main()
