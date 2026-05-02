"""
Prepare Branch B OSM-edge Granger-Gt STANDARD SERIES.

Purpose
-------
This replaces the old lagged-correlation Rt series with a Granger-style directed
predictive influence Gt series, while keeping the same standard file format used
by the existing Branch-B forecasting scripts:

    train|val|test/z.npy
    train|val|test/segment_ids.npy
    train|val|test/timestamps.npy
    train|val|test/G_series_meta.csv
    train|val|test/G_weight_series.npy       # [T, N, N], target x source
    train|val|test/G_best_lag_series.npy     # [T, N, N]

Graph semantics
---------------
G[t, target, source] is NOT correlation. It is Granger-style predictive influence:

    sign(source coefficient) * max(0, log((MSE_restricted + eps) / (MSE_full + eps)))

The graph is estimated from TRAIN only by time-of-day bucket. Each timestamp t is
assigned the Granger graph of its bucket, producing a standard Gt time series.
This lets the old method names true_gt / persistence_gt / ewma_gt / sparse_tvpvar_gt
run on Granger-Gt instead of lagged-correlation Rt.

Recommended quick test:
    python -u ml_core/src/data_processing/prepare_branchB_osm_edge_granger_series_like_branchA.py \
      --max-nodes 512 --granger-horizon 1 --granger-p 3 --bucket-minutes 60 --overwrite

Recommended output dir:
    outputs/branchB/osm_edge_granger_series_like_branchA/
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

# Granger utilities. Keep this file in the same folder as
# prepare_branchB_osm_edge_granger_like_branchA.py.
from prepare_branchB_osm_edge_granger_like_branchA import (  # type: ignore
    EPS,
    NumpyJsonEncoder,
    compute_granger_from_supervised_tensors,
    parse_int_list,
    save_json,
)

THIS_FILE = Path(__file__).resolve()
DATA_PROCESSING_DIR = THIS_FILE.parent
PROJECT_ROOT = DATA_PROCESSING_DIR.parents[2]
DEFAULT_SOURCE_NPZ = DATA_PROCESSING_DIR / "outputs" / "branchA" / "osm_edge_forecasting_dataset" / "train_val_test_split.npz"
DEFAULT_OUTPUT_DIR = DATA_PROCESSING_DIR / "outputs" / "branchB" / "osm_edge_granger_series_like_branchA"

_TIME_HHMM_COLON = re.compile(r"(?<!\d)([0-2]?\d):([0-5]\d)(?!\d)")
_TIME_HHMM_COMPACT = re.compile(r"(?<!\d)([0-2]\d)([0-5]\d)(?!\d)")


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


def maybe_iter(iterable, total: Optional[int] = None, desc: str = ""):
    if tqdm is not None:
        return tqdm(iterable, total=total, desc=desc)
    return iterable


def fmt_gb(nbytes: int) -> str:
    return f"{nbytes / (1024 ** 3):.2f} GB"


def parse_time_minutes(value: Any) -> Optional[int]:
    s = str(value)
    m = _TIME_HHMM_COLON.search(s)
    if m:
        hh, mm = int(m.group(1)), int(m.group(2))
        if 0 <= hh <= 23:
            return hh * 60 + mm
    m = _TIME_HHMM_COMPACT.search(s)
    if m:
        hh, mm = int(m.group(1)), int(m.group(2))
        if 0 <= hh <= 23:
            return hh * 60 + mm
    return None


def safe_to_datetime_array(arr: np.ndarray) -> pd.Series:
    vals = pd.Series(np.asarray(arr).astype(str))
    vals2 = vals.str.replace("__", " ", regex=False)
    return pd.to_datetime(vals2, errors="coerce")


def build_meta(timestamps: np.ndarray) -> pd.DataFrame:
    ts_raw = np.asarray(timestamps).astype(str)
    ts = safe_to_datetime_array(ts_raw)
    meta = pd.DataFrame({
        "time_index": np.arange(len(ts_raw), dtype=np.int64),
        "timestamp_raw": ts_raw,
        "timestamp_local": ts,
    })

    date_key: List[str] = []
    time_minutes: List[int] = []
    time_set: List[str] = []
    for raw, parsed in zip(ts_raw, ts):
        if pd.notna(parsed):
            date_key.append(str(parsed.date()))
            tm = int(parsed.hour) * 60 + int(parsed.minute)
            time_minutes.append(tm)
            time_set.append(f"{tm // 60:02d}:{tm % 60:02d}")
        else:
            parts = str(raw).replace("__", " ").split()
            date_key.append(parts[0] if parts else str(raw))
            tm = parse_time_minutes(raw)
            time_minutes.append(-1 if tm is None else int(tm))
            time_set.append("" if tm is None else f"{tm // 60:02d}:{tm % 60:02d}")

    meta["date_key"] = date_key
    meta["time_minutes"] = np.asarray(time_minutes, dtype=np.int32)
    meta["time_set"] = time_set
    meta["session_id"] = meta["date_key"].astype(str)
    return meta


def load_source_npz(source_npz: Path, primary_feature: str) -> Dict[str, Any]:
    if not source_npz.exists():
        raise FileNotFoundError(f"Missing source split NPZ: {source_npz}")
    log(f"Loading source NPZ: {source_npz}")
    data = np.load(source_npz, allow_pickle=True)

    feature_names = [str(x) for x in data["feature_names"].tolist()] if "feature_names" in data.files else []
    if primary_feature in feature_names:
        f_idx = feature_names.index(primary_feature)
    else:
        f_idx = 0
        log(f"[WARN] primary_feature={primary_feature} not found. Use feature index 0.")

    splits: Dict[str, Dict[str, Any]] = {}
    for split in ["train", "val", "test"]:
        x_key = f"X_{split}"
        if x_key not in data.files:
            raise KeyError(f"Missing {x_key} in {source_npz}")
        X = np.asarray(data[x_key])
        if X.ndim == 3:
            z = X[:, :, f_idx].astype(np.float32)
        elif X.ndim == 2:
            z = X.astype(np.float32)
        else:
            raise ValueError(f"{x_key} must be 2D or 3D, got shape={X.shape}")

        t_key = f"timestamps_{split}"
        timestamps = np.asarray(data[t_key]).astype(str) if t_key in data.files else np.asarray([f"{split}_{i}" for i in range(z.shape[0])])
        splits[split] = {"z": z, "timestamps": timestamps, "meta": build_meta(timestamps)}

    if "model_node_ids" in data.files:
        segment_ids = np.asarray(data["model_node_ids"], dtype=np.int64)
    elif "segment_ids" in data.files:
        segment_ids = np.asarray(data["segment_ids"], dtype=np.int64)
    else:
        segment_ids = np.arange(splits["train"]["z"].shape[1], dtype=np.int64)

    return {
        "splits": splits,
        "segment_ids": segment_ids,
        "feature_names": feature_names,
        "primary_feature": primary_feature,
        "primary_feature_index": int(f_idx),
        "source_npz": str(source_npz),
    }


def resolve_node_indices(
    segment_ids: np.ndarray,
    max_nodes: int = 0,
    node_indices_arg: Optional[str] = None,
    node_ids_arg: Optional[str] = None,
    node_sample: str = "first",
    seed: int = 42,
) -> Optional[np.ndarray]:
    N = int(len(segment_ids))
    selected: Optional[np.ndarray] = None

    if node_indices_arg:
        idx = np.asarray(parse_int_list(node_indices_arg), dtype=np.int64)
        if idx.size == 0:
            raise ValueError("--node-indices was provided but no valid index was parsed.")
        if idx.min() < 0 or idx.max() >= N:
            raise ValueError(f"node index out of range. N={N}, min={idx.min()}, max={idx.max()}")
        selected = idx

    if node_ids_arg:
        requested = np.asarray(parse_int_list(node_ids_arg), dtype=np.int64)
        pos = {int(v): i for i, v in enumerate(np.asarray(segment_ids, dtype=np.int64))}
        missing = [int(x) for x in requested if int(x) not in pos]
        if missing:
            raise ValueError(f"Some node ids are missing from segment_ids: {missing[:20]}")
        idx = np.asarray([pos[int(x)] for x in requested], dtype=np.int64)
        selected = idx if selected is None else np.intersect1d(selected, idx)

    if selected is None and int(max_nodes) > 0:
        max_nodes = min(int(max_nodes), N)
        if max_nodes < N:
            if node_sample == "first":
                selected = np.arange(max_nodes, dtype=np.int64)
            elif node_sample == "random":
                rng = np.random.default_rng(int(seed))
                selected = np.sort(rng.choice(N, size=max_nodes, replace=False).astype(np.int64))
            else:
                raise ValueError("--node-sample must be first or random")

    if selected is None:
        return None
    selected = np.asarray(sorted(set(map(int, selected.tolist()))), dtype=np.int64)
    if selected.size == 0:
        raise ValueError("Node selection is empty.")
    return selected


def subset_split(split_data: Dict[str, Any], node_idx: Optional[np.ndarray]) -> Dict[str, Any]:
    if node_idx is None:
        return dict(split_data)
    out = dict(split_data)
    out["z"] = np.asarray(split_data["z"], dtype=np.float32)[:, node_idx]
    return out


def session_groups(meta: pd.DataFrame) -> List[np.ndarray]:
    if "session_id" in meta.columns:
        groups = []
        for _, sub in meta.groupby("session_id", sort=False):
            idx = sub.index.to_numpy(dtype=np.int64)
            if len(idx):
                groups.append(idx)
        return groups
    return [np.arange(len(meta), dtype=np.int64)]


def build_supervised_tensors(z: np.ndarray, meta: pd.DataFrame, horizon: int, p: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build:
        X_lags[m, r, n] = z[origin-r, n], r=0..p-1
        Y[m, n]         = z[origin+horizon, n]
    Session boundaries are respected.
    """
    z = np.asarray(z, dtype=np.float32)
    h = int(horizon)
    p = int(p)
    X_rows: List[np.ndarray] = []
    Y_rows: List[np.ndarray] = []
    origins: List[int] = []
    targets: List[int] = []

    for idx in session_groups(meta):
        if len(idx) <= h + p:
            continue
        for pos in range(p - 1, len(idx) - h):
            origin = int(idx[pos])
            target = int(idx[pos + h])
            lag_indices = [int(idx[pos - r]) for r in range(p)]
            X_rows.append(z[lag_indices, :])
            Y_rows.append(z[target, :])
            origins.append(origin)
            targets.append(target)

    if not X_rows:
        N = int(z.shape[1])
        return (
            np.empty((0, p, N), dtype=np.float32),
            np.empty((0, N), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
        )

    return (
        np.stack(X_rows, axis=0).astype(np.float32),
        np.stack(Y_rows, axis=0).astype(np.float32),
        np.asarray(origins, dtype=np.int64),
        np.asarray(targets, dtype=np.int64),
    )


def build_bucket_table(bucket_minutes: int, start_minute: int, end_minute: int) -> pd.DataFrame:
    rows = []
    b = 0
    for s in range(int(start_minute), int(end_minute), int(bucket_minutes)):
        e = min(s + int(bucket_minutes), int(end_minute))
        rows.append({
            "bucket_id": b,
            "start_minute": int(s),
            "end_minute": int(e),
            "label": f"{s // 60:02d}:{s % 60:02d}-{e // 60:02d}:{e % 60:02d}",
        })
        b += 1
    return pd.DataFrame(rows)


def assign_bucket_ids(meta: pd.DataFrame, bucket_table: pd.DataFrame) -> np.ndarray:
    if "time_minutes" in meta.columns:
        tod = pd.to_numeric(meta["time_minutes"], errors="coerce").fillna(-1).to_numpy(dtype=np.int32)
    else:
        tod = np.arange(len(meta), dtype=np.int32)
    ids = np.zeros(len(meta), dtype=np.int16)
    starts = bucket_table["start_minute"].to_numpy(dtype=np.int32)
    ends = bucket_table["end_minute"].to_numpy(dtype=np.int32)
    for i, m in enumerate(tod):
        hit = np.where((starts <= int(m)) & (int(m) < ends))[0]
        if len(hit):
            ids[i] = int(bucket_table.iloc[int(hit[0])]["bucket_id"])
        elif int(m) < starts.min():
            ids[i] = int(bucket_table.iloc[0]["bucket_id"])
        else:
            ids[i] = int(bucket_table.iloc[-1]["bucket_id"])
    return ids


def estimate_bucket_granger_graphs(train_z: np.ndarray, train_meta: pd.DataFrame, args: argparse.Namespace) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame]:
    bucket_table = build_bucket_table(
        bucket_minutes=int(args.bucket_minutes),
        start_minute=int(args.start_minute),
        end_minute=int(args.end_minute),
    )
    n_buckets = len(bucket_table)
    N = int(train_z.shape[1])
    dtype = np.dtype(args.g_dtype)
    lag_dtype = np.dtype(args.lag_dtype)

    print_stage("ESTIMATE TRAIN-ONLY GRANGER BUCKET GRAPHS")
    log(f"N={N}, buckets={n_buckets}, granger_horizon={args.granger_horizon}, p={args.granger_p}")

    X_lags, Y, origins, _ = build_supervised_tensors(
        z=train_z,
        meta=train_meta,
        horizon=int(args.granger_horizon),
        p=int(args.granger_p),
    )
    if X_lags.shape[0] == 0:
        raise RuntimeError("No supervised samples to estimate Granger graphs.")

    origin_bucket_ids = assign_bucket_ids(train_meta, bucket_table)[origins]

    G_global, L_global, global_summary = compute_granger_from_supervised_tensors(
        X_lags=X_lags,
        Y=Y,
        horizon=int(args.granger_horizon),
        p=int(args.granger_p),
        max_candidates=int(args.max_candidates),
        candidate_block_size=int(args.candidate_block_size),
        ridge=float(args.ridge),
        fit_intercept=bool(args.fit_intercept),
        signed=not bool(args.unsigned),
        dtype=str(args.g_dtype),
        lag_dtype=str(args.lag_dtype),
        min_improvement=float(args.min_improvement),
        label="global_train",
    )

    G_bucket = np.zeros((n_buckets, N, N), dtype=dtype)
    L_bucket = np.zeros((n_buckets, N, N), dtype=lag_dtype)
    rows: List[Dict[str, Any]] = []

    for b in range(n_buckets):
        sample_mask = origin_bucket_ids == int(b)
        n_samples = int(np.count_nonzero(sample_mask))
        label = str(bucket_table.iloc[b]["label"])
        if n_samples < int(args.min_bucket_samples):
            log(f"bucket {b} {label}: n={n_samples} < min={args.min_bucket_samples}; use global graph")
            G_bucket[b] = G_global
            L_bucket[b] = L_global
            rows.append({
                "bucket_id": int(b),
                "label": label,
                "n_samples": n_samples,
                "used_global_fallback": True,
                "n_nonzero_edges": int(np.count_nonzero(G_global)),
            })
            continue

        G, L, summary = compute_granger_from_supervised_tensors(
            X_lags=X_lags[sample_mask],
            Y=Y[sample_mask],
            horizon=int(args.granger_horizon),
            p=int(args.granger_p),
            max_candidates=int(args.max_candidates),
            candidate_block_size=int(args.candidate_block_size),
            ridge=float(args.ridge),
            fit_intercept=bool(args.fit_intercept),
            signed=not bool(args.unsigned),
            dtype=str(args.g_dtype),
            lag_dtype=str(args.lag_dtype),
            min_improvement=float(args.min_improvement),
            label=f"bucket_{b}_{label}",
        )
        G_bucket[b] = G
        L_bucket[b] = L
        rows.append({
            "bucket_id": int(b),
            "label": label,
            "n_samples": n_samples,
            "used_global_fallback": False,
            "n_nonzero_edges": int(summary["n_nonzero_edges"]),
            "elapsed_seconds": float(summary.get("elapsed_seconds", 0.0)),
        })

    summary_df = pd.DataFrame(rows)
    global_summary_df = pd.DataFrame([global_summary])
    return bucket_table, G_bucket, L_bucket, pd.concat([global_summary_df.assign(bucket_id=-1, label="global"), summary_df], ignore_index=True)


def save_split_standard_series(
    split_name: str,
    split_data: Dict[str, Any],
    segment_ids: np.ndarray,
    out_dir: Path,
    bucket_table: pd.DataFrame,
    G_bucket: np.ndarray,
    L_bucket: np.ndarray,
    args: argparse.Namespace,
) -> None:
    split_out = out_dir / split_name
    if split_out.exists():
        shutil.rmtree(split_out)
    ensure_dir(split_out)

    z = np.asarray(split_data["z"], dtype=np.float32)
    timestamps = np.asarray(split_data["timestamps"]).astype(str)
    meta = split_data["meta"].copy()
    bucket_ids = assign_bucket_ids(meta, bucket_table)
    T, N = z.shape

    np.save(split_out / "z.npy", z.astype(np.float32))
    np.save(split_out / "segment_ids.npy", np.asarray(segment_ids, dtype=np.int64))
    np.save(split_out / "timestamps.npy", timestamps.astype(str))

    meta_out = meta.copy()
    meta_out["bucket_id"] = bucket_ids.astype(np.int16)
    meta_out["graph_type"] = "granger_predictive_influence_series"
    meta_out["granger_horizon"] = int(args.granger_horizon)
    meta_out["granger_p"] = int(args.granger_p)
    meta_out.to_csv(split_out / "G_series_meta.csv", index=False)

    G_path = split_out / "G_weight_series.npy"
    L_path = split_out / "G_best_lag_series.npy"
    G_mm = np.lib.format.open_memmap(G_path, mode="w+", dtype=np.dtype(args.g_dtype), shape=(T, N, N))
    L_mm = np.lib.format.open_memmap(L_path, mode="w+", dtype=np.dtype(args.lag_dtype), shape=(T, N, N))

    log(f"Writing {split_name} Gt series: T={T}, N={N}, size≈{fmt_gb(T*N*N*np.dtype(args.g_dtype).itemsize)}")
    for t in maybe_iter(range(T), total=T, desc=f"write {split_name}"):
        b = int(bucket_ids[t])
        G_mm[t] = G_bucket[b]
        L_mm[t] = L_bucket[b]
    G_mm.flush()
    L_mm.flush()

    split_summary = {
        "created_at": now_str(),
        "split": split_name,
        "graph_type": "granger_predictive_influence_series",
        "not_correlation": True,
        "series_source": "train_only_time_of_day_bucket_granger",
        "z_shape": list(map(int, z.shape)),
        "G_weight_series_shape": [int(T), int(N), int(N)],
        "G_dtype": str(np.dtype(args.g_dtype)),
        "L_dtype": str(np.dtype(args.lag_dtype)),
        "granger_horizon": int(args.granger_horizon),
        "granger_p": int(args.granger_p),
        "bucket_minutes": int(args.bucket_minutes),
        "max_candidates": int(args.max_candidates),
        "ridge": float(args.ridge),
        "fit_intercept": bool(args.fit_intercept),
        "signed": not bool(args.unsigned),
    }
    save_json(split_summary, split_out / "branchB_gt_split_summary.json")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-npz", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--primary-feature", type=str, default="average_speed")
    parser.add_argument("--granger-horizon", type=int, default=1, help="Horizon used to estimate the Granger influence graph series.")
    parser.add_argument("--granger-p", type=int, default=3)
    parser.add_argument("--bucket-minutes", type=int, default=60)
    parser.add_argument("--start-minute", type=int, default=6 * 60)
    parser.add_argument("--end-minute", type=int, default=12 * 60)
    parser.add_argument("--min-bucket-samples", type=int, default=40)
    parser.add_argument("--max-candidates", type=int, default=50)
    parser.add_argument("--candidate-block-size", type=int, default=256)
    parser.add_argument("--ridge", type=float, default=1e-4)
    parser.add_argument("--min-improvement", type=float, default=0.0)
    parser.add_argument("--unsigned", action="store_true")
    parser.add_argument("--fit-intercept", action="store_true")
    parser.add_argument("--g-dtype", type=str, default="float16")
    parser.add_argument("--lag-dtype", type=str, default="int8")
    parser.add_argument("--max-nodes", type=int, default=0)
    parser.add_argument("--node-indices", type=str, default=None)
    parser.add_argument("--node-ids", type=str, default=None)
    parser.add_argument("--node-sample", type=str, default="first", choices=["first", "random"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    source_npz = Path(args.source_npz) if args.source_npz else DEFAULT_SOURCE_NPZ
    output_dir = Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_DIR
    if not source_npz.is_absolute():
        source_npz = PROJECT_ROOT / source_npz
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    print_stage("PREPARE BRANCH B GRANGER-GT STANDARD SERIES")
    log(f"PROJECT_ROOT: {PROJECT_ROOT}")
    log(f"SOURCE_NPZ  : {source_npz}")
    log(f"OUTPUT_DIR  : {output_dir}")
    log("This creates Gt = Granger predictive influence, NOT lagged correlation Rt.")

    dataset = load_source_npz(source_npz, primary_feature=str(args.primary_feature))
    segment_ids_all = np.asarray(dataset["segment_ids"], dtype=np.int64)
    node_idx = resolve_node_indices(
        segment_ids=segment_ids_all,
        max_nodes=int(args.max_nodes),
        node_indices_arg=args.node_indices,
        node_ids_arg=args.node_ids,
        node_sample=str(args.node_sample),
        seed=int(args.seed),
    )
    if node_idx is None:
        segment_ids = segment_ids_all
        node_mode = "full"
    else:
        segment_ids = segment_ids_all[node_idx]
        node_mode = f"subset n={len(node_idx)}"

    splits = {name: subset_split(data, node_idx) for name, data in dataset["splits"].items()}
    N = int(splits["train"]["z"].shape[1])
    total_T = sum(int(splits[s]["z"].shape[0]) for s in ["train", "val", "test"])
    est_bytes = total_T * N * N * np.dtype(args.g_dtype).itemsize
    log(f"node mode: {node_mode}")
    log(f"N={N}, total T={total_T}, estimated G series storage={fmt_gb(est_bytes)}")

    if args.dry_run:
        log("DRY RUN: stop before computing graphs.")
        return

    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output exists: {output_dir}. Use --overwrite to replace it.")
        log(f"[CLEAN] removing old output: {output_dir}")
        shutil.rmtree(output_dir)
    ensure_dir(output_dir)
    ensure_dir(output_dir / "tables")

    bucket_table, G_bucket, L_bucket, bucket_summary = estimate_bucket_granger_graphs(
        train_z=splits["train"]["z"],
        train_meta=splits["train"]["meta"],
        args=args,
    )

    bucket_table.to_csv(output_dir / "tables" / "bucket_table.csv", index=False)
    bucket_summary.to_csv(output_dir / "tables" / "bucket_granger_summary.csv", index=False)
    np.save(output_dir / "tables" / "G_bucket.npy", G_bucket.astype(np.dtype(args.g_dtype)))
    np.save(output_dir / "tables" / "L_bucket.npy", L_bucket.astype(np.dtype(args.lag_dtype)))

    for split_name in ["train", "val", "test"]:
        save_split_standard_series(
            split_name=split_name,
            split_data=splits[split_name],
            segment_ids=segment_ids,
            out_dir=output_dir,
            bucket_table=bucket_table,
            G_bucket=G_bucket,
            L_bucket=L_bucket,
            args=args,
        )

    run_summary = {
        "created_at": now_str(),
        "graph_type": "granger_predictive_influence_series",
        "not_correlation": True,
        "project_root": str(PROJECT_ROOT),
        "source_npz": str(source_npz),
        "output_dir": str(output_dir),
        "node_mode": node_mode,
        "n_segments": int(N),
        "total_T": int(total_T),
        "estimated_storage_gb": float(est_bytes / (1024 ** 3)),
        "args": vars(args),
    }
    save_json(run_summary, output_dir / "branchB_granger_series_run_summary.json")
    (output_dir / "README_GRANGER_GT_SERIES.md").write_text(
        "# Branch B Granger-Gt standard series\n\n"
        "This directory replaces lagged-correlation Rt with Granger-style directed predictive influence Gt.\n"
        "It keeps the standard Branch-B file format so true_gt/persistence_gt/ewma_gt/sparse_tvpvar_gt/var_gt can run unchanged.\n",
        encoding="utf-8",
    )
    log("DONE.")
    log(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
