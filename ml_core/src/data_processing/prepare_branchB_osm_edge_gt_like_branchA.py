"""
Prepare Branch B OSM-edge True-Rt / lagged-correlation graph.

This file restores the old/correlation prepare pipeline.

Input:
    ml_core/src/data_processing/outputs/branchA/osm_edge_forecasting_dataset/train_val_test_split.npz

Output:
    ml_core/src/data_processing/outputs/branchB/osm_edge_gt_like_branchA/{train,val,test}/

For every split it saves:
    z.npy
    segment_ids.npy
    timestamps.npy
    G_series_meta.csv
    G_weight_series.npy       # shape: T x N x N, convention target x source
    G_best_lag_series.npy     # shape: T x N x N

Graph semantics:
    G[t, target, source] = lagged correlation between source past and target future
    best lag is selected by largest absolute correlation.

This is a correlation graph, NOT Granger causality.
"""

from __future__ import annotations

import argparse
import json
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
except Exception:
    tqdm = None


EPS = 1e-8
THIS_FILE = Path(__file__).resolve()
DATA_PROCESSING_DIR = THIS_FILE.parent
PROJECT_ROOT = DATA_PROCESSING_DIR.parents[2]
DEFAULT_SOURCE_NPZ = DATA_PROCESSING_DIR / "outputs" / "branchA" / "osm_edge_forecasting_dataset" / "train_val_test_split.npz"
DEFAULT_OUTPUT_DIR = DATA_PROCESSING_DIR / "outputs" / "branchB" / "osm_edge_gt_like_branchA"


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


def save_json(obj: Dict[str, Any], path: Path) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, cls=NumpyJsonEncoder)


def parse_int_list(s: Optional[str]) -> List[int]:
    if s is None:
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


def fmt_gb(nbytes: int) -> str:
    return f"{nbytes / (1024 ** 3):.2f} GB"


def safe_to_datetime_array(arr: np.ndarray) -> pd.Series:
    vals = pd.Series(np.asarray(arr).astype(str))
    # Many tensors store timestamp as "YYYY-MM-DD__HH:MM" or similar.
    vals2 = vals.str.replace("__", " ", regex=False)
    return pd.to_datetime(vals2, errors="coerce")


_TIME_HHMM_COLON = re.compile(r"(?<!\d)([0-2]?\d):([0-5]\d)(?!\d)")
_TIME_HHMM_COMPACT = re.compile(r"(?<!\d)([0-2]\d)([0-5]\d)(?!\d)")


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


def build_meta(timestamps: np.ndarray) -> pd.DataFrame:
    ts_raw = np.asarray(timestamps).astype(str)
    ts = safe_to_datetime_array(ts_raw)

    meta = pd.DataFrame({
        "time_index": np.arange(len(ts_raw), dtype=np.int64),
        "timestamp_raw": ts_raw,
        "timestamp_local": ts,
    })

    # Fallback parse date/time from raw if pd.to_datetime failed.
    date_key = []
    time_minutes = []
    time_set = []
    for raw, parsed in zip(ts_raw, ts):
        if pd.notna(parsed):
            date_key.append(str(parsed.date()))
            time_minutes.append(int(parsed.hour) * 60 + int(parsed.minute))
            time_set.append(f"{int(parsed.hour):02d}:{int(parsed.minute):02d}")
        else:
            parts = str(raw).replace("__", " ").split()
            date_key.append(parts[0] if parts else str(raw))
            tm = parse_time_minutes(raw)
            time_minutes.append(-1 if tm is None else int(tm))
            time_set.append("" if tm is None else f"{tm // 60:02d}:{tm % 60:02d}")

    meta["date_key"] = date_key
    meta["time_minutes"] = np.asarray(time_minutes, dtype=np.int32)
    meta["time_set"] = time_set
    # One session = one calendar date. This prevents horizon pairs crossing days.
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
        if feature_names:
            log(f"[WARN] primary_feature={primary_feature} not found. Use first feature: {feature_names[0]}")
        else:
            log("[WARN] feature_names not found. Use feature index 0.")

    split_map = {}
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
        if t_key in data.files:
            timestamps = np.asarray(data[t_key]).astype(str)
        else:
            timestamps = np.asarray([f"{split}_{i}" for i in range(z.shape[0])])

        split_map[split] = {
            "z": z,
            "timestamps": timestamps,
            "meta": build_meta(timestamps),
        }

    if "model_node_ids" in data.files:
        segment_ids = np.asarray(data["model_node_ids"], dtype=np.int64)
    elif "segment_ids" in data.files:
        segment_ids = np.asarray(data["segment_ids"], dtype=np.int64)
    else:
        segment_ids = np.arange(split_map["train"]["z"].shape[1], dtype=np.int64)

    return {
        "splits": split_map,
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


def lagged_corr_for_block(block: np.ndarray, max_lag: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    block: [M, N]
    returns:
        G[target, source]
        L[target, source] = best lag
    """
    block = np.asarray(block, dtype=np.float32)
    M, N = block.shape
    best_abs = np.zeros((N, N), dtype=np.float32)
    best_val = np.zeros((N, N), dtype=np.float32)
    best_lag = np.zeros((N, N), dtype=np.int16)

    max_lag_eff = min(int(max_lag), max(1, M - 3))
    for lag in range(1, max_lag_eff + 1):
        X = block[:-lag]
        Y = block[lag:]
        if X.shape[0] < 3:
            continue

        Xc = X - X.mean(axis=0, keepdims=True)
        Yc = Y - Y.mean(axis=0, keepdims=True)
        Xstd = Xc.std(axis=0, keepdims=True) + EPS
        Ystd = Yc.std(axis=0, keepdims=True) + EPS
        Xs = Xc / Xstd
        Ys = Yc / Ystd

        # source x target, then transpose to target x source.
        C_source_target = (Xs.T @ Ys) / max(1, Xs.shape[0] - 1)
        C = C_source_target.T.astype(np.float32)
        absC = np.abs(C)
        mask = absC > best_abs
        best_abs[mask] = absC[mask]
        best_val[mask] = C[mask]
        best_lag[mask] = int(lag)

    np.fill_diagonal(best_val, 0.0)
    np.fill_diagonal(best_lag, 0)
    best_val = np.nan_to_num(best_val, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return best_val, best_lag


def compute_g_series(z: np.ndarray, meta: pd.DataFrame, window: int, max_lag: int, dtype: str, lag_dtype: str) -> Tuple[np.ndarray, np.ndarray]:
    z = np.asarray(z, dtype=np.float32)
    T, N = z.shape
    G_series = np.zeros((T, N, N), dtype=np.dtype(dtype))
    L_series = np.zeros((T, N, N), dtype=np.dtype(lag_dtype))

    log(f"Computing lagged-correlation G series: T={T}, N={N}, window={window}, max_lag={max_lag}")
    for idx in maybe_iter(session_groups(meta), total=len(session_groups(meta)), desc="sessions"):
        if len(idx) == 0:
            continue
        session_z = z[idx]
        session_len = len(idx)

        # Session-wide fallback for early positions and short windows.
        fallback_G, fallback_L = lagged_corr_for_block(session_z, max_lag=max_lag)

        for pos, global_t in enumerate(idx):
            start = max(0, pos - int(window) + 1)
            block = session_z[start:pos + 1]
            if block.shape[0] <= min(max_lag + 2, 4):
                G, L = fallback_G, fallback_L
            else:
                G, L = lagged_corr_for_block(block, max_lag=max_lag)
                if not np.any(G):
                    G, L = fallback_G, fallback_L
            G_series[int(global_t)] = G.astype(dtype)
            L_series[int(global_t)] = L.astype(lag_dtype)

    return G_series, L_series


def save_split(out_dir: Path, split: str, split_data: Dict[str, Any], segment_ids: np.ndarray, G: np.ndarray, L: np.ndarray, summary: Dict[str, Any]) -> None:
    d = out_dir / split
    if d.exists():
        shutil.rmtree(d)
    ensure_dir(d)

    z = np.asarray(split_data["z"], dtype=np.float32)
    timestamps = np.asarray(split_data["timestamps"]).astype(str)
    meta = split_data["meta"].copy()

    np.save(d / "z.npy", z)
    np.save(d / "segment_ids.npy", np.asarray(segment_ids, dtype=np.int64))
    np.save(d / "timestamps.npy", timestamps)
    np.save(d / "G_weight_series.npy", G)
    np.save(d / "G_best_lag_series.npy", L)
    meta.to_csv(d / "G_series_meta.csv", index=False)

    split_summary = dict(summary)
    split_summary.update({
        "split": split,
        "z_shape": list(map(int, z.shape)),
        "G_weight_series_shape": list(map(int, G.shape)),
        "G_best_lag_series_shape": list(map(int, L.shape)),
        "n_nonzero_edges_total": int(np.count_nonzero(G)),
    })
    save_json(split_summary, d / "branchB_gt_split_summary.json")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-npz", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--primary-feature", type=str, default="average_speed")
    parser.add_argument("--window", type=int, default=20, help="Lookback window for local lagged correlation.")
    parser.add_argument("--max-lag", type=int, default=9)
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

    print_stage("PREPARE BRANCH B TRUE-RT / LAGGED CORRELATION")
    log(f"PROJECT_ROOT: {PROJECT_ROOT}")
    log(f"SOURCE_NPZ  : {source_npz}")
    log(f"OUTPUT_DIR  : {output_dir}")

    loaded = load_source_npz(source_npz, primary_feature=args.primary_feature)
    segment_ids_all = np.asarray(loaded["segment_ids"], dtype=np.int64)
    node_idx = resolve_node_indices(
        segment_ids=segment_ids_all,
        max_nodes=int(args.max_nodes),
        node_indices_arg=args.node_indices,
        node_ids_arg=args.node_ids,
        node_sample=args.node_sample,
        seed=int(args.seed),
    )
    segment_ids = segment_ids_all if node_idx is None else segment_ids_all[node_idx]

    splits = {
        name: subset_split(data, node_idx)
        for name, data in loaded["splits"].items()
    }
    N = int(len(segment_ids))
    T_total = int(sum(np.asarray(splits[s]["z"]).shape[0] for s in ["train", "val", "test"]))
    est_bytes = T_total * N * N * (np.dtype(args.g_dtype).itemsize + np.dtype(args.lag_dtype).itemsize)
    log(f"node mode: {'full' if node_idx is None else f'subset n={len(node_idx)}'}")
    log(f"N={N}, total T={T_total}, estimated graph storage={fmt_gb(est_bytes)}")

    if args.dry_run:
        log("DRY RUN: stop before computation.")
        return

    if output_dir.exists() and args.overwrite:
        log(f"[CLEAN] removing old output: {output_dir}")
        shutil.rmtree(output_dir)
    ensure_dir(output_dir)

    summary = {
        "created_at": now_str(),
        "mode": "lagged_correlation_true_rt",
        "source_npz": str(source_npz),
        "output_dir": str(output_dir),
        "primary_feature": str(args.primary_feature),
        "primary_feature_index": int(loaded["primary_feature_index"]),
        "window": int(args.window),
        "max_lag": int(args.max_lag),
        "g_dtype": str(args.g_dtype),
        "lag_dtype": str(args.lag_dtype),
        "n_segments": int(N),
        "node_mode": "full" if node_idx is None else f"subset n={len(node_idx)}",
        "semantics": "G[t,target,source] = lagged correlation; best lag by max absolute correlation",
    }

    for split in ["train", "val", "test"]:
        print_stage(f"COMPUTE SPLIT: {split}")
        z = np.asarray(splits[split]["z"], dtype=np.float32)
        G, L = compute_g_series(
            z=z,
            meta=splits[split]["meta"],
            window=int(args.window),
            max_lag=int(args.max_lag),
            dtype=str(args.g_dtype),
            lag_dtype=str(args.lag_dtype),
        )
        save_split(output_dir, split, splits[split], segment_ids, G, L, summary)

    save_json(summary, output_dir / "branchB_true_rt_run_summary.json")
    (output_dir / "README_TRUE_RT.md").write_text(
        "# Branch B True-Rt / lagged correlation\n\n"
        "This folder stores lagged-correlation graphs with convention G[t,target,source].\n"
        "This is not Granger causality.\n",
        encoding="utf-8",
    )
    log("DONE.")


if __name__ == "__main__":
    main()
