# ml_core/src/data_processing/prepare_branchB_osm_edge_gt_like_branchA.py
"""
Prepare Branch B OSM-edge directed GT in the SAME style as the old Kaggle notebook:
05_branchB_prepare_segment_segment_gt_standalone_window10_like_branchA.ipynb

Mục tiêu:
- Input mới: OSM edge tensor từ prepare_osm_edge_forecasting_dataset.py
- Output giữ FORMAT CŨ của Branch B để các file 06_branchB_run_* dễ dùng lại.

Mỗi split train/val/test sẽ có:
    G_weight_series.npy
    G_best_lag_series.npy
    z.npy
    segment_ids.npy
    timestamps.npy
    G_series_meta.csv
    raw_meta.csv
    branchB_gt_split_summary.json

Định nghĩa:
- node = matched OSM directed edge
- block = z[t-window+1 : t+1, :]
- Với mỗi cặp ordered pair (source, target), thử lag = 1..9.
- Chọn lag có abs(lagged correlation) lớn nhất.
- C nội bộ là source x target.
- G lưu ra là target x source để downstream dùng:
      y_target = G @ x_source

Path đặt file:
    UTraffic-ML/ml_core/src/data_processing/prepare_branchB_osm_edge_gt_like_branchA.py

Chạy từ root project:
    cd C:/AI/Thesis/UTraffic-ML

Chạy full:
    python ml_core/src/data_processing/prepare_branchB_osm_edge_gt_like_branchA.py --overwrite

Chạy thử nhẹ:
    python ml_core/src/data_processing/prepare_branchB_osm_edge_gt_like_branchA.py --max-nodes 512 --overwrite

Chỉ kiểm tra shape/dung lượng:
    python ml_core/src/data_processing/prepare_branchB_osm_edge_gt_like_branchA.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


# =============================================================================
# PATH CONFIG
# =============================================================================

THIS_FILE = Path(__file__).resolve()
DATA_PROCESSING_DIR = THIS_FILE.parent
SRC_ROOT = DATA_PROCESSING_DIR.parent
ML_CORE_ROOT = SRC_ROOT.parent
PROJECT_ROOT = ML_CORE_ROOT.parent

DEFAULT_INPUT_DIR = DATA_PROCESSING_DIR / "outputs" / "branchA" / "osm_edge_forecasting_dataset"
DEFAULT_OUTPUT_DIR = DATA_PROCESSING_DIR / "outputs" / "branchB" / "osm_edge_gt_like_branchA"


# =============================================================================
# DEFAULT CONFIG LIKE OLD NOTEBOOK
# =============================================================================

WINDOW = 10
RELATION_LAGS = list(range(1, 10))
MIN_OVERLAP = 4

G_DTYPE = "float16"
BEST_LAG_DTYPE = "int8"

SET_DIAGONAL_WEIGHT = 0.0
SET_DIAGONAL_LAG = 0

PRINT_EVERY = 5


# =============================================================================
# UTILS
# =============================================================================

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
        return super().default(obj)


def now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def print_stage(title: str) -> None:
    print("\n" + "=" * 92)
    print(f"{now_str()} | {title}")
    print("=" * 92)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(obj: Dict[str, Any], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, cls=NumpyJsonEncoder)


def decode_str_array(arr: np.ndarray) -> np.ndarray:
    out = []
    for x in arr:
        if isinstance(x, bytes):
            out.append(x.decode("utf-8"))
        else:
            out.append(str(x))
    return np.array(out)


def get_npz_key(data: np.lib.npyio.NpzFile, candidates: List[str]) -> Optional[str]:
    for k in candidates:
        if k in data.files:
            return k
    return None


def bytes_to_gb(n_bytes: int) -> float:
    return n_bytes / (1024 ** 3)


def fmt_gb(n_bytes: int) -> str:
    return f"{bytes_to_gb(n_bytes):,.2f} GB"


def maybe_iter(iterable, total: Optional[int] = None, desc: str = ""):
    if tqdm is not None:
        return tqdm(iterable, total=total, desc=desc)
    return iterable


def save_readme(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


# =============================================================================
# TIMESTAMP META
# =============================================================================

_TIME_HHMM_COLON = re.compile(r"(?<!\d)([0-2]?\d):([0-5]\d)(?!\d)")
_TIME_HHMM_COMPACT = re.compile(r"(?<!\d)([0-2]\d)([0-5]\d)(?!\d)")


def parse_start_minutes(time_label: Any) -> Optional[int]:
    if time_label is None:
        return None
    s = str(time_label)

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


def parse_timestamp_key(ts: str, fallback_idx: int) -> Dict[str, Any]:
    """
    Hỗ trợ timestamp dạng:
        2024-08-01__06:00-06:15
        2024-08-01__Slot_0600
        2024-08-01 06:00:00
    """
    s = str(ts)

    date_part = None
    time_part = None

    if "__" in s:
        date_part, time_part = s.split("__", 1)
    else:
        # Try pandas parse directly.
        dt_try = pd.to_datetime(s, errors="coerce")
        if pd.notna(dt_try):
            dt_try = pd.Timestamp(dt_try)
            return {
                "timestamp_local": dt_try,
                "date": str(dt_try.date()),
                "session_id": str(dt_try.date()),
                "time_set_id": dt_try.strftime("%H:%M"),
                "slot_index": fallback_idx,
                "tod_minutes": int(dt_try.hour * 60 + dt_try.minute),
            }
        date_part = s
        time_part = f"idx_{fallback_idx}"

    date_dt = pd.to_datetime(date_part, errors="coerce")
    if pd.isna(date_dt):
        # fallback synthetic date by raw index
        date_str = str(date_part)
        base = pd.Timestamp("1970-01-01")
        tod_minutes = fallback_idx
        timestamp_local = base + pd.Timedelta(minutes=int(tod_minutes))
    else:
        date_dt = pd.Timestamp(date_dt)
        date_str = str(date_dt.date())
        tod_minutes = parse_start_minutes(time_part)
        if tod_minutes is None:
            tod_minutes = fallback_idx
        timestamp_local = pd.Timestamp(date_dt.date()) + pd.Timedelta(minutes=int(tod_minutes))

    return {
        "timestamp_local": timestamp_local,
        "date": date_str,
        "session_id": date_str,
        "time_set_id": str(time_part),
        "slot_index": fallback_idx,
        "tod_minutes": int(tod_minutes),
    }


def build_meta_from_timestamps(timestamps: np.ndarray) -> pd.DataFrame:
    rows = []
    for i, ts in enumerate(timestamps):
        row = parse_timestamp_key(str(ts), fallback_idx=i)
        row["raw_timestamp_index"] = i
        rows.append(row)

    meta = pd.DataFrame(rows)
    meta["timestamp_local"] = pd.to_datetime(meta["timestamp_local"])
    meta = meta.sort_values(["timestamp_local", "raw_timestamp_index"]).reset_index(drop=True)

    # Vì tensor đã theo thứ tự thời gian, sau sort cần mapping lại. Thực tế thường không đổi.
    # Ở đây giữ raw_order_position để kiểm tra.
    meta["order_position"] = np.arange(len(meta), dtype=np.int64)
    meta["slot_index"] = meta.groupby("session_id").cumcount().astype(int)
    return meta



def safe_timestamp_value(x: Any) -> pd.Timestamp:
    """
    Convert timestamp value safely, including numpy.str_.
    """
    try:
        return pd.Timestamp(str(x))
    except Exception:
        dt = pd.to_datetime(str(x), errors="coerce")
        if pd.isna(dt):
            return pd.Timestamp("1970-01-01")
        return pd.Timestamp(dt)


def session_index_groups(meta: pd.DataFrame) -> List[np.ndarray]:
    groups = []
    for _, sub in meta.groupby("session_id", sort=False):
        idx = sub.index.to_numpy(dtype=np.int64)
        if len(idx):
            groups.append(idx)
    return groups if groups else [np.arange(len(meta), dtype=np.int64)]


def count_gt_samples(meta: pd.DataFrame, window: int) -> int:
    return int(sum(max(0, len(idx) - window + 1) for idx in session_index_groups(meta)))


# =============================================================================
# LOAD OSM EDGE TENSOR
# =============================================================================

def load_tensor_dataset(input_dir: Path, prefer: str) -> Dict[str, Any]:
    tensor_path = input_dir / "osm_edge_tensor.npz"
    if not tensor_path.exists():
        candidates = sorted(input_dir.glob("*.npz"))
        if not candidates:
            raise FileNotFoundError(f"Không tìm thấy osm_edge_tensor.npz hoặc *.npz trong {input_dir}")
        tensor_path = candidates[-1]

    print(f"Loading tensor dataset: {tensor_path}")
    data = np.load(str(tensor_path), allow_pickle=True, mmap_mode=None)

    if prefer == "norm":
        x_key = get_npz_key(data, ["X_norm", "X_normalized", "X_filled", "X"])
    elif prefer == "filled":
        x_key = get_npz_key(data, ["X_filled", "X_norm", "X_normalized", "X"])
    elif prefer == "raw":
        x_key = get_npz_key(data, ["X_raw", "X", "X_filled", "X_norm"])
    else:
        raise ValueError(f"prefer không hợp lệ: {prefer}")

    if x_key is None:
        raise KeyError(f"Không tìm thấy X trong {tensor_path}. Keys: {data.files}")

    X = data[x_key]
    if X.ndim not in [2, 3]:
        raise ValueError(f"X phải có shape [T,N] hoặc [T,N,F], nhận được {X.shape}")

    if "feature_names" in data.files:
        feature_names = decode_str_array(data["feature_names"])
    elif X.ndim == 3:
        feature_names = np.array([f"feature_{i}" for i in range(X.shape[2])])
    else:
        feature_names = np.array(["value"])

    if "timestamps" in data.files:
        timestamps = decode_str_array(data["timestamps"])
    else:
        timestamps = np.array([str(i) for i in range(X.shape[0])])

    if "model_node_ids" in data.files:
        model_node_ids = data["model_node_ids"].astype(np.int64)
    else:
        model_node_ids = np.arange(X.shape[1], dtype=np.int64)

    edge_key = get_npz_key(data, ["osm_edge_ids", "model_node_osm_edge_id", "model_node_osm_edge_ids"])
    if edge_key is not None:
        osm_edge_ids = decode_str_array(data[edge_key])
    else:
        osm_edge_ids = np.array([str(x) for x in model_node_ids])

    recommended_keep_mask = data["recommended_keep_mask"].astype(bool) if "recommended_keep_mask" in data.files else None

    train_idx = data["train_idx"].astype(np.int64) if "train_idx" in data.files else None
    val_idx = data["val_idx"].astype(np.int64) if "val_idx" in data.files else None
    test_idx = data["test_idx"].astype(np.int64) if "test_idx" in data.files else None

    return {
        "tensor_path": tensor_path,
        "x_key": x_key,
        "X": X,
        "feature_names": feature_names,
        "timestamps": timestamps,
        "model_node_ids": model_node_ids,
        "osm_edge_ids": osm_edge_ids,
        "recommended_keep_mask": recommended_keep_mask,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
    }


def select_feature(X: np.ndarray, feature_names: np.ndarray, feature: str) -> Tuple[np.ndarray, int, str]:
    if X.ndim == 2:
        return X.astype(np.float32, copy=False), 0, "value"

    names = [str(x) for x in feature_names.tolist()]
    if feature in names:
        f_idx = names.index(feature)
    else:
        print(f"WARNING: Không tìm thấy feature='{feature}', dùng feature đầu tiên: {names[0]}")
        f_idx = 0

    return X[:, :, f_idx].astype(np.float32, copy=False), f_idx, names[f_idx]


def apply_node_filter(
    X2: np.ndarray,
    model_node_ids: np.ndarray,
    osm_edge_ids: np.ndarray,
    recommended_keep_mask: Optional[np.ndarray],
    node_filter: str,
    max_nodes: Optional[int],
    input_dir: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    N = X2.shape[1]

    if node_filter == "recommended" and recommended_keep_mask is not None and recommended_keep_mask.shape[0] == N:
        keep = recommended_keep_mask.astype(bool)
    else:
        keep = np.ones(N, dtype=bool)

    selected_positions = np.where(keep)[0]

    if max_nodes is not None and max_nodes > 0 and len(selected_positions) > max_nodes:
        node_quality_path = input_dir / "tables" / "node_quality.csv"
        if node_quality_path.exists():
            q = pd.read_csv(node_quality_path)
            if "model_node_id" in q.columns:
                q = q[q["model_node_id"].isin(model_node_ids[selected_positions])]
                sort_cols, asc = [], []
                if "recommended_keep" in q.columns:
                    sort_cols.append("recommended_keep"); asc.append(False)
                if "valid_ratio" in q.columns:
                    sort_cols.append("valid_ratio"); asc.append(False)
                if "average_speed_std" in q.columns:
                    sort_cols.append("average_speed_std"); asc.append(False)
                if sort_cols:
                    q = q.sort_values(sort_cols, ascending=asc)
                chosen = set(map(int, q["model_node_id"].head(max_nodes).tolist()))
                selected_positions = np.array(
                    [i for i in selected_positions if int(model_node_ids[i]) in chosen],
                    dtype=np.int64,
                )[:max_nodes]
            else:
                selected_positions = selected_positions[:max_nodes]
        else:
            selected_positions = selected_positions[:max_nodes]

    return (
        X2[:, selected_positions],
        model_node_ids[selected_positions],
        osm_edge_ids[selected_positions],
        selected_positions.astype(np.int64),
    )


# =============================================================================
# OLD NOTEBOOK BRANCH B CORE LOGIC
# =============================================================================

def safe_standardize_local(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    mu = X.mean(axis=0, keepdims=True)
    Xc = X - mu

    if X.shape[0] > 1:
        std = Xc.std(axis=0, ddof=1, keepdims=True)
    else:
        std = np.ones((1, X.shape[1]), dtype=np.float32)

    std = np.where(np.isfinite(std) & (std > 1e-6), std, 1.0).astype(np.float32)
    return (Xc / std).astype(np.float32)


def lagged_corr_source_target(block: np.ndarray, lag: int, min_overlap: int) -> np.ndarray:
    """
    Return C[source, target] where source at time τ is compared with target at time τ+lag.
    """
    if block.shape[0] <= lag:
        N = block.shape[1]
        return np.zeros((N, N), dtype=np.float32)

    X_src = block[:-lag]
    Y_tgt = block[lag:]
    n = X_src.shape[0]

    if n < min_overlap:
        N = block.shape[1]
        return np.zeros((N, N), dtype=np.float32)

    Xs = safe_standardize_local(X_src)
    Ys = safe_standardize_local(Y_tgt)

    C = (Xs.T @ Ys) / float(max(n - 1, 1))
    C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(C, -1.0, 1.0).astype(np.float32)


def compute_directed_best_lag_matrix(
    block: np.ndarray,
    relation_lags: Sequence[int],
    min_overlap: int,
    set_diagonal_weight: float,
    set_diagonal_lag: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    For each ordered pair (source, target), choose lag with max absolute lagged correlation.

    Internal C is source x target.
    Saved G is target x source so that:
        (G @ x)[target] = sum_source G[target, source] * x[source]
    This matches downstream forecast feature G_t X_t.
    """
    block = np.asarray(block, dtype=np.float32)
    N = block.shape[1]

    best_abs = np.zeros((N, N), dtype=np.float32)
    best_signed = np.zeros((N, N), dtype=np.float32)
    best_lag = np.zeros((N, N), dtype=np.int8)

    for lag in relation_lags:
        if block.shape[0] - lag < min_overlap:
            continue

        C = lagged_corr_source_target(block, lag=lag, min_overlap=min_overlap)
        absC = np.abs(C)
        mask = absC > best_abs

        if mask.any():
            best_abs[mask] = absC[mask]
            best_signed[mask] = C[mask]
            best_lag[mask] = int(lag)

    G = best_signed.T.astype(np.float32)
    L = best_lag.T.astype(np.int8)

    if set_diagonal_weight is not None:
        np.fill_diagonal(G, float(set_diagonal_weight))
    if set_diagonal_lag is not None:
        np.fill_diagonal(L, int(set_diagonal_lag))

    return G, L


# =============================================================================
# SPLIT BUILDING
# =============================================================================

def make_split_indices(T: int, train_idx, val_idx, test_idx) -> Dict[str, np.ndarray]:
    if train_idx is not None and val_idx is not None and test_idx is not None:
        return {
            "train": train_idx.astype(np.int64),
            "val": val_idx.astype(np.int64),
            "test": test_idx.astype(np.int64),
        }

    train_end = int(0.70 * T)
    val_end = train_end + int(0.15 * T)
    return {
        "train": np.arange(0, train_end, dtype=np.int64),
        "val": np.arange(train_end, val_end, dtype=np.int64),
        "test": np.arange(val_end, T, dtype=np.int64),
    }


def build_split_data(
    X2: np.ndarray,
    timestamps: np.ndarray,
    split_indices: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Tạo z/meta riêng cho split, giữ đúng thứ tự thời gian theo split_indices.
    """
    split_indices = np.asarray(split_indices, dtype=np.int64)
    z = X2[split_indices].astype(np.float32, copy=False)
    ts = timestamps[split_indices].astype(str)

    meta = build_meta_from_timestamps(ts)
    meta["raw_global_timestamp_idx"] = split_indices
    meta["split_local_idx"] = np.arange(len(split_indices), dtype=np.int64)

    return z, ts, meta


def build_gt_series_to_disk(
    z_full: np.ndarray,
    timestamps_full: np.ndarray,
    meta_full: pd.DataFrame,
    segment_ids: np.ndarray,
    split_out: Path,
    split_name: str,
    window: int,
    relation_lags: Sequence[int],
    min_overlap: int,
    g_dtype: str,
    lag_dtype: str,
    overwrite: bool,
    print_every: int,
    set_diagonal_weight: float,
    set_diagonal_lag: int,
    dry_run: bool,
) -> Dict[str, Any]:

    if overwrite and split_out.exists() and not dry_run:
        print(f"[CLEAN] removing old split output: {split_out}")
        shutil.rmtree(split_out)

    split_out.mkdir(parents=True, exist_ok=True)

    T, N = z_full.shape
    n_samples = count_gt_samples(meta_full, window=window)

    if n_samples <= 0:
        raise RuntimeError(f"No valid Branch-B samples for split={split_name}. T={T}, window={window}")

    one_G_bytes = N * N * np.dtype(g_dtype).itemsize
    split_bytes = n_samples * one_G_bytes + n_samples * N * N * np.dtype(lag_dtype).itemsize

    print(f"\n[BUILD] split={split_name} T={T} N={N} n_samples={n_samples} window={window} lags={list(relation_lags)}")
    print(f"Estimated split output: {fmt_gb(split_bytes)}")
    print("G path:", split_out / "G_weight_series.npy")
    print("L path:", split_out / "G_best_lag_series.npy")

    if dry_run:
        return {
            "split": split_name,
            "T_raw": int(T),
            "n_segments": int(N),
            "n_samples": int(n_samples),
            "estimated_split_bytes": int(split_bytes),
            "estimated_split_gb": bytes_to_gb(split_bytes),
            "dry_run": True,
        }

    G_path = split_out / "G_weight_series.npy"
    L_path = split_out / "G_best_lag_series.npy"

    G_mem = np.lib.format.open_memmap(G_path, mode="w+", dtype=g_dtype, shape=(n_samples, N, N))
    L_mem = np.lib.format.open_memmap(L_path, mode="w+", dtype=lag_dtype, shape=(n_samples, N, N))

    rows = []
    valid_raw_indices = []
    cursor = 0
    t0 = time.time()

    groups = session_index_groups(meta_full)
    iterator_groups = groups

    for session_indices in iterator_groups:
        if len(session_indices) < window:
            continue

        for local_end in range(window - 1, len(session_indices)):
            block_idx = session_indices[local_end - window + 1: local_end + 1]
            global_t = int(session_indices[local_end])

            block = np.nan_to_num(
                z_full[block_idx],
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ).astype(np.float32)

            G, L = compute_directed_best_lag_matrix(
                block=block,
                relation_lags=relation_lags,
                min_overlap=min_overlap,
                set_diagonal_weight=set_diagonal_weight,
                set_diagonal_lag=set_diagonal_lag,
            )

            G_mem[cursor] = G.astype(g_dtype)
            L_mem[cursor] = L.astype(lag_dtype)

            src = meta_full.iloc[global_t]
            row = {
                "sample_id": int(cursor),
                "raw_row_idx": int(global_t),
                "window_end_idx": int(global_t),
                "session_step": int(local_end),
                "timestamp_local": safe_timestamp_value(timestamps_full[global_t]),
            }

            for c in [
                "session_id",
                "date",
                "time_set_id",
                "slot_index",
                "tod_minutes",
                "raw_global_timestamp_idx",
                "split_local_idx",
            ]:
                if c in meta_full.columns:
                    row[c] = src[c]

            rows.append(row)
            valid_raw_indices.append(global_t)

            cursor += 1
            if cursor % print_every == 0 or cursor == n_samples:
                elapsed = time.time() - t0
                speed = cursor / max(elapsed, 1e-9)
                eta = (n_samples - cursor) / max(speed, 1e-9)
                print(
                    f"  {split_name}: {cursor}/{n_samples} samples done "
                    f"| speed={speed:.2f} sample/s | ETA={eta/60:.1f} min "
                    f"| elapsed={elapsed/60:.1f} min"
                )

    G_mem.flush()
    L_mem.flush()
    del G_mem, L_mem

    valid_raw_indices = np.asarray(valid_raw_indices, dtype=np.int64)
    meta_out = pd.DataFrame(rows)
    meta_out["timestamp_local"] = pd.to_datetime(meta_out["timestamp_local"])

    # Quan trọng: z.npy cùng sample index space với G_series.
    z_out = z_full[valid_raw_indices].astype(np.float32)
    timestamps_out = np.asarray([safe_timestamp_value(x) for x in timestamps_full[valid_raw_indices]]).astype("datetime64[ns]")

    np.save(split_out / "z.npy", z_out)
    np.save(split_out / "segment_ids.npy", segment_ids.astype(np.int64))
    np.save(split_out / "timestamps.npy", timestamps_out)
    meta_out.to_csv(split_out / "G_series_meta.csv", index=False)
    meta_full.to_csv(split_out / "raw_meta.csv", index=False)

    session_counts = meta_out["session_id"].value_counts(sort=False).to_dict() if "session_id" in meta_out.columns else {}

    summary = {
        "split": split_name,
        "T_raw": int(T),
        "n_segments": int(N),
        "n_samples": int(n_samples),
        "window": int(window),
        "relation_lags": [int(x) for x in relation_lags],
        "min_overlap": int(min_overlap),
        "G_dtype": str(g_dtype),
        "best_lag_dtype": str(lag_dtype),
        "G_shape": [int(n_samples), int(N), int(N)],
        "z_shape": list(map(int, z_out.shape)),
        "min_timestamp": None if len(meta_out) == 0 else str(meta_out["timestamp_local"].min()),
        "max_timestamp": None if len(meta_out) == 0 else str(meta_out["timestamp_local"].max()),
        "min_samples_per_session": int(min(session_counts.values())) if session_counts else None,
        "max_samples_per_session": int(max(session_counts.values())) if session_counts else None,
        "can_evaluate_horizon_9": bool(session_counts and min(session_counts.values()) >= 10),
        "estimated_split_bytes": int(split_bytes),
        "estimated_split_gb": bytes_to_gb(split_bytes),
    }

    save_json(summary, split_out / "branchB_gt_split_summary.json")

    print(f"[DONE] {split_name}: saved required Branch-B files.")
    print("summary:", summary)
    return summary


def validate_required_outputs(common_dir: Path) -> None:
    required = [
        "G_weight_series.npy",
        "G_best_lag_series.npy",
        "z.npy",
        "segment_ids.npy",
        "timestamps.npy",
        "G_series_meta.csv",
    ]

    ok = True
    for split in ["train", "val", "test"]:
        split_dir = common_dir / split
        print("\n[CHECK]", split, split_dir)
        for f in required:
            exists = (split_dir / f).exists()
            print(f"  {f:24s} => {exists}")
            ok = ok and exists

        if (split_dir / "G_series_meta.csv").exists():
            meta = pd.read_csv(split_dir / "G_series_meta.csv")
            if "session_id" in meta.columns:
                vc = meta["session_id"].value_counts(sort=False)
                print("  rows:", len(meta), "| min samples/session:", int(vc.min()), "| max:", int(vc.max()))
                if vc.min() < 10:
                    print("  WARNING: min samples/session < 10, horizon 9 may not be available for every session.")

    if not ok:
        raise RuntimeError("Some required outputs are missing.")

    print("\nAll required Branch-B 05 outputs exist.")


# =============================================================================
# MAIN
# =============================================================================

def prepare(args: argparse.Namespace) -> None:
    input_dir = Path(args.input_dir).resolve()
    out_dir = Path(args.output_dir).resolve()

    print_stage("05 BRANCH B — PREPARE OSM EDGE GT LIKE OLD KAGGLE NOTEBOOK")
    print("PROJECT_ROOT :", PROJECT_ROOT)
    print("INPUT_DIR    :", input_dir)
    print("OUT_DIR      :", out_dir)
    print("WINDOW       :", args.window)
    print("REL_LAGS     :", args.relation_lags)
    print("MIN_OVERLAP  :", args.min_overlap)
    print("FEATURE      :", args.feature)
    print("PREFER       :", args.prefer)
    print("MAX_NODES    :", args.max_nodes)
    print("DRY_RUN      :", args.dry_run)

    ensure_dir(out_dir)

    dataset = load_tensor_dataset(input_dir, prefer=args.prefer)
    X = dataset["X"]
    X2, feature_idx, feature_used = select_feature(X, dataset["feature_names"], args.feature)

    X2, model_node_ids, osm_edge_ids, selected_positions = apply_node_filter(
        X2=X2,
        model_node_ids=dataset["model_node_ids"],
        osm_edge_ids=dataset["osm_edge_ids"],
        recommended_keep_mask=dataset["recommended_keep_mask"],
        node_filter=args.node_filter,
        max_nodes=args.max_nodes,
        input_dir=input_dir,
    )

    T, N = X2.shape
    print("\nLoaded:")
    print("  X key          :", dataset["x_key"])
    print("  Original shape :", X.shape)
    print("  Selected X2    :", X2.shape)
    print("  Feature used   :", feature_used)
    print("  Timestamps     :", len(dataset["timestamps"]))
    print("  Nodes          :", N)

    split_indices = make_split_indices(
        T=T,
        train_idx=dataset["train_idx"],
        val_idx=dataset["val_idx"],
        test_idx=dataset["test_idx"],
    )

    print("\nSplit sizes:")
    for split, idx in split_indices.items():
        print(f"  {split}: {len(idx)} timestamps")

    # Save common node mapping.
    tables_dir = ensure_dir(out_dir / "tables")
    node_map = pd.DataFrame({
        "position_in_G": np.arange(N, dtype=np.int64),
        "model_node_id": model_node_ids.astype(np.int64),
        "osm_edge_id": osm_edge_ids.astype(str),
        "original_position_in_tensor": selected_positions.astype(np.int64),
    })
    node_map.to_csv(tables_dir / "gt_node_mapping.csv", index=False, encoding="utf-8-sig")

    np.savez_compressed(
        out_dir / "gt_common_index.npz",
        model_node_ids=model_node_ids.astype(np.int64),
        osm_edge_ids=osm_edge_ids.astype(str),
        selected_positions=selected_positions.astype(np.int64),
        all_timestamps=dataset["timestamps"].astype(str),
        feature_name=np.array([feature_used]),
        window=np.array([args.window], dtype=np.int64),
        relation_lags=np.array(args.relation_lags, dtype=np.int64),
    )

    summaries = []

    # Estimate first.
    total_est_bytes = 0
    for split in ["train", "val", "test"]:
        z_split, ts_split, meta_split = build_split_data(X2, dataset["timestamps"], split_indices[split])
        n_samples = count_gt_samples(meta_split, window=args.window)
        one_G_bytes = N * N * np.dtype(args.g_dtype).itemsize
        one_L_bytes = N * N * np.dtype(args.lag_dtype).itemsize
        total_est_bytes += n_samples * (one_G_bytes + one_L_bytes)

    print(f"\nEstimated total Branch-B output: {fmt_gb(total_est_bytes)}")
    if bytes_to_gb(total_est_bytes) > args.warn_gb:
        print("WARNING:")
        print(f"  Output dự kiến lớn hơn {args.warn_gb} GB.")
        print("  Nếu muốn test trước, chạy --max-nodes 512.")
        print()

    for split in ["train", "val", "test"]:
        z_split, ts_split, meta_split = build_split_data(X2, dataset["timestamps"], split_indices[split])
        split_out = out_dir / split

        summary = build_gt_series_to_disk(
            z_full=z_split,
            timestamps_full=ts_split,
            meta_full=meta_split,
            segment_ids=model_node_ids,
            split_out=split_out,
            split_name=split,
            window=args.window,
            relation_lags=args.relation_lags,
            min_overlap=args.min_overlap,
            g_dtype=args.g_dtype,
            lag_dtype=args.lag_dtype,
            overwrite=args.overwrite,
            print_every=args.print_every,
            set_diagonal_weight=args.set_diagonal_weight,
            set_diagonal_lag=args.set_diagonal_lag,
            dry_run=args.dry_run,
        )
        summaries.append(summary)

    final_summary = {
        "project_root": str(PROJECT_ROOT),
        "input_dir": str(input_dir),
        "out_dir": str(out_dir),
        "tensor_path": str(dataset["tensor_path"]),
        "x_key": dataset["x_key"],
        "feature_requested": args.feature,
        "feature_used": feature_used,
        "feature_idx": int(feature_idx),
        "window": int(args.window),
        "relation_lags": [int(x) for x in args.relation_lags],
        "min_overlap": int(args.min_overlap),
        "g_dtype": args.g_dtype,
        "lag_dtype": args.lag_dtype,
        "num_timestamps": int(T),
        "num_nodes": int(N),
        "splits": summaries,
        "output_format": "same as old Kaggle Branch-B notebook",
        "required_files_per_split": [
            "G_weight_series.npy",
            "G_best_lag_series.npy",
            "z.npy",
            "segment_ids.npy",
            "timestamps.npy",
            "G_series_meta.csv",
            "raw_meta.csv",
        ],
    }
    save_json(final_summary, out_dir / "branchB_gt_prepare_summary.json")
    save_json(final_summary, out_dir / "metadata.json")

    readme = f"""# Branch B OSM Edge GT — Like old Kaggle notebook

This folder keeps the same output format as:
05_branchB_prepare_segment_segment_gt_standalone_window10_like_branchA.ipynb

## Definition

- One model node = one matched OSM directed edge.
- Feature used = {feature_used}
- Window = {args.window}
- Relation lags = {args.relation_lags}
- For every ordered pair source -> target, choose the lag with max absolute lagged correlation.
- G is saved as target x source so downstream can use `G @ x`.

## Required files per split

Each split folder train/val/test contains:

- G_weight_series.npy
- G_best_lag_series.npy
- z.npy
- segment_ids.npy
- timestamps.npy
- G_series_meta.csv
- raw_meta.csv
- branchB_gt_split_summary.json

## Load example

```python
from pathlib import Path
import numpy as np
import pandas as pd

split_dir = Path(r"{out_dir}") / "train"

G = np.load(split_dir / "G_weight_series.npy", mmap_mode="r")
L = np.load(split_dir / "G_best_lag_series.npy", mmap_mode="r")
z = np.load(split_dir / "z.npy")
segment_ids = np.load(split_dir / "segment_ids.npy")
meta = pd.read_csv(split_dir / "G_series_meta.csv")

print(G.shape, L.shape, z.shape)
```

## Notes

For full N≈3697, output can be large but follows the same working Kaggle format.
"""
    save_readme(out_dir / "README_outputs.md", readme)

    if not args.dry_run:
        validate_required_outputs(out_dir)

    print_stage("DONE")
    print("Output dir:", out_dir)
    print("Summary:", out_dir / "branchB_gt_prepare_summary.json")
    print("README :", out_dir / "README_outputs.md")


# =============================================================================
# CLI
# =============================================================================

def parse_lags(s: str) -> List[int]:
    out = []
    for part in str(s).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(list(range(int(a), int(b) + 1)))
        else:
            out.append(int(part))
    return sorted(set(out))


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Prepare Branch B OSM-edge GT like old Kaggle notebook.")

    p.add_argument("--input-dir", type=str, default=str(DEFAULT_INPUT_DIR))
    p.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))

    p.add_argument("--feature", type=str, default="average_speed")
    p.add_argument("--prefer", type=str, default="norm", choices=["norm", "filled", "raw"])
    p.add_argument("--window", type=int, default=WINDOW)
    p.add_argument("--relation-lags", type=parse_lags, default=RELATION_LAGS)
    p.add_argument("--min-overlap", type=int, default=MIN_OVERLAP)

    p.add_argument("--g-dtype", type=str, default=G_DTYPE)
    p.add_argument("--lag-dtype", type=str, default=BEST_LAG_DTYPE)

    p.add_argument("--set-diagonal-weight", type=float, default=SET_DIAGONAL_WEIGHT)
    p.add_argument("--set-diagonal-lag", type=int, default=SET_DIAGONAL_LAG)

    p.add_argument("--node-filter", type=str, default="all", choices=["all", "recommended"])
    p.add_argument("--max-nodes", type=int, default=None)

    p.add_argument("--print-every", type=int, default=PRINT_EVERY)
    p.add_argument("--warn-gb", type=float, default=50.0)

    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--overwrite", action="store_true")

    return p


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    prepare(args)


if __name__ == "__main__":
    main()
