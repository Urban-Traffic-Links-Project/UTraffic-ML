# ml_core/src/data_processing/prepare_osm_edge_forecasting_dataset.py
"""
Prepare OSM-edge forecasting dataset for Branch A / Branch B.

Mục tiêu:
- Đọc output từ run_osm_match_offline.py.
- Dùng mapping TomTom segment -> matched OSM directed edge.
- Aggregate dữ liệu TomTom theo thời gian sang matched OSM edge.
- Tạo tensor:
      X[t, n, f]
  trong đó:
      t = timestamp
      n = matched OSM directed edge
      f = traffic feature
- Xuất dataset chuẩn để các pipeline Branch A/B dùng chung.
- Xuất thêm bảng/biểu đồ thống kê để phân tích chất lượng dữ liệu.

Đặt file tại:
    UTraffic-ML/ml_core/src/data_processing/prepare_osm_edge_forecasting_dataset.py

Chạy từ thư mục gốc project:
    cd C:/AI/Thesis/UTraffic-ML
    python ml_core/src/data_processing/prepare_osm_edge_forecasting_dataset.py

Input mặc định:
    UTraffic-ML/ml_core/src/data_processing/outputs/branchA/

Output mặc định:
    UTraffic-ML/ml_core/src/data_processing/outputs/branchA/osm_edge_forecasting_dataset/

Ghi chú quan trọng:
- Node trong dataset này = matched OSM directed edge.
- Data động lấy từ TomTom.
- Địa lý/topology lấy từ OSM.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# Optional libraries
# =============================================================================

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


# =============================================================================
# Path auto-detect
# =============================================================================

THIS_FILE = Path(__file__).resolve()

# Expected:
#   UTraffic-ML/ml_core/src/data_processing/prepare_osm_edge_forecasting_dataset.py
DATA_PROCESSING_DIR = THIS_FILE.parent
SRC_ROOT = DATA_PROCESSING_DIR.parent
ML_CORE_ROOT = SRC_ROOT.parent
PROJECT_ROOT = ML_CORE_ROOT.parent

DEFAULT_INPUT_BASE = DATA_PROCESSING_DIR / "outputs" / "branchA"
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_BASE / "osm_edge_forecasting_dataset"


# =============================================================================
# User config
# =============================================================================

DEFAULT_FEATURES = [
    "average_speed",
    "harmonic_average_speed",
    "median_speed",
    "std_speed",
    "average_travel_time",
    "travel_time_ratio",
    "congestion_index",
    "speed_limit_ratio",
    "sample_size",
]

PRIMARY_FEATURE = "average_speed"

# Đây là ngưỡng dùng để khuyến nghị giữ/lược node, KHÔNG tự động xóa node.
MIN_VALID_RATIO_RECOMMENDED = 0.80

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42


# =============================================================================
# Logging utilities
# =============================================================================

def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


class Timer:
    def __init__(self, name: str):
        self.name = name
        self.start = time.time()

    def done(self) -> str:
        elapsed = time.time() - self.start
        return f"{self.name} completed in {elapsed:.2f}s"


def log(msg: str) -> None:
    print(f"[{now_str()}] {msg}", flush=True)


def log_section(title: str) -> None:
    print("\n" + "=" * 88, flush=True)
    print(title, flush=True)
    print("=" * 88, flush=True)


def iter_progress(iterable, total: Optional[int] = None, desc: str = ""):
    if tqdm is not None:
        return tqdm(iterable, total=total, desc=desc)
    return iterable


# =============================================================================
# File utilities
# =============================================================================

def latest_file(folder: Path, pattern: str) -> Path:
    files = sorted(folder.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No file matching {pattern} in {folder}")
    return files[-1]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(obj: Dict[str, Any], path: Path) -> None:
    class NpEncoder(json.JSONEncoder):
        def default(self, o: Any):
            if isinstance(o, np.integer):
                return int(o)
            if isinstance(o, np.floating):
                return float(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
            if isinstance(o, Path):
                return str(o)
            return super().default(o)

    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, cls=NpEncoder)


def read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_npz_metadata(npz_path: Path) -> Dict[str, Any]:
    with np.load(str(npz_path), allow_pickle=True) as data:
        if "_metadata" not in data.files:
            return {}
        raw = data["_metadata"][0]
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        return json.loads(str(raw))


def list_npz_columns(npz_path: Path) -> List[str]:
    with np.load(str(npz_path), allow_pickle=True) as data:
        return [k for k in data.files if k != "_metadata"]


def save_npz_compressed(path: Path, **kwargs) -> None:
    ensure_dir(path.parent)
    log(f"Saving NPZ: {path}")
    t = Timer("save_npz_compressed")
    np.savez_compressed(str(path), **kwargs)
    log(t.done())


def memory_mb(arr: np.ndarray) -> float:
    return arr.nbytes / 1024 / 1024


# =============================================================================
# Time utilities
# =============================================================================

_TIME_HHMM_COLON = re.compile(r"(?<!\d)([0-2]?\d):([0-5]\d)(?!\d)")
_TIME_HHMM_COMPACT = re.compile(r"(?<!\d)([0-2]\d)([0-5]\d)(?!\d)")


def parse_start_minutes(time_label: Any) -> Optional[int]:
    if time_label is None:
        return None
    s = str(time_label)

    m = _TIME_HHMM_COLON.search(s)
    if m:
        hh = int(m.group(1))
        mm = int(m.group(2))
        if 0 <= hh <= 23:
            return hh * 60 + mm

    m = _TIME_HHMM_COMPACT.search(s)
    if m:
        hh = int(m.group(1))
        mm = int(m.group(2))
        if 0 <= hh <= 23:
            return hh * 60 + mm

    return None


# =============================================================================
# Data loading
# =============================================================================

def locate_inputs(input_base: Path) -> Dict[str, Path]:
    paths = {
        "summary_json": input_base / "match_summary" / "matching_summary.json",
        "edge_meta_csv": input_base / "match_summary" / "matched_osm_edge_metadata.csv",
        "match_table_csv_gz": input_base / "match_summary" / "tomtom_to_osm_edge_matches.csv.gz",
        "traffic_features_npz": latest_file(input_base / "traffic_features", "traffic_features_*.npz"),
    }

    optional_graph_dir = input_base / "graph_structure"
    optional_osm_dir = input_base / "osm_graph"

    if optional_graph_dir.exists():
        paths["graph_structure_npz"] = latest_file(optional_graph_dir, "graph_structure_*.npz")
    if optional_osm_dir.exists():
        paths["osm_graph_npz"] = latest_file(optional_osm_dir, "osm_graph_*.npz")

    for key, p in paths.items():
        if not p.exists():
            raise FileNotFoundError(f"Missing input {key}: {p}")

    return paths


def load_traffic_npz_arrays(npz_path: Path, features: List[str]) -> Dict[str, np.ndarray]:
    log(f"Loading traffic NPZ: {npz_path}")
    t = Timer("load_traffic_npz_arrays")

    available_cols = list_npz_columns(npz_path)
    log(f"Available NPZ columns: {available_cols}")

    required_base = ["segment_id", "time_set"]
    date_col = "date_from" if "date_from" in available_cols else "date_range"
    if date_col not in available_cols:
        raise ValueError("Cannot find date column. Expected 'date_from' or 'date_range'.")

    used_features = [f for f in features if f in available_cols]
    missing_features = [f for f in features if f not in available_cols]

    if missing_features:
        log(f"WARNING: Missing feature columns will be skipped: {missing_features}")

    needed = required_base + [date_col] + used_features

    arrays: Dict[str, np.ndarray] = {}
    with np.load(str(npz_path), allow_pickle=True) as data:
        n_rows = len(data["segment_id"])
        log(f"Traffic rows: {n_rows:,}")

        for col in needed:
            arr = data[col]
            if col in used_features:
                arr = pd.to_numeric(pd.Series(arr), errors="coerce").to_numpy(dtype=np.float32)
            else:
                arr = arr.astype(str)
            arrays[col] = arr
            log(f"  loaded {col:<25} shape={arr.shape}, dtype={arr.dtype}")

    arrays["_date_col"] = np.array([date_col])
    arrays["_used_features"] = np.array(used_features)

    log(t.done())
    return arrays


def build_timestamp_index(date_values: np.ndarray, time_values: np.ndarray) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Build sorted timestamp_idx for each row.

    Returns:
        timestamp_idx_arr: [num_rows]
        timestamp_table  : columns timestamp_idx, timestamp_key, date_key, time_set, time_minutes
    """
    log("Building timestamp index...")
    t = Timer("build_timestamp_index")

    date_s = pd.Series(date_values, dtype="string")
    time_s = pd.Series(time_values, dtype="string")
    key_s = date_s + "__" + time_s

    codes, uniques = pd.factorize(key_s, sort=False)
    timestamp_table = pd.DataFrame({"timestamp_key": uniques.astype(str)})
    split = timestamp_table["timestamp_key"].str.split("__", n=1, expand=True)
    timestamp_table["date_key"] = split[0].astype(str)
    timestamp_table["time_set"] = split[1].astype(str)
    timestamp_table["date_parsed"] = pd.to_datetime(timestamp_table["date_key"], errors="coerce")
    timestamp_table["time_minutes"] = timestamp_table["time_set"].apply(parse_start_minutes)

    # Fallback if some time labels cannot be parsed.
    if timestamp_table["time_minutes"].isna().any():
        log("WARNING: Some time_set labels cannot be parsed. Fallback to lexical order for those values.")
        time_order = {v: i for i, v in enumerate(sorted(timestamp_table["time_set"].unique()))}
        fallback = timestamp_table["time_set"].map(time_order).astype(float)
        timestamp_table["time_minutes"] = timestamp_table["time_minutes"].fillna(fallback)

    # Sort by date then time.
    timestamp_table = timestamp_table.sort_values(
        ["date_parsed", "date_key", "time_minutes", "time_set"],
        kind="stable",
    ).reset_index(drop=True)
    timestamp_table["timestamp_idx"] = np.arange(len(timestamp_table), dtype=np.int64)

    key_to_idx = dict(zip(timestamp_table["timestamp_key"], timestamp_table["timestamp_idx"]))
    timestamp_idx_arr = key_s.map(key_to_idx).to_numpy(dtype=np.int64)

    timestamp_table = timestamp_table[
        ["timestamp_idx", "timestamp_key", "date_key", "time_set", "time_minutes"]
    ]

    log(f"Unique timestamps: {len(timestamp_table):,}")
    log(t.done())
    return timestamp_idx_arr, timestamp_table


def build_segment_to_model_mapping(match_df: pd.DataFrame, edge_meta: pd.DataFrame) -> pd.DataFrame:
    """
    Build mapping:
        TomTom segment_id -> model_node_id

    Nếu 1 TomTom segment cover nhiều OSM edges, nó sẽ có nhiều dòng mapping.
    """
    log("Building segment_id -> model_node_id mapping...")
    t = Timer("build_segment_to_model_mapping")

    match_df = match_df.copy()
    edge_meta = edge_meta.copy()

    match_df["segment_id"] = match_df["segment_id"].astype(str)

    for c in ["osm_u_idx", "osm_v_idx"]:
        match_df[c] = pd.to_numeric(match_df[c], errors="coerce").astype("Int64")
        edge_meta[c] = pd.to_numeric(edge_meta[c], errors="coerce").astype("Int64")

    needed_edge_cols = ["model_node_id", "osm_edge_id", "osm_u_idx", "osm_v_idx"]
    missing = [c for c in needed_edge_cols if c not in edge_meta.columns]
    if missing:
        raise ValueError(f"edge_meta missing columns: {missing}")

    mapping_df = (
        match_df[["segment_id", "osm_u_idx", "osm_v_idx"]]
        .dropna()
        .drop_duplicates()
        .merge(edge_meta[needed_edge_cols], on=["osm_u_idx", "osm_v_idx"], how="inner")
        .drop_duplicates(subset=["segment_id", "model_node_id"])
    )

    mapping_df["model_node_id"] = mapping_df["model_node_id"].astype(np.int64)
    mapping_df["segment_id"] = mapping_df["segment_id"].astype(str)

    log(f"Mapping rows: {len(mapping_df):,}")
    log(f"Unique TomTom segments in mapping: {mapping_df['segment_id'].nunique():,}")
    log(f"Unique model nodes in mapping: {mapping_df['model_node_id'].nunique():,}")
    log(t.done())

    return mapping_df


# =============================================================================
# Tensor creation
# =============================================================================

def build_osm_edge_tensor(
    arrays: Dict[str, np.ndarray],
    timestamp_idx_arr: np.ndarray,
    timestamp_table: pd.DataFrame,
    mapping_df: pd.DataFrame,
    edge_meta: pd.DataFrame,
    features: List[str],
    primary_feature: str,
    output_dir: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    """
    Build X_raw [T, N, F] from TomTom rows and mapping to OSM edges.

    Aggregation:
    - For sample_size: sum.
    - Other features: mean.
    """
    log_section("STAGE 1 — BUILD OSM EDGE TENSOR X[t, n, f]")

    segment_ids = arrays["segment_id"].astype(str)
    used_features = list(arrays["_used_features"].astype(str))

    if primary_feature not in used_features:
        raise ValueError(f"primary_feature={primary_feature} is not available in used_features={used_features}")

    model_node_ids = edge_meta.sort_values("model_node_id")["model_node_id"].to_numpy(dtype=np.int64)
    node_to_pos = {int(nid): i for i, nid in enumerate(model_node_ids)}
    n_nodes = len(model_node_ids)
    n_timestamps = len(timestamp_table)
    n_features = len(used_features)

    log(f"Timestamps T : {n_timestamps:,}")
    log(f"Model nodes N: {n_nodes:,}")
    log(f"Features F   : {n_features:,} -> {used_features}")

    X = np.full((n_timestamps, n_nodes, n_features), np.nan, dtype=np.float32)

    # Sort row indices by timestamp for efficient per-slot processing.
    log("Sorting rows by timestamp...")
    order = np.argsort(timestamp_idx_arr, kind="stable")
    counts = np.bincount(timestamp_idx_arr, minlength=n_timestamps)
    starts = np.concatenate([[0], np.cumsum(counts[:-1])])
    ends = np.cumsum(counts)

    mapping_small = mapping_df[["segment_id", "model_node_id"]].copy()
    mapping_small["segment_id"] = mapping_small["segment_id"].astype(str)

    timestamp_summaries: List[Dict[str, Any]] = []
    primary_idx = used_features.index(primary_feature)

    loop_iter = range(n_timestamps)
    for t_idx in iter_progress(loop_iter, total=n_timestamps, desc="Aggregating timestamps"):
        row_idx = order[starts[t_idx]:ends[t_idx]]

        if len(row_idx) == 0:
            timestamp_summaries.append({
                "timestamp_idx": int(t_idx),
                "input_rows": 0,
                "expanded_rows_after_mapping": 0,
                "observed_nodes": 0,
                "coverage": 0.0,
            })
            continue

        slot = pd.DataFrame({"segment_id": segment_ids[row_idx]})
        for f in used_features:
            slot[f] = arrays[f][row_idx]

        expanded = slot.merge(mapping_small, on="segment_id", how="inner")

        if expanded.empty:
            timestamp_summaries.append({
                "timestamp_idx": int(t_idx),
                "input_rows": int(len(slot)),
                "expanded_rows_after_mapping": 0,
                "observed_nodes": 0,
                "coverage": 0.0,
            })
            continue

        agg_dict: Dict[str, str] = {}
        for f in used_features:
            agg_dict[f] = "sum" if f == "sample_size" else "mean"

        grouped = expanded.groupby("model_node_id", as_index=False).agg(agg_dict)
        grouped["node_pos"] = grouped["model_node_id"].map(node_to_pos)
        grouped = grouped.dropna(subset=["node_pos"])
        grouped["node_pos"] = grouped["node_pos"].astype(np.int64)

        pos = grouped["node_pos"].to_numpy(dtype=np.int64)
        for f_idx, f in enumerate(used_features):
            X[t_idx, pos, f_idx] = grouped[f].to_numpy(dtype=np.float32)

        observed = int(np.isfinite(X[t_idx, :, primary_idx]).sum())
        timestamp_summaries.append({
            "timestamp_idx": int(t_idx),
            "input_rows": int(len(slot)),
            "expanded_rows_after_mapping": int(len(expanded)),
            "observed_nodes": observed,
            "coverage": float(observed / max(n_nodes, 1)),
        })

        if tqdm is None and ((t_idx + 1) % 25 == 0 or t_idx + 1 == n_timestamps):
            log(f"  processed {t_idx + 1}/{n_timestamps} timestamps")

    timestamp_summary_df = pd.DataFrame(timestamp_summaries)
    timestamp_summary_df = timestamp_summary_df.merge(timestamp_table, on="timestamp_idx", how="left")
    timestamp_summary_df.to_csv(output_dir / "tables" / "timestamp_coverage_summary.csv", index=False, encoding="utf-8-sig")

    valid_mask = np.isfinite(X[:, :, primary_idx])
    valid_ratio = valid_mask.mean(axis=0)

    log(f"X_raw shape: {X.shape}, memory={memory_mb(X):.2f} MB")
    log(f"Coverage mean/min/max by timestamp: "
        f"{timestamp_summary_df['coverage'].mean():.4f} / "
        f"{timestamp_summary_df['coverage'].min():.4f} / "
        f"{timestamp_summary_df['coverage'].max():.4f}")
    log(f"Node valid ratio mean/min/max: "
        f"{valid_ratio.mean():.4f} / {valid_ratio.min():.4f} / {valid_ratio.max():.4f}")

    return X, valid_mask, model_node_ids, timestamp_summary_df, pd.DataFrame({
        "model_node_id": model_node_ids,
        "valid_ratio": valid_ratio,
        "missing_ratio": 1.0 - valid_ratio,
    })


def fill_missing_values(X_raw: np.ndarray, feature_names: List[str], output_dir: Path) -> np.ndarray:
    """
    Fill missing values along time axis for each node and feature:
    interpolate -> ffill -> bfill -> global mean -> 0.
    """
    log_section("STAGE 2 — FILL MISSING VALUES")
    t = Timer("fill_missing_values")

    X_filled = np.empty_like(X_raw, dtype=np.float32)

    for f_idx, fname in enumerate(feature_names):
        log(f"Filling feature {f_idx + 1}/{len(feature_names)}: {fname}")

        df = pd.DataFrame(X_raw[:, :, f_idx])
        df = df.interpolate(axis=0, limit_direction="both")
        df = df.ffill().bfill()

        # Fill columns still NaN by feature global mean.
        global_mean = np.nanmean(X_raw[:, :, f_idx])
        if not np.isfinite(global_mean):
            global_mean = 0.0
        df = df.fillna(float(global_mean)).fillna(0.0)

        X_filled[:, :, f_idx] = df.to_numpy(dtype=np.float32)

    log(f"X_filled shape: {X_filled.shape}, memory={memory_mb(X_filled):.2f} MB")
    log(t.done())
    return X_filled


def normalize_using_train(
    X_filled: np.ndarray,
    train_end: int,
    eps: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalize per feature using train split statistics.

    Mean/std are computed over axes (T, N) on train only.
    """
    log_section("STAGE 3 — NORMALIZE USING TRAIN STATISTICS")
    t = Timer("normalize_using_train")

    train = X_filled[:train_end]
    mean = train.mean(axis=(0, 1), keepdims=True).astype(np.float32)  # [1,1,F]
    std = train.std(axis=(0, 1), keepdims=True).astype(np.float32)
    std = np.where(std < eps, 1.0, std).astype(np.float32)

    X_norm = ((X_filled - mean) / std).astype(np.float32)

    log(f"X_norm shape: {X_norm.shape}, memory={memory_mb(X_norm):.2f} MB")
    log(f"Feature train mean: {mean.reshape(-1)}")
    log(f"Feature train std : {std.reshape(-1)}")
    log(t.done())

    return X_norm, mean.reshape(-1), std.reshape(-1)


def compute_node_quality(
    X_raw: np.ndarray,
    valid_mask: np.ndarray,
    model_node_ids: np.ndarray,
    edge_meta: pd.DataFrame,
    feature_names: List[str],
    primary_feature: str,
    min_valid_ratio: float,
) -> pd.DataFrame:
    log_section("STAGE 4 — COMPUTE NODE QUALITY TABLE")
    t = Timer("compute_node_quality")

    primary_idx = feature_names.index(primary_feature)
    quality = pd.DataFrame({
        "model_node_id": model_node_ids,
        "valid_ratio": valid_mask.mean(axis=0),
        "missing_ratio": 1.0 - valid_mask.mean(axis=0),
        "recommended_keep": valid_mask.mean(axis=0) >= min_valid_ratio,
        f"{primary_feature}_mean": np.nanmean(X_raw[:, :, primary_idx], axis=0),
        f"{primary_feature}_std": np.nanstd(X_raw[:, :, primary_idx], axis=0),
    })

    if "sample_size" in feature_names:
        sidx = feature_names.index("sample_size")
        quality["sample_size_mean"] = np.nanmean(X_raw[:, :, sidx], axis=0)
        quality["sample_size_sum"] = np.nansum(X_raw[:, :, sidx], axis=0)

    # Add static metadata.
    meta_cols = [
        "model_node_id",
        "osm_edge_id",
        "osm_u_id",
        "osm_v_id",
        "osm_u_idx",
        "osm_v_idx",
        "u_lat",
        "u_lon",
        "v_lat",
        "v_lon",
        "mid_lat",
        "mid_lon",
        "osm_length_m",
        "osm_maxspeed",
        "osm_lanes",
        "osm_highway_type",
        "tomtom_unique_segments",
        "tomtom_match_rows",
        "mean_match_dist_m",
        "median_match_dist_m",
        "street_names",
        "tomtom_segment_ids",
    ]
    meta_cols = [c for c in meta_cols if c in edge_meta.columns]
    quality = quality.merge(edge_meta[meta_cols], on="model_node_id", how="left")

    log(f"Recommended keep nodes: {int(quality['recommended_keep'].sum()):,}/{len(quality):,}")
    log(t.done())

    return quality


def compute_network_time_summary(
    X_raw: np.ndarray,
    valid_mask: np.ndarray,
    timestamp_table: pd.DataFrame,
    feature_names: List[str],
) -> pd.DataFrame:
    log_section("STAGE 5 — COMPUTE NETWORK TIME SUMMARY")
    rows = []

    for t_idx in range(X_raw.shape[0]):
        row: Dict[str, Any] = {
            "timestamp_idx": int(t_idx),
            "observed_nodes": int(valid_mask[t_idx].sum()),
            "coverage": float(valid_mask[t_idx].mean()),
        }
        for f_idx, fname in enumerate(feature_names):
            vals = X_raw[t_idx, :, f_idx]
            row[f"{fname}_mean"] = float(np.nanmean(vals)) if np.isfinite(vals).any() else np.nan
            row[f"{fname}_std"] = float(np.nanstd(vals)) if np.isfinite(vals).any() else np.nan
        rows.append(row)

    out = pd.DataFrame(rows).merge(timestamp_table, on="timestamp_idx", how="left")
    return out


def make_splits(T: int, train_ratio: float, val_ratio: float) -> Tuple[int, int, np.ndarray, np.ndarray, np.ndarray]:
    train_end = max(1, int(T * train_ratio))
    val_end = min(T, train_end + max(1, int(T * val_ratio)))

    train_idx = np.arange(0, train_end, dtype=np.int64)
    val_idx = np.arange(train_end, val_end, dtype=np.int64)
    test_idx = np.arange(val_end, T, dtype=np.int64)

    return train_end, val_end, train_idx, val_idx, test_idx


# =============================================================================
# Plots
# =============================================================================

def save_plot(path: Path) -> None:
    if plt is None:
        return
    ensure_dir(path.parent)
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    log(f"Saved plot: {path}")


def make_basic_plots(
    output_dir: Path,
    node_quality: pd.DataFrame,
    network_time_summary: pd.DataFrame,
    primary_feature: str,
) -> None:
    if plt is None:
        log("matplotlib not available. Skip plots.")
        return

    log_section("STAGE 6 — MAKE BASIC PLOTS")

    plot_dir = ensure_dir(output_dir / "plots")

    plt.figure(figsize=(8, 4))
    node_quality["valid_ratio"].hist(bins=50)
    plt.title("Node valid ratio distribution")
    plt.xlabel("Valid ratio")
    plt.ylabel("Count")
    save_plot(plot_dir / "node_valid_ratio_distribution.png")

    if f"{primary_feature}_mean" in node_quality.columns:
        plt.figure(figsize=(8, 4))
        node_quality[f"{primary_feature}_mean"].dropna().hist(bins=60)
        plt.title(f"Node mean {primary_feature} distribution")
        plt.xlabel(f"Mean {primary_feature}")
        plt.ylabel("Count")
        save_plot(plot_dir / f"node_mean_{primary_feature}_distribution.png")

    if f"{primary_feature}_mean" in network_time_summary.columns:
        plt.figure(figsize=(12, 4))
        plt.plot(network_time_summary["timestamp_idx"], network_time_summary[f"{primary_feature}_mean"])
        plt.title(f"Network mean {primary_feature} over time")
        plt.xlabel("Timestamp index")
        plt.ylabel(f"Mean {primary_feature}")
        save_plot(plot_dir / f"network_mean_{primary_feature}_over_time.png")

    plt.figure(figsize=(12, 4))
    plt.plot(network_time_summary["timestamp_idx"], network_time_summary["coverage"])
    plt.title("Network coverage over time")
    plt.xlabel("Timestamp index")
    plt.ylabel("Coverage")
    save_plot(plot_dir / "network_coverage_over_time.png")

    if "time_set" in network_time_summary.columns and f"{primary_feature}_mean" in network_time_summary.columns:
        grouped = list(network_time_summary.groupby("time_set", sort=False))
        labels = [str(k) for k, _ in grouped]
        values = [g[f"{primary_feature}_mean"].dropna().values for _, g in grouped]

        plt.figure(figsize=(14, 5))
        plt.boxplot(values, labels=labels, showfliers=False)
        plt.title(f"Network mean {primary_feature} by time slot")
        plt.xlabel("Time slot")
        plt.ylabel(f"Mean {primary_feature}")
        plt.xticks(rotation=90)
        save_plot(plot_dir / f"network_mean_{primary_feature}_by_time_slot_boxplot.png")


# =============================================================================
# Main
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare OSM-edge forecasting tensor dataset for Branch A/B."
    )
    parser.add_argument(
        "--input-base",
        type=str,
        default=str(DEFAULT_INPUT_BASE),
        help="Path to outputs/branchA generated by run_osm_match_offline.py",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory for forecasting dataset",
    )
    parser.add_argument(
        "--features",
        type=str,
        default=",".join(DEFAULT_FEATURES),
        help="Comma-separated feature list",
    )
    parser.add_argument(
        "--primary-feature",
        type=str,
        default=PRIMARY_FEATURE,
        help="Feature used to define valid_mask and coverage",
    )
    parser.add_argument(
        "--min-valid-ratio",
        type=float,
        default=MIN_VALID_RATIO_RECOMMENDED,
        help="Recommended threshold for node keep mask. Does not drop nodes.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=TRAIN_RATIO,
        help="Temporal train ratio",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=VAL_RATIO,
        help="Temporal validation ratio",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable plot generation",
    )
    parser.add_argument(
        "--save-raw",
        action="store_true",
        help="Also save X_raw with NaNs in osm_edge_tensor.npz. This increases file size.",
    )
    return parser.parse_args()


def main() -> None:
    np.random.seed(RANDOM_SEED)

    args = parse_args()

    input_base = Path(args.input_base).resolve()
    output_dir = Path(args.output_dir).resolve()
    tables_dir = ensure_dir(output_dir / "tables")
    matrices_dir = ensure_dir(output_dir / "matrices")
    ensure_dir(output_dir)

    features = [x.strip() for x in args.features.split(",") if x.strip()]
    primary_feature = args.primary_feature

    log_section("PREPARE OSM EDGE FORECASTING DATASET")
    log(f"Project root     : {PROJECT_ROOT}")
    log(f"Input base       : {input_base}")
    log(f"Output dir       : {output_dir}")
    log(f"Features         : {features}")
    log(f"Primary feature  : {primary_feature}")
    log(f"Min valid ratio  : {args.min_valid_ratio}")
    log(f"Train/Val/Test   : {args.train_ratio}/{args.val_ratio}/{1 - args.train_ratio - args.val_ratio:.2f}")

    if not input_base.exists():
        raise FileNotFoundError(f"Input base does not exist: {input_base}")

    # -------------------------------------------------------------------------
    # Locate and load inputs
    # -------------------------------------------------------------------------
    log_section("LOAD INPUT FILES")
    paths = locate_inputs(input_base)
    for k, v in paths.items():
        log(f"{k:<22}: {v}")

    match_summary = read_json(paths["summary_json"])
    edge_meta = pd.read_csv(paths["edge_meta_csv"])
    match_df = pd.read_csv(paths["match_table_csv_gz"], compression="gzip")

    edge_meta["model_node_id"] = pd.to_numeric(edge_meta["model_node_id"], errors="coerce").astype(np.int64)
    edge_meta = edge_meta.sort_values("model_node_id").reset_index(drop=True)

    log(f"edge_meta shape: {edge_meta.shape}")
    log(f"match_df shape : {match_df.shape}")

    arrays = load_traffic_npz_arrays(paths["traffic_features_npz"], features)

    used_features = list(arrays["_used_features"].astype(str))
    if primary_feature not in used_features:
        raise ValueError(
            f"Primary feature '{primary_feature}' not in available feature list: {used_features}"
        )

    date_col = str(arrays["_date_col"][0])

    # -------------------------------------------------------------------------
    # Timestamp index and mapping
    # -------------------------------------------------------------------------
    timestamp_idx_arr, timestamp_table = build_timestamp_index(
        arrays[date_col],
        arrays["time_set"],
    )
    timestamp_table.to_csv(tables_dir / "timestamp_table.csv", index=False, encoding="utf-8-sig")
    log(f"Saved timestamp table: {tables_dir / 'timestamp_table.csv'}")

    mapping_df = build_segment_to_model_mapping(match_df, edge_meta)
    mapping_df.to_csv(tables_dir / "segment_to_model_node_mapping.csv", index=False, encoding="utf-8-sig")
    log(f"Saved mapping table: {tables_dir / 'segment_to_model_node_mapping.csv'}")

    # -------------------------------------------------------------------------
    # Tensor X
    # -------------------------------------------------------------------------
    X_raw, valid_mask, model_node_ids, timestamp_coverage, base_node_quality = build_osm_edge_tensor(
        arrays=arrays,
        timestamp_idx_arr=timestamp_idx_arr,
        timestamp_table=timestamp_table,
        mapping_df=mapping_df,
        edge_meta=edge_meta,
        features=features,
        primary_feature=primary_feature,
        output_dir=output_dir,
    )

    feature_names = used_features
    timestamps = timestamp_table["timestamp_key"].astype(str).to_numpy()

    # -------------------------------------------------------------------------
    # Fill missing and normalize
    # -------------------------------------------------------------------------
    X_filled = fill_missing_values(X_raw, feature_names, output_dir)

    T = X_filled.shape[0]
    train_end, val_end, train_idx, val_idx, test_idx = make_splits(
        T,
        args.train_ratio,
        args.val_ratio,
    )
    X_norm, scaler_mean, scaler_std = normalize_using_train(X_filled, train_end)

    # -------------------------------------------------------------------------
    # Quality and summaries
    # -------------------------------------------------------------------------
    node_quality = compute_node_quality(
        X_raw=X_raw,
        valid_mask=valid_mask,
        model_node_ids=model_node_ids,
        edge_meta=edge_meta,
        feature_names=feature_names,
        primary_feature=primary_feature,
        min_valid_ratio=args.min_valid_ratio,
    )
    node_quality.to_csv(tables_dir / "node_quality.csv", index=False, encoding="utf-8-sig")
    node_quality.sort_values("valid_ratio").head(100).to_csv(
        tables_dir / "node_quality_lowest_valid_ratio_top100.csv",
        index=False,
        encoding="utf-8-sig",
    )

    network_time_summary = compute_network_time_summary(
        X_raw=X_raw,
        valid_mask=valid_mask,
        timestamp_table=timestamp_table,
        feature_names=feature_names,
    )
    network_time_summary.to_csv(tables_dir / "network_time_summary.csv", index=False, encoding="utf-8-sig")

    if not args.no_plots:
        make_basic_plots(
            output_dir=output_dir,
            node_quality=node_quality,
            network_time_summary=network_time_summary,
            primary_feature=primary_feature,
        )

    # -------------------------------------------------------------------------
    # Save arrays
    # -------------------------------------------------------------------------
    log_section("SAVE OUTPUT ARRAYS")

    osm_edge_ids = (
        edge_meta.sort_values("model_node_id")["osm_edge_id"].astype(str).to_numpy()
        if "osm_edge_id" in edge_meta.columns
        else np.array([str(x) for x in model_node_ids])
    )

    recommended_keep_mask = node_quality.sort_values("model_node_id")["recommended_keep"].to_numpy(dtype=bool)

    tensor_npz_path = output_dir / "osm_edge_tensor.npz"
    tensor_payload = {
        "X_filled": X_filled.astype(np.float32),
        "X_norm": X_norm.astype(np.float32),
        "valid_mask": valid_mask.astype(bool),
        "recommended_keep_mask": recommended_keep_mask.astype(bool),
        "timestamps": timestamps,
        "timestamp_idx": timestamp_table["timestamp_idx"].to_numpy(dtype=np.int64),
        "model_node_ids": model_node_ids.astype(np.int64),
        "osm_edge_ids": osm_edge_ids,
        "feature_names": np.array(feature_names),
        "scaler_mean": scaler_mean.astype(np.float32),
        "scaler_std": scaler_std.astype(np.float32),
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
    }
    if args.save_raw:
        tensor_payload["X_raw"] = X_raw.astype(np.float32)

    save_npz_compressed(tensor_npz_path, **tensor_payload)

    split_npz_path = output_dir / "train_val_test_split.npz"
    save_npz_compressed(
        split_npz_path,
        X_train=X_norm[train_idx].astype(np.float32),
        X_val=X_norm[val_idx].astype(np.float32),
        X_test=X_norm[test_idx].astype(np.float32),
        X_train_filled=X_filled[train_idx].astype(np.float32),
        X_val_filled=X_filled[val_idx].astype(np.float32),
        X_test_filled=X_filled[test_idx].astype(np.float32),
        timestamps_train=timestamps[train_idx],
        timestamps_val=timestamps[val_idx],
        timestamps_test=timestamps[test_idx],
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        model_node_ids=model_node_ids.astype(np.int64),
        osm_edge_ids=osm_edge_ids,
        feature_names=np.array(feature_names),
        scaler_mean=scaler_mean.astype(np.float32),
        scaler_std=scaler_std.astype(np.float32),
        recommended_keep_mask=recommended_keep_mask.astype(bool),
    )

    # Save small matrices separately for direct use.
    np.save(matrices_dir / "model_node_ids.npy", model_node_ids.astype(np.int64))
    np.save(matrices_dir / "recommended_keep_mask.npy", recommended_keep_mask.astype(bool))
    np.save(matrices_dir / "timestamps.npy", timestamps)
    np.save(matrices_dir / "feature_names.npy", np.array(feature_names))

    # -------------------------------------------------------------------------
    # Metadata
    # -------------------------------------------------------------------------
    metadata = {
        "created_at": now_str(),
        "script": str(THIS_FILE),
        "input_base": str(input_base),
        "output_dir": str(output_dir),
        "input_files": {k: str(v) for k, v in paths.items()},
        "match_summary": match_summary.get("match", {}),
        "definition": {
            "model_node": "matched OSM directed edge",
            "traffic_data_source": "TomTom",
            "geography_topology_source": "OpenStreetMap",
            "aggregation": {
                "sample_size": "sum over TomTom segments mapped to the same OSM edge at each timestamp",
                "other_features": "mean over TomTom segments mapped to the same OSM edge at each timestamp",
            },
        },
        "shape": {
            "T": int(X_filled.shape[0]),
            "N": int(X_filled.shape[1]),
            "F": int(X_filled.shape[2]),
            "X_filled": list(X_filled.shape),
            "X_norm": list(X_norm.shape),
            "valid_mask": list(valid_mask.shape),
        },
        "features": feature_names,
        "primary_feature": primary_feature,
        "coverage": {
            "timestamp_coverage_mean": float(timestamp_coverage["coverage"].mean()),
            "timestamp_coverage_min": float(timestamp_coverage["coverage"].min()),
            "timestamp_coverage_max": float(timestamp_coverage["coverage"].max()),
            "node_valid_ratio_mean": float(node_quality["valid_ratio"].mean()),
            "node_valid_ratio_min": float(node_quality["valid_ratio"].min()),
            "node_valid_ratio_max": float(node_quality["valid_ratio"].max()),
            "recommended_keep_nodes": int(node_quality["recommended_keep"].sum()),
            "total_nodes": int(len(node_quality)),
            "min_valid_ratio_recommended": float(args.min_valid_ratio),
        },
        "split": {
            "train_ratio": float(args.train_ratio),
            "val_ratio": float(args.val_ratio),
            "test_ratio": float(1.0 - args.train_ratio - args.val_ratio),
            "train_start": 0,
            "train_end_exclusive": int(train_end),
            "val_start": int(train_end),
            "val_end_exclusive": int(val_end),
            "test_start": int(val_end),
            "test_end_exclusive": int(T),
            "train_size": int(len(train_idx)),
            "val_size": int(len(val_idx)),
            "test_size": int(len(test_idx)),
        },
        "outputs": {
            "tensor_npz": str(tensor_npz_path),
            "split_npz": str(split_npz_path),
            "tables_dir": str(tables_dir),
            "matrices_dir": str(matrices_dir),
            "plots_dir": str(output_dir / "plots"),
        },
    }

    metadata_path = output_dir / "metadata.json"
    save_json(metadata, metadata_path)

    # Human-readable summary.
    summary_txt = output_dir / "README_outputs.txt"
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write("OSM Edge Forecasting Dataset\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Created at: {metadata['created_at']}\n")
        f.write(f"Shape X: T={X_filled.shape[0]}, N={X_filled.shape[1]}, F={X_filled.shape[2]}\n")
        f.write(f"Features: {feature_names}\n")
        f.write(f"Primary feature: {primary_feature}\n")
        f.write(f"Train/Val/Test: {len(train_idx)}/{len(val_idx)}/{len(test_idx)}\n")
        f.write(f"Recommended keep nodes: {metadata['coverage']['recommended_keep_nodes']}/{metadata['coverage']['total_nodes']}\n\n")
        f.write("Important files:\n")
        f.write(f"- {tensor_npz_path.name}: full tensor arrays\n")
        f.write(f"- {split_npz_path.name}: temporal train/val/test split\n")
        f.write("- tables/node_quality.csv: node quality and OSM metadata\n")
        f.write("- tables/network_time_summary.csv: network-level temporal statistics\n")
        f.write("- metadata.json: full machine-readable metadata\n")

    log_section("DONE")
    log(f"Tensor NPZ       : {tensor_npz_path}")
    log(f"Split NPZ        : {split_npz_path}")
    log(f"Metadata JSON    : {metadata_path}")
    log(f"Tables dir       : {tables_dir}")
    log(f"Plots dir        : {output_dir / 'plots'}")
    log(f"README           : {summary_txt}")
    log(f"Final shape      : X = {X_filled.shape}")
    log(f"Train/Val/Test   : {len(train_idx)}/{len(val_idx)}/{len(test_idx)}")
    log(f"Recommended nodes: {metadata['coverage']['recommended_keep_nodes']:,}/{metadata['coverage']['total_nodes']:,}")


if __name__ == "__main__":
    main()
