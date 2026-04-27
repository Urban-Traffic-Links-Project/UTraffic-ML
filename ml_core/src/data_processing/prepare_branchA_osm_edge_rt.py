# ml_core/src/data_processing/prepare_branchA_osm_edge_rt.py
"""
Prepare Branch A dataset: rolling segment-segment correlation matrices R_t
for matched OSM directed edges.

Mục tiêu:
- Đọc tensor X[t, n, f] đã tạo từ prepare_osm_edge_forecasting_dataset.py.
- Chọn feature chính, mặc định average_speed.
- Tính rolling correlation R_t[n, n] với window=10.
- Lưu dạng memmap để tránh nổ RAM.
- Xuất metadata, split index, bảng thống kê và plot phục vụ phân tích.

Vị trí đặt file:
    UTraffic-ML/ml_core/src/data_processing/prepare_branchA_osm_edge_rt.py

Chạy từ thư mục gốc project:
    cd C:/AI/Thesis/UTraffic-ML
    python ml_core/src/data_processing/prepare_branchA_osm_edge_rt.py

Input mặc định:
    UTraffic-ML/ml_core/src/data_processing/outputs/branchA/osm_edge_forecasting_dataset/osm_edge_tensor.npz

Output mặc định:
    UTraffic-ML/ml_core/src/data_processing/outputs/branchA/osm_edge_rt/

Ghi chú quan trọng:
- Với N khoảng 3,697, mỗi R_t float32 có kích thước khoảng 54.7 MB.
- Với khoảng 711 rolling windows, R_all có thể khoảng 39 GB.
- Script dùng np.memmap để ghi trực tiếp xuống disk, không giữ toàn bộ R_all trong RAM.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


# =============================================================================
# PATH CONFIG
# =============================================================================

THIS_FILE = Path(__file__).resolve()

# File này nằm tại: UTraffic-ML/ml_core/src/data_processing/prepare_branchA_osm_edge_rt.py
DATA_PROCESSING_DIR = THIS_FILE.parent                 # .../ml_core/src/data_processing
SRC_ROOT = DATA_PROCESSING_DIR.parent                  # .../ml_core/src
ML_CORE_ROOT = SRC_ROOT.parent                         # .../ml_core
PROJECT_ROOT = ML_CORE_ROOT.parent                     # .../UTraffic-ML

DEFAULT_INPUT_DIR = DATA_PROCESSING_DIR / "outputs" / "branchA" / "osm_edge_forecasting_dataset"
DEFAULT_OUTPUT_DIR = DATA_PROCESSING_DIR / "outputs" / "branchA" / "osm_edge_rt"


# =============================================================================
# UTILS
# =============================================================================

def now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def print_stage(title: str) -> None:
    print("\n" + "=" * 88)
    print(f"{now_str()} | {title}")
    print("=" * 88)


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
        return super().default(obj)


def save_json(obj: Dict[str, Any], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, cls=NumpyJsonEncoder)


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def bytes_to_gb(n_bytes: int) -> float:
    return n_bytes / (1024 ** 3)


def fmt_gb(n_bytes: int) -> str:
    return f"{bytes_to_gb(n_bytes):,.2f} GB"


def latest_file(folder: Path, pattern: str) -> Path:
    files = sorted(folder.glob(pattern))
    if not files:
        raise FileNotFoundError(f"Không tìm thấy {pattern} trong {folder}")
    return files[-1]


def decode_str_array(arr: np.ndarray) -> np.ndarray:
    """
    Chuẩn hóa array string có thể là bytes/object.
    """
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


def parse_metadata_from_npz(data: np.lib.npyio.NpzFile) -> Dict[str, Any]:
    if "_metadata" not in data.files:
        return {}
    raw = data["_metadata"][0]
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    try:
        return json.loads(str(raw))
    except Exception:
        return {"_metadata_raw": str(raw)}


def save_readme(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def maybe_iter(iterable, total: Optional[int] = None, desc: str = ""):
    if tqdm is not None:
        return tqdm(iterable, total=total, desc=desc)
    return iterable


def sample_upper_triangle(R: np.ndarray, max_samples: int, rng: np.random.Generator) -> np.ndarray:
    """
    Sample off-diagonal upper-triangle values from R.
    """
    n = R.shape[0]
    total_pairs = n * (n - 1) // 2
    if total_pairs <= 0:
        return np.array([], dtype=np.float32)

    if total_pairs <= max_samples:
        iu = np.triu_indices(n, k=1)
        return R[iu].astype(np.float32, copy=False)

    i = rng.integers(0, n - 1, size=max_samples, endpoint=False)
    # j must be > i
    # j = i + 1 + random integer in [0, n-i-2]
    span = n - i - 1
    j = i + 1 + rng.integers(0, span)
    return R[i, j].astype(np.float32, copy=False)


def compute_corr_window(W: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Tính correlation matrix cho window W [window, N].

    Công thức:
        Z = (W - mean) / (std * sqrt(window - 1))
        R = Z.T @ Z

    Ưu điểm:
        - Nhanh hơn np.corrcoef trong vòng lặp lớn.
        - Kiểm soát được node có std gần 0.
    """
    W = W.astype(np.float32, copy=False)
    W = np.nan_to_num(W, nan=0.0, posinf=0.0, neginf=0.0)

    win = W.shape[0]
    if win < 2:
        raise ValueError("Window phải >= 2 để tính correlation.")

    mu = W.mean(axis=0, keepdims=True, dtype=np.float32)
    Xc = W - mu

    # ddof=1: std mẫu
    std = Xc.std(axis=0, ddof=1, keepdims=True).astype(np.float32)
    good = std.reshape(-1) > eps

    denom = std * math.sqrt(max(win - 1, 1))
    Z = np.zeros_like(Xc, dtype=np.float32)
    np.divide(Xc, denom, out=Z, where=denom > eps)

    R = (Z.T @ Z).astype(np.float32, copy=False)
    R = np.nan_to_num(R, nan=0.0, posinf=0.0, neginf=0.0)
    np.clip(R, -1.0, 1.0, out=R)

    # Diagonal = 1 cho tất cả node để đúng dạng correlation matrix.
    # Nếu muốn node std=0 có diag=0, sửa dòng này.
    np.fill_diagonal(R, 1.0)

    return R


def split_rt_indices_by_end_timestamp(
    rt_end_indices: np.ndarray,
    train_idx: Optional[np.ndarray],
    val_idx: Optional[np.ndarray],
    test_idx: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    R_t tại vị trí k được gán split theo end timestamp của rolling window.
    """
    if train_idx is None or val_idx is None or test_idx is None:
        n = len(rt_end_indices)
        train_end = int(0.70 * n)
        val_end = train_end + int(0.15 * n)
        return (
            np.arange(0, train_end, dtype=np.int64),
            np.arange(train_end, val_end, dtype=np.int64),
            np.arange(val_end, n, dtype=np.int64),
        )

    train_set = set(map(int, train_idx.tolist()))
    val_set = set(map(int, val_idx.tolist()))
    test_set = set(map(int, test_idx.tolist()))

    train_rt, val_rt, test_rt = [], [], []
    for pos, end_ts in enumerate(rt_end_indices):
        end_ts = int(end_ts)
        if end_ts in train_set:
            train_rt.append(pos)
        elif end_ts in val_set:
            val_rt.append(pos)
        elif end_ts in test_set:
            test_rt.append(pos)
        else:
            # fallback: không bỏ mất R_t nếu split index không khớp
            test_rt.append(pos)

    return (
        np.array(train_rt, dtype=np.int64),
        np.array(val_rt, dtype=np.int64),
        np.array(test_rt, dtype=np.int64),
    )


def make_basic_plots(
    output_dir: Path,
    per_t_df: pd.DataFrame,
    sample_values: np.ndarray,
    stability_df: pd.DataFrame,
) -> None:
    if plt is None:
        print("matplotlib chưa sẵn sàng, bỏ qua plot.")
        return

    plots_dir = ensure_dir(output_dir / "plots")

    if not per_t_df.empty:
        plt.figure(figsize=(12, 4))
        plt.plot(per_t_df["rt_pos"], per_t_df["mean_abs_corr"])
        plt.title("Branch A — Mean |R_t| over time")
        plt.xlabel("R_t position")
        plt.ylabel("Mean |correlation|")
        plt.tight_layout()
        plt.savefig(plots_dir / "mean_abs_corr_over_time.png", dpi=160)
        plt.close()

        plt.figure(figsize=(12, 4))
        plt.plot(per_t_df["rt_pos"], per_t_df["q95_abs_corr"])
        plt.title("Branch A — q95 |R_t| over time")
        plt.xlabel("R_t position")
        plt.ylabel("q95 |correlation|")
        plt.tight_layout()
        plt.savefig(plots_dir / "q95_abs_corr_over_time.png", dpi=160)
        plt.close()

    if sample_values is not None and len(sample_values) > 0:
        plt.figure(figsize=(8, 4))
        plt.hist(sample_values, bins=100)
        plt.title("Branch A — Distribution of sampled off-diagonal correlations")
        plt.xlabel("Correlation")
        plt.ylabel("Sample count")
        plt.tight_layout()
        plt.savefig(plots_dir / "sampled_corr_distribution.png", dpi=160)
        plt.close()

        plt.figure(figsize=(8, 4))
        plt.hist(np.abs(sample_values), bins=100)
        plt.title("Branch A — Distribution of sampled |correlation|")
        plt.xlabel("|Correlation|")
        plt.ylabel("Sample count")
        plt.tight_layout()
        plt.savefig(plots_dir / "sampled_abs_corr_distribution.png", dpi=160)
        plt.close()

    if not stability_df.empty:
        plt.figure(figsize=(12, 4))
        plt.plot(stability_df["rt_pos"], stability_df["frobenius_diff_norm"])
        plt.title("Branch A — Rolling correlation stability")
        plt.xlabel("R_t position")
        plt.ylabel("Normalized Frobenius diff from previous R_t")
        plt.tight_layout()
        plt.savefig(plots_dir / "rt_stability_frobenius_diff.png", dpi=160)
        plt.close()

        plt.figure(figsize=(12, 4))
        plt.plot(stability_df["rt_pos"], stability_df["upper_triangle_similarity"])
        plt.title("Branch A — Similarity between consecutive R_t")
        plt.xlabel("R_t position")
        plt.ylabel("Upper-triangle Pearson similarity")
        plt.tight_layout()
        plt.savefig(plots_dir / "rt_stability_similarity.png", dpi=160)
        plt.close()

    print(f"Saved plots to: {plots_dir}")


# =============================================================================
# LOAD INPUT
# =============================================================================

def load_tensor_dataset(input_dir: Path, prefer: str) -> Dict[str, Any]:
    tensor_path = input_dir / "osm_edge_tensor.npz"
    if not tensor_path.exists():
        # fallback: latest npz containing tensor
        candidates = sorted(input_dir.glob("*.npz"))
        if not candidates:
            raise FileNotFoundError(f"Không tìm thấy osm_edge_tensor.npz hoặc *.npz trong {input_dir}")
        tensor_path = candidates[-1]

    print(f"Loading tensor dataset: {tensor_path}")
    data = np.load(str(tensor_path), allow_pickle=True, mmap_mode=None)

    # Chọn X theo ưu tiên.
    if prefer == "norm":
        x_key = get_npz_key(data, ["X_norm", "X_normalized", "X_filled", "X"])
    elif prefer == "filled":
        x_key = get_npz_key(data, ["X_filled", "X_norm", "X_normalized", "X"])
    elif prefer == "raw":
        x_key = get_npz_key(data, ["X_raw", "X", "X_filled", "X_norm"])
    else:
        raise ValueError(f"prefer không hợp lệ: {prefer}")

    if x_key is None:
        raise KeyError(f"Không tìm thấy tensor X trong {tensor_path}. Keys: {data.files}")

    X = data[x_key]
    if X.ndim not in [2, 3]:
        raise ValueError(f"X phải có shape [T,N] hoặc [T,N,F], nhận được {X.shape}")

    metadata = parse_metadata_from_npz(data)

    feature_names = None
    if "feature_names" in data.files:
        feature_names = decode_str_array(data["feature_names"])
    elif X.ndim == 3:
        feature_names = np.array([f"feature_{i}" for i in range(X.shape[2])])
    else:
        feature_names = np.array(["value"])

    timestamps = None
    if "timestamps" in data.files:
        timestamps = decode_str_array(data["timestamps"])
    else:
        timestamps = np.array([str(i) for i in range(X.shape[0])])

    model_node_ids = None
    if "model_node_ids" in data.files:
        model_node_ids = data["model_node_ids"].astype(np.int64)
    else:
        model_node_ids = np.arange(X.shape[1], dtype=np.int64)

    osm_edge_ids = None
    edge_key = get_npz_key(data, ["osm_edge_ids", "model_node_osm_edge_id", "model_node_osm_edge_ids"])
    if edge_key is not None:
        osm_edge_ids = decode_str_array(data[edge_key])
    else:
        osm_edge_ids = np.array([str(x) for x in model_node_ids])

    recommended_keep_mask = None
    if "recommended_keep_mask" in data.files:
        recommended_keep_mask = data["recommended_keep_mask"].astype(bool)

    valid_mask = None
    if "valid_mask" in data.files:
        valid_mask = data["valid_mask"].astype(bool)

    train_idx = data["train_idx"].astype(np.int64) if "train_idx" in data.files else None
    val_idx = data["val_idx"].astype(np.int64) if "val_idx" in data.files else None
    test_idx = data["test_idx"].astype(np.int64) if "test_idx" in data.files else None

    return {
        "tensor_path": tensor_path,
        "x_key": x_key,
        "X": X,
        "metadata": metadata,
        "feature_names": feature_names,
        "timestamps": timestamps,
        "model_node_ids": model_node_ids,
        "osm_edge_ids": osm_edge_ids,
        "recommended_keep_mask": recommended_keep_mask,
        "valid_mask": valid_mask,
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
        print(f"WARNING: Không tìm thấy feature='{feature}'. Dùng feature đầu tiên: {names[0]}")
        f_idx = 0

    return X[:, :, f_idx].astype(np.float32, copy=False), f_idx, names[f_idx]


def apply_node_filter(
    X2: np.ndarray,
    model_node_ids: np.ndarray,
    osm_edge_ids: np.ndarray,
    recommended_keep_mask: Optional[np.ndarray],
    valid_mask: Optional[np.ndarray],
    node_filter: str,
    max_nodes: Optional[int],
    input_dir: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    N = X2.shape[1]

    if node_filter == "recommended" and recommended_keep_mask is not None:
        keep = recommended_keep_mask.astype(bool).copy()
        if keep.shape[0] != N:
            print("WARNING: recommended_keep_mask không khớp N, dùng all nodes.")
            keep = np.ones(N, dtype=bool)
    else:
        keep = np.ones(N, dtype=bool)

    selected_positions = np.where(keep)[0]

    # Nếu max_nodes được set, chọn node chất lượng cao nhất nếu có node_quality.csv.
    if max_nodes is not None and max_nodes > 0 and len(selected_positions) > max_nodes:
        node_quality_path = input_dir / "tables" / "node_quality.csv"
        if node_quality_path.exists():
            q = pd.read_csv(node_quality_path)
            if "model_node_id" in q.columns:
                q = q[q["model_node_id"].isin(model_node_ids[selected_positions])]
                sort_cols = []
                ascending = []
                if "recommended_keep" in q.columns:
                    sort_cols.append("recommended_keep")
                    ascending.append(False)
                if "valid_ratio" in q.columns:
                    sort_cols.append("valid_ratio")
                    ascending.append(False)
                if "average_speed_std" in q.columns:
                    sort_cols.append("average_speed_std")
                    ascending.append(False)

                if sort_cols:
                    q = q.sort_values(sort_cols, ascending=ascending)
                chosen_ids = q["model_node_id"].head(max_nodes).to_numpy()
                chosen_set = set(map(int, chosen_ids))
                selected_positions = np.array(
                    [i for i in selected_positions if int(model_node_ids[i]) in chosen_set],
                    dtype=np.int64,
                )
                selected_positions = selected_positions[:max_nodes]
            else:
                selected_positions = selected_positions[:max_nodes]
        else:
            selected_positions = selected_positions[:max_nodes]

    X_sel = X2[:, selected_positions]
    node_ids_sel = model_node_ids[selected_positions]
    edge_ids_sel = osm_edge_ids[selected_positions]

    return X_sel, node_ids_sel, edge_ids_sel, selected_positions


# =============================================================================
# MAIN PREPARE RT
# =============================================================================

def prepare_branchA_rt(args: argparse.Namespace) -> None:
    input_dir = Path(args.input_dir).resolve()
    output_dir = ensure_dir(Path(args.output_dir).resolve())
    tables_dir = ensure_dir(output_dir / "tables")
    matrices_dir = ensure_dir(output_dir / "matrices")

    print_stage("BRANCH A — PREPARE OSM EDGE R_t DATASET")
    print("Project root :", PROJECT_ROOT)
    print("Input dir    :", input_dir)
    print("Output dir   :", output_dir)
    print("Feature      :", args.feature)
    print("Window       :", args.window)
    print("Prefer X     :", args.prefer)
    print("Node filter  :", args.node_filter)
    print("Max nodes    :", args.max_nodes)
    print("Dry run      :", args.dry_run)

    # -------------------------------------------------------------------------
    # Stage 1 — Load tensor
    # -------------------------------------------------------------------------
    print_stage("STAGE 1 — LOAD OSM EDGE TENSOR")

    dataset = load_tensor_dataset(input_dir, prefer=args.prefer)
    X = dataset["X"]
    feature_names = dataset["feature_names"]
    timestamps = dataset["timestamps"]
    model_node_ids = dataset["model_node_ids"]
    osm_edge_ids = dataset["osm_edge_ids"]

    print(f"Loaded X key       : {dataset['x_key']}")
    print(f"Original X shape   : {X.shape}")
    print(f"Feature names      : {feature_names.tolist()[:20]}")
    print(f"Num timestamps     : {len(timestamps):,}")
    print(f"Num model nodes    : {len(model_node_ids):,}")

    X2, feature_idx, feature_used = select_feature(X, feature_names, args.feature)
    print(f"Selected feature   : {feature_used} (index={feature_idx})")
    print(f"X2 shape [T,N]     : {X2.shape}")

    X2, node_ids_sel, edge_ids_sel, selected_positions = apply_node_filter(
        X2=X2,
        model_node_ids=model_node_ids,
        osm_edge_ids=osm_edge_ids,
        recommended_keep_mask=dataset["recommended_keep_mask"],
        valid_mask=dataset["valid_mask"],
        node_filter=args.node_filter,
        max_nodes=args.max_nodes,
        input_dir=input_dir,
    )

    T, N = X2.shape
    W = int(args.window)
    if W < 2:
        raise ValueError("window phải >= 2.")
    if T < W:
        raise ValueError(f"Số timestamp T={T} nhỏ hơn window={W}.")

    T_rt = T - W + 1
    rt_end_indices = np.arange(W - 1, T, dtype=np.int64)
    rt_start_indices = np.arange(0, T_rt, dtype=np.int64)

    print(f"Selected X2 shape  : {X2.shape}")
    print(f"R_t count          : {T_rt:,}")
    print(f"R_t shape each     : [{N:,}, {N:,}]")

    r_one_bytes = N * N * np.dtype(np.float32).itemsize
    r_all_bytes = T_rt * r_one_bytes
    print(f"One R_t size       : {fmt_gb(r_one_bytes)}")
    print(f"R_all size estimate: {fmt_gb(r_all_bytes)}")

    if r_all_bytes > args.warn_gb * (1024 ** 3):
        print("\nWARNING:")
        print(f"  R_all dự kiến {fmt_gb(r_all_bytes)}, lớn hơn ngưỡng cảnh báo {args.warn_gb} GB.")
        print("  Script vẫn chạy bằng memmap, nhưng cần đảm bảo ổ cứng còn đủ dung lượng.")
        print("  Có thể chạy thử với --max-nodes 512 trước.")
        print()

    # Split R_t theo end timestamp.
    train_rt_idx, val_rt_idx, test_rt_idx = split_rt_indices_by_end_timestamp(
        rt_end_indices=rt_end_indices,
        train_idx=dataset["train_idx"],
        val_idx=dataset["val_idx"],
        test_idx=dataset["test_idx"],
    )

    print(f"R_train count      : {len(train_rt_idx):,}")
    print(f"R_val count        : {len(val_rt_idx):,}")
    print(f"R_test count       : {len(test_rt_idx):,}")

    # Lưu mapping/index trước.
    np.savez_compressed(
        output_dir / "rt_index.npz",
        rt_start_indices=rt_start_indices,
        rt_end_indices=rt_end_indices,
        train_rt_idx=train_rt_idx,
        val_rt_idx=val_rt_idx,
        test_rt_idx=test_rt_idx,
        selected_positions=selected_positions.astype(np.int64),
        model_node_ids=node_ids_sel.astype(np.int64),
        osm_edge_ids=edge_ids_sel.astype(str),
        timestamps=timestamps.astype(str),
        rt_timestamps=timestamps[rt_end_indices].astype(str),
        feature_name=np.array([feature_used]),
        window=np.array([W], dtype=np.int64),
    )

    node_map_df = pd.DataFrame({
        "position_in_R": np.arange(N, dtype=np.int64),
        "model_node_id": node_ids_sel.astype(np.int64),
        "osm_edge_id": edge_ids_sel.astype(str),
        "original_position_in_tensor": selected_positions.astype(np.int64),
    })
    node_map_df.to_csv(tables_dir / "rt_node_mapping.csv", index=False, encoding="utf-8-sig")

    rt_time_df = pd.DataFrame({
        "rt_pos": np.arange(T_rt, dtype=np.int64),
        "window_start_timestamp_idx": rt_start_indices,
        "window_end_timestamp_idx": rt_end_indices,
        "window_end_timestamp": timestamps[rt_end_indices].astype(str),
        "split": "unknown",
    })
    rt_time_df.loc[train_rt_idx, "split"] = "train"
    rt_time_df.loc[val_rt_idx, "split"] = "val"
    rt_time_df.loc[test_rt_idx, "split"] = "test"
    rt_time_df.to_csv(tables_dir / "rt_time_index.csv", index=False, encoding="utf-8-sig")

    metadata = {
        "created_at": now_str(),
        "project_root": str(PROJECT_ROOT),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "tensor_path": str(dataset["tensor_path"]),
        "x_key": dataset["x_key"],
        "feature_requested": args.feature,
        "feature_used": feature_used,
        "feature_idx": int(feature_idx),
        "window": int(W),
        "original_shape": list(X.shape),
        "x2_shape": [int(T), int(N)],
        "num_timestamps": int(T),
        "num_nodes": int(N),
        "num_rt": int(T_rt),
        "rt_shape": [int(T_rt), int(N), int(N)],
        "dtype": "float32",
        "node_filter": args.node_filter,
        "max_nodes": args.max_nodes,
        "estimated_one_rt_bytes": int(r_one_bytes),
        "estimated_r_all_bytes": int(r_all_bytes),
        "estimated_one_rt_gb": bytes_to_gb(r_one_bytes),
        "estimated_r_all_gb": bytes_to_gb(r_all_bytes),
        "train_rt_count": int(len(train_rt_idx)),
        "val_rt_count": int(len(val_rt_idx)),
        "test_rt_count": int(len(test_rt_idx)),
        "model_node_definition": "one node = one matched OSM directed edge",
        "matrix_definition": "R_t[i,j] = Pearson correlation between selected feature of node i and node j over rolling window",
    }

    save_json(metadata, output_dir / "metadata_precompute.json")

    if args.dry_run:
        print_stage("DRY RUN COMPLETE")
        print("Đã xuất metadata/index, chưa tính R_t.")
        return

    # -------------------------------------------------------------------------
    # Stage 2 — Compute rolling R_t and save memmap
    # -------------------------------------------------------------------------
    print_stage("STAGE 2 — COMPUTE R_t MEMMAP")

    r_path = matrices_dir / "R_all_float32.dat"
    if r_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"{r_path} đã tồn tại. Dùng --overwrite nếu muốn ghi đè."
        )

    R_mem = np.memmap(
        r_path,
        dtype=np.float32,
        mode="w+",
        shape=(T_rt, N, N),
    )

    rng = np.random.default_rng(args.seed)
    per_t_rows: List[Dict[str, Any]] = []
    stability_rows: List[Dict[str, Any]] = []
    sample_values_all: List[np.ndarray] = []

    prev_sample = None
    prev_R = None
    t0 = time.time()

    iterator = maybe_iter(range(T_rt), total=T_rt, desc="Computing R_t")

    for rt_pos in iterator:
        start = rt_pos
        end = rt_pos + W
        W_data = X2[start:end, :]

        R_t = compute_corr_window(W_data)
        R_mem[rt_pos, :, :] = R_t

        # Sampling để thống kê phân phối, tránh đọc lại toàn bộ sau này.
        vals = sample_upper_triangle(R_t, args.sample_pairs_per_t, rng)
        vals = vals[np.isfinite(vals)]
        abs_vals = np.abs(vals)

        if len(vals) > 0:
            row = {
                "rt_pos": int(rt_pos),
                "window_start_timestamp_idx": int(start),
                "window_end_timestamp_idx": int(end - 1),
                "window_end_timestamp": str(timestamps[end - 1]),
                "split": rt_time_df.loc[rt_pos, "split"],
                "sample_n_pairs": int(len(vals)),
                "mean_corr": float(np.mean(vals)),
                "std_corr": float(np.std(vals)),
                "mean_abs_corr": float(np.mean(abs_vals)),
                "q05_corr": float(np.quantile(vals, 0.05)),
                "q50_corr": float(np.quantile(vals, 0.50)),
                "q95_corr": float(np.quantile(vals, 0.95)),
                "q95_abs_corr": float(np.quantile(abs_vals, 0.95)),
                "ratio_corr_gt_0_5": float(np.mean(vals > 0.5)),
                "ratio_abs_corr_gt_0_5": float(np.mean(abs_vals > 0.5)),
            }
        else:
            row = {
                "rt_pos": int(rt_pos),
                "window_start_timestamp_idx": int(start),
                "window_end_timestamp_idx": int(end - 1),
                "window_end_timestamp": str(timestamps[end - 1]),
                "split": rt_time_df.loc[rt_pos, "split"],
                "sample_n_pairs": 0,
                "mean_corr": np.nan,
                "std_corr": np.nan,
                "mean_abs_corr": np.nan,
                "q05_corr": np.nan,
                "q50_corr": np.nan,
                "q95_corr": np.nan,
                "q95_abs_corr": np.nan,
                "ratio_corr_gt_0_5": np.nan,
                "ratio_abs_corr_gt_0_5": np.nan,
            }

        per_t_rows.append(row)

        if args.store_distribution_samples and len(vals) > 0:
            # Giới hạn tổng sample để không tạo file quá lớn.
            if len(sample_values_all) < args.max_sample_blocks:
                sample_values_all.append(vals.astype(np.float32, copy=False))

        # Stability: so sánh với R_t trước đó.
        if prev_R is not None:
            diff = R_t - prev_R
            frob = float(np.linalg.norm(diff) / math.sqrt(diff.size))

            # Similarity trên cùng sample indices để nhẹ hơn.
            if args.stability_sample_pairs > 0:
                s1 = sample_upper_triangle(prev_R, args.stability_sample_pairs, rng)
                s2 = sample_upper_triangle(R_t, args.stability_sample_pairs, rng)
                # Lưu ý: hai lần sample khác nhau sẽ làm similarity nhiễu.
                # Do đó dùng Frobenius diff là chỉ số chính.
                if np.std(s1) > 1e-8 and np.std(s2) > 1e-8:
                    sim = float(np.corrcoef(s1, s2)[0, 1])
                else:
                    sim = np.nan
            else:
                sim = np.nan

            stability_rows.append({
                "rt_pos": int(rt_pos),
                "prev_rt_pos": int(rt_pos - 1),
                "window_end_timestamp_idx": int(end - 1),
                "window_end_timestamp": str(timestamps[end - 1]),
                "split": rt_time_df.loc[rt_pos, "split"],
                "frobenius_diff_norm": frob,
                "upper_triangle_similarity": sim,
            })

        prev_R = R_t.copy() if args.compute_stability else None

        if (rt_pos + 1) % args.flush_every == 0:
            R_mem.flush()
            elapsed = time.time() - t0
            speed = (rt_pos + 1) / max(elapsed, 1e-9)
            remaining = (T_rt - rt_pos - 1) / max(speed, 1e-9)
            print(
                f"[{now_str()}] progress={rt_pos + 1:,}/{T_rt:,} "
                f"speed={speed:.2f} R/s ETA={remaining/60:.1f} min"
            )

    R_mem.flush()

    elapsed_total = time.time() - t0
    print(f"\nFinished computing R_t in {elapsed_total/60:.2f} minutes.")
    print(f"R memmap path: {r_path}")

    # -------------------------------------------------------------------------
    # Stage 3 — Save summary tables
    # -------------------------------------------------------------------------
    print_stage("STAGE 3 — SAVE SUMMARY TABLES")

    per_t_df = pd.DataFrame(per_t_rows)
    per_t_path = tables_dir / "rt_per_timestamp_distribution_summary.csv"
    per_t_df.to_csv(per_t_path, index=False, encoding="utf-8-sig")
    print("Saved:", per_t_path)

    stability_df = pd.DataFrame(stability_rows)
    stability_path = tables_dir / "rt_stability_summary.csv"
    stability_df.to_csv(stability_path, index=False, encoding="utf-8-sig")
    print("Saved:", stability_path)

    if sample_values_all:
        sample_values = np.concatenate(sample_values_all)
    else:
        sample_values = np.array([], dtype=np.float32)

    sample_path = matrices_dir / "rt_distribution_sample_values.npy"
    np.save(sample_path, sample_values)
    print("Saved:", sample_path, "shape=", sample_values.shape)

    if len(sample_values) > 0:
        global_dist = {
            "sample_n": int(len(sample_values)),
            "mean_corr": float(np.mean(sample_values)),
            "std_corr": float(np.std(sample_values)),
            "mean_abs_corr": float(np.mean(np.abs(sample_values))),
            "q01_corr": float(np.quantile(sample_values, 0.01)),
            "q05_corr": float(np.quantile(sample_values, 0.05)),
            "q25_corr": float(np.quantile(sample_values, 0.25)),
            "q50_corr": float(np.quantile(sample_values, 0.50)),
            "q75_corr": float(np.quantile(sample_values, 0.75)),
            "q95_corr": float(np.quantile(sample_values, 0.95)),
            "q99_corr": float(np.quantile(sample_values, 0.99)),
            "ratio_corr_gt_0_3": float(np.mean(sample_values > 0.3)),
            "ratio_corr_gt_0_5": float(np.mean(sample_values > 0.5)),
            "ratio_corr_gt_0_7": float(np.mean(sample_values > 0.7)),
            "ratio_abs_corr_gt_0_5": float(np.mean(np.abs(sample_values) > 0.5)),
        }
    else:
        global_dist = {}

    save_json(global_dist, tables_dir / "rt_global_distribution_summary.json")

    # Split summary.
    split_summary = (
        per_t_df
        .groupby("split", as_index=False)
        .agg(
            n_rt=("rt_pos", "count"),
            mean_abs_corr=("mean_abs_corr", "mean"),
            q95_abs_corr=("q95_abs_corr", "mean"),
            ratio_abs_corr_gt_0_5=("ratio_abs_corr_gt_0_5", "mean"),
        )
    )
    split_summary.to_csv(tables_dir / "rt_split_distribution_summary.csv", index=False, encoding="utf-8-sig")
    print("Saved split summary.")

    # -------------------------------------------------------------------------
    # Stage 4 — Save final metadata and README
    # -------------------------------------------------------------------------
    print_stage("STAGE 4 — SAVE FINAL METADATA")

    final_metadata = dict(metadata)
    final_metadata.update({
        "R_all_memmap_path": str(r_path),
        "R_all_memmap_dtype": "float32",
        "R_all_memmap_shape": [int(T_rt), int(N), int(N)],
        "rt_index_path": str(output_dir / "rt_index.npz"),
        "rt_node_mapping_csv": str(tables_dir / "rt_node_mapping.csv"),
        "rt_time_index_csv": str(tables_dir / "rt_time_index.csv"),
        "rt_per_timestamp_distribution_summary_csv": str(per_t_path),
        "rt_stability_summary_csv": str(stability_path),
        "rt_distribution_sample_values_npy": str(sample_path),
        "elapsed_minutes": elapsed_total / 60.0,
        "global_distribution_summary": global_dist,
    })
    save_json(final_metadata, output_dir / "metadata.json")

    readme = f"""# Branch A OSM Edge R_t Dataset

Created at: {now_str()}

## Definition

- One model node = one matched OSM directed edge.
- R_t[i, j] = Pearson correlation between node i and node j over a rolling window.
- Feature used: {feature_used}
- Window: {W}
- R_all shape: [{T_rt}, {N}, {N}]
- dtype: float32

## Main files

- matrices/R_all_float32.dat
  Memmap binary file storing R_all with shape [{T_rt}, {N}, {N}], dtype float32.

- rt_index.npz
  Contains rt_start_indices, rt_end_indices, train_rt_idx, val_rt_idx, test_rt_idx,
  model_node_ids, osm_edge_ids, timestamps, rt_timestamps.

- metadata.json
  Full metadata for loading R_all.

- tables/rt_node_mapping.csv
  Mapping from R position to model_node_id and osm_edge_id.

- tables/rt_time_index.csv
  Mapping from R_t position to rolling window start/end timestamp and split.

- tables/rt_per_timestamp_distribution_summary.csv
  Per-R_t sampled correlation distribution.

- tables/rt_global_distribution_summary.json
  Global sampled distribution summary.

## How to load R_all

```python
import json
import numpy as np
from pathlib import Path

out_dir = Path(r"{output_dir}")
meta = json.load(open(out_dir / "metadata.json", "r", encoding="utf-8"))

R = np.memmap(
    meta["R_all_memmap_path"],
    dtype=np.float32,
    mode="r",
    shape=tuple(meta["R_all_memmap_shape"]),
)

idx = np.load(out_dir / "rt_index.npz", allow_pickle=True)
train_rt_idx = idx["train_rt_idx"]
val_rt_idx = idx["val_rt_idx"]
test_rt_idx = idx["test_rt_idx"]

# Example: first train matrix
R0 = R[train_rt_idx[0]]
```

## Notes

This dataset can be large. Do not load the full R_all into RAM with np.array(R).
Use memmap slicing instead.
"""
    save_readme(output_dir / "README_outputs.md", readme)

    # -------------------------------------------------------------------------
    # Stage 5 — Basic plots
    # -------------------------------------------------------------------------
    if not args.no_plots:
        print_stage("STAGE 5 — MAKE BASIC PLOTS")
        make_basic_plots(output_dir, per_t_df, sample_values, stability_df)

    print_stage("DONE")
    print("Output dir:", output_dir)
    print("Important files:")
    print("  -", r_path)
    print("  -", output_dir / "metadata.json")
    print("  -", output_dir / "rt_index.npz")
    print("  -", tables_dir / "rt_per_timestamp_distribution_summary.csv")
    print("  -", tables_dir / "rt_global_distribution_summary.json")
    print("  -", output_dir / "README_outputs.md")


# =============================================================================
# CLI
# =============================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Prepare Branch A rolling correlation matrices R_t for matched OSM edges."
    )

    p.add_argument(
        "--input-dir",
        type=str,
        default=str(DEFAULT_INPUT_DIR),
        help="Folder containing osm_edge_tensor.npz from prepare_osm_edge_forecasting_dataset.py.",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output folder for Branch A R_t dataset.",
    )
    p.add_argument(
        "--feature",
        type=str,
        default="average_speed",
        help="Feature to compute correlation on, e.g. average_speed, congestion_index, travel_time_ratio.",
    )
    p.add_argument(
        "--window",
        type=int,
        default=10,
        help="Rolling window length. Default 10.",
    )
    p.add_argument(
        "--prefer",
        type=str,
        default="norm",
        choices=["norm", "filled", "raw"],
        help="Which X to prefer from osm_edge_tensor.npz. Default uses X_norm.",
    )
    p.add_argument(
        "--node-filter",
        type=str,
        default="all",
        choices=["all", "recommended"],
        help="Use all nodes or recommended_keep_mask if available.",
    )
    p.add_argument(
        "--max-nodes",
        type=int,
        default=None,
        help="Optional debug mode: keep only top N nodes to test quickly.",
    )
    p.add_argument(
        "--sample-pairs-per-t",
        type=int,
        default=20000,
        help="Number of off-diagonal pairs sampled per R_t for distribution summary.",
    )
    p.add_argument(
        "--stability-sample-pairs",
        type=int,
        default=20000,
        help="Number of pairs sampled for consecutive R_t similarity. Frobenius diff is always exact.",
    )
    p.add_argument(
        "--max-sample-blocks",
        type=int,
        default=400,
        help="Maximum number of per-R_t sample blocks stored in rt_distribution_sample_values.npy.",
    )
    p.add_argument(
        "--no-store-distribution-samples",
        dest="store_distribution_samples",
        action="store_false",
        help="Do not store sampled correlation values to rt_distribution_sample_values.npy. "
             "Per-timestamp summary is still computed.",
    )
    p.add_argument(
        "--flush-every",
        type=int,
        default=10,
        help="Flush memmap and print progress every K R_t matrices.",
    )
    p.add_argument(
        "--warn-gb",
        type=float,
        default=30.0,
        help="Warn if estimated output file size exceeds this GB.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Only inspect input and write index/metadata. Do not compute R_t.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing R_all_float32.dat.",
    )
    p.add_argument(
        "--no-plots",
        action="store_true",
        help="Do not generate plots.",
    )
    p.add_argument(
        "--no-stability",
        dest="compute_stability",
        action="store_false",
        help="Disable stability calculation against previous R_t.",
    )
    p.set_defaults(compute_stability=True, store_distribution_samples=True)

    return p


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    prepare_branchA_rt(args)


if __name__ == "__main__":
    main()
