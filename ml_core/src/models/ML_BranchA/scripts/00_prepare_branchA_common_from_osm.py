# ml_core/src/models/ML_BranchA/scripts/00_prepare_branchA_common_from_osm.py
"""
Create Branch A common input format from OSM-edge tensor with whole-date train/val/test split.

This script converts:
    ml_core/src/data_processing/outputs/branchA/osm_edge_forecasting_dataset/osm_edge_tensor.npz

into the old Branch A expected folder:
    ml_core/src/models/ML_BranchA/data/05_branchA_prepare_segment_segment_rt/

Each split train/val/test contains:
    R_series.npy
    z.npy
    segment_ids.npy
    timestamps.npy
    R_series_meta.csv
    raw_meta.csv
    traffic_tensor_resid.npz
    traffic_tensor_resid_meta.csv

Run:
    cd C:/AI/Thesis/UTraffic-ML
    python ml_core/src/models/ML_BranchA/scripts/00_prepare_branchA_common_from_osm.py --overwrite

Split rule:
    - train/val/test are split by whole dates.
    - No date is allowed to appear in more than one split.
    - If input npz already has train_idx/val_idx/test_idx, this script validates them.
    - If input npz does not have split indices, it creates by-date split fallback.

Debug:
    python ml_core/src/models/ML_BranchA/scripts/00_prepare_branchA_common_from_osm.py --max-nodes 512 --overwrite
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

WINDOW = 10
DEFAULT_FEATURE = "average_speed"
DEFAULT_DTYPE = "float16"

THIS_FILE = Path(__file__).resolve()
ML_BRANCH_A_ROOT = THIS_FILE.parents[1]  # .../ML_BranchA
PROJECT_ROOT = THIS_FILE.parents[5]      # .../UTraffic-ML

DEFAULT_INPUT = PROJECT_ROOT / "ml_core" / "src" / "data_processing" / "outputs" / "branchA" / "osm_edge_forecasting_dataset" / "osm_edge_tensor.npz"
DEFAULT_OUT = ML_BRANCH_A_ROOT / "data" / "05_branchA_prepare_segment_segment_rt"


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def print_stage(x):
    print("\n" + "=" * 90)
    print(f"{now()} | {x}")
    print("=" * 90)


def decode_str_array(arr):
    out = []
    for x in arr:
        out.append(x.decode("utf-8") if isinstance(x, bytes) else str(x))
    return np.array(out)


def get_npz_key(data, candidates):
    for k in candidates:
        if k in data.files:
            return k
    return None


_TIME_HHMM_COLON = re.compile(r"(?<!\d)([0-2]?\d):([0-5]\d)(?!\d)")
_TIME_HHMM_COMPACT = re.compile(r"(?<!\d)([0-2]\d)([0-5]\d)(?!\d)")


def parse_start_minutes(time_label):
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


def parse_timestamp_key(ts: str, fallback_idx: int):
    s = str(ts)
    if "__" in s:
        date_part, time_part = s.split("__", 1)
    else:
        dt = pd.to_datetime(s, errors="coerce")
        if pd.notna(dt):
            dt = pd.Timestamp(dt)
            return {
                "timestamp_local": dt,
                "date": str(dt.date()),
                "session_id": str(dt.date()),
                "time_set_id": dt.strftime("%H:%M"),
                "slot_index": fallback_idx,
                "tod_minutes": int(dt.hour * 60 + dt.minute),
            }
        date_part, time_part = s, f"idx_{fallback_idx}"

    date_dt = pd.to_datetime(date_part, errors="coerce")
    tod = parse_start_minutes(time_part)
    if pd.isna(date_dt):
        base = pd.Timestamp("1970-01-01")
        tod = fallback_idx if tod is None else tod
        timestamp_local = base + pd.Timedelta(minutes=int(tod))
        date_str = str(base.date())
    else:
        date_dt = pd.Timestamp(date_dt)
        tod = fallback_idx if tod is None else tod
        timestamp_local = pd.Timestamp(date_dt.date()) + pd.Timedelta(minutes=int(tod))
        date_str = str(date_dt.date())

    return {
        "timestamp_local": timestamp_local,
        "date": date_str,
        "session_id": date_str,
        "time_set_id": str(time_part),
        "slot_index": fallback_idx,
        "tod_minutes": int(tod),
    }



def safe_timestamp_value(x):
    try:
        return pd.Timestamp(str(x))
    except Exception:
        dt = pd.to_datetime(str(x), errors="coerce")
        if pd.isna(dt):
            return pd.Timestamp("1970-01-01")
        return pd.Timestamp(dt)


def build_meta(timestamps):
    rows = []
    for i, ts in enumerate(timestamps):
        row = parse_timestamp_key(str(ts), i)
        row["raw_timestamp_index"] = i
        rows.append(row)
    meta = pd.DataFrame(rows)
    meta["timestamp_local"] = pd.to_datetime(meta["timestamp_local"])
    meta = meta.sort_values(["timestamp_local", "raw_timestamp_index"]).reset_index(drop=True)
    meta["slot_index"] = meta.groupby("session_id").cumcount().astype(int)
    return meta


def split_indices_by_date_from_timestamps(
    timestamps: np.ndarray,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any], pd.DataFrame]:
    """
    Fallback split by whole dates if input npz has no train_idx/val_idx/test_idx.
    """
    meta = build_meta(timestamps)
    date_table = (
        meta[["date"]]
        .drop_duplicates()
        .assign(date_parsed=lambda df: pd.to_datetime(df["date"], errors="coerce"))
        .sort_values(["date_parsed", "date"], kind="stable")
        .reset_index(drop=True)
    )

    n_days = int(len(date_table))
    if n_days < 3:
        raise ValueError(f"Need at least 3 dates for train/val/test split, got {n_days}.")

    test_ratio = 1.0 - float(train_ratio) - float(val_ratio)
    if train_ratio <= 0 or val_ratio <= 0 or test_ratio <= 0:
        raise ValueError(f"Invalid ratios: train={train_ratio}, val={val_ratio}, test={test_ratio}")

    n_train = int(round(n_days * float(train_ratio)))
    n_val = int(round(n_days * float(val_ratio)))
    n_train = max(1, min(n_train, n_days - 2))
    n_val = max(1, min(n_val, n_days - n_train - 1))
    n_test = n_days - n_train - n_val

    train_dates = date_table.iloc[:n_train]["date"].astype(str).tolist()
    val_dates = date_table.iloc[n_train:n_train + n_val]["date"].astype(str).tolist()
    test_dates = date_table.iloc[n_train + n_val:]["date"].astype(str).tolist()

    split_dates = {
        "train": train_dates,
        "val": val_dates,
        "test": test_dates,
    }

    splits = {}
    for split, dates in split_dates.items():
        splits[split] = meta.index[meta["date"].astype(str).isin(dates)].to_numpy(dtype=np.int64)

    day_rows = []
    for split, dates in split_dates.items():
        for d in dates:
            day_rows.append({
                "date": d,
                "split": split,
                "n_timestamps": int((meta["date"].astype(str) == d).sum()),
            })
    split_day_table = pd.DataFrame(day_rows)
    split_day_table["date_parsed"] = pd.to_datetime(split_day_table["date"], errors="coerce")
    split_day_table = split_day_table.sort_values(["date_parsed", "split"], kind="stable").reset_index(drop=True)

    split_info = {
        "split_mode": "by_whole_date_fallback_in_00_prepare",
        "n_days": int(n_days),
        "n_train_days": int(n_train),
        "n_val_days": int(n_val),
        "n_test_days": int(n_test),
        "train_dates": train_dates,
        "val_dates": val_dates,
        "test_dates": test_dates,
        "train_date_start": train_dates[0],
        "train_date_end": train_dates[-1],
        "val_date_start": val_dates[0],
        "val_date_end": val_dates[-1],
        "test_date_start": test_dates[0],
        "test_date_end": test_dates[-1],
    }

    return splits, split_info, split_day_table


def validate_splits_are_whole_dates(
    timestamps: np.ndarray,
    splits: Dict[str, np.ndarray],
) -> Dict[str, Any]:
    """
    Validate that no date appears in multiple splits.
    """
    full_meta = build_meta(timestamps)
    date_sets: Dict[str, set] = {}
    split_info: Dict[str, Any] = {"split_mode": "from_input_indices_validated_by_whole_date"}

    for split in ["train", "val", "test"]:
        idx = np.asarray(splits[split], dtype=np.int64)
        sub = full_meta.iloc[idx].copy()
        dates = sorted(sub["date"].astype(str).unique().tolist())
        date_sets[split] = set(dates)
        split_info[f"{split}_dates"] = dates
        split_info[f"n_{split}_days"] = int(len(dates))
        split_info[f"{split}_size"] = int(len(idx))
        split_info[f"{split}_date_start"] = dates[0] if dates else None
        split_info[f"{split}_date_end"] = dates[-1] if dates else None

    overlaps = {
        "train_val": sorted(date_sets["train"] & date_sets["val"]),
        "train_test": sorted(date_sets["train"] & date_sets["test"]),
        "val_test": sorted(date_sets["val"] & date_sets["test"]),
    }
    bad = {k: v for k, v in overlaps.items() if v}
    if bad:
        raise ValueError(
            "Invalid split: at least one date appears in multiple splits. "
            f"Overlaps={bad}. Re-run prepare_osm_edge_forecasting_dataset.py with by-date split."
        )

    split_info["n_days"] = int(sum(len(date_sets[x]) for x in ["train", "val", "test"]))
    split_info["overlap_check"] = "passed_no_date_overlap"
    return split_info


def session_groups(meta):
    groups = []
    if "session_id" not in meta.columns:
        return [np.arange(len(meta), dtype=np.int64)]
    for _, sub in meta.groupby("session_id", sort=False):
        idx = sub.index.to_numpy(dtype=np.int64)
        if len(idx):
            groups.append(idx)
    return groups or [np.arange(len(meta), dtype=np.int64)]


def count_rt(meta, window):
    return int(sum(max(0, len(idx) - window + 1) for idx in session_groups(meta)))


def corr_window(W, eps=1e-8):
    W = W.astype(np.float32, copy=False)
    W = np.nan_to_num(W, nan=0.0, posinf=0.0, neginf=0.0)
    n = W.shape[0]
    mu = W.mean(axis=0, keepdims=True, dtype=np.float32)
    Xc = W - mu
    std = Xc.std(axis=0, ddof=1, keepdims=True).astype(np.float32)
    denom = std * math.sqrt(max(n - 1, 1))
    Z = np.zeros_like(Xc, dtype=np.float32)
    np.divide(Xc, denom, out=Z, where=denom > eps)
    R = (Z.T @ Z).astype(np.float32)
    R = np.nan_to_num(R, nan=0.0, posinf=0.0, neginf=0.0)
    np.clip(R, -1.0, 1.0, out=R)
    np.fill_diagonal(R, 1.0)
    return R


def choose_feature(X, feature_names, feature):
    if X.ndim == 2:
        return X.astype(np.float32), 0, "value"
    names = [str(x) for x in feature_names.tolist()]
    if feature not in names:
        print(f"WARNING: feature={feature} not found. Use {names[0]}")
        idx = 0
    else:
        idx = names.index(feature)
    return X[:, :, idx].astype(np.float32), idx, names[idx]


def load_input(input_npz, prefer, feature):
    if not input_npz.exists():
        raise FileNotFoundError(
            f"Cannot find input tensor: {input_npz}\n"
            "Run this first:\n"
            "  python ml_core/src/data_processing/prepare_osm_edge_forecasting_dataset.py"
        )
    data = np.load(input_npz, allow_pickle=True)
    if prefer == "norm":
        x_key = get_npz_key(data, ["X_norm", "X_normalized", "X_filled", "X"])
    elif prefer == "filled":
        x_key = get_npz_key(data, ["X_filled", "X_norm", "X_normalized", "X"])
    else:
        x_key = get_npz_key(data, ["X_raw", "X", "X_filled", "X_norm"])
    if x_key is None:
        raise KeyError(f"No X key found in {input_npz}. Keys={data.files}")

    X = data[x_key]
    feature_names = decode_str_array(data["feature_names"]) if "feature_names" in data.files else np.array([f"feature_{i}" for i in range(X.shape[2])])
    X2, feature_idx, feature_used = choose_feature(X, feature_names, feature)

    timestamps = decode_str_array(data["timestamps"]) if "timestamps" in data.files else np.array([str(i) for i in range(X2.shape[0])])
    model_node_ids = data["model_node_ids"].astype(np.int64) if "model_node_ids" in data.files else np.arange(X2.shape[1], dtype=np.int64)

    train_idx = data["train_idx"].astype(np.int64) if "train_idx" in data.files else None
    val_idx = data["val_idx"].astype(np.int64) if "val_idx" in data.files else None
    test_idx = data["test_idx"].astype(np.int64) if "test_idx" in data.files else None

    if train_idx is None or val_idx is None or test_idx is None:
        print("[SPLIT] Input npz has no train_idx/val_idx/test_idx. Creating fallback split by whole dates.")
        splits, split_info, split_day_table = split_indices_by_date_from_timestamps(timestamps)
    else:
        splits = {"train": train_idx, "val": val_idx, "test": test_idx}
        split_info = validate_splits_are_whole_dates(timestamps, splits)
        split_day_table = pd.concat(
            [
                pd.DataFrame({"date": split_info[f"{split}_dates"], "split": split})
                for split in ["train", "val", "test"]
            ],
            ignore_index=True,
        )
        split_day_table["date_parsed"] = pd.to_datetime(split_day_table["date"], errors="coerce")
        split_day_table = split_day_table.sort_values(["date_parsed", "split"], kind="stable").reset_index(drop=True)

    return X2, timestamps, model_node_ids, splits, x_key, feature_idx, feature_used, split_info, split_day_table


def build_split(out_dir, split, X2, timestamps, model_node_ids, idx, window, dtype, overwrite):
    split_dir = out_dir / split
    if overwrite and split_dir.exists():
        shutil.rmtree(split_dir)
    split_dir.mkdir(parents=True, exist_ok=True)

    z = X2[idx].astype(np.float32)
    ts = timestamps[idx].astype(str)
    meta = build_meta(ts)
    meta["raw_global_timestamp_idx"] = idx
    meta["split_local_idx"] = np.arange(len(idx), dtype=np.int64)

    n_samples = count_rt(meta, window)
    T, N = z.shape
    print(f"\n[BUILD] {split}: T={T}, N={N}, R samples={n_samples}, dtype={dtype}")

    R_path = split_dir / "R_series.npy"
    R_mem = np.lib.format.open_memmap(R_path, mode="w+", dtype=dtype, shape=(n_samples, N, N))

    rows = []
    cursor = 0
    t0 = time.time()

    groups = session_groups(meta)
    for sidx in groups:
        if len(sidx) < window:
            continue
        for local_end in range(window - 1, len(sidx)):
            block_idx = sidx[local_end - window + 1: local_end + 1]
            global_t = int(sidx[local_end])
            R = corr_window(z[block_idx])
            R_mem[cursor] = R.astype(dtype)

            src = meta.iloc[global_t]
            row = {
                "sample_id": int(cursor),
                "raw_row_idx": int(global_t),
                "window_end_idx": int(global_t),
                "session_step": int(local_end),
                "timestamp_local": safe_timestamp_value(ts[global_t]),
            }
            for c in ["session_id", "date", "time_set_id", "slot_index", "tod_minutes", "raw_global_timestamp_idx", "split_local_idx"]:
                if c in meta.columns:
                    row[c] = src[c]
            rows.append(row)
            cursor += 1

            if cursor % 5 == 0 or cursor == n_samples:
                elapsed = time.time() - t0
                speed = cursor / max(elapsed, 1e-9)
                eta = (n_samples - cursor) / max(speed, 1e-9)
                print(f"  {split}: {cursor}/{n_samples} | {speed:.2f} R/s | ETA={eta/60:.1f} min")

    R_mem.flush()
    del R_mem

    R_meta = pd.DataFrame(rows)

    np.save(split_dir / "z.npy", z.astype(np.float32))
    np.save(split_dir / "segment_ids.npy", model_node_ids.astype(np.int64))
    np.save(split_dir / "timestamps.npy", np.asarray([safe_timestamp_value(x) for x in ts]).astype("datetime64[ns]"))

    R_meta.to_csv(split_dir / "R_series_meta.csv", index=False)
    meta.to_csv(split_dir / "raw_meta.csv", index=False)

    # Extra compatibility with old functions.
    np.savez_compressed(
        split_dir / "traffic_tensor_resid.npz",
        z=z.astype(np.float32),
        resid=z.astype(np.float32),
        speed=z.astype(np.float32),
        segment_ids=model_node_ids.astype(np.int64),
        timestamp_local=np.asarray([safe_timestamp_value(x) for x in ts]).astype("datetime64[ns]"),
    )
    meta.to_csv(split_dir / "traffic_tensor_resid_meta.csv", index=False)

    summary = {
        "split": split,
        "T": int(T),
        "N": int(N),
        "n_Rt": int(n_samples),
        "R_shape": [int(n_samples), int(N), int(N)],
        "R_path": str(R_path),
        "dtype": dtype,
    }
    with open(split_dir / "branchA_common_split_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("[DONE]", split, summary)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default=str(DEFAULT_INPUT))
    ap.add_argument("--output", type=str, default=str(DEFAULT_OUT))
    ap.add_argument("--feature", type=str, default=DEFAULT_FEATURE)
    ap.add_argument("--prefer", type=str, default="norm", choices=["norm", "filled", "raw"])
    ap.add_argument("--window", type=int, default=WINDOW)
    ap.add_argument("--dtype", type=str, default=DEFAULT_DTYPE)
    ap.add_argument("--max-nodes", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    input_npz = Path(args.input).resolve()
    out_dir = Path(args.output).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print_stage("PREPARE BRANCH A COMMON FORMAT FROM OSM EDGE TENSOR")
    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("INPUT       :", input_npz)
    print("OUTPUT      :", out_dir)

    X2, timestamps, node_ids, splits, x_key, feature_idx, feature_used, split_info, split_day_table = load_input(input_npz, args.prefer, args.feature)

    print("[SPLIT] mode:", split_info.get("split_mode"))
    print("[SPLIT] train:", split_info.get("train_date_start"), "->", split_info.get("train_date_end"), f"({split_info.get('n_train_days')} days)")
    print("[SPLIT] val  :", split_info.get("val_date_start"), "->", split_info.get("val_date_end"), f"({split_info.get('n_val_days')} days)")
    print("[SPLIT] test :", split_info.get("test_date_start"), "->", split_info.get("test_date_end"), f"({split_info.get('n_test_days')} days)")
    split_day_table.to_csv(out_dir / "split_by_date.csv", index=False)

    if args.max_nodes is not None and args.max_nodes > 0 and X2.shape[1] > args.max_nodes:
        print(f"[DEBUG] keep first {args.max_nodes} nodes")
        X2 = X2[:, :args.max_nodes]
        node_ids = node_ids[:args.max_nodes]

    summaries = []
    for split in ["train", "val", "test"]:
        summaries.append(build_split(
            out_dir=out_dir,
            split=split,
            X2=X2,
            timestamps=timestamps,
            model_node_ids=node_ids,
            idx=splits[split],
            window=args.window,
            dtype=args.dtype,
            overwrite=args.overwrite,
        ))

    summary = {
        "input": str(input_npz),
        "x_key": x_key,
        "feature_used": feature_used,
        "feature_idx": int(feature_idx),
        "window": int(args.window),
        "dtype": args.dtype,
        "shape_X2": list(map(int, X2.shape)),
        "split_info": split_info,
        "splits": summaries,
    }
    with open(out_dir / "branchA_common_prepare_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print_stage("DONE")
    print("Output:", out_dir)


if __name__ == "__main__":
    main()
