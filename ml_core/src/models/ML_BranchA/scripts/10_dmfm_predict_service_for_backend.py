# ml_core/src/models/ML_BranchA/scripts/10_dmfm_predict_service_backend.py
"""
Backend on-demand DMFM prediction service for Branch A OSM-edge correlation matrices.

Purpose
-------
Given a date, time in day, and forecast step/horizon, this script:
1. Finds the correct non-train split (val/test) containing that timestamp.
2. Loads the current R_t from R_series.npy.
3. Uses the exported DMFM model to predict R_{t+h}.
4. Returns/saves:
      R_pred[i, j] = predicted correlation between segment_ids[i] and segment_ids[j]
      segment_ids[i] = OSM/model node id for matrix row/column i

Important backend rule
----------------------
Requests that belong to the TRAIN date range are rejected.
The backend service must not accept train timestamps for inference.

Run examples
------------
cd C:/AI/Thesis/UTraffic-ML

# Example: predict step 5 from 10/8 at 11:00, and save a self-contained NPZ bundle.
python ml_core/src/models/ML_BranchA/scripts/10_dmfm_predict_service_backend.py ^
  --date 10/8 --time 11h --step 5 --save-npz

# With explicit output path:
python ml_core/src/models/ML_BranchA/scripts/10_dmfm_predict_service_backend.py ^
  --date 2024-08-28 --time 11:00 --step 5 ^
  --output ml_core/src/models/ML_BranchA/artifacts/backend_predictions/demo_h5.npz

Python usage
------------
Because this filename starts with "10_", normal Python import syntax cannot import it directly.
For backend code, either:
- call it as a subprocess, or
- copy/rename this file to dmfm_predict_service_backend.py for direct import, then call:

    result = predict_by_datetime(date="10/8", time_in_day="11h", step=5)
    R_pred = result["R_pred"]
    segment_ids = result["segment_ids"]
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


THIS_FILE = Path(__file__).resolve()
ML_BRANCH_A_ROOT = THIS_FILE.parents[1]
PROJECT_ROOT = THIS_FILE.parents[5]

DEFAULT_COMMON_DIR = ML_BRANCH_A_ROOT / "data" / "05_branchA_prepare_segment_segment_rt"
DEFAULT_MODEL_PATH = ML_BRANCH_A_ROOT / "artifacts" / "dmfm_model" / "dmfm_model.npz"
DEFAULT_OUTPUT_DIR = ML_BRANCH_A_ROOT / "artifacts" / "backend_predictions"

ALLOWED_INFERENCE_SPLITS = ("val", "test")


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def print_stage(title: str) -> None:
    print("\n" + "=" * 96)
    print(f"{now()} | {title}")
    print("=" * 96)


def sym_clip_diag(R: np.ndarray, diag_value: float = 1.0) -> np.ndarray:
    R = np.asarray(R, dtype=np.float32)
    R = 0.5 * (R + R.T)
    np.nan_to_num(R, copy=False, nan=0.0, posinf=1.0, neginf=-1.0)
    np.clip(R, -1.0, 1.0, out=R)
    np.fill_diagonal(R, diag_value)
    return R.astype(np.float32, copy=False)


def load_model(model_path: Path) -> Dict[str, Any]:
    if not model_path.exists():
        raise FileNotFoundError(
            f"Cannot find DMFM model: {model_path}\n"
            "Run 09_train_dmfm_export_model.py first."
        )

    data = np.load(model_path, allow_pickle=False)
    return {
        "n": int(data["n"][0]),
        "rank": int(data["rank"][0]),
        "mean_vec": data["mean_vec"].astype(np.float32),
        "components": data["components"].astype(np.float32),
        "A": data["A"].astype(np.float32),
        "segment_ids": data["segment_ids"].astype(np.int64),
    }


def load_split_meta(split_dir: Path) -> pd.DataFrame:
    meta_path = split_dir / "R_series_meta.csv"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing R_series_meta.csv: {meta_path}")

    meta = pd.read_csv(meta_path)
    if "timestamp_local" not in meta.columns:
        raise ValueError(f"{meta_path} must contain timestamp_local column.")

    meta["timestamp_local"] = pd.to_datetime(meta["timestamp_local"], errors="coerce")
    meta = meta.dropna(subset=["timestamp_local"]).reset_index(drop=True)
    meta["_date"] = meta["timestamp_local"].dt.date.astype(str)
    meta["_time_hhmm"] = meta["timestamp_local"].dt.strftime("%H:%M")

    if "sample_id" not in meta.columns:
        meta["sample_id"] = np.arange(len(meta), dtype=np.int64)

    return meta


def load_split_date_set(split_dir: Path, meta: pd.DataFrame) -> set:
    """
    Prefer raw_meta.csv because it contains every timestamp in the split,
    including early slots that cannot form R_t yet.
    """
    raw_meta_path = split_dir / "raw_meta.csv"
    if raw_meta_path.exists():
        raw = pd.read_csv(raw_meta_path)
        if "date" in raw.columns:
            return set(raw["date"].astype(str).dropna().unique().tolist())
        if "timestamp_local" in raw.columns:
            ts = pd.to_datetime(raw["timestamp_local"], errors="coerce")
            return set(ts.dropna().dt.date.astype(str).unique().tolist())

    return set(meta["_date"].astype(str).unique().tolist())


def load_common_data(common_dir: Path, include_r_series: Sequence[str] = ALLOWED_INFERENCE_SPLITS) -> Dict[str, Any]:
    if not common_dir.exists():
        raise FileNotFoundError(
            f"Cannot find common Branch A directory: {common_dir}\n"
            "Run 00_prepare_branchA_common_from_osm.py first."
        )

    splits: Dict[str, Dict[str, Any]] = {}
    for split in ["train", "val", "test"]:
        split_dir = common_dir / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Missing split directory: {split_dir}")

        meta = load_split_meta(split_dir)
        date_set = load_split_date_set(split_dir, meta)

        split_obj: Dict[str, Any] = {
            "split": split,
            "dir": split_dir,
            "meta": meta,
            "date_set": date_set,
        }

        if split in include_r_series:
            r_path = split_dir / "R_series.npy"
            ids_path = split_dir / "segment_ids.npy"
            if not r_path.exists():
                raise FileNotFoundError(f"Missing R_series.npy: {r_path}")
            if not ids_path.exists():
                raise FileNotFoundError(f"Missing segment_ids.npy: {ids_path}")

            split_obj["R_series"] = np.load(r_path, mmap_mode="r")
            split_obj["segment_ids"] = np.load(ids_path).astype(np.int64)

        splits[split] = split_obj

    return {"common_dir": common_dir, "splits": splits}


_TIME_H_RE = re.compile(r"^\s*(\d{1,2})\s*[hH]\s*(\d{1,2})?\s*$")
_TIME_COLON_RE = re.compile(r"^\s*(\d{1,2})\s*:\s*(\d{1,2})(?::\d{1,2})?\s*$")
_TIME_COMPACT_RE = re.compile(r"^\s*(\d{2})(\d{2})\s*$")


def normalize_time_to_hhmm(time_in_day: str) -> str:
    """
    Accept common Vietnamese/backend forms:
    - 11h
    - 11h30
    - 11:00
    - 11:00:00
    - 1100
    """
    s = str(time_in_day).strip()

    m = _TIME_H_RE.match(s)
    if m:
        hh = int(m.group(1))
        mm = int(m.group(2) or 0)
        if 0 <= hh <= 23 and 0 <= mm <= 59:
            return f"{hh:02d}:{mm:02d}"

    m = _TIME_COLON_RE.match(s)
    if m:
        hh = int(m.group(1))
        mm = int(m.group(2))
        if 0 <= hh <= 23 and 0 <= mm <= 59:
            return f"{hh:02d}:{mm:02d}"

    m = _TIME_COMPACT_RE.match(s)
    if m:
        hh = int(m.group(1))
        mm = int(m.group(2))
        if 0 <= hh <= 23 and 0 <= mm <= 59:
            return f"{hh:02d}:{mm:02d}"

    raise ValueError(f"Cannot parse time_in_day={time_in_day!r}. Use examples: 11h, 11:00, 11h30, 1100.")


def available_dates_from_common(common_data: Dict[str, Any]) -> List[pd.Timestamp]:
    dates: List[pd.Timestamp] = []
    for split_obj in common_data["splits"].values():
        for d in split_obj["date_set"]:
            dt = pd.to_datetime(str(d), errors="coerce")
            if pd.notna(dt):
                dates.append(pd.Timestamp(dt).normalize())
    return sorted(set(dates))


def parse_request_date(date: str, common_data: Dict[str, Any], default_year: Optional[int] = None) -> pd.Timestamp:
    """
    Parse date input.

    Supported:
    - 2024-08-10
    - 10/8/2024
    - 10/8       -> infer year from prepared data
    - 10-8       -> infer year from prepared data
    """
    s = str(date).strip()
    available_dates = available_dates_from_common(common_data)
    if not available_dates:
        raise ValueError("Cannot infer date because no available split dates were found.")

    # Explicit ISO or year-containing input.
    if re.search(r"\d{4}", s):
        dt = pd.to_datetime(s, errors="coerce", dayfirst=True)
        if pd.isna(dt):
            raise ValueError(f"Cannot parse date={date!r}.")
        return pd.Timestamp(dt).normalize()

    # Day/month without year.
    m = re.match(r"^\s*(\d{1,2})\s*[/-]\s*(\d{1,2})\s*$", s)
    if not m:
        dt = pd.to_datetime(s, errors="coerce", dayfirst=True)
        if pd.isna(dt):
            raise ValueError(f"Cannot parse date={date!r}. Use examples: 2024-08-10 or 10/8.")
        return pd.Timestamp(dt).normalize()

    day = int(m.group(1))
    month = int(m.group(2))

    # If the exact day/month exists in prepared data, use that year.
    candidates = [d for d in available_dates if d.day == day and d.month == month]
    if candidates:
        return pd.Timestamp(candidates[0]).normalize()

    if default_year is None:
        years = sorted(set(d.year for d in available_dates))
        if len(years) == 1:
            default_year = years[0]
        else:
            raise ValueError(
                f"Date {date!r} has no year and available data has multiple years={years}. "
                "Pass --year explicitly."
            )

    return pd.Timestamp(year=int(default_year), month=month, day=day)


def parse_request_timestamp(
    date: str,
    time_in_day: str,
    common_data: Dict[str, Any],
    default_year: Optional[int] = None,
) -> pd.Timestamp:
    d = parse_request_date(date, common_data, default_year=default_year)
    hhmm = normalize_time_to_hhmm(time_in_day)
    hh, mm = map(int, hhmm.split(":"))
    return pd.Timestamp(year=d.year, month=d.month, day=d.day, hour=hh, minute=mm)


def predict_dmfm_matrix(model: Dict[str, Any], R_origin: np.ndarray, horizon: int) -> np.ndarray:
    if horizon <= 0:
        raise ValueError(f"step/horizon must be positive, got {horizon}.")

    N = model["n"]
    iu = np.triu_indices(N, k=1)

    R_origin = sym_clip_diag(np.asarray(R_origin, dtype=np.float32))
    vec = R_origin[iu].astype(np.float32)

    mean_vec = model["mean_vec"]
    components = model["components"]
    A = model["A"]

    score = (vec - mean_vec) @ components
    A_pow = np.linalg.matrix_power(A, int(horizon)).astype(np.float32)
    pred_score = score @ A_pow
    pred_vec = mean_vec + pred_score @ components.T

    R_pred = np.eye(N, dtype=np.float32)
    R_pred[iu] = pred_vec.astype(np.float32)
    R_pred[(iu[1], iu[0])] = pred_vec.astype(np.float32)

    return sym_clip_diag(R_pred)


def find_available_times_message(split_meta: pd.DataFrame, query_date: str, max_items: int = 20) -> str:
    same_day = split_meta[split_meta["_date"] == query_date].copy()
    if same_day.empty:
        return "No available R_t timestamp on that date in this split."
    times = same_day["_time_hhmm"].drop_duplicates().tolist()
    head = ", ".join(times[:max_items])
    more = "" if len(times) <= max_items else f", ... ({len(times)} times total)"
    return f"Available R_t times on {query_date}: {head}{more}"


def locate_inference_sample(
    query_ts: pd.Timestamp,
    common_data: Dict[str, Any],
    allowed_splits: Sequence[str] = ALLOWED_INFERENCE_SPLITS,
) -> Tuple[str, int, Dict[str, Any]]:
    query_date = str(query_ts.date())
    query_hhmm = query_ts.strftime("%H:%M")
    splits = common_data["splits"]

    # Hard rule: backend must not accept training dates.
    if query_date in splits["train"]["date_set"]:
        train_dates = sorted(splits["train"]["date_set"])
        raise PermissionError(
            f"Rejected: {query_ts} belongs to TRAIN date range/date set. "
            "Backend inference is only allowed for validation/test dates. "
            f"Train dates: {train_dates[0]} -> {train_dates[-1]}."
        )

    candidate_splits = []
    for split in allowed_splits:
        if query_date in splits[split]["date_set"]:
            candidate_splits.append(split)

    if not candidate_splits:
        val_dates = sorted(splits["val"]["date_set"])
        test_dates = sorted(splits["test"]["date_set"])
        raise LookupError(
            f"Date {query_date} does not belong to val/test split. "
            f"Val dates: {val_dates[0]} -> {val_dates[-1]}; "
            f"Test dates: {test_dates[0]} -> {test_dates[-1]}."
        )

    for split in candidate_splits:
        meta = splits[split]["meta"]
        hit = meta[meta["timestamp_local"] == query_ts]
        if not hit.empty:
            row = hit.iloc[0].to_dict()
            sample_id = int(row.get("sample_id", int(hit.index[0])))
            return split, sample_id, row

    # Date is allowed, but exact time has no R_t. Usually because rolling window=10
    # means early morning slots such as 06:00-08:00 cannot form R_t yet.
    messages = []
    for split in candidate_splits:
        messages.append(f"[{split}] " + find_available_times_message(splits[split]["meta"], query_date))
    raise LookupError(
        f"No R_t found for {query_ts} (date={query_date}, time={query_hhmm}). "
        "This may happen if the requested time is too early to have a rolling window. "
        + " | ".join(messages)
    )


def predict_by_datetime(
    date: str,
    time_in_day: str,
    step: int,
    common_dir: Path | str = DEFAULT_COMMON_DIR,
    model_path: Path | str = DEFAULT_MODEL_PATH,
    default_year: Optional[int] = None,
    allowed_splits: Sequence[str] = ALLOWED_INFERENCE_SPLITS,
    return_current: bool = False,
) -> Dict[str, Any]:
    """
    Backend-friendly function.

    Args:
        date: e.g. "10/8", "10/8/2024", "2024-08-10"
        time_in_day: e.g. "11h", "11:00", "11h30"
        step: forecast horizon h. Example: 5 means predict R_{t+5} from current R_t.
        common_dir: prepared Branch A common directory.
        model_path: exported DMFM model npz.
        default_year: optional year used when date has no year.
        allowed_splits: default ("val", "test"). Train is always rejected by date.
        return_current: include R_current in returned dict.

    Returns:
        dict with R_pred, segment_ids, split, source_sample_id, request_timestamp, metadata.
    """
    common_dir = Path(common_dir).resolve()
    model_path = Path(model_path).resolve()

    model = load_model(model_path)
    common_data = load_common_data(common_dir, include_r_series=allowed_splits)
    query_ts = parse_request_timestamp(date, time_in_day, common_data, default_year=default_year)

    split, sample_id, meta_row = locate_inference_sample(query_ts, common_data, allowed_splits=allowed_splits)
    split_data = common_data["splits"][split]

    segment_ids = split_data["segment_ids"]
    if not np.array_equal(model["segment_ids"], segment_ids):
        raise ValueError(
            "segment_ids in model and requested split do not match. "
            "Re-run 09_train_dmfm_export_model.py using the same prepared data."
        )

    R_series = split_data["R_series"]
    if sample_id < 0 or sample_id >= R_series.shape[0]:
        raise IndexError(f"sample_id={sample_id} out of range for split={split}, R_series shape={R_series.shape}")

    R_current = np.asarray(R_series[sample_id], dtype=np.float32)
    R_pred = predict_dmfm_matrix(model, R_current, int(step))

    result: Dict[str, Any] = {
        "R_pred": R_pred,
        "segment_ids": segment_ids.astype(np.int64),
        "split": split,
        "source_sample_id": int(sample_id),
        "step": int(step),
        "request_timestamp": str(query_ts),
        "timestamp_local": str(meta_row.get("timestamp_local", query_ts)),
        "matrix_contract": "R_pred[i,j] is predicted correlation between segment_ids[i] and segment_ids[j].",
        "model_path": str(model_path),
        "common_dir": str(common_dir),
        "meta_row": meta_row,
    }
    if return_current:
        result["R_current"] = sym_clip_diag(R_current)

    return result


def save_prediction_bundle(
    result: Dict[str, Any],
    output_path: Path,
    dtype: str = "float16",
    include_current: bool = False,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, Any] = {
        "R_pred": result["R_pred"].astype(dtype),
        "segment_ids": result["segment_ids"].astype(np.int64),
        "matrix_index": np.arange(len(result["segment_ids"]), dtype=np.int64),
        "split": np.array(str(result["split"])),
        "source_sample_id": np.array([int(result["source_sample_id"])], dtype=np.int64),
        "step": np.array([int(result["step"])], dtype=np.int64),
        "request_timestamp": np.array(str(result["request_timestamp"])),
        "timestamp_local": np.array(str(result["timestamp_local"])),
        "matrix_contract": np.array(str(result["matrix_contract"])),
    }
    if include_current and "R_current" in result:
        payload["R_current"] = result["R_current"].astype(dtype)

    np.savez_compressed(output_path, **payload)

    meta_json = output_path.with_suffix(".json")
    json_payload = {
        k: v
        for k, v in result.items()
        if k not in {"R_pred", "segment_ids", "R_current"}
    }
    with open(meta_json, "w", encoding="utf-8") as f:
        json.dump(json_payload, f, ensure_ascii=False, indent=2, default=str)

    return output_path


def build_default_output_path(output_dir: Path, result: Dict[str, Any]) -> Path:
    ts = pd.Timestamp(result["request_timestamp"])
    name = f"dmfm_backend_pred_{result['split']}_{ts:%Y%m%d_%H%M}_h{int(result['step'])}.npz"
    return output_dir / name


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Backend on-demand DMFM predictor for Branch A R_t matrices.")
    ap.add_argument("--common-dir", type=str, default=str(DEFAULT_COMMON_DIR))
    ap.add_argument("--model-path", type=str, default=str(DEFAULT_MODEL_PATH))
    ap.add_argument("--date", type=str, required=True, help="Date, e.g. 2024-08-10, 10/8/2024, or 10/8.")
    ap.add_argument("--time", type=str, required=True, help="Time in day, e.g. 11h, 11:00, 11h30.")
    ap.add_argument("--step", type=int, required=True, help="Forecast horizon h, e.g. 5.")
    ap.add_argument("--year", type=int, default=None, help="Optional year if --date has no year.")
    ap.add_argument("--dtype", type=str, default="float16", choices=["float16", "float32"])
    ap.add_argument("--output", type=str, default=None, help="Optional output .npz path.")
    ap.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    ap.add_argument("--save-npz", action="store_true", help="Save a self-contained NPZ bundle for backend/OSM plotting.")
    ap.add_argument("--include-current", action="store_true", help="Also save/return current R_t as R_current.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    print_stage("DMFM BACKEND ON-DEMAND PREDICT")
    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("COMMON_DIR  :", Path(args.common_dir).resolve())
    print("MODEL_PATH  :", Path(args.model_path).resolve())
    print("date/time   :", args.date, args.time)
    print("step        :", args.step)
    print("rule        : reject train dates; allow val/test only")

    result = predict_by_datetime(
        date=args.date,
        time_in_day=args.time,
        step=args.step,
        common_dir=args.common_dir,
        model_path=args.model_path,
        default_year=args.year,
        return_current=args.include_current,
    )

    print("[OK] split             :", result["split"])
    print("[OK] source_sample_id  :", result["source_sample_id"])
    print("[OK] timestamp_local   :", result["timestamp_local"])
    print("[OK] R_pred shape      :", result["R_pred"].shape)
    print("[OK] segment_ids shape :", result["segment_ids"].shape)
    print("[OK] contract          :", result["matrix_contract"])

    should_save = bool(args.save_npz or args.output)
    if should_save:
        if args.output:
            out_path = Path(args.output).resolve()
        else:
            out_path = build_default_output_path(Path(args.output_dir).resolve(), result)
        save_prediction_bundle(
            result=result,
            output_path=out_path,
            dtype=args.dtype,
            include_current=args.include_current,
        )
        print("[SAVED] bundle:", out_path)
        print("[SAVED] meta  :", out_path.with_suffix(".json"))

    print_stage("DONE")


if __name__ == "__main__":
    main()
