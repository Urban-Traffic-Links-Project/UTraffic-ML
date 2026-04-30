# ml_core/src/models/ML_BranchA/scripts/11_batch_export_dmfm_predictions.py
"""
Batch export DMFM R_pred with sliding origin time.

Default behavior:
- User selects start day and end day, e.g. 27 -> 31.
- Origin time slides through every available R_t in that day, starting from --start-time.
- max_step defaults to 9.
- For each origin, export h=1..max_step only if the target timestamp exists inside the same R_t window.
- If target timestamp exceeds available R_t window, stop horizons for that origin.

Example:
    python ml_core/src/models/ML_BranchA/scripts/11_batch_export_dmfm_predictions.py ^
      --start-day 27 ^
      --end-day 31 ^
      --month 8 ^
      --year 2024 ^
      --start-time 08:15

If available R_t is 08:15 -> 11:45 and max_step=9:
    08:15 exports h1..h9
    11:30 exports h1 only
    11:45 exports nothing
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


THIS_FILE = Path(__file__).resolve()
SCRIPT_DIR = THIS_FILE.parent

SERVICE_CANDIDATES = [
    SCRIPT_DIR / "10_function_helper.py",
]


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def print_stage(title: str) -> None:
    print("\n" + "=" * 96)
    print(f"{now()} | {title}")
    print("=" * 96)


def load_service_module():
    service_path = None
    for p in SERVICE_CANDIDATES:
        if p.exists():
            service_path = p
            break

    if service_path is None:
        raise FileNotFoundError(
            "Cannot find DMFM service file. Expected one of:\n"
            + "\n".join(str(p) for p in SERVICE_CANDIDATES)
        )

    spec = importlib.util.spec_from_file_location("dmfm_predict_service_backend", service_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import service module from: {service_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, service_path


service, SERVICE_PATH = load_service_module()


def build_date_string(year: int, month: int, day: int) -> str:
    return f"{int(year):04d}-{int(month):02d}-{int(day):02d}"


def hhmm_to_minutes(hhmm: str) -> int:
    hhmm = service.normalize_time_to_hhmm(hhmm)
    hh, mm = map(int, hhmm.split(":"))
    return hh * 60 + mm


def clean_time_tag(hhmm: str) -> str:
    return service.normalize_time_to_hhmm(hhmm).replace(":", "")


def default_output_dir(args: argparse.Namespace) -> Path:
    folder = (
        f"sliding_d{args.start_day:02d}_d{args.end_day:02d}_"
        f"from_{clean_time_tag(args.start_time)}_h1_h{args.max_step}"
    )
    return Path(service.DEFAULT_OUTPUT_DIR) / folder


def save_json(obj: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Batch export sliding DMFM R_pred matrices for backend/OSM plotting."
    )

    ap.add_argument("--common-dir", type=str, default=str(service.DEFAULT_COMMON_DIR))
    ap.add_argument("--model-path", type=str, default=str(service.DEFAULT_MODEL_PATH))

    ap.add_argument("--year", type=int, default=2024)
    ap.add_argument("--month", type=int, default=8)
    ap.add_argument("--start-day", type=int, required=True)
    ap.add_argument("--end-day", type=int, required=True)

    ap.add_argument(
        "--start-time",
        type=str,
        default="08:15",
        help="First origin R_t time to export from. Default: 08:15.",
    )
    ap.add_argument(
        "--end-time",
        type=str,
        default=None,
        help="Optional last origin time. If omitted, use all available R_t times until the end of day.",
    )
    ap.add_argument(
        "--time-interval-min",
        type=int,
        default=15,
        help="Forecast step size in minutes. Default: 15.",
    )
    ap.add_argument(
        "--max-step",
        type=int,
        default=9,
        help="Maximum forecast horizon. Default: 9.",
    )

    ap.add_argument("--dtype", type=str, default="float16", choices=["float16", "float32"])
    ap.add_argument("--output-dir", type=str, default=None)

    ap.add_argument("--include-current", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--stop-on-error", action="store_true")

    return ap.parse_args()


def get_split_for_date(
    date_str: str,
    common_data: Dict[str, Any],
    allowed_splits: Sequence[str],
) -> str:
    splits = common_data["splits"]

    if date_str in splits["train"]["date_set"]:
        train_dates = sorted(splits["train"]["date_set"])
        raise PermissionError(
            f"Rejected: {date_str} belongs to TRAIN date range/date set. "
            f"Train dates: {train_dates[0]} -> {train_dates[-1]}."
        )

    matched = []
    for split in allowed_splits:
        if date_str in splits[split]["date_set"]:
            matched.append(split)

    if not matched:
        val_dates = sorted(splits["val"]["date_set"])
        test_dates = sorted(splits["test"]["date_set"])
        raise LookupError(
            f"Date {date_str} does not belong to val/test split. "
            f"Val dates: {val_dates[0]} -> {val_dates[-1]}; "
            f"Test dates: {test_dates[0]} -> {test_dates[-1]}."
        )

    return matched[0]


def get_available_origins_for_date(
    split_data: Dict[str, Any],
    date_str: str,
    start_time: str,
    end_time: Optional[str],
) -> pd.DataFrame:
    """
    Return available R_t origins for one date.

    Important fix:
    Some prepared R_series_meta.csv may have timestamp_local = 1970-01-01,
    while date/time_set_id/tod_minutes are correct. Therefore, we rebuild
    timestamp_local from date + tod_minutes when possible.
    """
    meta = split_data["meta"].copy()

    # 1) Choose real date column
    if "date" in meta.columns:
        meta["_real_date"] = meta["date"].astype(str)
    elif "session_id" in meta.columns:
        meta["_real_date"] = meta["session_id"].astype(str)
    else:
        meta["_real_date"] = pd.to_datetime(
            meta["timestamp_local"], errors="coerce"
        ).dt.date.astype(str)

    # 2) Rebuild timestamp_local if tod_minutes exists
    if "tod_minutes" in meta.columns:
        base_date = pd.to_datetime(meta["_real_date"], errors="coerce")
        tod = pd.to_numeric(meta["tod_minutes"], errors="coerce")

        rebuilt_ts = base_date + pd.to_timedelta(tod.fillna(0).astype(int), unit="m")

        # Use rebuilt timestamp when valid
        meta["timestamp_local"] = rebuilt_ts
    else:
        meta["timestamp_local"] = pd.to_datetime(
            meta["timestamp_local"], errors="coerce"
        )

    # 3) Create date/time columns from rebuilt timestamp
    meta["_date"] = meta["_real_date"].astype(str)
    meta["_time_hhmm"] = pd.to_datetime(meta["timestamp_local"]).dt.strftime("%H:%M")
    meta["_minutes"] = meta["_time_hhmm"].apply(hhmm_to_minutes)

    same_day = meta[meta["_date"].astype(str) == str(date_str)].copy()
    if same_day.empty:
        return same_day

    start_m = hhmm_to_minutes(start_time)
    same_day = same_day[same_day["_minutes"] >= start_m]

    if end_time is not None:
        end_m = hhmm_to_minutes(end_time)
        same_day = same_day[same_day["_minutes"] <= end_m]

    same_day = same_day.sort_values("timestamp_local").reset_index(drop=True)
    return same_day

def main() -> None:
    args = parse_args()

    if args.start_day > args.end_day:
        raise ValueError("--start-day must be <= --end-day.")
    if args.max_step <= 0:
        raise ValueError("--max-step must be positive.")
    if args.time_interval_min <= 0:
        raise ValueError("--time-interval-min must be positive.")

    common_dir = Path(args.common_dir).resolve()
    model_path = Path(args.model_path).resolve()
    out_dir = Path(args.output_dir).resolve() if args.output_dir else default_output_dir(args).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    days = list(range(int(args.start_day), int(args.end_day) + 1))

    print_stage("BATCH EXPORT SLIDING DMFM R_PRED")
    print("SERVICE_PATH      :", SERVICE_PATH)
    print("COMMON_DIR        :", common_dir)
    print("MODEL_PATH        :", model_path)
    print("OUTPUT_DIR        :", out_dir)
    print("YEAR/MONTH        :", args.year, args.month)
    print("DAYS              :", days)
    print("START_TIME        :", args.start_time)
    print("END_TIME          :", args.end_time if args.end_time else "auto: last available R_t")
    print("MAX_STEP          :", args.max_step)
    print("TIME_INTERVAL_MIN :", args.time_interval_min)
    print("DTYPE             :", args.dtype)
    print("OVERWRITE         :", args.overwrite)
    print("DRY_RUN           :", args.dry_run)

    print_stage("LOAD MODEL AND COMMON DATA ONCE")
    model = service.load_model(model_path)
    common_data = service.load_common_data(
        common_dir,
        include_r_series=service.ALLOWED_INFERENCE_SPLITS,
    )

    success_rows: List[Dict[str, Any]] = []
    skip_rows: List[Dict[str, Any]] = []
    error_rows: List[Dict[str, Any]] = []

    for day in days:
        date_str = build_date_string(args.year, args.month, day)

        print_stage(f"DATE {date_str}")

        try:
            split = get_split_for_date(
                date_str=date_str,
                common_data=common_data,
                allowed_splits=service.ALLOWED_INFERENCE_SPLITS,
            )

            split_data = common_data["splits"][split]
            segment_ids = split_data["segment_ids"]

            if not np.array_equal(model["segment_ids"], segment_ids):
                raise ValueError(
                    "segment_ids in model and requested split do not match. "
                    "Re-run 09_train_dmfm_export_model.py using the same prepared data."
                )

            day_meta = get_available_origins_for_date(
                split_data=split_data,
                date_str=date_str,
                start_time=args.start_time,
                end_time=args.end_time,
            )

            if day_meta.empty:
                print(f"[WARN] No available R_t origin for {date_str} from {args.start_time}.")
                skip_rows.append({
                    "date": date_str,
                    "reason": f"No available R_t origin from {args.start_time}",
                })
                continue

            available_ts = set(pd.to_datetime(day_meta["timestamp_local"]).tolist())
            first_ts = day_meta["timestamp_local"].iloc[0]
            last_ts = day_meta["timestamp_local"].iloc[-1]

            print(f"[INFO] split={split}")
            print(f"[INFO] available origins: {len(day_meta)}")
            print(f"[INFO] first origin     : {first_ts}")
            print(f"[INFO] last origin      : {last_ts}")

            for _, origin_row in day_meta.iterrows():
                origin_ts = pd.Timestamp(origin_row["timestamp_local"])
                origin_time = origin_ts.strftime("%H:%M")
                sample_id = int(origin_row.get("sample_id", origin_row.name))

                R_series = split_data["R_series"]
                if sample_id < 0 or sample_id >= R_series.shape[0]:
                    raise IndexError(
                        f"sample_id={sample_id} out of range for split={split}, "
                        f"R_series shape={R_series.shape}"
                    )

                R_current = np.asarray(R_series[sample_id], dtype=np.float32)

                print(f"\n[ORIGIN] {date_str} {origin_time} | sample_id={sample_id}")

                for h in range(1, int(args.max_step) + 1):
                    target_ts = origin_ts + pd.Timedelta(
                        minutes=int(h) * int(args.time_interval_min)
                    )

                    if target_ts not in available_ts:
                        print(
                            f"  [STOP] h={h}: target={target_ts.strftime('%H:%M')} "
                            f"outside available R_t window. Stop this origin."
                        )
                        skip_rows.append({
                            "date": date_str,
                            "origin_time": origin_time,
                            "step": int(h),
                            "target_timestamp": str(target_ts),
                            "reason": "target outside available R_t window",
                        })
                        break

                    result: Dict[str, Any] = {
                        "segment_ids": segment_ids.astype(np.int64),
                        "split": split,
                        "source_sample_id": int(sample_id),
                        "step": int(h),
                        "request_timestamp": str(origin_ts),
                        "timestamp_local": str(origin_ts),
                        "target_timestamp": str(target_ts),
                        "time_interval_min": int(args.time_interval_min),
                        "matrix_contract": (
                            "R_pred[i,j] is predicted correlation between "
                            "segment_ids[i] and segment_ids[j]."
                        ),
                        "model_path": str(model_path),
                        "common_dir": str(common_dir),
                        "meta_row": origin_row.to_dict(),
                    }

                    out_path = service.build_default_output_path(out_dir, result)

                    if out_path.exists() and not args.overwrite:
                        print(f"  [SKIP] h={h}: exists -> {out_path.name}")
                        success_rows.append({
                            "status": "skipped_exists",
                            "date": date_str,
                            "origin_time": origin_time,
                            "step": int(h),
                            "target_timestamp": str(target_ts),
                            "split": split,
                            "sample_id": int(sample_id),
                            "output_path": str(out_path),
                        })
                        continue

                    if args.dry_run:
                        print(
                            f"  [DRY] h={h}: target={target_ts.strftime('%H:%M')} "
                            f"-> {out_path.name}"
                        )
                        success_rows.append({
                            "status": "dry_run",
                            "date": date_str,
                            "origin_time": origin_time,
                            "step": int(h),
                            "target_timestamp": str(target_ts),
                            "split": split,
                            "sample_id": int(sample_id),
                            "output_path": str(out_path),
                        })
                        continue

                    R_pred = service.predict_dmfm_matrix(model, R_current, int(h))
                    result["R_pred"] = R_pred

                    if args.include_current:
                        result["R_current"] = service.sym_clip_diag(R_current)

                    saved_path = service.save_prediction_bundle(
                        result=result,
                        output_path=out_path,
                        dtype=args.dtype,
                        include_current=args.include_current,
                    )

                    print(
                        f"  [SAVED] h={h}: target={target_ts.strftime('%H:%M')} "
                        f"-> {saved_path.name}"
                    )

                    success_rows.append({
                        "status": "saved",
                        "date": date_str,
                        "origin_time": origin_time,
                        "step": int(h),
                        "target_timestamp": str(target_ts),
                        "split": split,
                        "sample_id": int(sample_id),
                        "output_path": str(saved_path),
                    })

        except Exception as e:
            msg = str(e)
            print(f"[ERROR] date={date_str}: {msg}")
            error_rows.append({
                "date": date_str,
                "error": msg,
            })
            if args.stop_on_error:
                raise

    manifest_path = out_dir / "batch_manifest.csv"
    skipped_path = out_dir / "batch_skipped.csv"
    errors_path = out_dir / "batch_errors.csv"
    summary_path = out_dir / "batch_summary.json"

    pd.DataFrame(success_rows).to_csv(manifest_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(skip_rows).to_csv(skipped_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(error_rows).to_csv(errors_path, index=False, encoding="utf-8-sig")

    summary = {
        "service_path": str(SERVICE_PATH),
        "common_dir": str(common_dir),
        "model_path": str(model_path),
        "output_dir": str(out_dir),
        "year": int(args.year),
        "month": int(args.month),
        "start_day": int(args.start_day),
        "end_day": int(args.end_day),
        "start_time": str(args.start_time),
        "end_time": str(args.end_time) if args.end_time else None,
        "max_step": int(args.max_step),
        "time_interval_min": int(args.time_interval_min),
        "dtype": args.dtype,
        "include_current": bool(args.include_current),
        "n_success_or_skipped_exists_or_dry": len(success_rows),
        "n_stopped_or_skipped": len(skip_rows),
        "n_errors": len(error_rows),
        "manifest_csv": str(manifest_path),
        "skipped_csv": str(skipped_path),
        "errors_csv": str(errors_path),
    }
    save_json(summary, summary_path)

    print_stage("DONE")
    print(f"Success/dry/existing : {len(success_rows):,}")
    print(f"Stopped/skipped      : {len(skip_rows):,}")
    print(f"Errors               : {len(error_rows):,}")
    print(f"Manifest             : {manifest_path}")
    print(f"Skipped CSV          : {skipped_path}")
    print(f"Errors CSV           : {errors_path}")
    print(f"Summary JSON         : {summary_path}")


if __name__ == "__main__":
    main()