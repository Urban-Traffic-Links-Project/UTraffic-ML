# ml_core/src/models/ML_BranchA/scripts/10_precompute_dmfm_full_matrices_with_axis.py
"""
Precompute full DMFM predicted correlation matrices WITH axis mapping.

This script saves full predicted matrices plus the exact ID order for rows/columns.

Core data contract:
    R_pred_series[pred_idx, i, j]
        = predicted correlation between segment_ids[i] and segment_ids[j]

Outputs:
    artifacts/dmfm_predictions_full/<split>/h<horizon>/
      R_pred_series.npy              # shape [P, N, N]
      segment_ids.npy                # shape [N]
      matrix_axis.csv                # matrix_index, segment_id
      R_pred_meta.csv                # pred_idx -> timestamp/source sample
      prediction_summary.json
      bundles/
        dmfm_pred_<split>_h<h>_idx000000.npz   # optional, one bundle per matrix

Run local:
    python ml_core/src/models/ML_BranchA/scripts/10_precompute_dmfm_full_matrices_with_axis.py --split test --horizons 1,3,6,9 --dtype float16 --save-npz-bundles --overwrite

Run Kaggle:
    !python -u ml_core/src/models/ML_BranchA/scripts/10_precompute_dmfm_full_matrices_with_axis.py --split test --horizons 1,3,6,9 --dtype float16 --save-npz-bundles --overwrite 2>&1 | tee logs_A10_precompute_dmfm_axis.txt
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd


THIS_FILE = Path(__file__).resolve()
ML_BRANCH_A_ROOT = THIS_FILE.parents[1]
PROJECT_ROOT = THIS_FILE.parents[5]

DEFAULT_COMMON_DIR = ML_BRANCH_A_ROOT / "data" / "05_branchA_prepare_segment_segment_rt"
DEFAULT_MODEL_PATH = ML_BRANCH_A_ROOT / "artifacts" / "dmfm_model" / "dmfm_model.npz"
DEFAULT_OUT_DIR = ML_BRANCH_A_ROOT / "artifacts" / "dmfm_predictions_full"


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def print_stage(title: str) -> None:
    print("\n" + "=" * 96)
    print(f"{now()} | {title}")
    print("=" * 96)


def parse_int_list(s: str) -> List[int]:
    out = []
    for x in str(s).split(","):
        x = x.strip()
        if not x:
            continue
        if "-" in x:
            a, b = x.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(x))
    return sorted(set(out))


def sym_clip_diag(R: np.ndarray, diag_value: float = 1.0) -> np.ndarray:
    R = np.asarray(R, dtype=np.float32)
    R = 0.5 * (R + R.T)
    np.nan_to_num(R, copy=False, nan=0.0, posinf=1.0, neginf=-1.0)
    np.clip(R, -1.0, 1.0, out=R)
    np.fill_diagonal(R, diag_value)
    return R.astype(np.float32, copy=False)


def load_model(model_path: Path) -> Dict[str, Any]:
    if not model_path.exists():
        raise FileNotFoundError(f"Cannot find DMFM model: {model_path}\nRun 09_train_dmfm_export_model.py first.")

    data = np.load(model_path, allow_pickle=False)
    return {
        "n": int(data["n"][0]),
        "rank": int(data["rank"][0]),
        "mean_vec": data["mean_vec"].astype(np.float32),
        "components": data["components"].astype(np.float32),
        "A": data["A"].astype(np.float32),
        "segment_ids": data["segment_ids"].astype(np.int64),
    }


def load_split(common_dir: Path, split: str) -> Dict[str, Any]:
    split_dir = common_dir / split
    required = [
        split_dir / "R_series.npy",
        split_dir / "R_series_meta.csv",
        split_dir / "segment_ids.npy",
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing split files:\n" + "\n".join(str(p) for p in missing))

    R_series = np.load(split_dir / "R_series.npy", mmap_mode="r")
    meta = pd.read_csv(split_dir / "R_series_meta.csv")
    if "timestamp_local" in meta.columns:
        meta["timestamp_local"] = pd.to_datetime(meta["timestamp_local"])
    segment_ids = np.load(split_dir / "segment_ids.npy").astype(np.int64)

    return {"R_series": R_series, "meta": meta, "segment_ids": segment_ids}


def choose_sample_indices(T: int, sample_ids: Optional[List[int]], max_samples: int) -> np.ndarray:
    if sample_ids:
        idx = np.asarray(sample_ids, dtype=np.int64)
        idx = idx[(idx >= 0) & (idx < T)]
        return np.unique(idx)

    if max_samples and max_samples > 0 and max_samples < T:
        return np.arange(max_samples, dtype=np.int64)

    return np.arange(T, dtype=np.int64)


def predict_dmfm_matrix(model: Dict[str, Any], R_origin: np.ndarray, horizon: int) -> np.ndarray:
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


def save_matrix_axis(h_dir: Path, segment_ids: np.ndarray) -> None:
    np.save(h_dir / "segment_ids.npy", segment_ids.astype(np.int64))
    pd.DataFrame({
        "matrix_index": np.arange(len(segment_ids), dtype=np.int64),
        "segment_id": segment_ids.astype(np.int64),
        "axis_role": "row_and_column",
    }).to_csv(h_dir / "matrix_axis.csv", index=False)


def save_npz_bundle(
    bundle_path: Path,
    R_pred: np.ndarray,
    segment_ids: np.ndarray,
    split: str,
    horizon: int,
    pred_idx: int,
    source_sample_id: int,
    timestamp_local: str,
    dtype: str,
) -> None:
    """
    One self-contained file for backend.

    Backend can do:
        data = np.load(bundle_path)
        R = data["R_pred"]
        ids = data["segment_ids"]
        R[i,j] -> ids[i], ids[j]
    """
    np.savez_compressed(
        bundle_path,
        R_pred=R_pred.astype(dtype),
        segment_ids=segment_ids.astype(np.int64),
        matrix_index=np.arange(len(segment_ids), dtype=np.int64),
        split=np.array(split),
        horizon=np.array([int(horizon)], dtype=np.int64),
        pred_idx=np.array([int(pred_idx)], dtype=np.int64),
        source_sample_id=np.array([int(source_sample_id)], dtype=np.int64),
        timestamp_local=np.array(str(timestamp_local)),
        matrix_contract=np.array("R_pred[i,j] is predicted correlation between segment_ids[i] and segment_ids[j]."),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--common-dir", type=str, default=str(DEFAULT_COMMON_DIR))
    ap.add_argument("--model-path", type=str, default=str(DEFAULT_MODEL_PATH))
    ap.add_argument("--output-dir", type=str, default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    ap.add_argument("--horizons", type=parse_int_list, default=[1, 3, 6, 9])
    ap.add_argument("--dtype", type=str, default="float16", choices=["float16", "float32"])
    ap.add_argument("--max-samples", type=int, default=0, help="0 = all R_t samples in split.")
    ap.add_argument("--sample-ids", type=parse_int_list, default=None, help="Optional comma list, e.g. 0,10,20.")
    ap.add_argument("--save-npz-bundles", action="store_true", help="Save one self-contained .npz bundle per predicted matrix.")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    common_dir = Path(args.common_dir).resolve()
    model_path = Path(args.model_path).resolve()
    out_root = Path(args.output_dir).resolve()

    print_stage("PRECOMPUTE FULL DMFM MATRICES WITH AXIS MAPPING")
    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("COMMON_DIR  :", common_dir)
    print("MODEL_PATH  :", model_path)
    print("OUTPUT_ROOT :", out_root)
    print("split       :", args.split)
    print("horizons    :", args.horizons)
    print("dtype       :", args.dtype)
    print("save bundles:", args.save_npz_bundles)

    model = load_model(model_path)
    split_data = load_split(common_dir, args.split)

    N = int(model["n"])
    R_series = split_data["R_series"]
    T = int(R_series.shape[0])

    if R_series.shape[1] != N or R_series.shape[2] != N:
        raise ValueError(f"Shape mismatch: model N={N}, R_series shape={R_series.shape}")

    if not np.array_equal(model["segment_ids"], split_data["segment_ids"]):
        raise ValueError("segment_ids in model and split do not match. Re-train DMFM using the same prepared data.")

    sample_indices = choose_sample_indices(T, args.sample_ids, args.max_samples)
    print(f"Selected samples: {len(sample_indices)}/{T}")
    print(f"One matrix size estimate ({args.dtype}): {N*N*np.dtype(args.dtype).itemsize/1024**2:.2f} MB")
    print("Matrix contract: R_pred_series[pred_idx, i, j] = corr(segment_ids[i], segment_ids[j])")

    for h in args.horizons:
        h_dir = out_root / args.split / f"h{h}"
        if args.overwrite and h_dir.exists():
            print("[CLEAN] Removing:", h_dir)
            shutil.rmtree(h_dir)
        h_dir.mkdir(parents=True, exist_ok=True)

        bundle_dir = h_dir / "bundles"
        if args.save_npz_bundles:
            bundle_dir.mkdir(parents=True, exist_ok=True)

        out_path = h_dir / "R_pred_series.npy"
        print_stage(f"PREDICT split={args.split}, horizon={h}")
        print("Output matrix series:", out_path)

        R_mem = np.lib.format.open_memmap(
            out_path,
            mode="w+",
            dtype=args.dtype,
            shape=(len(sample_indices), N, N),
        )

        save_matrix_axis(h_dir, model["segment_ids"])

        rows = []
        t0 = time.time()

        for out_i, sample_id in enumerate(sample_indices):
            R_origin = np.asarray(R_series[int(sample_id)], dtype=np.float32)
            R_pred = predict_dmfm_matrix(model, R_origin, int(h))
            R_mem[out_i] = R_pred.astype(args.dtype)

            meta_row = split_data["meta"].iloc[int(sample_id)].to_dict()
            timestamp_local = str(meta_row.get("timestamp_local", ""))

            row = {
                "pred_idx": int(out_i),
                "source_sample_id": int(sample_id),
                "horizon": int(h),
                "matrix_file": "R_pred_series.npy",
                "segment_ids_file": "segment_ids.npy",
                "matrix_axis_file": "matrix_axis.csv",
                "matrix_contract": "R_pred_series[pred_idx,i,j] = corr(segment_ids[i], segment_ids[j])",
            }
            for k, v in meta_row.items():
                row[k] = v

            if args.save_npz_bundles:
                bundle_name = f"dmfm_pred_{args.split}_h{h}_idx{out_i:06d}.npz"
                bundle_path = bundle_dir / bundle_name
                save_npz_bundle(
                    bundle_path=bundle_path,
                    R_pred=R_pred,
                    segment_ids=model["segment_ids"],
                    split=args.split,
                    horizon=int(h),
                    pred_idx=int(out_i),
                    source_sample_id=int(sample_id),
                    timestamp_local=timestamp_local,
                    dtype=args.dtype,
                )
                row["bundle_file"] = str(Path("bundles") / bundle_name)

            rows.append(row)

            if (out_i + 1) % 1 == 0 or (out_i + 1) == len(sample_indices):
                elapsed = time.time() - t0
                speed = (out_i + 1) / max(elapsed, 1e-9)
                eta = (len(sample_indices) - out_i - 1) / max(speed, 1e-9)
                print(f"  h={h}: {out_i+1}/{len(sample_indices)} | {speed:.3f} matrix/s | ETA={eta/60:.1f} min")

        R_mem.flush()
        del R_mem

        pred_meta = pd.DataFrame(rows)
        pred_meta.to_csv(h_dir / "R_pred_meta.csv", index=False)

        summary = {
            "split": args.split,
            "horizon": int(h),
            "dtype": args.dtype,
            "shape": [int(len(sample_indices)), int(N), int(N)],
            "N": int(N),
            "n_predictions": int(len(sample_indices)),
            "R_pred_series_path": str(out_path),
            "segment_ids_path": str(h_dir / "segment_ids.npy"),
            "matrix_axis_path": str(h_dir / "matrix_axis.csv"),
            "model_path": str(model_path),
            "save_npz_bundles": bool(args.save_npz_bundles),
            "matrix_contract": "R_pred_series[pred_idx,i,j] is predicted correlation between segment_ids[i] and segment_ids[j].",
        }
        with open(h_dir / "prediction_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    print_stage("DONE")
    print("Predictions saved under:", out_root / args.split)


if __name__ == "__main__":
    main()
