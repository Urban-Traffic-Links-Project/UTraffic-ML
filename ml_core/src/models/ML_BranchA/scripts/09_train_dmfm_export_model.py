# ml_core/src/models/ML_BranchA/scripts/09_train_dmfm_export_model.py
"""
Train DMFM model from Branch A R_series and export model artifact.

Input:
    ml_core/src/models/ML_BranchA/data/05_branchA_prepare_segment_segment_rt/train/R_series.npy
    ml_core/src/models/ML_BranchA/data/05_branchA_prepare_segment_segment_rt/train/segment_ids.npy

Output:
    ml_core/src/models/ML_BranchA/artifacts/dmfm_model/
      dmfm_model.npz
      dmfm_config.json
      segment_ids.npy

Run local:
    python ml_core/src/models/ML_BranchA/scripts/09_train_dmfm_export_model.py --train-samples 120 --rank 12 --overwrite

Run Kaggle:
    !python -u ml_core/src/models/ML_BranchA/scripts/09_train_dmfm_export_model.py --train-samples 120 --rank 12 --overwrite 2>&1 | tee logs_A09_train_dmfm.txt
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd

try:
    from sklearn.utils.extmath import randomized_svd
except Exception:
    randomized_svd = None


THIS_FILE = Path(__file__).resolve()
ML_BRANCH_A_ROOT = THIS_FILE.parents[1]
PROJECT_ROOT = THIS_FILE.parents[5]

DEFAULT_COMMON_DIR = ML_BRANCH_A_ROOT / "data" / "05_branchA_prepare_segment_segment_rt"
DEFAULT_OUT_DIR = ML_BRANCH_A_ROOT / "artifacts" / "dmfm_model"


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


def load_train(common_dir: Path) -> Dict[str, Any]:
    split_dir = common_dir / "train"
    required = [
        split_dir / "R_series.npy",
        split_dir / "segment_ids.npy",
        split_dir / "R_series_meta.csv",
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing Branch A train R_series files:\n"
            + "\n".join(str(p) for p in missing)
            + "\nRun 00_prepare_branchA_common_from_osm.py first."
        )

    R_series = np.load(split_dir / "R_series.npy", mmap_mode="r")
    segment_ids = np.load(split_dir / "segment_ids.npy").astype(np.int64)
    meta = pd.read_csv(split_dir / "R_series_meta.csv")
    if "timestamp_local" in meta.columns:
        meta["timestamp_local"] = pd.to_datetime(meta["timestamp_local"])

    return {"R_series": R_series, "segment_ids": segment_ids, "meta": meta}


def choose_train_indices(T: int, train_samples: int, mode: str, seed: int) -> np.ndarray:
    if train_samples is None or train_samples <= 0 or train_samples >= T:
        return np.arange(T, dtype=np.int64)

    if mode == "uniform":
        idx = np.linspace(0, T - 1, train_samples).round().astype(np.int64)
        return np.unique(idx)

    rng = np.random.default_rng(seed)
    idx = rng.choice(T, size=train_samples, replace=False)
    return np.sort(idx.astype(np.int64))


def fit_dmfm(train_data: Dict[str, Any], rank: int, train_samples: int, sample_mode: str, seed: int) -> Dict[str, Any]:
    R_series = train_data["R_series"]
    T = int(R_series.shape[0])
    N = int(R_series.shape[1])

    idx = choose_train_indices(T, train_samples, sample_mode, seed)
    iu = np.triu_indices(N, k=1)
    P = len(iu[0])

    print(f"R_series shape : {R_series.shape}")
    print(f"N nodes        : {N:,}")
    print(f"Upper pairs    : {P:,}")
    print(f"Train samples  : {len(idx):,}/{T:,}")
    print(f"DMFM rank      : {rank}")

    X = np.empty((len(idx), P), dtype=np.float32)
    t0 = time.time()

    for row, t in enumerate(idx):
        R = sym_clip_diag(np.asarray(R_series[int(t)], dtype=np.float32))
        X[row] = R[iu]
        if (row + 1) % 5 == 0 or (row + 1) == len(idx):
            elapsed = time.time() - t0
            speed = (row + 1) / max(elapsed, 1e-9)
            eta = (len(idx) - row - 1) / max(speed, 1e-9)
            print(f"  load R: {row + 1}/{len(idx)} | {speed:.2f} R/s | ETA={eta/60:.1f} min")

    print("Computing mean_vec and centering X in-place ...")
    mean_vec = X.mean(axis=0).astype(np.float32)
    X -= mean_vec[None, :]

    k = int(min(rank, max(1, X.shape[0] - 1), X.shape[1]))

    print("Running SVD ...")
    if randomized_svd is not None:
        _, _, VT = randomized_svd(X, n_components=k, random_state=seed)
    else:
        print("WARNING: sklearn randomized_svd not available. Falling back to np.linalg.svd.")
        _, _, VT = np.linalg.svd(X, full_matrices=False)
        VT = VT[:k]

    components = VT.T.astype(np.float32)
    factors = (X @ components).astype(np.float32)

    print("Fitting latent VAR(1): f_{t+1} = A f_t ...")
    if len(factors) < 2:
        A = np.eye(k, dtype=np.float32)
    else:
        X_prev = factors[:-1]
        X_next = factors[1:]
        ridge = 1e-3 * np.eye(k, dtype=np.float32)
        A = np.linalg.solve(X_prev.T @ X_prev + ridge, X_prev.T @ X_next).astype(np.float32)

    return {
        "n": np.array([N], dtype=np.int64),
        "rank": np.array([k], dtype=np.int64),
        "mean_vec": mean_vec.astype(np.float32),
        "components": components.astype(np.float32),
        "A": A.astype(np.float32),
        "segment_ids": train_data["segment_ids"].astype(np.int64),
        "train_sample_indices": idx.astype(np.int64),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--common-dir", type=str, default=str(DEFAULT_COMMON_DIR))
    ap.add_argument("--output-dir", type=str, default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--rank", type=int, default=12)
    ap.add_argument("--train-samples", type=int, default=120, help="0 = all train samples. Use 80/120 for memory safety.")
    ap.add_argument("--sample-mode", type=str, choices=["uniform", "random"], default="uniform")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    common_dir = Path(args.common_dir).resolve()
    out_dir = Path(args.output_dir).resolve()

    print_stage("TRAIN DMFM AND EXPORT MODEL")
    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("COMMON_DIR  :", common_dir)
    print("OUTPUT_DIR  :", out_dir)
    print("rank        :", args.rank)
    print("train_samples:", args.train_samples)

    if args.overwrite and out_dir.exists():
        print("[CLEAN] Removing:", out_dir)
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train = load_train(common_dir)
    model = fit_dmfm(
        train_data=train,
        rank=args.rank,
        train_samples=args.train_samples,
        sample_mode=args.sample_mode,
        seed=args.seed,
    )

    model_path = out_dir / "dmfm_model.npz"
    print("Saving model:", model_path)
    np.savez(
        model_path,
        n=model["n"],
        rank=model["rank"],
        mean_vec=model["mean_vec"],
        components=model["components"],
        A=model["A"],
        segment_ids=model["segment_ids"],
        train_sample_indices=model["train_sample_indices"],
    )
    np.save(out_dir / "segment_ids.npy", model["segment_ids"])

    # This file is for backend/data contract: row/column index -> OSM/model_node_id.
    axis_path = out_dir / "matrix_axis.csv"
    pd.DataFrame({
        "matrix_index": np.arange(len(model["segment_ids"]), dtype=np.int64),
        "segment_id": model["segment_ids"].astype(np.int64),
        "axis_role": "row_and_column",
    }).to_csv(axis_path, index=False)

    config = {
        "common_dir": str(common_dir),
        "model_path": str(model_path),
        "rank": int(model["rank"][0]),
        "n": int(model["n"][0]),
        "train_samples": int(args.train_samples),
        "sample_mode": args.sample_mode,
        "seed": int(args.seed),
        "matrix_contract": "R_pred[i,j] is predicted correlation between segment_ids[i] and segment_ids[j].",
    }
    with open(out_dir / "dmfm_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print_stage("DONE")
    print("Model saved:", model_path)
    print("Axis saved :", axis_path)


if __name__ == "__main__":
    main()
