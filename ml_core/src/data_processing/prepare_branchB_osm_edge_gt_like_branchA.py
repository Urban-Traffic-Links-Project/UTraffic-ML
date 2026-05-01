"""
Prepare Branch B OSM-edge DYNAMIC Granger-based directed predictive graph.

This is the dynamic/bucket version of the Granger pipeline.

Why bucket-dynamic instead of full rolling Granger?
- Full rolling Granger per timestamp is very noisy and expensive with short windows.
- This version estimates a Granger graph per time-of-day bucket and horizon using TRAIN only.
- At prediction time, the graph depends on the origin timestamp bucket:
      G_used = G_bucket[h, bucket(origin_time)]

Output is intentionally separate from the old static/correlation folders:
    outputs/branchB/osm_edge_granger_dynamic_like_branchA/

Graph semantics:
    G_bucket_hXXX.npy[bucket, target, source]

Weight definition:
    G[target, source] = sign(source coefficient) * max(0, log((MSE_R+eps)/(MSE_F+eps)))

This is NOT a correlation matrix. It is a directed predictive influence graph.
"""

from __future__ import annotations

import argparse
import json
import math
import os
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

# Reuse the static Granger utilities. Keep this file in the same folder as
# prepare_branchB_osm_edge_granger_like_branchA.py.
from prepare_branchB_osm_edge_granger_like_branchA import (  # type: ignore
    EPS,
    DATA_PROCESSING_DIR,
    PROJECT_ROOT,
    DEFAULT_SOURCE_DIR,
    NumpyJsonEncoder,
    now_str,
    log,
    print_stage,
    ensure_dir,
    save_json,
    parse_int_list,
    maybe_iter,
    fmt_gb,
    load_existing_split,
    subset_nodes,
    resolve_node_indices,
    build_supervised_tensors,
    fit_restricted_residuals,
    choose_candidates_by_partial_corr,
    ridge_beta,
    copy_basic_split_files,
    write_readme as _write_static_readme,
)

DEFAULT_OUTPUT_DIR = DATA_PROCESSING_DIR / "outputs" / "branchB" / "osm_edge_granger_dynamic_like_branchA"


def get_tod_minutes(meta: pd.DataFrame) -> np.ndarray:
    if "tod_minutes" in meta.columns:
        vals = pd.to_numeric(meta["tod_minutes"], errors="coerce")
        if vals.notna().all():
            return vals.to_numpy(dtype=np.int32)

    if "timestamp_local" in meta.columns:
        ts = pd.to_datetime(meta["timestamp_local"])
        return (ts.dt.hour * 60 + ts.dt.minute).to_numpy(dtype=np.int32)

    # fallback: use slot_index if available, assume 15-minute slots from 06:00
    if "slot_index" in meta.columns:
        slot = pd.to_numeric(meta["slot_index"], errors="coerce").fillna(0).to_numpy(dtype=np.int32)
        return (6 * 60 + slot * 15).astype(np.int32)

    return np.arange(len(meta), dtype=np.int32)


def build_bucket_table(bucket_minutes: int, start_minute: int, end_minute: int) -> pd.DataFrame:
    rows = []
    b = 0
    for s in range(int(start_minute), int(end_minute), int(bucket_minutes)):
        e = min(s + int(bucket_minutes), int(end_minute))
        rows.append({
            "bucket_id": b,
            "start_minute": int(s),
            "end_minute": int(e),
            "label": f"{s//60:02d}:{s%60:02d}-{e//60:02d}:{e%60:02d}",
        })
        b += 1
    return pd.DataFrame(rows)


def assign_bucket_ids(meta: pd.DataFrame, bucket_table: pd.DataFrame) -> np.ndarray:
    tod = get_tod_minutes(meta)
    ids = np.zeros(len(meta), dtype=np.int16)
    starts = bucket_table["start_minute"].to_numpy(dtype=np.int32)
    ends = bucket_table["end_minute"].to_numpy(dtype=np.int32)
    for i, m in enumerate(tod):
        # assign to nearest valid bucket if outside range
        hit = np.where((starts <= int(m)) & (int(m) < ends))[0]
        if len(hit):
            ids[i] = int(bucket_table.iloc[int(hit[0])]["bucket_id"])
        elif int(m) < starts.min():
            ids[i] = int(bucket_table.iloc[0]["bucket_id"])
        else:
            ids[i] = int(bucket_table.iloc[-1]["bucket_id"])
    return ids


def compute_granger_from_supervised_tensors(
    X_lags: np.ndarray,
    Y: np.ndarray,
    horizon: int,
    p: int,
    max_candidates: int,
    candidate_block_size: int,
    ridge: float,
    fit_intercept: bool,
    signed: bool,
    dtype: str,
    lag_dtype: str,
    min_improvement: float,
    label: str,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Compute candidate-pruned Granger graph from already built supervised samples."""
    t0 = time.time()
    X_lags = np.asarray(X_lags, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.float32)
    M, p_used, N = X_lags.shape
    if M <= (2 * p_used + 2):
        raise RuntimeError(f"Too few samples for Granger {label}: M={M}, p={p_used}")

    log(f"{label}: X_lags={X_lags.shape}, Y={Y.shape}")
    residuals, mse_r, _ = fit_restricted_residuals(
        X_lags=X_lags,
        Y=Y,
        ridge=float(ridge),
        fit_intercept=bool(fit_intercept),
    )
    candidates = choose_candidates_by_partial_corr(
        X_lags=X_lags,
        residuals=residuals,
        max_candidates=int(max_candidates),
        block_size=int(candidate_block_size),
    )

    G = np.zeros((N, N), dtype=np.float32)
    L = np.zeros((N, N), dtype=np.int16)

    for target in maybe_iter(range(N), total=N, desc=f"granger {label}"):
        own = X_lags[:, :, target]
        y = Y[:, target]
        mse_base = float(mse_r[target])
        if not np.isfinite(mse_base) or mse_base <= EPS:
            continue

        for source in candidates[target]:
            source = int(source)
            if source == target:
                continue
            src = X_lags[:, :, source]
            Xf = np.concatenate([own, src], axis=1)
            if fit_intercept:
                Xf = np.concatenate([np.ones((M, 1), dtype=np.float32), Xf], axis=1)
            beta = ridge_beta(Xf, y, ridge=float(ridge))
            pred = Xf @ beta
            mse_f = float(np.mean((y - pred) ** 2))
            if not np.isfinite(mse_f) or mse_f <= 0:
                continue
            score = math.log((mse_base + EPS) / (mse_f + EPS))
            if score <= float(min_improvement):
                continue

            src_offset = (1 if fit_intercept else 0) + p_used
            src_coefs = np.asarray(beta[src_offset: src_offset + p_used], dtype=np.float64)
            if src_coefs.size == 0:
                continue
            best_lag_idx = int(np.argmax(np.abs(src_coefs)))
            coef = float(src_coefs[best_lag_idx])
            w = math.copysign(score, coef if coef != 0.0 else 1.0) if signed else score
            G[target, source] = float(w)
            L[target, source] = int(best_lag_idx)

    np.fill_diagonal(G, 0.0)
    np.fill_diagonal(L, 0)
    elapsed = time.time() - t0
    summary = {
        "label": str(label),
        "horizon": int(horizon),
        "n_samples": int(M),
        "n_segments": int(N),
        "granger_p": int(p),
        "max_candidates": int(max_candidates),
        "candidate_block_size": int(candidate_block_size),
        "ridge": float(ridge),
        "fit_intercept": bool(fit_intercept),
        "signed": bool(signed),
        "min_improvement": float(min_improvement),
        "n_nonzero_edges": int(np.count_nonzero(G)),
        "mean_restricted_mse": float(np.mean(mse_r)),
        "median_restricted_mse": float(np.median(mse_r)),
        "elapsed_seconds": float(elapsed),
    }
    log(f"{label} DONE | nonzero={summary['n_nonzero_edges']:,} | elapsed={elapsed/60:.2f} min")
    return G.astype(dtype), L.astype(lag_dtype), summary


def save_dynamic_granger_outputs(
    out_dir: Path,
    source_dir: Path,
    splits: Dict[str, Dict[str, Any]],
    bucket_table: pd.DataFrame,
    bucket_ids_by_split: Dict[str, np.ndarray],
    G_by_h_bucket: Dict[int, np.ndarray],
    L_by_h_bucket: Dict[int, np.ndarray],
    horizons: Sequence[int],
    dtype: str,
    lag_dtype: str,
    node_idx: Optional[np.ndarray],
    run_summary: Dict[str, Any],
) -> None:
    graphs_dir = ensure_dir(out_dir / "graphs")
    bucket_table.to_csv(graphs_dir / "bucket_table.csv", index=False)

    for h in horizons:
        np.save(graphs_dir / f"G_bucket_h{int(h):03d}.npy", G_by_h_bucket[int(h)].astype(dtype))
        np.save(graphs_dir / f"L_bucket_h{int(h):03d}.npy", L_by_h_bucket[int(h)].astype(lag_dtype))

    np.save(graphs_dir / "available_horizons.npy", np.asarray(list(map(int, horizons)), dtype=np.int16))

    for split_name, data in splits.items():
        split_out = out_dir / split_name
        source_split_dir = source_dir / split_name
        if split_out.exists():
            shutil.rmtree(split_out)
        ensure_dir(split_out)
        copy_basic_split_files(source_split_dir, split_out, data, node_idx=node_idx)
        np.save(split_out / "origin_bucket_ids.npy", np.asarray(bucket_ids_by_split[split_name], dtype=np.int16))

        split_summary = dict(run_summary)
        split_summary.update({
            "split": split_name,
            "dynamic_graph_format": "graphs/G_bucket_hXXX.npy[bucket,target,source] + split/origin_bucket_ids.npy[t]",
            "z_shape": list(map(int, np.asarray(data["z"]).shape)),
            "segment_ids_shape": list(map(int, np.asarray(data["segment_ids"]).shape)),
        })
        save_json(split_summary, split_out / "branchB_granger_dynamic_split_summary.json")
        save_json(split_summary, split_out / "branchB_gt_split_summary.json")

    save_json(run_summary, out_dir / "branchB_granger_dynamic_run_summary.json")


def write_dynamic_readme(out_dir: Path, args: argparse.Namespace) -> None:
    text = f"""# Branch B Dynamic Granger-style directed predictive graph

Created by `prepare_branchB_osm_edge_granger_dynamic_like_branchA.py`.

Graph format:

- `graphs/G_bucket_hXXX.npy[bucket, target, source]`
- `graphs/L_bucket_hXXX.npy[bucket, target, source]`
- `train|val|test/origin_bucket_ids.npy[t]`

At forecast origin time `t` and horizon `h`, downstream uses:

`G = G_bucket_hXXX[origin_bucket_ids[t]]`

This is dynamic by time-of-day bucket and horizon, estimated from TRAIN only.
It is not a correlation matrix.

Args:

```json
{json.dumps(vars(args), ensure_ascii=False, indent=2)}
```
"""
    (out_dir / "README_GRANGER_DYNAMIC.md").write_text(text, encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--horizons", type=str, default="1-9")
    parser.add_argument("--granger-p", type=int, default=3)
    parser.add_argument("--bucket-minutes", type=int, default=60, help="Time-of-day bucket size. Recommended: 60 or 30.")
    parser.add_argument("--start-minute", type=int, default=6 * 60)
    parser.add_argument("--end-minute", type=int, default=12 * 60)
    parser.add_argument("--min-bucket-samples", type=int, default=40, help="Fallback to global graph if bucket samples are fewer than this.")
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

    source_dir = Path(args.source_dir) if args.source_dir else DEFAULT_SOURCE_DIR
    output_dir = Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_DIR
    if not source_dir.is_absolute():
        source_dir = PROJECT_ROOT / source_dir
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    horizons = parse_int_list(args.horizons)
    if not horizons:
        raise ValueError("No horizons parsed from --horizons")

    print_stage("LOAD EXISTING BRANCH-B DATA")
    log(f"PROJECT_ROOT: {PROJECT_ROOT}")
    log(f"SOURCE_DIR  : {source_dir}")
    log(f"OUTPUT_DIR  : {output_dir}")
    log(f"HORIZONS    : {horizons}")

    train0 = load_existing_split(source_dir, "train", mmap=False)
    node_idx = resolve_node_indices(
        train_segment_ids=np.asarray(train0["segment_ids"], dtype=np.int64),
        max_nodes=int(args.max_nodes),
        node_indices_arg=args.node_indices,
        node_ids_arg=args.node_ids,
        node_sample=args.node_sample,
        seed=int(args.seed),
    )
    splits = {
        "train": subset_nodes(train0, node_idx),
        "val": subset_nodes(load_existing_split(source_dir, "val", mmap=False), node_idx),
        "test": subset_nodes(load_existing_split(source_dir, "test", mmap=False), node_idx),
    }
    N = int(np.asarray(splits["train"]["z"]).shape[1])
    bucket_table = build_bucket_table(int(args.bucket_minutes), int(args.start_minute), int(args.end_minute))
    B = int(len(bucket_table))
    bucket_ids_by_split = {name: assign_bucket_ids(data["meta"], bucket_table) for name, data in splits.items()}

    log(f"train z shape: {np.asarray(splits['train']['z']).shape}")
    log(f"node mode: {'full' if node_idx is None else f'subset n={len(node_idx)}'}")
    log(f"bucket table:\n{bucket_table}")
    graph_bank_bytes = len(horizons) * B * N * N * np.dtype(args.g_dtype).itemsize
    log(f"Estimated graph bank size: {fmt_gb(graph_bank_bytes)}")

    if args.dry_run:
        log("DRY RUN: stop before computation.")
        return

    if output_dir.exists() and args.overwrite:
        log(f"[CLEAN] removing old output: {output_dir}")
        shutil.rmtree(output_dir)
    ensure_dir(output_dir)

    print_stage("COMPUTE BUCKET-DYNAMIC TRAIN-ONLY GRANGER GRAPHS")
    train = splits["train"]
    z_train = np.asarray(train["z"], dtype=np.float32)
    train_meta = train["meta"]
    train_bucket_ids = bucket_ids_by_split["train"]

    G_by_h_bucket: Dict[int, np.ndarray] = {}
    L_by_h_bucket: Dict[int, np.ndarray] = {}
    all_summaries: List[Dict[str, Any]] = []

    for h in horizons:
        print_stage(f"HORIZON h={h}")
        X_all, Y_all, origins, targets = build_supervised_tensors(z_train, train_meta, horizon=int(h), p=int(args.granger_p))
        origin_buckets = train_bucket_ids[origins]

        # Global fallback graph for this horizon.
        G_global, L_global, sum_global = compute_granger_from_supervised_tensors(
            X_lags=X_all,
            Y=Y_all,
            horizon=int(h),
            p=int(args.granger_p),
            max_candidates=int(args.max_candidates),
            candidate_block_size=int(args.candidate_block_size),
            ridge=float(args.ridge),
            fit_intercept=bool(args.fit_intercept),
            signed=not bool(args.unsigned),
            dtype=str(args.g_dtype),
            lag_dtype=str(args.lag_dtype),
            min_improvement=float(args.min_improvement),
            label=f"h={h} global fallback",
        )
        sum_global["bucket_id"] = -1
        sum_global["bucket_label"] = "global_fallback"
        all_summaries.append(sum_global)

        G_bank = np.zeros((B, N, N), dtype=np.dtype(args.g_dtype))
        L_bank = np.zeros((B, N, N), dtype=np.dtype(args.lag_dtype))

        for _, brow in bucket_table.iterrows():
            b = int(brow["bucket_id"])
            mask = origin_buckets == b
            n_samples = int(mask.sum())
            if n_samples < int(args.min_bucket_samples):
                log(f"h={h} bucket={b} {brow['label']}: samples={n_samples} < min={args.min_bucket_samples}; use global fallback")
                G_bank[b] = G_global
                L_bank[b] = L_global
                all_summaries.append({
                    "horizon": int(h),
                    "bucket_id": int(b),
                    "bucket_label": str(brow["label"]),
                    "n_samples": int(n_samples),
                    "fallback": True,
                })
                continue

            G_b, L_b, sum_b = compute_granger_from_supervised_tensors(
                X_lags=X_all[mask],
                Y=Y_all[mask],
                horizon=int(h),
                p=int(args.granger_p),
                max_candidates=int(args.max_candidates),
                candidate_block_size=int(args.candidate_block_size),
                ridge=float(args.ridge),
                fit_intercept=bool(args.fit_intercept),
                signed=not bool(args.unsigned),
                dtype=str(args.g_dtype),
                lag_dtype=str(args.lag_dtype),
                min_improvement=float(args.min_improvement),
                label=f"h={h} bucket={b} {brow['label']}",
            )
            G_bank[b] = G_b
            L_bank[b] = L_b
            sum_b["bucket_id"] = int(b)
            sum_b["bucket_label"] = str(brow["label"])
            sum_b["fallback"] = False
            all_summaries.append(sum_b)

        G_by_h_bucket[int(h)] = G_bank
        L_by_h_bucket[int(h)] = L_bank

    run_summary = {
        "created_at": now_str(),
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "mode": "bucket_dynamic_train_only_granger",
        "horizons": [int(x) for x in horizons],
        "n_segments": int(N),
        "node_mode": "full" if node_idx is None else f"subset n={len(node_idx)}",
        "bucket_minutes": int(args.bucket_minutes),
        "start_minute": int(args.start_minute),
        "end_minute": int(args.end_minute),
        "min_bucket_samples": int(args.min_bucket_samples),
        "n_buckets": int(B),
        "granger_p": int(args.granger_p),
        "max_candidates": int(args.max_candidates),
        "candidate_block_size": int(args.candidate_block_size),
        "ridge": float(args.ridge),
        "fit_intercept": bool(args.fit_intercept),
        "signed": not bool(args.unsigned),
        "min_improvement": float(args.min_improvement),
        "G_dtype": str(args.g_dtype),
        "lag_dtype": str(args.lag_dtype),
        "summaries": all_summaries,
    }

    print_stage("SAVE DYNAMIC GRANGER OUTPUTS")
    save_dynamic_granger_outputs(
        out_dir=output_dir,
        source_dir=source_dir,
        splits=splits,
        bucket_table=bucket_table,
        bucket_ids_by_split=bucket_ids_by_split,
        G_by_h_bucket=G_by_h_bucket,
        L_by_h_bucket=L_by_h_bucket,
        horizons=horizons,
        dtype=str(args.g_dtype),
        lag_dtype=str(args.lag_dtype),
        node_idx=node_idx,
        run_summary=run_summary,
    )
    write_dynamic_readme(output_dir, args)
    log("DONE. Next run 06B with --data-dir pointing to this output and --methods no_gt,granger_dynamic_gt")


if __name__ == "__main__":
    main()
