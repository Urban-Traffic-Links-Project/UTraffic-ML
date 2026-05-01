"""
Prepare Branch B OSM-edge STATIC Granger-based directed predictive graph.

This file also provides helper functions used by:
    prepare_branchB_osm_edge_granger_dynamic_like_branchA.py

Input:
    outputs/branchB/osm_edge_gt_like_branchA/

Output:
    outputs/branchB/osm_edge_granger_like_branchA/

Graph semantics:
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


EPS = 1e-8
THIS_FILE = Path(__file__).resolve()
DATA_PROCESSING_DIR = THIS_FILE.parent
PROJECT_ROOT = DATA_PROCESSING_DIR.parents[2]
DEFAULT_SOURCE_DIR = DATA_PROCESSING_DIR / "outputs" / "branchB" / "osm_edge_gt_like_branchA"
DEFAULT_OUTPUT_DIR = DATA_PROCESSING_DIR / "outputs" / "branchB" / "osm_edge_granger_like_branchA"


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


def load_existing_split(source_dir: Path, split: str, mmap: bool = True) -> Dict[str, Any]:
    d = Path(source_dir) / split
    required = ["z.npy", "segment_ids.npy", "timestamps.npy", "G_series_meta.csv"]
    missing = [d / name for name in required if not (d / name).exists()]
    if missing:
        raise FileNotFoundError("Missing Branch-B split files:\n" + "\n".join(map(str, missing)))

    mmap_mode = "r" if mmap else None
    z = np.load(d / "z.npy", mmap_mode=mmap_mode)
    segment_ids = np.asarray(np.load(d / "segment_ids.npy"), dtype=np.int64)
    timestamps = np.asarray(np.load(d / "timestamps.npy")).astype(str)
    meta = pd.read_csv(d / "G_series_meta.csv")
    if "timestamp_local" in meta.columns:
        meta["timestamp_local"] = pd.to_datetime(meta["timestamp_local"], errors="coerce")

    out = {
        "z": z,
        "segment_ids": segment_ids,
        "timestamps": timestamps,
        "meta": meta,
    }
    # Keep standard Rt if available.
    if (d / "G_weight_series.npy").exists():
        out["G_weight_series"] = np.load(d / "G_weight_series.npy", mmap_mode=mmap_mode)
    if (d / "G_best_lag_series.npy").exists():
        out["G_best_lag_series"] = np.load(d / "G_best_lag_series.npy", mmap_mode=mmap_mode)
    return out


def resolve_node_indices(
    train_segment_ids: np.ndarray,
    max_nodes: int = 0,
    node_indices_arg: Optional[str] = None,
    node_ids_arg: Optional[str] = None,
    node_sample: str = "first",
    seed: int = 42,
) -> Optional[np.ndarray]:
    seg = np.asarray(train_segment_ids, dtype=np.int64)
    N = int(len(seg))
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
        pos = {int(v): i for i, v in enumerate(seg)}
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


def subset_nodes(data: Dict[str, Any], node_idx: Optional[np.ndarray]) -> Dict[str, Any]:
    if node_idx is None:
        return dict(data)
    idx = np.asarray(node_idx, dtype=np.int64)
    out = dict(data)
    out["segment_ids"] = np.asarray(data["segment_ids"], dtype=np.int64)[idx]
    out["z"] = np.asarray(data["z"], dtype=np.float32)[:, idx]
    if "G_weight_series" in data:
        out["G_weight_series"] = np.asarray(data["G_weight_series"][:, idx, :][:, :, idx], dtype=np.float32)
    if "G_best_lag_series" in data:
        out["G_best_lag_series"] = np.asarray(data["G_best_lag_series"][:, idx, :][:, :, idx])
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


def build_supervised_tensors(
    z: np.ndarray,
    meta: pd.DataFrame,
    horizon: int,
    p: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build:
        X_lags[m, r, n] = z[origin-r, n], r=0..p-1
        Y[m, n]         = z[origin+horizon, n]
    Session boundaries are respected.
    """
    z = np.asarray(z, dtype=np.float32)
    h = int(horizon)
    p = int(p)
    X_rows = []
    Y_rows = []
    origins = []
    targets = []

    for idx in session_groups(meta):
        if len(idx) <= h + p:
            continue
        # pos is position of origin inside session.
        for pos in range(p - 1, len(idx) - h):
            origin = int(idx[pos])
            target = int(idx[pos + h])
            lag_indices = [int(idx[pos - r]) for r in range(p)]
            X_rows.append(z[lag_indices, :])
            Y_rows.append(z[target, :])
            origins.append(origin)
            targets.append(target)

    if not X_rows:
        N = int(z.shape[1])
        return (
            np.empty((0, p, N), dtype=np.float32),
            np.empty((0, N), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
        )

    return (
        np.stack(X_rows, axis=0).astype(np.float32),
        np.stack(Y_rows, axis=0).astype(np.float32),
        np.asarray(origins, dtype=np.int64),
        np.asarray(targets, dtype=np.int64),
    )


def ridge_beta(X: np.ndarray, y: np.ndarray, ridge: float = 1e-4) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    p = X.shape[1]
    A = X.T @ X + float(ridge) * np.eye(p, dtype=np.float64)
    b = X.T @ y
    try:
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(A) @ b


def fit_restricted_residuals(
    X_lags: np.ndarray,
    Y: np.ndarray,
    ridge: float = 1e-4,
    fit_intercept: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Per target i:
        restricted: y_i <- own past x_i(t), x_i(t-1), ...
    Returns residuals[M,N], mse_r[N], betas[N,p(+1)].
    """
    X_lags = np.asarray(X_lags, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.float32)
    M, p, N = X_lags.shape
    residuals = np.zeros((M, N), dtype=np.float32)
    mse = np.zeros(N, dtype=np.float32)
    betas = np.zeros((N, p + (1 if fit_intercept else 0)), dtype=np.float32)

    for i in range(N):
        X = X_lags[:, :, i]
        if fit_intercept:
            X = np.concatenate([np.ones((M, 1), dtype=np.float32), X], axis=1)
        beta = ridge_beta(X, Y[:, i], ridge=ridge)
        pred = X @ beta
        res = Y[:, i] - pred
        residuals[:, i] = res.astype(np.float32)
        mse[i] = float(np.mean(res ** 2))
        betas[i, :len(beta)] = beta.astype(np.float32)

    return residuals, mse, betas


def choose_candidates_by_partial_corr(
    X_lags: np.ndarray,
    residuals: np.ndarray,
    max_candidates: int = 50,
    block_size: int = 256,
) -> np.ndarray:
    """
    Candidate sources per target using correlation between source lag features and restricted residual.

    Score(source,target) = max_r |corr(X_lags[:,r,source], residual[:,target])|
    """
    X_lags = np.asarray(X_lags, dtype=np.float32)
    residuals = np.asarray(residuals, dtype=np.float32)
    M, p, N = X_lags.shape
    K = min(int(max_candidates), max(1, N - 1))
    scores = np.zeros((N, N), dtype=np.float32)  # target x source

    R = residuals - residuals.mean(axis=0, keepdims=True)
    R = R / (R.std(axis=0, keepdims=True) + EPS)

    for r in range(p):
        X = X_lags[:, r, :]
        X = X - X.mean(axis=0, keepdims=True)
        X = X / (X.std(axis=0, keepdims=True) + EPS)

        # Process targets by block to limit transient memory.
        for start in range(0, N, int(block_size)):
            end = min(N, start + int(block_size))
            # source x target_block -> transpose target_block x source
            C = (X.T @ R[:, start:end]) / max(1, M - 1)
            scores[start:end, :] = np.maximum(scores[start:end, :], np.abs(C.T).astype(np.float32))

    diag = np.arange(N)
    scores[diag, diag] = -np.inf
    idx = np.argpartition(scores, -K, axis=1)[:, -K:]
    return idx.astype(np.int64)


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


def copy_basic_split_files(source_split_dir: Path, split_out: Path, data: Dict[str, Any], node_idx: Optional[np.ndarray]) -> None:
    ensure_dir(split_out)
    np.save(split_out / "z.npy", np.asarray(data["z"], dtype=np.float32))
    np.save(split_out / "segment_ids.npy", np.asarray(data["segment_ids"], dtype=np.int64))
    np.save(split_out / "timestamps.npy", np.asarray(data["timestamps"]).astype(str))
    data["meta"].to_csv(split_out / "G_series_meta.csv", index=False)


def write_readme(out_dir: Path, args: argparse.Namespace) -> None:
    text = f"""# Branch B Static Granger-style directed predictive graph

Created by `prepare_branchB_osm_edge_granger_like_branchA.py`.

This folder stores a train-only static Granger graph per horizon.

Graph format:
- `graphs/G_hXXX.npy[target, source]`
- `graphs/L_hXXX.npy[target, source]`

It is not a correlation matrix.

Args:
```json
{json.dumps(vars(args), ensure_ascii=False, indent=2)}
```
"""
    (out_dir / "README_GRANGER_STATIC.md").write_text(text, encoding="utf-8")


def save_static_outputs(
    out_dir: Path,
    source_dir: Path,
    splits: Dict[str, Dict[str, Any]],
    G_by_h: Dict[int, np.ndarray],
    L_by_h: Dict[int, np.ndarray],
    horizons: Sequence[int],
    dtype: str,
    lag_dtype: str,
    node_idx: Optional[np.ndarray],
    run_summary: Dict[str, Any],
) -> None:
    graphs_dir = ensure_dir(out_dir / "graphs")
    for h in horizons:
        np.save(graphs_dir / f"G_h{int(h):03d}.npy", G_by_h[int(h)].astype(dtype))
        np.save(graphs_dir / f"L_h{int(h):03d}.npy", L_by_h[int(h)].astype(lag_dtype))
    np.save(graphs_dir / "available_horizons.npy", np.asarray(list(map(int, horizons)), dtype=np.int16))

    for split_name, data in splits.items():
        split_out = out_dir / split_name
        if split_out.exists():
            shutil.rmtree(split_out)
        ensure_dir(split_out)
        copy_basic_split_files(source_dir / split_name, split_out, data, node_idx=node_idx)

        # For compatibility with standard checkers/methods, also store one graph per timestamp.
        # The actual granger_gt method can use graphs/G_hXXX.npy; these series are a fallback.
        T = np.asarray(data["z"]).shape[0]
        G_series = np.zeros((T, len(data["segment_ids"]), len(data["segment_ids"])), dtype=np.dtype(dtype))
        L_series = np.zeros((T, len(data["segment_ids"]), len(data["segment_ids"])), dtype=np.dtype(lag_dtype))
        # Fill with h=1 graph as a generic fallback.
        h0 = int(list(horizons)[0])
        G_series[:] = G_by_h[h0].astype(dtype)
        L_series[:] = L_by_h[h0].astype(lag_dtype)
        np.save(split_out / "G_weight_series.npy", G_series)
        np.save(split_out / "G_best_lag_series.npy", L_series)

        split_summary = dict(run_summary)
        split_summary.update({
            "split": split_name,
            "z_shape": list(map(int, np.asarray(data["z"]).shape)),
            "static_graph_format": "graphs/G_hXXX.npy[target,source]",
        })
        save_json(split_summary, split_out / "branchB_granger_static_split_summary.json")
        save_json(split_summary, split_out / "branchB_gt_split_summary.json")

    save_json(run_summary, out_dir / "branchB_granger_static_run_summary.json")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--horizons", type=str, default="1-9")
    parser.add_argument("--granger-p", type=int, default=3)
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

    print_stage("LOAD EXISTING BRANCH-B TRUE-RT DATA")
    log(f"PROJECT_ROOT: {PROJECT_ROOT}")
    log(f"SOURCE_DIR  : {source_dir}")
    log(f"OUTPUT_DIR  : {output_dir}")

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
    est_bytes = len(horizons) * N * N * (np.dtype(args.g_dtype).itemsize + np.dtype(args.lag_dtype).itemsize)
    log(f"node mode: {'full' if node_idx is None else f'subset n={len(node_idx)}'}")
    log(f"N={N}, estimated graph storage={fmt_gb(est_bytes)}")

    if args.dry_run:
        log("DRY RUN: stop before computation.")
        return

    if output_dir.exists() and args.overwrite:
        log(f"[CLEAN] removing old output: {output_dir}")
        shutil.rmtree(output_dir)
    ensure_dir(output_dir)

    print_stage("COMPUTE STATIC TRAIN-ONLY GRANGER GRAPHS")
    train = splits["train"]
    z_train = np.asarray(train["z"], dtype=np.float32)
    train_meta = train["meta"]

    G_by_h: Dict[int, np.ndarray] = {}
    L_by_h: Dict[int, np.ndarray] = {}
    summaries: List[Dict[str, Any]] = []

    for h in horizons:
        print_stage(f"HORIZON h={h}")
        X_lags, Y, origins, targets = build_supervised_tensors(z_train, train_meta, horizon=int(h), p=int(args.granger_p))
        G, L, summary = compute_granger_from_supervised_tensors(
            X_lags=X_lags,
            Y=Y,
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
            label=f"h={h} static",
        )
        G_by_h[int(h)] = G
        L_by_h[int(h)] = L
        summaries.append(summary)

    run_summary = {
        "created_at": now_str(),
        "mode": "static_train_only_granger",
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "horizons": [int(x) for x in horizons],
        "n_segments": int(N),
        "node_mode": "full" if node_idx is None else f"subset n={len(node_idx)}",
        "granger_p": int(args.granger_p),
        "max_candidates": int(args.max_candidates),
        "candidate_block_size": int(args.candidate_block_size),
        "ridge": float(args.ridge),
        "fit_intercept": bool(args.fit_intercept),
        "signed": not bool(args.unsigned),
        "min_improvement": float(args.min_improvement),
        "G_dtype": str(args.g_dtype),
        "lag_dtype": str(args.lag_dtype),
        "summaries": summaries,
    }

    print_stage("SAVE STATIC GRANGER OUTPUTS")
    save_static_outputs(
        out_dir=output_dir,
        source_dir=source_dir,
        splits=splits,
        G_by_h=G_by_h,
        L_by_h=L_by_h,
        horizons=horizons,
        dtype=str(args.g_dtype),
        lag_dtype=str(args.lag_dtype),
        node_idx=node_idx,
        run_summary=run_summary,
    )
    write_readme(output_dir, args)
    log("DONE.")


if __name__ == "__main__":
    main()
