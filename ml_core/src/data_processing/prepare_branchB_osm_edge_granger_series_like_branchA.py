"""
Prepare Branch B OSM-edge Granger-based directed predictive graph.

This script replaces the old lagged-correlation GT with a practical Granger-style
predictive influence graph, while keeping the downstream Branch-B file format.

Main idea
---------
For each forecast horizon h and directed pair source j -> target i, estimate:

    Restricted: x_i[t+h] <- past/current of x_i
    Full      : x_i[t+h] <- past/current of x_i + past/current of x_j

Then define the edge weight:

    G[i, j, h] = sign(source coefficient) * max(0, log((MSE_R + eps)/(MSE_F + eps)))

where G uses the existing convention:

    G[target, source]

Important:
- This is NOT a correlation matrix.
- It is a directed predictive influence / Granger-style adjacency matrix.
- Default mode is static train-only Granger: graph is estimated from train split
  and reused for train/val/test to avoid leakage.

Recommended quick test:
    python -u ml_core/src/data_processing/prepare_branchB_osm_edge_granger_like_branchA.py \
      --max-nodes 512 --horizons 1-9 --granger-p 3 --max-candidates 50 --overwrite

Recommended full run, safer first:
    python -u ml_core/src/data_processing/prepare_branchB_osm_edge_granger_like_branchA.py \
      --horizons 1-9 --granger-p 3 --max-candidates 50 --candidate-block-size 256 --overwrite
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# Keep BLAS threads controlled. The heavy parts are matrix multiplications and
# small least-squares solves; avoid accidental oversubscription in notebooks.
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
SRC_ROOT = DATA_PROCESSING_DIR.parent
ML_CORE_ROOT = SRC_ROOT.parent
PROJECT_ROOT = ML_CORE_ROOT.parent

DEFAULT_SOURCE_DIR = DATA_PROCESSING_DIR / "outputs" / "branchB" / "osm_edge_gt_like_branchA"
DEFAULT_OUTPUT_DIR = DATA_PROCESSING_DIR / "outputs" / "branchB" / "osm_edge_granger_like_branchA"


# =============================================================================
# Utilities
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
        if isinstance(obj, Path):
            return str(obj)
        return super().default(obj)


def now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    print(f"[{now_str()}] {msg}", flush=True)


def print_stage(title: str) -> None:
    print("\n" + "=" * 96, flush=True)
    print(f"{now_str()} | {title}", flush=True)
    print("=" * 96, flush=True)


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


def memory_mb(arr: np.ndarray) -> float:
    return arr.nbytes / 1024 / 1024


def bytes_to_gb(n_bytes: int) -> float:
    return n_bytes / (1024 ** 3)


def fmt_gb(n_bytes: int) -> str:
    return f"{bytes_to_gb(n_bytes):,.2f} GB"


def decode_np_datetime(arr: np.ndarray) -> np.ndarray:
    try:
        return arr.astype("datetime64[ns]")
    except Exception:
        return np.asarray(pd.to_datetime(arr), dtype="datetime64[ns]")


# =============================================================================
# Loading existing Branch-B prepared data
# =============================================================================
def load_existing_split(source_dir: Path, split: str, mmap: bool = True) -> Dict[str, Any]:
    d = source_dir / split
    required = [
        d / "z.npy",
        d / "segment_ids.npy",
        d / "timestamps.npy",
        d / "G_series_meta.csv",
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing source Branch-B files:\n" + "\n".join(map(str, missing)))

    mmap_mode = "r" if mmap else None
    z = np.load(d / "z.npy", mmap_mode=mmap_mode)
    segment_ids = np.asarray(np.load(d / "segment_ids.npy"), dtype=np.int64)
    timestamps = decode_np_datetime(np.load(d / "timestamps.npy"))
    meta = pd.read_csv(d / "G_series_meta.csv")
    if "timestamp_local" in meta.columns:
        meta["timestamp_local"] = pd.to_datetime(meta["timestamp_local"])

    return {
        "split_dir": d,
        "z": z,
        "segment_ids": segment_ids,
        "timestamps": timestamps,
        "meta": meta,
    }


def subset_nodes(data: Dict[str, Any], node_idx: Optional[np.ndarray]) -> Dict[str, Any]:
    if node_idx is None:
        return data
    idx = np.asarray(node_idx, dtype=np.int64)
    out = dict(data)
    out["z"] = np.asarray(data["z"], dtype=np.float32)[:, idx]
    out["segment_ids"] = np.asarray(data["segment_ids"], dtype=np.int64)[idx]
    return out


def resolve_node_indices(
    train_segment_ids: np.ndarray,
    max_nodes: int = 0,
    node_indices_arg: Optional[str] = None,
    node_ids_arg: Optional[str] = None,
    node_sample: str = "first",
    seed: int = 42,
) -> Optional[np.ndarray]:
    N = int(len(train_segment_ids))
    if not node_indices_arg and not node_ids_arg and int(max_nodes) <= 0:
        return None

    selected: Optional[np.ndarray] = None

    if node_indices_arg:
        idx = np.asarray(parse_int_list(node_indices_arg), dtype=np.int64)
        if len(idx) == 0:
            raise ValueError("--node-indices provided but parsed no indices")
        if idx.min() < 0 or idx.max() >= N:
            raise ValueError(f"node index out of range. N={N}, min={idx.min()}, max={idx.max()}")
        selected = idx

    if node_ids_arg:
        requested = np.asarray(parse_int_list(node_ids_arg), dtype=np.int64)
        id_to_pos = {int(v): i for i, v in enumerate(train_segment_ids)}
        missing = [int(x) for x in requested if int(x) not in id_to_pos]
        if missing:
            raise ValueError(f"Some node ids are not in train segment_ids: {missing[:20]}")
        idx = np.asarray([id_to_pos[int(x)] for x in requested], dtype=np.int64)
        selected = idx if selected is None else np.intersect1d(selected, idx)

    if selected is None:
        max_nodes = int(max_nodes)
        if max_nodes <= 0 or max_nodes >= N:
            return None
        if node_sample == "first":
            selected = np.arange(max_nodes, dtype=np.int64)
        elif node_sample == "random":
            rng = np.random.default_rng(int(seed))
            selected = np.sort(rng.choice(N, size=max_nodes, replace=False).astype(np.int64))
        else:
            raise ValueError("--node-sample must be first or random")
    else:
        selected = np.asarray(sorted(set(map(int, selected.tolist()))), dtype=np.int64)
        if int(max_nodes) > 0 and len(selected) > int(max_nodes):
            selected = selected[: int(max_nodes)]

    if len(selected) == 0:
        raise ValueError("Node selection is empty")
    return selected


# =============================================================================
# Session-aware supervised samples
# =============================================================================
def session_groups(meta: pd.DataFrame) -> List[np.ndarray]:
    if "session_id" in meta.columns:
        groups = []
        for _, sub in meta.groupby("session_id", sort=False):
            idx = sub.index.to_numpy(dtype=np.int64)
            if len(idx):
                groups.append(idx)
        return groups
    return [np.arange(len(meta), dtype=np.int64)]


def iter_eval_pairs_with_history(meta: pd.DataFrame, horizon: int, p: int) -> List[Tuple[int, int]]:
    pairs: List[Tuple[int, int]] = []
    h = int(horizon)
    p = int(p)
    for idx in session_groups(meta):
        if len(idx) <= h:
            continue
        for pos in range(p - 1, len(idx) - h):
            origin = int(idx[pos])
            target = int(idx[pos + h])
            # Ensure lags origin-r stay inside the same session by construction.
            pairs.append((origin, target))
    return pairs


def build_supervised_tensors(z: np.ndarray, meta: pd.DataFrame, horizon: int, p: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pairs = iter_eval_pairs_with_history(meta, horizon=horizon, p=p)
    if not pairs:
        raise RuntimeError(f"No valid samples for horizon={horizon}, p={p}")

    origins = np.asarray([a for a, _ in pairs], dtype=np.int64)
    targets = np.asarray([b for _, b in pairs], dtype=np.int64)
    z_arr = np.asarray(z, dtype=np.float32)

    lagged = []
    for r in range(int(p)):
        lagged.append(z_arr[origins - r])
    X_lags = np.stack(lagged, axis=1).astype(np.float32)  # [M, p, N]
    Y = z_arr[targets].astype(np.float32)                 # [M, N]
    return X_lags, Y, origins, targets


# =============================================================================
# Linear algebra helpers
# =============================================================================
def standardize_columns(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    mu = X.mean(axis=0, keepdims=True)
    Xc = X - mu
    std = Xc.std(axis=0, ddof=1, keepdims=True) if X.shape[0] > 1 else np.ones_like(mu)
    std = np.where(np.isfinite(std) & (std > 1e-8), std, 1.0).astype(np.float32)
    return np.nan_to_num(Xc / std, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def ridge_beta(X: np.ndarray, y: np.ndarray, ridge: float) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    p = X.shape[1]
    A = X.T @ X
    if ridge > 0:
        A = A + float(ridge) * np.eye(p, dtype=np.float64)
    b = X.T @ y
    try:
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(A, b, rcond=None)[0]


def fit_restricted_residuals(
    X_lags: np.ndarray,
    Y: np.ndarray,
    ridge: float,
    fit_intercept: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit y_i <- own lags of i for every target i."""
    M, p, N = X_lags.shape
    residuals = np.empty((M, N), dtype=np.float32)
    mse_r = np.empty(N, dtype=np.float32)
    coefs = np.empty((N, p + (1 if fit_intercept else 0)), dtype=np.float32)

    for i in maybe_iter(range(N), total=N, desc="restricted models"):
        Xi = X_lags[:, :, i]
        if fit_intercept:
            Xi = np.concatenate([np.ones((M, 1), dtype=np.float32), Xi], axis=1)
        y = Y[:, i]
        beta = ridge_beta(Xi, y, ridge=ridge)
        pred = Xi @ beta
        res = y - pred
        residuals[:, i] = res.astype(np.float32)
        mse_r[i] = float(np.mean(res ** 2))
        coefs[i] = beta.astype(np.float32)

    return residuals, mse_r, coefs


def choose_candidates_by_partial_corr(
    X_lags: np.ndarray,
    residuals: np.ndarray,
    max_candidates: int,
    block_size: int,
) -> List[np.ndarray]:
    """
    Candidate pruning for Granger tests.

    We use correlation between source lag variables and the restricted residual
    of each target. This is only a fast candidate filter; final weights are
    Granger predictive-improvement scores.
    """
    M, p, N = X_lags.shape
    K = max(1, min(int(max_candidates), max(N - 1, 1)))
    B = max(1, int(block_size))

    log(f"Standardizing source lag matrices for candidate search: p={p}, N={N}")
    X_std_by_lag = [standardize_columns(X_lags[:, r, :]) for r in range(p)]
    R_std = standardize_columns(residuals)

    candidates: List[Optional[np.ndarray]] = [None] * N
    denom = float(max(M - 1, 1))

    for start in maybe_iter(range(0, N, B), total=math.ceil(N / B), desc="candidate blocks"):
        end = min(start + B, N)
        rb = R_std[:, start:end]
        score = np.zeros((N, end - start), dtype=np.float32)
        for Xs in X_std_by_lag:
            corr = (Xs.T @ rb).astype(np.float32) / denom
            np.maximum(score, np.abs(corr), out=score)
        # remove self candidates
        for local, target in enumerate(range(start, end)):
            score[target, local] = -np.inf

        k_eff = min(K, N - 1)
        idx = np.argpartition(score, -k_eff, axis=0)[-k_eff:, :]  # [K, B]
        # Sort each target's candidates by descending candidate score for reproducibility.
        for local, target in enumerate(range(start, end)):
            cand = idx[:, local]
            vals = score[cand, local]
            order = np.argsort(-vals)
            candidates[target] = cand[order].astype(np.int64)

    return [np.asarray(c, dtype=np.int64) for c in candidates]  # type: ignore[arg-type]


def compute_granger_for_horizon(
    z: np.ndarray,
    meta: pd.DataFrame,
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
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Return G[target, source] and best source-lag matrix for one horizon."""
    t0 = time.time()
    X_lags, Y, origins, targets = build_supervised_tensors(z, meta, horizon=horizon, p=p)
    M, p_used, N = X_lags.shape
    log(f"h={horizon}: supervised tensors X_lags={X_lags.shape}, Y={Y.shape}")

    if M <= (2 * p_used + 2):
        raise RuntimeError(f"Too few samples for Granger h={horizon}: M={M}, p={p_used}")

    residuals, mse_r, _ = fit_restricted_residuals(
        X_lags=X_lags,
        Y=Y,
        ridge=float(ridge),
        fit_intercept=bool(fit_intercept),
    )
    candidates = choose_candidates_by_partial_corr(
        X_lags=X_lags,
        residuals=residuals,
        max_candidates=max_candidates,
        block_size=candidate_block_size,
    )

    G = np.zeros((N, N), dtype=np.float32)
    L = np.zeros((N, N), dtype=np.int16)

    n_edges_positive = 0
    source_coef_abs_sum = 0.0
    feature_count = 2 * p_used + (1 if fit_intercept else 0)

    log(f"h={horizon}: running pairwise candidate Granger tests | N={N}, candidates/target={max_candidates}")
    for target in maybe_iter(range(N), total=N, desc=f"granger h={horizon}"):
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
            beta = ridge_beta(Xf, y, ridge=ridge)
            pred = Xf @ beta
            mse_f = float(np.mean((y - pred) ** 2))
            if not np.isfinite(mse_f) or mse_f <= 0:
                continue

            score = math.log((mse_base + EPS) / (mse_f + EPS))
            if score <= float(min_improvement):
                continue

            # Source coefficients are the last p entries unless intercept shifts offset.
            src_offset = (1 if fit_intercept else 0) + p_used
            src_coefs = np.asarray(beta[src_offset: src_offset + p_used], dtype=np.float64)
            if src_coefs.size == 0:
                continue
            best_lag_idx = int(np.argmax(np.abs(src_coefs)))  # 0 means x_t, 1 means x_{t-1}
            coef = float(src_coefs[best_lag_idx])
            if signed:
                w = math.copysign(score, coef if coef != 0.0 else 1.0)
            else:
                w = score

            G[target, source] = float(w)
            L[target, source] = int(best_lag_idx)
            n_edges_positive += 1
            source_coef_abs_sum += abs(coef)

    np.fill_diagonal(G, 0.0)
    np.fill_diagonal(L, 0)

    elapsed = time.time() - t0
    summary = {
        "horizon": int(horizon),
        "n_samples": int(M),
        "n_segments": int(N),
        "granger_p": int(p_used),
        "max_candidates": int(max_candidates),
        "candidate_block_size": int(candidate_block_size),
        "ridge": float(ridge),
        "fit_intercept": bool(fit_intercept),
        "signed": bool(signed),
        "min_improvement": float(min_improvement),
        "n_nonzero_edges": int(np.count_nonzero(G)),
        "candidate_edges_tested_approx": int(N * min(max_candidates, max(N - 1, 1))),
        "mean_restricted_mse": float(np.mean(mse_r)),
        "median_restricted_mse": float(np.median(mse_r)),
        "elapsed_seconds": float(elapsed),
    }
    log(f"h={horizon} DONE | nonzero={summary['n_nonzero_edges']:,} | elapsed={elapsed/60:.2f} min")

    return G.astype(dtype), L.astype(lag_dtype), summary


# =============================================================================
# Saving output in downstream-compatible structure
# =============================================================================
def copy_basic_split_files(
    source_split_dir: Path,
    out_split_dir: Path,
    data: Dict[str, Any],
    node_idx: Optional[np.ndarray],
) -> None:
    ensure_dir(out_split_dir)

    # Save z/segment_ids/timestamps after optional node subset.
    np.save(out_split_dir / "z.npy", np.asarray(data["z"], dtype=np.float32))
    np.save(out_split_dir / "segment_ids.npy", np.asarray(data["segment_ids"], dtype=np.int64))
    np.save(out_split_dir / "timestamps.npy", np.asarray(data["timestamps"]).astype("datetime64[ns]"))

    # Meta files are unchanged by node subset.
    meta_path = source_split_dir / "G_series_meta.csv"
    raw_meta_path = source_split_dir / "raw_meta.csv"
    if meta_path.exists():
        shutil.copy2(meta_path, out_split_dir / "G_series_meta.csv")
    else:
        pd.DataFrame(data["meta"]).to_csv(out_split_dir / "G_series_meta.csv", index=False)

    if raw_meta_path.exists():
        shutil.copy2(raw_meta_path, out_split_dir / "raw_meta.csv")

    if node_idx is not None:
        np.save(out_split_dir / "selected_node_indices.npy", np.asarray(node_idx, dtype=np.int64))


def save_static_granger_to_splits(
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
    h_max = int(max(horizons))
    N = int(next(iter(G_by_h.values())).shape[0])

    for split_name, data in splits.items():
        split_out = out_dir / split_name
        source_split_dir = source_dir / split_name
        if split_out.exists():
            shutil.rmtree(split_out)
        ensure_dir(split_out)
        copy_basic_split_files(source_split_dir, split_out, data, node_idx=node_idx)

        G_path = split_out / "G_weight_series.npy"
        L_path = split_out / "G_best_lag_series.npy"
        G_mem = np.lib.format.open_memmap(G_path, mode="w+", dtype=dtype, shape=(h_max + 1, N, N))
        L_mem = np.lib.format.open_memmap(L_path, mode="w+", dtype=lag_dtype, shape=(h_max + 1, N, N))
        G_mem[:] = 0
        L_mem[:] = 0
        for h in horizons:
            G_mem[int(h)] = G_by_h[int(h)].astype(dtype)
            L_mem[int(h)] = L_by_h[int(h)].astype(lag_dtype)
        G_mem.flush(); L_mem.flush()
        del G_mem, L_mem

        split_summary = dict(run_summary)
        split_summary.update({
            "split": split_name,
            "G_shape": [int(h_max + 1), int(N), int(N)],
            "G_semantics": "static train-only Granger/predictive influence graph indexed by horizon; G[h,target,source]",
            "z_shape": list(map(int, np.asarray(data["z"]).shape)),
            "segment_ids_shape": list(map(int, np.asarray(data["segment_ids"]).shape)),
        })
        save_json(split_summary, split_out / "branchB_granger_split_summary.json")
        # Also save with old name for compatibility with checks.
        save_json(split_summary, split_out / "branchB_gt_split_summary.json")

    # Save global horizon summaries.
    save_json(run_summary, out_dir / "branchB_granger_run_summary.json")


def write_readme(out_dir: Path, args: argparse.Namespace) -> None:
    text = f"""# Branch B Granger-style directed predictive graph

This directory was created by `prepare_branchB_osm_edge_granger_like_branchA.py`.

It keeps the downstream Branch-B file names but changes graph semantics:

- OLD: `G_weight_series.npy` from lagged correlation.
- NEW: `G_weight_series.npy[h, target, source]` is a static train-only Granger-style predictive influence graph for horizon `h`.

Weight definition:

`G[target, source, h] = sign(source coefficient) * max(0, log((MSE_R + eps)/(MSE_F + eps)))`

where:

- Restricted model predicts target from its own lags.
- Full model adds source lags.
- Candidate pruning is used before pairwise Granger tests.

Args:

```json
{json.dumps(vars(args), ensure_ascii=False, indent=2)}
```
"""
    (out_dir / "README_GRANGER.md").write_text(text, encoding="utf-8")


# =============================================================================
# CLI
# =============================================================================
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=str, default=None, help="Existing lagged-correlation Branch-B prepared dir used only for z/meta/splits.")
    parser.add_argument("--output-dir", type=str, default=None, help="Output Granger Branch-B prepared dir.")
    parser.add_argument("--horizons", type=str, default="1-9", help="Forecast horizons, e.g. 1-9 or 1,2,3.")
    parser.add_argument("--granger-p", type=int, default=3, help="Number of own/source lags: x_t, x_{t-1}, ..., x_{t-p+1}.")
    parser.add_argument("--max-candidates", type=int, default=50, help="Candidate sources per target before Granger testing.")
    parser.add_argument("--candidate-block-size", type=int, default=256, help="Target block size for candidate search.")
    parser.add_argument("--ridge", type=float, default=1e-4, help="Small ridge regularization for OLS solves.")
    parser.add_argument("--min-improvement", type=float, default=0.0, help="Minimum log(MSE_R/MSE_F) to keep an edge.")
    parser.add_argument("--unsigned", action="store_true", help="Use nonnegative Granger strengths instead of signed source coefficients.")
    parser.add_argument("--fit-intercept", action="store_true", help="Include intercept in restricted/full Granger regressions.")
    parser.add_argument("--g-dtype", type=str, default="float16")
    parser.add_argument("--lag-dtype", type=str, default="int8")
    parser.add_argument("--max-nodes", type=int, default=0, help="Use first/random N nodes for quick test. 0 means full.")
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
    if min(horizons) < 1:
        raise ValueError("Horizons must be >= 1")

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

    train = splits["train"]
    z_train = np.asarray(train["z"], dtype=np.float32)
    N = int(z_train.shape[1])
    log(f"train z shape: {z_train.shape}")
    log(f"node mode: {'full' if node_idx is None else f'subset n={len(node_idx)}'}")

    h_max = max(horizons)
    one_graph_bytes = (h_max + 1) * N * N * np.dtype(args.g_dtype).itemsize
    log(f"Estimated one split G file: {fmt_gb(one_graph_bytes)}; all 3 splits: {fmt_gb(one_graph_bytes * 3)}")

    if args.dry_run:
        log("DRY RUN: stop before computation.")
        return

    if output_dir.exists() and args.overwrite:
        log(f"[CLEAN] removing old output: {output_dir}")
        shutil.rmtree(output_dir)
    ensure_dir(output_dir)

    print_stage("COMPUTE STATIC TRAIN-ONLY GRANGER GRAPHS")
    G_by_h: Dict[int, np.ndarray] = {}
    L_by_h: Dict[int, np.ndarray] = {}
    horizon_summaries: List[Dict[str, Any]] = []

    for h in horizons:
        G, L, summary = compute_granger_for_horizon(
            z=z_train,
            meta=train["meta"],
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
        )
        G_by_h[int(h)] = G
        L_by_h[int(h)] = L
        horizon_summaries.append(summary)

    run_summary = {
        "created_at": now_str(),
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "mode": "static_train_only_granger",
        "horizons": [int(x) for x in horizons],
        "max_horizon": int(h_max),
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
        "horizon_summaries": horizon_summaries,
    }

    print_stage("SAVE DOWNSTREAM-COMPATIBLE OUTPUTS")
    save_static_granger_to_splits(
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
    log("DONE. Next run 06B with --data-dir pointing to this output and --methods no_gt,granger_gt")


if __name__ == "__main__":
    main()


# =============================================================================
# Compatibility helper for Granger-Gt standard series
# =============================================================================
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
    label: str = "granger",
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Compute Granger-style directed predictive influence graph from prebuilt
    supervised tensors.

    This is the same core estimator as compute_granger_for_horizon(), but it
    accepts X_lags and Y directly. It is required by
    prepare_branchB_osm_edge_granger_series_like_branchA.py for bucket-wise Gt.

    X_lags[m, r, n] = x_n at origin-r, r=0..p-1
    Y[m, n]         = target x_n
    Output convention: G[target, source]
    """
    t0 = time.time()
    X_lags = np.asarray(X_lags, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.float32)
    if X_lags.ndim != 3 or Y.ndim != 2:
        raise ValueError(f"X_lags must be [M,p,N] and Y [M,N], got {X_lags.shape}, {Y.shape}")
    M, p_used, N = X_lags.shape
    if Y.shape != (M, N):
        raise ValueError(f"Y shape mismatch: expected {(M, N)}, got {Y.shape}")
    if M <= (2 * p_used + 2):
        raise RuntimeError(f"Too few samples for Granger {label}: M={M}, p={p_used}")

    log(f"{label}: supervised tensors X_lags={X_lags.shape}, Y={Y.shape}")

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

    log(f"{label}: running candidate Granger tests | N={N}, candidates/target={max_candidates}")
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
        "granger_p": int(p_used),
        "max_candidates": int(max_candidates),
        "candidate_block_size": int(candidate_block_size),
        "ridge": float(ridge),
        "fit_intercept": bool(fit_intercept),
        "signed": bool(signed),
        "min_improvement": float(min_improvement),
        "n_nonzero_edges": int(np.count_nonzero(G)),
        "candidate_edges_tested_approx": int(N * min(int(max_candidates), max(N - 1, 1))),
        "mean_restricted_mse": float(np.mean(mse_r)),
        "median_restricted_mse": float(np.median(mse_r)),
        "elapsed_seconds": float(elapsed),
    }
    log(f"{label} DONE | nonzero={summary['n_nonzero_edges']:,} | elapsed={elapsed/60:.2f} min")
    return G.astype(dtype), L.astype(lag_dtype), summary
