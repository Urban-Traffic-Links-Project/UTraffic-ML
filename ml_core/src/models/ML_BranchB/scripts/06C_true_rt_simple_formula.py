# ml_core/src/models/ML_BranchB/scripts/06C_true_rt_simple_formula.py
"""
Branch B — True-Rt XT forecast using the SIMPLE formula only.

Simple formula tested:
    z_hat[t+h] = A_h z[t] + B_h( W_true[t->t+h] z[t] )

In this implementation:
- z is the prepared normalized velocity tensor from Branch-B data.
- W_true[t->t+h] is built from True-Rt at target_idx = t+h.
- By default, we keep only edges whose best_lag == h, because for ell=0
  the theoretical relation delay is d = h + ell = h.
- Then W is Top-K filtered, row-L1-normalized, and multiplied by z[t].
- Features are [z_t, W_h z_t].
- Only True-Rt is run. No-Rt must already exist if you want to compare in plot.

Output is compatible with:
    07_branchB_plot_xt_forecast_results.py

Example:
    python ml_core/src/models/ML_BranchB/scripts/06C_true_rt_simple_formula.py \
      --topk 50 --lags 1-9 --max-nodes 512 --overwrite

Full:
    python ml_core/src/models/ML_BranchB/scripts/06C_true_rt_simple_formula.py \
      --topk 50 --lags 1-9 --overwrite
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from sklearn.linear_model import MultiTaskElasticNet, Ridge

EPS = 1e-8


# =============================================================================
# Basic utilities
# =============================================================================

def now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    print(f"[{now_str()}] {msg}", flush=True)


def find_project_root() -> Path:
    cwd = Path.cwd().resolve()
    candidates = [
        cwd,
        *cwd.parents,
        Path("/kaggle/working/UTraffic-ML"),
        Path("/kaggle/working"),
    ]
    for p in candidates:
        if (p / "ml_core").exists():
            return p if p.name == "UTraffic-ML" else p
        if (p / "UTraffic-ML" / "ml_core").exists():
            return p / "UTraffic-ML"
    return cwd


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


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj: Dict[str, Any], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# =============================================================================
# Data loading and node subset
# =============================================================================

def load_split(common_dir: Path, split: str) -> Dict[str, Any]:
    d = common_dir / split
    required = [
        d / "G_weight_series.npy",
        d / "G_best_lag_series.npy",
        d / "z.npy",
        d / "segment_ids.npy",
        d / "timestamps.npy",
        d / "G_series_meta.csv",
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing Branch-B files:\n" + "\n".join(map(str, missing)))

    meta = pd.read_csv(d / "G_series_meta.csv")
    if "timestamp_local" in meta.columns:
        meta["timestamp_local"] = pd.to_datetime(meta["timestamp_local"])

    return {
        "G_weight_series": np.load(d / "G_weight_series.npy", mmap_mode="r"),
        "G_best_lag_series": np.load(d / "G_best_lag_series.npy", mmap_mode="r"),
        "z": np.asarray(np.load(d / "z.npy"), dtype=np.float32),
        "segment_ids": np.asarray(np.load(d / "segment_ids.npy"), dtype=np.int64),
        "timestamps": np.load(d / "timestamps.npy", allow_pickle=True),
        "meta": meta,
    }


def resolve_node_indices(
    common_dir: Path,
    max_nodes: int = 0,
    node_indices_arg: Optional[str] = None,
    node_ids_arg: Optional[str] = None,
    node_sample: str = "first",
    seed: int = 42,
) -> Optional[np.ndarray]:
    if not node_indices_arg and not node_ids_arg and int(max_nodes) <= 0:
        return None

    seg_path = common_dir / "train" / "segment_ids.npy"
    if not seg_path.exists():
        raise FileNotFoundError(f"Missing segment_ids for node selection: {seg_path}")

    segment_ids = np.asarray(np.load(seg_path), dtype=np.int64)
    N = int(len(segment_ids))
    selected: Optional[np.ndarray] = None

    if node_indices_arg:
        idx = np.asarray(parse_int_list(node_indices_arg), dtype=np.int64)
        if len(idx) == 0:
            raise ValueError("--node-indices was provided but no valid index was parsed.")
        if idx.min() < 0 or idx.max() >= N:
            raise ValueError(f"node index out of range. N={N}, min={idx.min()}, max={idx.max()}")
        selected = idx

    if node_ids_arg:
        requested = np.asarray(parse_int_list(node_ids_arg), dtype=np.int64)
        id_to_pos = {int(v): i for i, v in enumerate(segment_ids)}
        missing = [int(x) for x in requested if int(x) not in id_to_pos]
        if missing:
            raise ValueError(f"Some --node-ids are not in train/segment_ids.npy: {missing[:20]}")
        idx = np.asarray([id_to_pos[int(x)] for x in requested], dtype=np.int64)
        selected = idx if selected is None else np.intersect1d(selected, idx)

    if selected is None:
        max_nodes = min(int(max_nodes), N)
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
        raise ValueError("Node selection is empty.")
    return selected


def subset_split_data(data: Dict[str, Any], node_idx: Optional[np.ndarray]) -> Dict[str, Any]:
    if node_idx is None:
        return data

    idx = np.asarray(node_idx, dtype=np.int64)
    out = dict(data)
    out["segment_ids"] = np.asarray(data["segment_ids"], dtype=np.int64)[idx]
    out["z"] = np.asarray(data["z"], dtype=np.float32)[:, idx]
    out["G_weight_series"] = np.asarray(data["G_weight_series"][:, idx, :][:, :, idx], dtype=np.float32)
    out["G_best_lag_series"] = np.asarray(data["G_best_lag_series"][:, idx, :][:, :, idx], dtype=np.int16)
    return out


# =============================================================================
# Pair creation
# =============================================================================

def session_col(meta: pd.DataFrame) -> Optional[str]:
    for c in ["session_id", "date", "date_key"]:
        if c in meta.columns:
            return c
    return None


def iter_eval_pairs(meta: pd.DataFrame, horizon: int) -> List[Tuple[int, int]]:
    """
    Return origin_idx, target_idx pairs without crossing sessions/days.
    This mimics the intended Branch-B split behavior.
    """
    h = int(horizon)
    if h <= 0:
        raise ValueError("horizon must be positive")

    col = session_col(meta)
    pairs: List[Tuple[int, int]] = []

    if col is None:
        T = len(meta)
        return [(i, i + h) for i in range(max(0, T - h))]

    for _, sub in meta.groupby(col, sort=False):
        idx = sub.index.to_numpy(dtype=np.int64)
        if len(idx) <= h:
            continue
        for p in range(0, len(idx) - h):
            pairs.append((int(idx[p]), int(idx[p + h])))
    return pairs


# =============================================================================
# Graph signal
# =============================================================================

def topk_graph_signal(
    G: np.ndarray,
    x_t: np.ndarray,
    topk: int = 50,
    normalize: bool = True,
    remove_self_loop: bool = True,
) -> np.ndarray:
    """
    G convention: target x source.
    Signal: gx[target] = sum_source W[target, source] * x_t[source].
    """
    G = np.asarray(G, dtype=np.float32)
    x_t = np.asarray(x_t, dtype=np.float32)
    if G.ndim != 2 or G.shape[0] != G.shape[1]:
        raise ValueError(f"G must be square, got {G.shape}")
    N = int(G.shape[0])
    if x_t.shape[0] != N:
        raise ValueError(f"x_t shape mismatch: {x_t.shape} vs G={G.shape}")

    G_work = G.astype(np.float32, copy=True)
    if remove_self_loop:
        diag = np.arange(N)
        G_work[diag, diag] = 0.0

    k = int(topk)
    if k <= 0 or k >= N:
        G_new = G_work
    else:
        G_new = np.zeros_like(G_work, dtype=np.float32)
        absG = np.abs(G_work)
        idx = np.argpartition(absG, -k, axis=1)[:, -k:]
        rows = np.arange(N)[:, None]
        G_new[rows, idx] = G_work[rows, idx]

    if normalize:
        denom = np.sum(np.abs(G_new), axis=1, keepdims=True) + EPS
        G_new = G_new / denom

    gx = G_new @ x_t
    return np.nan_to_num(gx, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def true_W_for_horizon(
    data: Dict[str, Any],
    target_idx: int,
    horizon: int,
    lag_mode: str = "matching",
) -> np.ndarray:
    """
    Build W_true[t->t+h] from True-Rt.

    lag_mode='matching':
        keep only edges where G_best_lag == horizon.
        This matches the simple formula W^{(0)} with delay d=h.
    lag_mode='all':
        use the full mixed-best-lag True-Rt matrix, similar to older True-Rt runs.
    """
    G = np.asarray(data["G_weight_series"][target_idx], dtype=np.float32)
    if lag_mode == "all":
        return G
    if lag_mode == "matching":
        L = np.asarray(data["G_best_lag_series"][target_idx])
        return np.where(L == int(horizon), G, 0.0).astype(np.float32)
    raise ValueError("--lag-mode must be matching or all")


# =============================================================================
# Dataset, model, metrics
# =============================================================================

def build_dataset_for_horizon(
    split_data: Dict[str, Any],
    horizon: int,
    topk: int,
    normalize: bool,
    remove_self_loop: bool,
    lag_mode: str,
) -> Tuple[np.ndarray, np.ndarray]:
    z = np.asarray(split_data["z"], dtype=np.float32)
    meta = split_data["meta"]
    pairs = iter_eval_pairs(meta, int(horizon))

    X_rows: List[np.ndarray] = []
    Y_rows: List[np.ndarray] = []

    for origin_idx, target_idx in pairs:
        x_t = z[origin_idx]
        y = z[target_idx]

        W = true_W_for_horizon(split_data, target_idx, int(horizon), lag_mode=lag_mode)
        gx = topk_graph_signal(W, x_t, topk=topk, normalize=normalize, remove_self_loop=remove_self_loop)

        feat = np.concatenate([x_t, gx], axis=0).astype(np.float32)
        X_rows.append(feat)
        Y_rows.append(y.astype(np.float32))

    if not X_rows:
        N = z.shape[1]
        return np.empty((0, 2 * N), dtype=np.float32), np.empty((0, N), dtype=np.float32)

    return np.vstack(X_rows).astype(np.float32), np.vstack(Y_rows).astype(np.float32)


def fit_model(X: np.ndarray, Y: np.ndarray, model_type: str, fit_intercept: bool):
    if model_type == "elasticnet":
        model = MultiTaskElasticNet(
            alpha=0.001,
            l1_ratio=0.5,
            fit_intercept=bool(fit_intercept),
            max_iter=500,
            tol=1e-2,
            selection="random",
            random_state=42,
        )
    elif model_type == "ridge":
        model = Ridge(alpha=1.0, fit_intercept=bool(fit_intercept), random_state=42)
    else:
        raise ValueError("--model-type must be elasticnet or ridge")
    model.fit(X, Y)
    return model


def vector_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    err = np.asarray(y_pred, dtype=np.float32) - np.asarray(y_true, dtype=np.float32)
    mae = float(np.mean(np.abs(err)))
    mse = float(np.mean(err ** 2))
    rmse = float(np.sqrt(mse))
    return {"mae": mae, "mse": mse, "rmse": rmse}


def run_horizon(
    train: Dict[str, Any],
    val: Dict[str, Any],
    test: Dict[str, Any],
    horizon: int,
    method_name: str,
    topk: int,
    normalize: bool,
    remove_self_loop: bool,
    lag_mode: str,
    model_type: str,
    fit_intercept: bool,
) -> List[Dict[str, Any]]:
    log(f"[h={horizon}] build train dataset")
    X_train, Y_train = build_dataset_for_horizon(
        train, horizon, topk, normalize, remove_self_loop, lag_mode
    )
    if len(X_train) == 0:
        log(f"[h={horizon}] no train samples, skip.")
        return []

    log(f"[h={horizon}] X_train={X_train.shape}, Y_train={Y_train.shape}")
    model = fit_model(X_train, Y_train, model_type=model_type, fit_intercept=fit_intercept)

    rows: List[Dict[str, Any]] = []
    for split_name, split_data in [("val", val), ("test", test)]:
        X_split, Y_split = build_dataset_for_horizon(
            split_data, horizon, topk, normalize, remove_self_loop, lag_mode
        )
        if len(X_split) == 0:
            continue

        Y_pred = model.predict(X_split).astype(np.float32)
        metrics = vector_metrics(Y_split, Y_pred)
        row = {
            "method": method_name,
            "base_method": "true_gt",
            "formula": "simple",
            "split": split_name,
            "lag": int(horizon),
            "n_samples": int(len(X_split)),
            "n_segments": int(Y_split.shape[1]),
            "topk": int(topk),
            "normalize": bool(normalize),
            "remove_self_loop": bool(remove_self_loop),
            "fit_intercept": bool(fit_intercept),
            "gamma_removed": not bool(fit_intercept),
            "lag_mode": lag_mode,
            "model_type": model_type,
            **metrics,
        }
        log(f"[h={horizon}] {split_name}: {metrics}")
        rows.append(row)

    return rows


# =============================================================================
# Main
# =============================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default=None, help="Prepared Branch-B data directory.")
    parser.add_argument("--results-dir", type=str, default=None, help="Base results dir compatible with 07 plot.")
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--lags", type=str, default="1-9")
    parser.add_argument("--lag-mode", type=str, default="matching", choices=["matching", "all"])
    parser.add_argument("--model-type", type=str, default="elasticnet", choices=["elasticnet", "ridge"])
    parser.add_argument("--fit-intercept", action="store_true")
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--keep-self-loop", action="store_true")
    parser.add_argument("--max-nodes", type=int, default=0)
    parser.add_argument("--node-indices", type=str, default=None)
    parser.add_argument("--node-ids", type=str, default=None)
    parser.add_argument("--node-sample", type=str, default="first", choices=["first", "random"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    project_root = find_project_root()

    common_dir = (
        Path(args.data_dir)
        if args.data_dir
        else project_root / "ml_core" / "src" / "data_processing" / "outputs" / "branchB" / "osm_edge_gt_like_branchA"
    )
    if not common_dir.is_absolute():
        common_dir = project_root / common_dir

    base_results_dir = (
        Path(args.results_dir)
        if args.results_dir
        else project_root / "ml_core" / "src" / "models" / "ML_BranchB" / "results" / "06_branchB_run_xt_forecast"
    )
    if not base_results_dir.is_absolute():
        base_results_dir = project_root / base_results_dir
    ensure_dir(base_results_dir)

    horizons = parse_int_list(args.lags)
    normalize = not bool(args.no_normalize)
    remove_self_loop = not bool(args.keep_self_loop)

    node_idx = resolve_node_indices(
        common_dir=common_dir,
        max_nodes=int(args.max_nodes),
        node_indices_arg=args.node_indices,
        node_ids_arg=args.node_ids,
        node_sample=args.node_sample,
        seed=int(args.seed),
    )
    node_tag = "full" if node_idx is None else f"nodes{len(node_idx)}"
    norm_tag = "norm" if normalize else "nonorm"
    self_tag = "noself" if remove_self_loop else "self"
    intercept_tag = "withbias" if args.fit_intercept else "nogamma"

    method_name = (
        f"formula_simple_true_rt_{args.lag_mode}_topk{int(args.topk)}_"
        f"{norm_tag}_{self_tag}_{intercept_tag}_{node_tag}"
    )

    out_dir = ensure_dir(base_results_dir / method_name)
    metrics_path = out_dir / f"{method_name}_xt_per_lag_metrics.csv"

    if metrics_path.exists() and not args.overwrite:
        log(f"Metrics already exists: {metrics_path}")
        log("Use --overwrite to rerun.")
        return

    log("PROJECT_ROOT : " + str(project_root))
    log("COMMON_DIR   : " + str(common_dir))
    log("RESULTS_DIR  : " + str(base_results_dir))
    log("METHOD_NAME  : " + method_name)
    log("HORIZONS     : " + str(horizons))
    log("NODE MODE    : " + node_tag)

    train = subset_split_data(load_split(common_dir, "train"), node_idx)
    val = subset_split_data(load_split(common_dir, "val"), node_idx)
    test = subset_split_data(load_split(common_dir, "test"), node_idx)

    if not np.array_equal(train["segment_ids"], val["segment_ids"]):
        raise ValueError("train and val segment_ids differ.")
    if not np.array_equal(train["segment_ids"], test["segment_ids"]):
        raise ValueError("train and test segment_ids differ.")

    rows: List[Dict[str, Any]] = []
    for h in horizons:
        rows.extend(
            run_horizon(
                train=train,
                val=val,
                test=test,
                horizon=int(h),
                method_name=method_name,
                topk=int(args.topk),
                normalize=normalize,
                remove_self_loop=remove_self_loop,
                lag_mode=str(args.lag_mode),
                model_type=str(args.model_type),
                fit_intercept=bool(args.fit_intercept),
            )
        )

    df = pd.DataFrame(rows)
    df.to_csv(metrics_path, index=False)
    log(f"Saved metrics: {metrics_path}")

    save_json(
        {
            "method_name": method_name,
            "formula": "z_hat[t+h] = A_h z[t] + B_h(TopK(W_true_h) z[t])",
            "lag_mode": args.lag_mode,
            "topk": int(args.topk),
            "normalize": normalize,
            "remove_self_loop": remove_self_loop,
            "fit_intercept": bool(args.fit_intercept),
            "model_type": args.model_type,
            "node_mode": node_tag,
            "horizons": horizons,
            "common_dir": str(common_dir),
            "metrics_path": str(metrics_path),
        },
        out_dir / "run_config.json",
    )

    log("Next plot command:")
    log(
        "python ml_core/src/models/ML_BranchB/scripts/07_branchB_plot_xt_forecast_results.py "
        f"--topk {int(args.topk)} --nogamma-only"
    )


if __name__ == "__main__":
    main()
