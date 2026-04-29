# ml_core/src/models/ML_BranchB/scripts/06B_check_xt_graph_orientation.py

from pathlib import Path
from typing import Dict, List, Optional
import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import MultiTaskElasticNet


# ============================================================
# Config giống các file 06 hiện tại
# ============================================================
EPS = 1e-8
DEFAULT_HORIZONS = list(range(1, 10))

ALPHA = 0.001
L1_RATIO = 0.5
MAX_ITER = 500
TOL = 1e-2
SELECTION = "random"
RANDOM_STATE = 42


# ============================================================
# Path / loading
# ============================================================
def find_project_root() -> Path:
    cwd = Path.cwd().resolve()
    candidates = [cwd, *cwd.parents, Path("/kaggle/working/UTraffic-ML"), Path("/kaggle/working")]
    for p in candidates:
        if (p / "ml_core").exists() and (p / "dataset").exists():
            return p
        if p.name == "UTraffic-ML":
            return p
        if (p / "UTraffic-ML").exists():
            pp = p / "UTraffic-ML"
            if (pp / "ml_core").exists():
                return pp
    return cwd


def check_branchB_common_dir_ready(common_dir: Path) -> None:
    required = []
    for split in ["train", "val", "test"]:
        for name in [
            "G_weight_series.npy",
            "G_best_lag_series.npy",
            "z.npy",
            "segment_ids.npy",
            "timestamps.npy",
            "G_series_meta.csv",
            "raw_meta.csv",
        ]:
            required.append(common_dir / split / name)

    missing = [p for p in required if not p.exists()]
    if missing:
        print("\nMissing files:")
        for p in missing[:30]:
            print(" -", p)
        raise FileNotFoundError(
            "Branch B prepared data is incomplete. Run:\n"
            "python ml_core/src/data_processing/prepare_branchB_osm_edge_gt_like_branchA.py --overwrite"
        )


def load_gt_split(common_dir: Path, split_name: str, mmap_mode: str = "r") -> Dict[str, object]:
    split_dir = common_dir / split_name

    G_weight_series = np.load(split_dir / "G_weight_series.npy", mmap_mode=mmap_mode)
    z = np.load(split_dir / "z.npy", mmap_mode=mmap_mode)
    segment_ids = np.load(split_dir / "segment_ids.npy")
    timestamps = pd.to_datetime(np.load(split_dir / "timestamps.npy"))

    meta = pd.read_csv(split_dir / "G_series_meta.csv")
    if "timestamp_local" in meta.columns:
        meta["timestamp_local"] = pd.to_datetime(meta["timestamp_local"])

    return {
        "G_weight_series": G_weight_series,
        "z": z,
        "segment_ids": segment_ids.astype(np.int64),
        "timestamps": timestamps,
        "meta": meta,
    }


# ============================================================
# Pair generation
# ============================================================
def iter_eval_pairs(meta: pd.DataFrame, horizon: int):
    T = len(meta)
    sess = meta["session_id"].to_numpy() if "session_id" in meta.columns else None

    for origin_idx in range(T - horizon):
        target_idx = origin_idx + horizon

        # Không cho nhảy qua session/ngày khác
        if sess is not None and sess[origin_idx] != sess[target_idx]:
            continue

        yield origin_idx, target_idx


# ============================================================
# Metrics
# ============================================================
def batch_vector_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    yt = np.asarray(y_true, dtype=np.float32)
    yp = np.asarray(y_pred, dtype=np.float32)

    diff = yp - yt
    mse = float(np.mean(diff ** 2))
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(mse))

    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
    }


# ============================================================
# G methods cần kiểm tra orientation
# ============================================================
def predict_G_basic(
    method_name: str,
    split_data: Dict[str, object],
    origin_idx: int,
    target_idx: int,
) -> np.ndarray:
    G = split_data["G_weight_series"]

    if method_name == "true_gt":
        # Oracle: lấy G tại thời điểm target
        return np.asarray(G[target_idx], dtype=np.float32)

    if method_name == "persistence_gt":
        # Persistence: lấy G tại origin
        return np.asarray(G[origin_idx], dtype=np.float32)

    raise ValueError(f"Unsupported method for orientation check: {method_name}")


def graph_signal(G_used: np.ndarray, x_t: np.ndarray, orientation: str) -> np.ndarray:
    """
    orientation='old':
        gx = G @ x

    orientation='transpose':
        gx = G.T @ x

    Nếu G[i,j] = i -> j thì thường gx_j = sum_i G[i,j] x_i,
    tức là nên dùng G.T @ x.
    """
    if orientation == "old":
        return np.asarray(G_used @ x_t, dtype=np.float32)

    if orientation == "transpose":
        return np.asarray(G_used.T @ x_t, dtype=np.float32)

    raise ValueError(f"Unknown orientation: {orientation}")


# ============================================================
# Build Xt dataset
# ============================================================
def build_xt_dataset_for_horizon(
    method_name: str,
    split_data: Dict[str, object],
    horizon: int,
    use_gt: bool,
    orientation: str = "old",
    max_pairs: Optional[int] = None,
):
    z = np.asarray(split_data["z"], dtype=np.float32)
    meta = split_data["meta"]

    X_rows, Y_rows = [], []

    for k, (origin_idx, target_idx) in enumerate(iter_eval_pairs(meta, horizon)):
        if max_pairs is not None and k >= max_pairs:
            break

        x_t = np.asarray(z[origin_idx], dtype=np.float32)
        y_true = np.asarray(z[target_idx], dtype=np.float32)

        if use_gt:
            G_used = predict_G_basic(
                method_name=method_name,
                split_data=split_data,
                origin_idx=origin_idx,
                target_idx=target_idx,
            )

            gx = graph_signal(G_used, x_t, orientation=orientation)
            feat = np.concatenate([x_t, gx], axis=0)
        else:
            feat = x_t

        X_rows.append(feat.astype(np.float32))
        Y_rows.append(y_true.astype(np.float32))

    if not X_rows:
        n = z.shape[1]
        in_dim = n * (2 if use_gt else 1)
        return (
            np.empty((0, in_dim), dtype=np.float32),
            np.empty((0, n), dtype=np.float32),
        )

    return (
        np.stack(X_rows, axis=0).astype(np.float32),
        np.stack(Y_rows, axis=0).astype(np.float32),
    )


# ============================================================
# Fit Xt model
# ============================================================
def fit_direct_xt_model(X_train: np.ndarray, Y_train: np.ndarray) -> MultiTaskElasticNet:
    model = MultiTaskElasticNet(
        alpha=float(ALPHA),
        l1_ratio=float(L1_RATIO),
        fit_intercept=True,
        max_iter=int(MAX_ITER),
        tol=float(TOL),
        selection=SELECTION,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, Y_train)
    return model


# ============================================================
# Run one variant
# ============================================================
def run_one_variant(
    label: str,
    base_method: str,
    train: Dict[str, object],
    val: Dict[str, object],
    test: Dict[str, object],
    horizons: List[int],
    use_gt: bool,
    orientation: str,
    max_pairs: Optional[int] = None,
) -> pd.DataFrame:
    n_segments = int(len(train["segment_ids"]))
    rows = []

    print("\n" + "=" * 90)
    print(f"RUN VARIANT: {label}")
    print(f"base_method={base_method} | use_gt={use_gt} | orientation={orientation}")
    print("=" * 90)

    for horizon in horizons:
        print(f"\n[HORIZON {horizon}] build train feature...")

        X_train, Y_train = build_xt_dataset_for_horizon(
            method_name=base_method,
            split_data=train,
            horizon=horizon,
            use_gt=use_gt,
            orientation=orientation,
            max_pairs=max_pairs,
        )

        if len(X_train) == 0:
            print("No train samples. Skip.")
            continue

        print("X_train:", X_train.shape, "Y_train:", Y_train.shape)
        model = fit_direct_xt_model(X_train, Y_train)

        for split_name, split_data in [("val", val), ("test", test)]:
            print(f"[HORIZON {horizon}] evaluate {split_name}...")

            X_split, Y_split = build_xt_dataset_for_horizon(
                method_name=base_method,
                split_data=split_data,
                horizon=horizon,
                use_gt=use_gt,
                orientation=orientation,
                max_pairs=max_pairs,
            )

            if len(X_split) == 0:
                continue

            Y_pred = model.predict(X_split).astype(np.float32)
            metrics = batch_vector_metrics(Y_split, Y_pred)

            row = {
                "method": label,
                "base_method": base_method,
                "orientation": orientation,
                "split": split_name,
                "lag": int(horizon),
                "n_samples": int(len(X_split)),
                "n_segments": int(n_segments),
                **metrics,
            }
            rows.append(row)

            print(split_name, metrics)

        del X_train, Y_train, model

    return pd.DataFrame(rows)


# ============================================================
# Main
# ============================================================
def parse_list_int(s: str) -> List[int]:
    out = []
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


def parse_list_str(s: str) -> List[str]:
    return [x.strip() for x in str(s).split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--methods",
        type=str,
        default="true_gt,persistence_gt",
        help="Methods to check. Recommended: true_gt,persistence_gt",
    )
    parser.add_argument(
        "--horizons",
        type=str,
        default="1-9",
        help="Horizons, e.g. 1-9 or 1,2,3,4",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="Quick debug only. Example: --max-pairs 80. Leave empty for full evaluation.",
    )

    args = parser.parse_args()

    methods = parse_list_str(args.methods)
    horizons = parse_list_int(args.horizons)

    project_root = find_project_root()
    branchb_root = project_root / "ml_core" / "src" / "models" / "ML_BranchB"
    common_dir = (
        project_root
        / "ml_core"
        / "src"
        / "data_processing"
        / "outputs"
        / "branchB"
        / "osm_edge_gt_like_branchA"
    )

    out_dir = branchb_root / "results" / "06_branchB_xt_orientation_check"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("PROJECT_ROOT:", project_root)
    print("COMMON_DIR  :", common_dir)
    print("OUT_DIR     :", out_dir)
    print("METHODS     :", methods)
    print("HORIZONS    :", horizons)
    print("MAX_PAIRS   :", args.max_pairs)

    check_branchB_common_dir_ready(common_dir)

    train = load_gt_split(common_dir, "train")
    val = load_gt_split(common_dir, "val")
    test = load_gt_split(common_dir, "test")

    assert np.array_equal(train["segment_ids"], val["segment_ids"])
    assert np.array_equal(train["segment_ids"], test["segment_ids"])

    all_results = []

    # Baseline No-G / No-Rt
    all_results.append(
        run_one_variant(
            label="no_gt",
            base_method="true_gt",   # không dùng vì use_gt=False
            train=train,
            val=val,
            test=test,
            horizons=horizons,
            use_gt=False,
            orientation="none",
            max_pairs=args.max_pairs,
        )
    )

    # Check old vs transpose
    for method in methods:
        for orientation in ["old", "transpose"]:
            label = f"{method}_{orientation}"

            all_results.append(
                run_one_variant(
                    label=label,
                    base_method=method,
                    train=train,
                    val=val,
                    test=test,
                    horizons=horizons,
                    use_gt=True,
                    orientation=orientation,
                    max_pairs=args.max_pairs,
                )
            )

    df = pd.concat(all_results, ignore_index=True)
    df = df.sort_values(["split", "lag", "method"]).reset_index(drop=True)

    raw_path = out_dir / "branchB_xt_orientation_check_metrics.csv"
    df.to_csv(raw_path, index=False)
    print("\nSaved raw metrics:", raw_path)

    # Add delta vs No-G
    base = df[df["method"] == "no_gt"][["split", "lag", "mae", "rmse"]].rename(
        columns={"mae": "no_gt_mae", "rmse": "no_gt_rmse"}
    )

    cmp_df = df.merge(base, on=["split", "lag"], how="left")
    cmp_df["delta_mae_vs_no_gt"] = cmp_df["mae"] - cmp_df["no_gt_mae"]
    cmp_df["delta_rmse_vs_no_gt"] = cmp_df["rmse"] - cmp_df["no_gt_rmse"]

    cmp_path = out_dir / "branchB_xt_orientation_check_compare_vs_no_gt.csv"
    cmp_df.to_csv(cmp_path, index=False)
    print("Saved compare metrics:", cmp_path)

    # Print compact table
    compact = cmp_df[
        [
            "method",
            "split",
            "lag",
            "mae",
            "no_gt_mae",
            "delta_mae_vs_no_gt",
            "rmse",
            "delta_rmse_vs_no_gt",
            "n_samples",
        ]
    ].sort_values(["split", "lag", "method"])

    print("\n===== COMPACT RESULT =====")
    print(compact.to_string(index=False))

    print("\nDone.")
    print("Lower MAE/RMSE is better.")
    print("delta_mae_vs_no_gt < 0 means the method is better than No-G.")


if __name__ == "__main__":
    main()