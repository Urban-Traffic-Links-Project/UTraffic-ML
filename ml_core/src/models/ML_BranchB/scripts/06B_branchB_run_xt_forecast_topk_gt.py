# ml_core/src/models/ML_BranchB/scripts/06B_branchB_run_xt_forecast_topk_gt.py

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Dict, Optional, List

import numpy as np
import pandas as pd


EPS = 1e-8


METHOD_TO_SCRIPT = {
    "true_gt": "06_branchB_run_xt_forecast_true_gt.py",
    "persistence_gt": "06_branchB_run_xt_forecast_persistence_gt.py",
    "ewma_gt": "06_branchB_run_xt_forecast_ewma_gt.py",
    "sparse_tvpvar_gt": "06_branchB_run_xt_forecast_sparse_tvpvar_gt.py",
    "factorized_var_gt": "06_branchB_run_xt_forecast_factorized_var_gt.py",
    "factorized_mar_gt": "06_branchB_run_xt_forecast_factorized_mar_gt.py",
    "factorized_tvpvar_gt": "06_branchB_run_xt_forecast_factorized_tvpvar_gt.py",
    "dense_tvpvar_gt": "06_branchB_run_xt_forecast_dense_tvpvar_gt.py",
}


def find_project_root() -> Path:
    cwd = Path.cwd().resolve()
    candidates = [
        cwd,
        *cwd.parents,
        Path("/kaggle/working/UTraffic-ML"),
        Path("/kaggle/working"),
    ]

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


def parse_int_list(s: str) -> List[int]:
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


def load_module_from_path(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, str(path))

    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def topk_graph_signal(
    G: np.ndarray,
    x_t: np.ndarray,
    topk: int,
    gamma: float,
    normalize: bool = True,
    remove_self_loop: bool = True,
) -> np.ndarray:
    """
    G shape: (N, N), convention: target x source.

    Original graph signal:
        gx = G @ x_t

    Top-K graph signal:
        gx_j = gamma * sum_{i in TopK(j)} G[j, i] * x_i

    If normalize=True:
        gx_j = gamma * weighted_mean(...)
    """

    G = np.asarray(G, dtype=np.float32)
    x_t = np.asarray(x_t, dtype=np.float32)

    if G.ndim != 2 or G.shape[0] != G.shape[1]:
        raise ValueError(f"G must be square matrix, got shape={G.shape}")

    N = G.shape[0]
    k = int(topk)

    if x_t.shape[0] != N:
        raise ValueError(f"x_t shape mismatch: x_t={x_t.shape}, G={G.shape}")

    # Nếu topk <= 0: giữ full G, chỉ áp dụng normalize/gamma nếu cần.
    if k <= 0 or k >= N:
        G_work = G.copy()

        if remove_self_loop:
            diag = np.arange(N)
            G_work[diag, diag] = 0.0

        gx = G_work @ x_t

        if normalize:
            denom = np.sum(np.abs(G_work), axis=1) + EPS
            gx = gx / denom

        return (float(gamma) * gx).astype(np.float32)

    absG = np.abs(G).astype(np.float32, copy=True)

    if remove_self_loop:
        diag = np.arange(N)
        absG[diag, diag] = -np.inf

    # idx[j] = top-k source indices for target row j
    idx = np.argpartition(absG, -k, axis=1)[:, -k:]

    rows = np.arange(N)[:, None]

    weights = G[rows, idx]   # shape: N x k
    x_src = x_t[idx]         # shape: N x k

    gx = np.sum(weights * x_src, axis=1)

    if normalize:
        denom = np.sum(np.abs(weights), axis=1) + EPS
        gx = gx / denom

    return (float(gamma) * gx).astype(np.float32)


def make_topk_build_dataset(
    module,
    method_name: str,
    topk: int,
    gamma: float,
    normalize: bool,
    remove_self_loop: bool,
):
    """
    Monkey-patch hàm build_xt_dataset_for_horizon của module gốc.

    Giữ toàn bộ logic gốc:
        G_used = predict_G_method(...)

    Chỉ thay:
        gx = G_used @ x_t

    Thành:
        gx = gamma * TopK(G_used) @ x_t
    """

    def build_xt_dataset_for_horizon(
        method_name_arg: str,
        g_model: Dict[str, object],
        train_data: Dict[str, object],
        split_name: str,
        split_data: Dict[str, object],
        horizon: int,
        use_gt: bool,
    ):
        z = np.asarray(split_data["z"], dtype=np.float32)
        meta = split_data["meta"]

        X_rows, Y_rows = [], []

        for origin_idx, target_idx in module.iter_eval_pairs(meta, horizon):
            x_t = np.asarray(z[origin_idx], dtype=np.float32)
            y_true = np.asarray(z[target_idx], dtype=np.float32)

            if use_gt:
                G_used = module.predict_G_method(
                    method_name,
                    g_model,
                    split_name,
                    split_data,
                    origin_idx,
                    target_idx,
                    horizon,
                )

                gx = topk_graph_signal(
                    G=G_used,
                    x_t=x_t,
                    topk=topk,
                    gamma=gamma,
                    normalize=normalize,
                    remove_self_loop=remove_self_loop,
                )

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

    return build_xt_dataset_for_horizon


def patch_metric_files(out_dir: Path, new_method_name: str) -> None:
    """
    Sau khi module gốc chạy xong, sửa cột method trong CSV để file 07 plot phân biệt được.
    """

    metric_files = list(out_dir.glob("*_xt_per_lag_metrics.csv"))

    if not metric_files:
        print(f"[WARN] No metric CSV found in {out_dir}")
        return

    for p in metric_files:
        df = pd.read_csv(p)
        df["method"] = new_method_name

        # Ghi đè file cũ để plot 07 đọc đúng method.
        df.to_csv(p, index=False)

        # Lưu thêm bản tên mới rõ ràng.
        new_path = out_dir / f"{new_method_name}_xt_per_lag_metrics.csv"
        df.to_csv(new_path, index=False)

        print(f"Patched method column: {p}")
        print(f"Saved labeled copy   : {new_path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=sorted(METHOD_TO_SCRIPT.keys()),
        help="Base G method to run with Top-K graph feature.",
    )

    parser.add_argument(
        "--topk",
        type=int,
        default=4,
        help="Keep top-k source edges per target row. Use <=0 to keep full G.",
    )

    parser.add_argument(
        "--gamma",
        type=float,
        default=1.0,
        help="Scale graph signal: gx = gamma * TopK(G) @ x_t.",
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Optional prepared Branch-B data dir. If omitted, use repo default.",
    )

    parser.add_argument(
        "--lags",
        type=str,
        default=None,
        help="Optional horizons, e.g. 1-9 or 1,2,3,4. If omitted, use base script HORIZONS.",
    )

    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Disable row L1 normalization after Top-K.",
    )

    parser.add_argument(
        "--keep-self-loop",
        action="store_true",
        help="Keep diagonal/self-loop edges. Default removes self-loop.",
    )

    args = parser.parse_args()

    project_root = find_project_root()

    scripts_dir = (
        project_root
        / "ml_core"
        / "src"
        / "models"
        / "ML_BranchB"
        / "scripts"
    )

    branchb_root = (
        project_root
        / "ml_core"
        / "src"
        / "models"
        / "ML_BranchB"
    )

    if args.data_dir is None:
        common_dir = (
            project_root
            / "ml_core"
            / "src"
            / "data_processing"
            / "outputs"
            / "branchB"
            / "osm_edge_gt_like_branchA"
        )
    else:
        common_dir = Path(args.data_dir)
        if not common_dir.is_absolute():
            common_dir = project_root / common_dir

    method_name = args.method
    topk = int(args.topk)
    gamma = float(args.gamma)
    normalize = not bool(args.no_normalize)
    remove_self_loop = not bool(args.keep_self_loop)

    gamma_tag = str(gamma).replace(".", "p")
    norm_tag = "norm" if normalize else "nonorm"
    self_tag = "noself" if remove_self_loop else "self"

    new_method_name = f"{method_name}_topk{topk}_g{gamma_tag}_{norm_tag}_{self_tag}"

    out_dir = (
        branchb_root
        / "results"
        / "06_branchB_run_xt_forecast"
        / new_method_name
    )

    script_path = scripts_dir / METHOD_TO_SCRIPT[method_name]

    print("PROJECT_ROOT:", project_root)
    print("COMMON_DIR  :", common_dir)
    print("SCRIPT_PATH :", script_path)
    print("OUT_DIR     :", out_dir)
    print("BASE METHOD :", method_name)
    print("NEW METHOD  :", new_method_name)
    print("TOPK        :", topk)
    print("GAMMA       :", gamma)
    print("NORMALIZE   :", normalize)
    print("REMOVE SELF :", remove_self_loop)

    if not script_path.exists():
        raise FileNotFoundError(f"Missing base script: {script_path}")

    if not common_dir.exists():
        raise FileNotFoundError(f"Missing common_dir: {common_dir}")

    module = load_module_from_path(script_path)

    if args.lags is not None:
        horizons = parse_int_list(args.lags)
        print("PATCH HORIZONS:", horizons)
        module.HORIZONS = horizons

    # Patch build dataset của module gốc.
    module.build_xt_dataset_for_horizon = make_topk_build_dataset(
        module=module,
        method_name=method_name,
        topk=topk,
        gamma=gamma,
        normalize=normalize,
        remove_self_loop=remove_self_loop,
    )

    # Chạy pipeline gốc nhưng với graph feature Top-K.
    module.run_branchB_xt_forecast(
        method_name=method_name,
        common_dir=common_dir,
        out_dir=out_dir,
        use_gt=True,
    )

    # Sửa method label trong CSV để file 07 plot nhận diện được.
    patch_metric_files(out_dir, new_method_name)

    print("\nDONE.")
    print("Saved to:", out_dir)


if __name__ == "__main__":
    main()