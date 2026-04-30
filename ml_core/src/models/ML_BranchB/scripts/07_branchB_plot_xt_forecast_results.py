# ml_core/src/models/ML_BranchB/scripts/07_branchB_plot_xt_forecast_results.py
"""
Plot Branch B XT forecast metrics.

Updated version:
- Can filter to Top-K runs, e.g. --topk 20.
- Can filter to no-gamma runs, e.g. --nogamma-only.
- Dynamic labels for *_topk20_* methods.
- Keeps No-Rt and True-Rt baselines visible in the same plots.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def find_project_root() -> Path:
    cwd = Path.cwd().resolve()
    for p in [cwd, *cwd.parents, Path("/kaggle/working/UTraffic-ML"), Path("/kaggle/working")]:
        if (p / "ml_core").exists() and (p / "dataset").exists():
            return p
        if p.name == "UTraffic-ML":
            return p
        if (p / "UTraffic-ML").exists():
            pp = p / "UTraffic-ML"
            if (pp / "ml_core").exists():
                return pp
    return cwd


BASE_METHOD_LABELS: Dict[str, str] = {
    "no_gt": "No-Rt",
    "true_gt": "True-Rt",
    "persistence_gt": "Persistence-Rt",
    "ewma_gt": "EWMA-Rt",
    "sparse_tvpvar_gt": "Sparse TVP-VAR-Rt",
    "factorized_var_gt": "Factorized VAR-Rt",
    "factorized_mar_gt": "Factorized MAR-Rt",
    "factorized_tvpvar_gt": "Factorized TVP-VAR-Rt",
    "dense_tvpvar_gt": "Dense TVP-VAR-Rt",
}

BASE_ORDER = [
    "no_gt",
    "true_gt",
    "persistence_gt",
    "ewma_gt",
    "sparse_tvpvar_gt",
    "factorized_var_gt",
    "factorized_mar_gt",
    "factorized_tvpvar_gt",
    "dense_tvpvar_gt",
]


def base_method_from_name(method: str) -> str:
    if method.startswith("no_gt"):
        return "no_gt"
    for base in sorted(BASE_METHOD_LABELS, key=len, reverse=True):
        if method == base or method.startswith(base + "_"):
            return base
    return method


def pretty_method_label(method: str) -> str:
    base = base_method_from_name(method)
    label = BASE_METHOD_LABELS.get(base, base)

    if method.startswith("no_gt"):
        if "nogamma" in method:
            label += " (no gamma)"
        return label

    m_topk = re.search(r"topk(\d+)", method)
    if m_topk:
        label += f" TopK-{m_topk.group(1)}"

    if "nogamma" in method:
        label += " (no gamma)"
    elif "withbias" in method:
        label += " (with bias)"

    if "full" in method:
        pass
    else:
        m_nodes = re.search(r"nodes(\d+)", method)
        if m_nodes:
            label += f" [{m_nodes.group(1)} nodes]"

    return label


def method_sort_key(method: str) -> tuple:
    base = base_method_from_name(method)
    try:
        base_pos = BASE_ORDER.index(base)
    except ValueError:
        base_pos = len(BASE_ORDER) + 1

    # Prefer no-gamma/topk runs over older raw runs when both exist.
    is_topk = 0 if "topk" in method else 1
    is_nogamma = 0 if "nogamma" in method else 1
    return (base_pos, is_topk, is_nogamma, method)


def load_metric_files(base_dir: Path) -> pd.DataFrame:
    metric_files = sorted(base_dir.glob("*/*_xt_per_lag_metrics.csv"))
    if not metric_files:
        raise FileNotFoundError(
            f"No *_xt_per_lag_metrics.csv files found under {base_dir}. "
            "Run Branch B model scripts first."
        )

    dfs = []
    seen_files = set()
    for p in metric_files:
        # Avoid reading duplicate legacy copies inside the same directory if possible.
        key = str(p.resolve())
        if key in seen_files:
            continue
        seen_files.add(key)

        df = pd.read_csv(p)
        if df.empty:
            continue
        if "method" not in df.columns:
            method = p.name.replace("_xt_per_lag_metrics.csv", "")
            df["method"] = method
        df["source_file"] = str(p)
        dfs.append(df)

    if not dfs:
        raise FileNotFoundError(f"Metric files exist but all are empty under {base_dir}")

    all_df = pd.concat(dfs, ignore_index=True)
    required = {"method", "split", "lag", "mae", "mse", "rmse"}
    missing = required - set(all_df.columns)
    if missing:
        raise ValueError(f"Missing required metric columns: {missing}")

    all_df["lag"] = all_df["lag"].astype(int)
    all_df["base_method"] = all_df["method"].apply(base_method_from_name)
    all_df["method_label"] = all_df["method"].apply(pretty_method_label)
    return all_df


def filter_metrics(
    all_df: pd.DataFrame,
    topk: Optional[int],
    nogamma_only: bool,
    include_old_baselines: bool,
) -> pd.DataFrame:
    df = all_df.copy()

    if topk is not None:
        topk_tag = f"topk{int(topk)}"
        # Keep all matching Top-K methods plus No-Rt baselines.
        mask = df["method"].str.contains(topk_tag, regex=False) | df["base_method"].eq("no_gt")
        if include_old_baselines:
            # Also keep old true_gt/no_gt if user wants comparison with legacy full Rt/no Rt.
            mask = mask | df["method"].isin(["true_gt", "no_gt"])
        df = df[mask].copy()

    if nogamma_only:
        # Keep no-gamma methods. For legacy no_gt without tag, keep only if no no_gt_nogamma exists.
        has_no_gt_nogamma = df["method"].str.startswith("no_gt_nogamma").any()
        mask = df["method"].str.contains("nogamma", regex=False)
        if not has_no_gt_nogamma:
            mask = mask | df["method"].eq("no_gt")
        df = df[mask].copy()

    if df.empty:
        raise ValueError("No metrics left after filtering. Check --topk/--nogamma-only settings.")
    return df


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=str, default=None, help="Directory containing method metric folders.")
    parser.add_argument("--plot-dir", type=str, default=None, help="Output plot directory.")
    parser.add_argument("--topk", type=int, default=None, help="Only plot methods containing topkK plus No-Rt baseline.")
    parser.add_argument("--nogamma-only", action="store_true", help="Only plot no-gamma runs.")
    parser.add_argument(
        "--include-old-baselines",
        action="store_true",
        help="When --topk is set, also keep legacy true_gt/no_gt metrics if present.",
    )
    parser.add_argument("--fig-width", type=float, default=12.0)
    parser.add_argument("--fig-height", type=float, default=5.5)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    project_root = find_project_root()
    branchb_root = project_root / "ml_core" / "src" / "models" / "ML_BranchB"

    base_dir = Path(args.base_dir) if args.base_dir else branchb_root / "results" / "06_branchB_run_xt_forecast"
    if not base_dir.is_absolute():
        base_dir = project_root / base_dir

    plot_dir = Path(args.plot_dir) if args.plot_dir else branchb_root / "results" / "plots"
    if not plot_dir.is_absolute():
        plot_dir = project_root / plot_dir
    plot_dir.mkdir(parents=True, exist_ok=True)

    print("PROJECT_ROOT:", project_root)
    print("BASE_DIR    :", base_dir)
    print("PLOT_DIR    :", plot_dir)
    print("FILTER TOPK :", args.topk)
    print("NOGAMMA ONLY:", bool(args.nogamma_only))

    all_df = load_metric_files(base_dir)
    all_df.to_csv(plot_dir / "branchB_all_xt_per_lag_metrics.csv", index=False)

    plot_df = filter_metrics(
        all_df,
        topk=args.topk,
        nogamma_only=bool(args.nogamma_only),
        include_old_baselines=bool(args.include_old_baselines),
    )
    plot_df.to_csv(plot_dir / "branchB_filtered_xt_per_lag_metrics.csv", index=False)
    print("Loaded all metrics     :", all_df.shape)
    print("Filtered plot metrics  :", plot_df.shape)

    summary = (
        plot_df
        .groupby(["split", "method", "method_label", "base_method"], as_index=False)
        .agg(
            mae_mean=("mae", "mean"),
            mse_mean=("mse", "mean"),
            rmse_mean=("rmse", "mean"),
            n_lags=("lag", "nunique"),
        )
    )

    summary["rmse_improvement_vs_no_rt_pct"] = np.nan
    summary["mae_improvement_vs_no_rt_pct"] = np.nan

    for split in summary["split"].unique():
        mask = summary["split"].eq(split)
        no_rt = summary[mask & summary["base_method"].eq("no_gt")].sort_values("method")
        if len(no_rt) >= 1:
            base_rmse = float(no_rt["rmse_mean"].iloc[0])
            base_mae = float(no_rt["mae_mean"].iloc[0])
            if base_rmse != 0:
                summary.loc[mask, "rmse_improvement_vs_no_rt_pct"] = (
                    (base_rmse - summary.loc[mask, "rmse_mean"]) / base_rmse * 100.0
                )
            if base_mae != 0:
                summary.loc[mask, "mae_improvement_vs_no_rt_pct"] = (
                    (base_mae - summary.loc[mask, "mae_mean"]) / base_mae * 100.0
                )

    summary = summary.sort_values(["split", "rmse_mean"]).reset_index(drop=True)
    summary_path = plot_dir / "branchB_method_split_summary.csv"
    summary.to_csv(summary_path, index=False)
    print("Saved summary:", summary_path)
    print(summary)

    metric_cols = [c for c in ["mae", "mse", "rmse"] if c in plot_df.columns]
    available_methods = sorted(plot_df["method"].unique(), key=method_sort_key)

    suffix_parts = []
    if args.topk is not None:
        suffix_parts.append(f"topk{int(args.topk)}")
    if args.nogamma_only:
        suffix_parts.append("nogamma")
    suffix = "_" + "_".join(suffix_parts) if suffix_parts else ""

    for split in sorted(plot_df["split"].unique()):
        df_split = plot_df[plot_df["split"].eq(split)]
        for metric in metric_cols:
            plt.figure(figsize=(args.fig_width, args.fig_height))
            for method in available_methods:
                g = df_split[df_split["method"].eq(method)].sort_values("lag")
                if g.empty:
                    continue
                label = g["method_label"].iloc[0]
                plt.plot(g["lag"], g[metric], marker="o", label=label)
            plt.title(f"Branch B XT forecast — {metric.upper()} by lag ({split})")
            plt.xlabel("Lag / horizon")
            plt.ylabel(metric.upper())
            plt.grid(True, alpha=0.3)
            plt.legend(fontsize=8, ncol=2)
            plt.tight_layout()
            out = plot_dir / f"branchB_{metric}_by_lag_{split}{suffix}.png"
            plt.savefig(out, dpi=160)
            plt.close()
            print("Saved:", out)

    print("All outputs saved to:", plot_dir)


if __name__ == "__main__":
    main()
