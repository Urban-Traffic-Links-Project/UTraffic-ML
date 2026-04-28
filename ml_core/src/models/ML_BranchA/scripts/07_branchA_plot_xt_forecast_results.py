# ml_core/src/models/ML_BranchA/scripts/07_branchA_plot_xt_forecast_results.py
"""
Plot Branch A XT forecast results.

Mục tiêu:
- Đọc tất cả file *_xt_per_lag_metrics.csv trong:
    ml_core/src/models/ML_BranchA/results/06_branchA_run_xt_forecast/
- Xuất bảng tổng hợp.
- Plot riêng MAE và RMSE cho từng tập:
    val
    test

Chạy trên Kaggle:
    %cd /kaggle/working/UTraffic-ML
    !python -u ml_core/src/models/ML_BranchA/scripts/07_branchA_plot_xt_forecast_results.py 2>&1 | tee logs_A_plot.txt

Chạy local:
    cd C:/AI/Thesis/UTraffic-ML
    python ml_core/src/models/ML_BranchA/scripts/07_branchA_plot_xt_forecast_results.py
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def find_project_root() -> Path:
    cwd = Path.cwd().resolve()
    for p in [cwd, *cwd.parents, Path("/kaggle/working/UTraffic-ML")]:
        if (p / "ml_core").exists():
            return p
        if p.name == "UTraffic-ML":
            return p
    return cwd


PROJECT_ROOT = find_project_root()
BRANCHA_ROOT = PROJECT_ROOT / "ml_core" / "src" / "models" / "ML_BranchA"
BASE_DIR = BRANCHA_ROOT / "results" / "06_branchA_run_xt_forecast"
PLOT_DIR = BRANCHA_ROOT / "results" / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

print("PROJECT_ROOT:", PROJECT_ROOT)
print("BASE_DIR    :", BASE_DIR)
print("PLOT_DIR    :", PLOT_DIR)


METHOD_ORDER = [
    "no_rt",
    "true_rt",
    "persistence",
    "ewma",
    "dcc",
    "prodcc",
    "dmfm",
    "factorized_uut",
]

METHOD_LABELS = {
    "no_rt": "No R",
    "true_rt": "True R",
    "persistence": "Persistence",
    "ewma": "EWMA",
    "dcc": "DCC",
    "prodcc": "Pro-DCC",
    "dmfm": "DMFM",
    "factorized_uut": "Factorized UUT",
}


metric_files = sorted(BASE_DIR.glob("*/*_xt_per_lag_metrics.csv"))

if not metric_files:
    raise FileNotFoundError(
        f"No *_xt_per_lag_metrics.csv files found under {BASE_DIR}. "
        "Run Branch A model scripts first."
    )

dfs = []
for p in metric_files:
    df = pd.read_csv(p)
    df["source_file"] = str(p)

    if "method" not in df.columns:
        method = p.name.replace("_xt_per_lag_metrics.csv", "")
        df["method"] = method

    dfs.append(df)

all_df = pd.concat(dfs, ignore_index=True)

required_cols = {"method", "split", "lag", "mae", "rmse"}
missing = required_cols - set(all_df.columns)
if missing:
    raise ValueError(f"Missing required columns in metric CSV files: {missing}")

all_df["lag"] = all_df["lag"].astype(int)
all_df["method"] = all_df["method"].astype(str)
all_df["split"] = all_df["split"].astype(str)
all_df["method_label"] = all_df["method"].map(METHOD_LABELS).fillna(all_df["method"])

all_metrics_path = PLOT_DIR / "branchA_all_xt_per_lag_metrics.csv"
all_df.to_csv(all_metrics_path, index=False)

print("\nLoaded metrics:", all_df.shape)
print("Saved:", all_metrics_path)
print(all_df.head())


summary = (
    all_df
    .groupby(["split", "method", "method_label"], as_index=False)
    .agg(
        mae_mean=("mae", "mean"),
        rmse_mean=("rmse", "mean"),
        n_lags=("lag", "nunique"),
        n_rows=("lag", "count"),
    )
)

if "mse" in all_df.columns:
    mse_summary = (
        all_df
        .groupby(["split", "method"], as_index=False)
        .agg(mse_mean=("mse", "mean"))
    )
    summary = summary.merge(mse_summary, on=["split", "method"], how="left")

summary["mae_improvement_vs_no_rt_pct"] = np.nan
summary["rmse_improvement_vs_no_rt_pct"] = np.nan

for split in summary["split"].unique():
    mask = summary["split"] == split
    base = summary[mask & (summary["method"] == "no_rt")]
    if len(base) == 1:
        base_mae = float(base["mae_mean"].iloc[0])
        base_rmse = float(base["rmse_mean"].iloc[0])

        if base_mae != 0:
            summary.loc[mask, "mae_improvement_vs_no_rt_pct"] = (
                (base_mae - summary.loc[mask, "mae_mean"]) / base_mae * 100.0
            )

        if base_rmse != 0:
            summary.loc[mask, "rmse_improvement_vs_no_rt_pct"] = (
                (base_rmse - summary.loc[mask, "rmse_mean"]) / base_rmse * 100.0
            )

summary = summary.sort_values(["split", "rmse_mean"]).reset_index(drop=True)

summary_path = PLOT_DIR / "branchA_method_split_summary.csv"
summary.to_csv(summary_path, index=False)

print("\nSaved summary:", summary_path)
print(summary)


for metric in ["mae", "rmse"]:
    pivot = all_df.pivot_table(
        index=["method"],
        columns=["split", "lag"],
        values=metric,
        aggfunc="mean",
    )
    order = [m for m in METHOD_ORDER if m in pivot.index] + [m for m in pivot.index if m not in METHOD_ORDER]
    pivot = pivot.reindex(order)
    pivot_path = PLOT_DIR / f"branchA_{metric}_pivot_method_split_lag.csv"
    pivot.to_csv(pivot_path)
    print("Saved:", pivot_path)


def ordered_methods(df: pd.DataFrame):
    existing = list(df["method"].unique())
    ordered = [m for m in METHOD_ORDER if m in existing]
    ordered += [m for m in sorted(existing) if m not in ordered]
    return ordered


def plot_metric_by_lag_for_split(df: pd.DataFrame, split: str, metric: str):
    df_split = df[df["split"] == split].copy()
    if df_split.empty:
        print(f"Skip plot {metric} {split}: no data")
        return

    plt.figure(figsize=(11, 6))

    for method in ordered_methods(df_split):
        g = df_split[df_split["method"] == method].sort_values("lag")
        if g.empty:
            continue

        label = METHOD_LABELS.get(method, method)
        plt.plot(
            g["lag"],
            g[metric],
            marker="o",
            linewidth=2,
            label=label,
        )

    plt.title(f"Branch A - {metric.upper()} by horizon on {split.upper()} set")
    plt.xlabel("Horizon / lag")
    plt.ylabel(metric.upper())
    plt.xticks(sorted(df_split["lag"].unique()))
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()

    out = PLOT_DIR / f"branchA_{metric}_by_lag_{split}.png"
    plt.savefig(out, dpi=180)
    plt.close()

    print("Saved:", out)


def plot_metric_bar_for_split(summary_df: pd.DataFrame, split: str, metric_mean_col: str):
    df_split = summary_df[summary_df["split"] == split].copy()
    if df_split.empty:
        print(f"Skip bar plot {metric_mean_col} {split}: no data")
        return

    df_split = df_split.sort_values(metric_mean_col, ascending=True)
    labels = df_split["method_label"].tolist()
    values = df_split[metric_mean_col].to_numpy(dtype=float)

    metric_name = metric_mean_col.replace("_mean", "").upper()

    plt.figure(figsize=(11, 6))
    plt.bar(labels, values)
    plt.title(f"Branch A - Average {metric_name} on {split.upper()} set")
    plt.ylabel(metric_name)
    plt.xticks(rotation=35, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    out = PLOT_DIR / f"branchA_{metric_mean_col}_bar_{split}.png"
    plt.savefig(out, dpi=180)
    plt.close()

    print("Saved:", out)


def plot_improvement_bar_for_split(summary_df: pd.DataFrame, split: str, col: str):
    df_split = summary_df[summary_df["split"] == split].copy()
    if df_split.empty or col not in df_split.columns:
        return

    df_split = df_split.sort_values(col, ascending=False)
    labels = df_split["method_label"].tolist()
    values = df_split[col].to_numpy(dtype=float)

    plt.figure(figsize=(11, 6))
    plt.bar(labels, values)
    plt.axhline(0.0, linewidth=1)
    plt.title(f"Branch A - {col.replace('_', ' ')} on {split.upper()} set")
    plt.ylabel("Improvement (%)")
    plt.xticks(rotation=35, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    out = PLOT_DIR / f"branchA_{col}_{split}.png"
    plt.savefig(out, dpi=180)
    plt.close()

    print("Saved:", out)


target_splits = [s for s in ["val", "test"] if s in set(all_df["split"])]
if not target_splits:
    target_splits = sorted(all_df["split"].unique())

for split in target_splits:
    plot_metric_by_lag_for_split(all_df, split, "mae")
    plot_metric_by_lag_for_split(all_df, split, "rmse")

    plot_metric_bar_for_split(summary, split, "mae_mean")
    plot_metric_bar_for_split(summary, split, "rmse_mean")

    plot_improvement_bar_for_split(summary, split, "mae_improvement_vs_no_rt_pct")
    plot_improvement_bar_for_split(summary, split, "rmse_improvement_vs_no_rt_pct")


for metric in ["mae", "rmse"]:
    plt.figure(figsize=(11, 6))

    for split in target_splits:
        df_split = all_df[all_df["split"] == split]
        avg = df_split.groupby("lag", as_index=False)[metric].mean().sort_values("lag")
        plt.plot(
            avg["lag"],
            avg[metric],
            marker="o",
            linewidth=2,
            label=f"{split.upper()} mean across methods",
        )

    plt.title(f"Branch A - VAL vs TEST mean {metric.upper()} comparison")
    plt.xlabel("Horizon / lag")
    plt.ylabel(metric.upper())
    plt.xticks(sorted(all_df["lag"].unique()))
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out = PLOT_DIR / f"branchA_{metric}_val_test_mean_comparison.png"
    plt.savefig(out, dpi=180)
    plt.close()

    print("Saved:", out)


print("\nDONE. All Branch A plots saved to:", PLOT_DIR)
