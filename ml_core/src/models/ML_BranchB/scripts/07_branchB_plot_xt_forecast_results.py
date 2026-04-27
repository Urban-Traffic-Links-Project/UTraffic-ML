# ml_core/src/models/ML_BranchB/scripts/07_branchB_plot_xt_forecast_results.py
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def find_project_root() -> Path:
    cwd = Path.cwd().resolve()
    for p in [cwd, *cwd.parents, Path("/kaggle/working/UTraffic-ML")]:
        if (p / "ml_core").exists() and (p / "dataset").exists():
            return p
        if p.name == "UTraffic-ML":
            return p
    return cwd

PROJECT_ROOT = find_project_root()
BRANCHB_ROOT = PROJECT_ROOT / "ml_core" / "src" / "models" / "ML_BranchB"
BASE_DIR = BRANCHB_ROOT / "results" / "06_branchB_run_xt_forecast"
PLOT_DIR = BRANCHB_ROOT / "results" / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

METHOD_ORDER = [
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

METHOD_LABELS = {
    "no_gt": "No-G",
    "true_gt": "True-G",
    "persistence_gt": "Persistence-G",
    "ewma_gt": "EWMA-G",
    "sparse_tvpvar_gt": "Sparse TVP-VAR-G",
    "factorized_var_gt": "Factorized VAR-G",
    "factorized_mar_gt": "Factorized MAR-G",
    "factorized_tvpvar_gt": "Factorized TVP-VAR-G",
    "dense_tvpvar_gt": "Dense TVP-VAR-G",
}

print("PROJECT_ROOT:", PROJECT_ROOT)
print("BASE_DIR    :", BASE_DIR)
print("PLOT_DIR    :", PLOT_DIR)

metric_files = sorted(BASE_DIR.glob("*/*_xt_per_lag_metrics.csv"))
if not metric_files:
    raise FileNotFoundError(
        f"No *_xt_per_lag_metrics.csv files found under {BASE_DIR}. "
        "Run Branch B model scripts first."
    )

dfs = []
for p in metric_files:
    df = pd.read_csv(p)
    if "method" not in df.columns:
        method = p.name.replace("_xt_per_lag_metrics.csv", "")
        df["method"] = method
    df["source_file"] = str(p)
    dfs.append(df)

all_df = pd.concat(dfs, ignore_index=True)
required = {"method", "split", "lag", "mae", "mse", "rmse"}
missing = required - set(all_df.columns)
if missing:
    raise ValueError(f"Missing required metric columns: {missing}")

all_df["lag"] = all_df["lag"].astype(int)
all_df["method_label"] = all_df["method"].map(METHOD_LABELS).fillna(all_df["method"])

all_df.to_csv(PLOT_DIR / "branchB_all_xt_per_lag_metrics.csv", index=False)
print("Loaded metrics:", all_df.shape)
print(all_df.head())

metric_cols = [c for c in ["mae", "mse", "rmse"] if c in all_df.columns]

summary = (
    all_df
    .groupby(["split", "method", "method_label"], as_index=False)
    .agg(
        mae_mean=("mae", "mean"),
        mse_mean=("mse", "mean"),
        rmse_mean=("rmse", "mean"),
        n_lags=("lag", "nunique"),
        total_samples=("n_samples", "sum") if "n_samples" in all_df.columns else ("lag", "count"),
    )
)

summary["rmse_improvement_vs_no_gt_pct"] = np.nan
summary["mae_improvement_vs_no_gt_pct"] = np.nan

for split in summary["split"].unique():
    mask = summary["split"] == split
    base = summary[mask & (summary["method"] == "no_gt")]
    if len(base) == 1:
        base_rmse = float(base["rmse_mean"].iloc[0])
        base_mae = float(base["mae_mean"].iloc[0])
        if base_rmse != 0:
            summary.loc[mask, "rmse_improvement_vs_no_gt_pct"] = (
                (base_rmse - summary.loc[mask, "rmse_mean"]) / base_rmse * 100.0
            )
        if base_mae != 0:
            summary.loc[mask, "mae_improvement_vs_no_gt_pct"] = (
                (base_mae - summary.loc[mask, "mae_mean"]) / base_mae * 100.0
            )

summary = summary.sort_values(["split", "rmse_mean"])
summary.to_csv(PLOT_DIR / "branchB_method_split_summary.csv", index=False)
print("Saved summary:", PLOT_DIR / "branchB_method_split_summary.csv")
print(summary)

available_methods = [m for m in METHOD_ORDER if m in set(all_df["method"])]
extra_methods = [m for m in sorted(all_df["method"].unique()) if m not in available_methods]
available_methods += extra_methods

for split in sorted(all_df["split"].unique()):
    df_split = all_df[all_df["split"] == split]
    for metric in metric_cols:
        plt.figure(figsize=(10, 5))
        for method in available_methods:
            g = df_split[df_split["method"] == method].sort_values("lag")
            if g.empty:
                continue
            plt.plot(g["lag"], g[metric], marker="o", label=METHOD_LABELS.get(method, method))
        plt.title(f"Branch B XT forecast — {metric.upper()} by lag ({split})")
        plt.xlabel("Lag / horizon")
        plt.ylabel(metric.upper())
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=8, ncol=2)
        plt.tight_layout()
        out = PLOT_DIR / f"branchB_{metric}_by_lag_{split}.png"
        plt.savefig(out, dpi=160)
        plt.close()
        print("Saved:", out)

    # Bar plot average RMSE
    s = summary[summary["split"] == split].copy()
    if not s.empty:
        s["method_label"] = s["method"].map(METHOD_LABELS).fillna(s["method"])
        plt.figure(figsize=(10, 5))
        plt.bar(s["method_label"], s["rmse_mean"])
        plt.title(f"Branch B average RMSE ({split})")
        plt.ylabel("RMSE")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        out = PLOT_DIR / f"branchB_average_rmse_{split}.png"
        plt.savefig(out, dpi=160)
        plt.close()
        print("Saved:", out)

        if "rmse_improvement_vs_no_gt_pct" in s.columns:
            plt.figure(figsize=(10, 5))
            plt.bar(s["method_label"], s["rmse_improvement_vs_no_gt_pct"])
            plt.axhline(0.0, linewidth=1)
            plt.title(f"Branch B RMSE improvement vs No-G ({split})")
            plt.ylabel("Improvement (%)")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            out = PLOT_DIR / f"branchB_rmse_improvement_vs_no_gt_{split}.png"
            plt.savefig(out, dpi=160)
            plt.close()
            print("Saved:", out)

print("All outputs saved to:", PLOT_DIR)
