# Auto-converted from: 07_branchA_plot_xt_forecast_results_standalone(4).ipynb
# Folder target: ml_core/src/models/ML_BranchA/scripts
# Results are saved under: ml_core/src/models/ML_BranchA/results
# Generated for OSM-edge Branch A workflow.

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def find_project_root() -> Path:
    cwd = Path.cwd().resolve()
    for p in [cwd, *cwd.parents]:
        if (p / "ml_core").exists() and (p / "dataset").exists():
            return p
        if p.name == "UTraffic-ML":
            return p
    return cwd

PROJECT_ROOT = find_project_root()
BRANCHA_ROOT = PROJECT_ROOT / "ml_core" / "src" / "models" / "ML_BranchA"
BASE_DIR = BRANCHA_ROOT / "results" / "06_branchA_run_xt_forecast"
PLOT_DIR = BRANCHA_ROOT / "results" / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

print('BASE_DIR :', BASE_DIR)
print('PLOT_DIR :', PLOT_DIR)

metric_files = sorted(BASE_DIR.glob("*/*_xt_per_lag_metrics.csv"))
if not metric_files:
    raise FileNotFoundError(f"No metric files found under {BASE_DIR}. Run Branch A model scripts first.")

dfs = []
for p in metric_files:
    df = pd.read_csv(p)
    df["source_file"] = str(p)
    if "method" not in df.columns:
        df["method"] = p.parent.name
    dfs.append(df)

all_df = pd.concat(dfs, ignore_index=True)
all_df.to_csv(PLOT_DIR / "branchA_all_xt_per_lag_metrics.csv", index=False)

metric_cols = [c for c in ["mae", "mse", "rmse", "mae_off", "rmse_off"] if c in all_df.columns]
print("Loaded metrics:", all_df.shape)
print(all_df.head())

for metric in metric_cols:
    plt.figure(figsize=(10, 5))
    for (method, split), g in all_df.groupby(["method", "split"]):
        g = g.sort_values("lag")
        plt.plot(g["lag"], g[metric], marker="o", label=f"{method}-{split}")
    plt.title(f"Branch A XT forecast — {metric} by lag")
    plt.xlabel("Lag / horizon")
    plt.ylabel(metric)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    out = PLOT_DIR / f"branchA_{metric}_by_lag.png"
    plt.savefig(out, dpi=160)
    plt.close()
    print("Saved:", out)

summary = (
    all_df
    .groupby(["method", "split"], as_index=False)[metric_cols]
    .mean()
    .sort_values(metric_cols[0] if metric_cols else "method")
)
summary.to_csv(PLOT_DIR / "branchA_method_split_summary.csv", index=False)
print("Saved summary:", PLOT_DIR / "branchA_method_split_summary.csv")
print(summary)
