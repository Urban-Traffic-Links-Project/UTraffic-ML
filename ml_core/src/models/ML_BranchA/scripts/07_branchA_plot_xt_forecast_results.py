# ml_core/src/models/ML_BranchA/scripts/07_branchA_plot_xt_forecast_results.py
"""
Plot đúng theo yêu cầu:

Tạo đúng 4 ảnh:
1. RMSE trên tập val  -> tất cả phương pháp nằm chung 1 ảnh
2. MAE  trên tập val  -> tất cả phương pháp nằm chung 1 ảnh
3. RMSE trên tập test -> tất cả phương pháp nằm chung 1 ảnh
4. MAE  trên tập test -> tất cả phương pháp nằm chung 1 ảnh

Mỗi ảnh:
- trục x: lag / horizon
- trục y: metric (MAE hoặc RMSE)
- mỗi đường: một phương pháp

Ngoài ra script vẫn lưu:
- branchA_all_xt_per_lag_metrics.csv
- branchA_method_split_summary.csv
"""

from pathlib import Path
import pandas as pd
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
]

METHOD_LABELS = {
    "no_rt": "No R",
    "true_rt": "True R",
    "persistence": "Persistence",
    "ewma": "EWMA",
    "dcc": "DCC",
    "prodcc": "Pro-DCC",
    "dmfm": "DMFM",
}


def get_method_order(existing_methods):
    ordered = [m for m in METHOD_ORDER if m in existing_methods]
    ordered += [m for m in sorted(existing_methods) if m not in ordered]
    return ordered


metric_files = sorted(BASE_DIR.glob("*/*_xt_per_lag_metrics.csv"))
if not metric_files:
    raise FileNotFoundError(
        f"No *_xt_per_lag_metrics.csv found under {BASE_DIR}. "
        "Please run Branch A forecasting scripts first."
    )

dfs = []
for p in metric_files:
    df = pd.read_csv(p)
    if "method" not in df.columns:
        df["method"] = p.name.replace("_xt_per_lag_metrics.csv", "")
    df["source_file"] = str(p)
    dfs.append(df)

all_df = pd.concat(dfs, ignore_index=True)

# Removed from thesis/reporting pipeline: do not plot Factorized UUT even if old result files exist.
EXCLUDED_METHODS = {"factorized_uut"}
all_df = all_df[~all_df["method"].astype(str).isin(EXCLUDED_METHODS)].copy()
if all_df.empty:
    raise RuntimeError("No metrics left after excluding methods: factorized_uut")

required_cols = {"method", "split", "lag", "mae", "rmse"}
missing = required_cols - set(all_df.columns)
if missing:
    raise ValueError(f"Missing required columns: {missing}")

all_df["lag"] = all_df["lag"].astype(int)
all_df["method"] = all_df["method"].astype(str)
all_df["split"] = all_df["split"].astype(str)

all_metrics_path = PLOT_DIR / "branchA_all_xt_per_lag_metrics.csv"
all_df.to_csv(all_metrics_path, index=False)
print("Saved:", all_metrics_path)

summary = (
    all_df.groupby(["split", "method"], as_index=False)
    .agg(
        mae_mean=("mae", "mean"),
        rmse_mean=("rmse", "mean"),
        n_lags=("lag", "nunique"),
    )
    .sort_values(["split", "rmse_mean", "mae_mean"])
)
summary_path = PLOT_DIR / "branchA_method_split_summary.csv"
summary.to_csv(summary_path, index=False)
print("Saved:", summary_path)
print(summary)


def plot_one(metric: str, split: str, out_name: str):
    df = all_df[all_df["split"] == split].copy()
    if df.empty:
        print(f"Skip {metric}-{split}: no data")
        return

    plt.figure(figsize=(11, 6))

    for method in get_method_order(df["method"].unique()):
        g = df[df["method"] == method].sort_values("lag")
        if g.empty:
            continue
        label = METHOD_LABELS.get(method, method)
        plt.plot(g["lag"], g[metric], marker="o", linewidth=2, label=label)

    plt.title(f"Branch A - {metric.upper()} on {split.upper()} set")
    plt.xlabel("Lag / horizon")
    plt.ylabel(metric.upper())
    plt.xticks(sorted(df["lag"].unique()))
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()

    out = PLOT_DIR / out_name
    plt.savefig(out, dpi=180)
    plt.close()
    print("Saved:", out)


# Đúng 4 ảnh theo yêu cầu
plot_one("rmse", "val",  "branchA_rmse_val.png")
plot_one("mae",  "val",  "branchA_mae_val.png")
plot_one("rmse", "test", "branchA_rmse_test.png")
plot_one("mae",  "test", "branchA_mae_test.png")

print("\nDONE.")
print("Expected output files:")
print(" -", PLOT_DIR / "branchA_rmse_val.png")
print(" -", PLOT_DIR / "branchA_mae_val.png")
print(" -", PLOT_DIR / "branchA_rmse_test.png")
print(" -", PLOT_DIR / "branchA_mae_test.png")
