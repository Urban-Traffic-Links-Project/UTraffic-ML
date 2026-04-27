# ml_core/src/models/ML_BranchB/scripts/run_all_branchB.py
"""
Run Branch B methods sequentially.

Run from project root:
    cd C:/AI/Thesis/UTraffic-ML
    python ml_core/src/models/ML_BranchB/scripts/run_all_branchB.py

This script first checks that prepare_branchB_osm_edge_gt_like_branchA.py has been run.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

THIS_FILE = Path(__file__).resolve()
SCRIPTS_DIR = THIS_FILE.parent

METHOD_SCRIPTS = [
    "06_branchB_run_xt_forecast_no_gt.py",
    "06_branchB_run_xt_forecast_true_gt.py",
    "06_branchB_run_xt_forecast_persistence_gt.py",
    "06_branchB_run_xt_forecast_ewma_gt.py",
    "06_branchB_run_xt_forecast_sparse_tvpvar_gt.py",
    "06_branchB_run_xt_forecast_factorized_var_gt.py",
    "06_branchB_run_xt_forecast_factorized_mar_gt.py",
    "06_branchB_run_xt_forecast_factorized_tvpvar_gt.py",
    # Dense can be slow/heavy; run separately if needed.
    # "06_branchB_run_xt_forecast_dense_tvpvar_gt.py",
]

def run_script(name: str):
    path = SCRIPTS_DIR / name
    print("\n" + "=" * 90)
    print("RUN:", path)
    print("=" * 90)
    subprocess.run([sys.executable, str(path)], check=True)

def main():
    run_script("00_check_branchB_prepared_data.py")

    for name in METHOD_SCRIPTS:
        run_script(name)

    run_script("07_branchB_plot_xt_forecast_results.py")

if __name__ == "__main__":
    main()
