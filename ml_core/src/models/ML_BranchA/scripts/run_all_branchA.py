# ml_core/src/models/ML_BranchA/scripts/run_all_branchA.py
"""
Run Branch A methods sequentially.

Run from project root:
    cd C:/AI/Thesis/UTraffic-ML
    python ml_core/src/models/ML_BranchA/scripts/run_all_branchA.py

This script always prepares the Branch A common data first, so it will not fail
because of a missing train/R_series.npy folder.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

THIS_FILE = Path(__file__).resolve()
SCRIPTS_DIR = THIS_FILE.parent

PREPARE_SCRIPT = "00_prepare_branchA_common_from_osm.py"

METHOD_SCRIPTS = [
    "06_branchA_run_xt_forecast_no_rt.py",
    "06_branchA_run_xt_forecast_true_rt.py",
    "06_branchA_run_xt_forecast_persistence.py",
    "06_branchA_run_xt_forecast_ewma.py",
    "06_branchA_run_xt_forecast_dcc.py",
    "06_branchA_run_xt_forecast_prodcc.py",
    "06_branchA_run_xt_forecast_dmfm.py",
    # "06_branchA_run_xt_forecast_factorized_uut.py",
]

def run_script(name: str, *args: str):
    path = SCRIPTS_DIR / name
    print("\n" + "=" * 90)
    print("RUN:", path)
    print("=" * 90)
    subprocess.run([sys.executable, str(path), *args], check=True)

def main():
    run_script(PREPARE_SCRIPT, "--overwrite")

    for name in METHOD_SCRIPTS:
        run_script(name)

    run_script("07_branchA_plot_xt_forecast_results.py")

if __name__ == "__main__":
    main()
