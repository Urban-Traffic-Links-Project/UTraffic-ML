"""
Run Branch B Granger-Gt pipeline.

This orchestrates:
1) Prepare Granger-Gt standard series from Branch-A OSM edge forecasting NPZ.
2) Run selected graph methods on Gt:
   true_gt, persistence_gt, ewma_gt, sparse_tvpvar_gt, sparse_var_gt
3) Run DMFM paper-style XT forecast separately.
4) Plot Gt method metrics.

Example:
    python -u ml_core/src/models/ML_BranchB/scripts/run_all_branchB_gt_pipeline.py \
      --max-nodes 512 --topk 20 --lags 1-9
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List


def find_project_root() -> Path:
    cwd = Path.cwd().resolve()
    for p in [cwd, *cwd.parents, Path('/kaggle/working/UTraffic-ML'), Path('/kaggle/working')]:
        if (p / 'ml_core').exists() and (p / 'dataset').exists():
            return p
        if p.name == 'UTraffic-ML':
            return p
        if (p / 'UTraffic-ML').exists():
            pp = p / 'UTraffic-ML'
            if (pp / 'ml_core').exists():
                return pp
    return cwd


def run(cmd: List[str]) -> None:
    print('\n' + '=' * 100, flush=True)
    print('RUN:', ' '.join(cmd), flush=True)
    print('=' * 100, flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-nodes', type=int, default=512, help='0 means full nodes. Recommended: 512 first.')
    parser.add_argument('--topk', type=int, default=20)
    parser.add_argument('--lags', type=str, default='1-9')
    parser.add_argument('--granger-p', type=int, default=3)
    parser.add_argument('--granger-horizon', type=int, default=1)
    parser.add_argument('--bucket-minutes', type=int, default=60)
    parser.add_argument('--max-candidates', type=int, default=50)
    parser.add_argument('--n-jobs', type=int, default=2)
    parser.add_argument('--skip-dmfm', action='store_true')
    parser.add_argument('--dmfm-rank', type=str, default='8,8')
    parser.add_argument('--matrix-shape', type=str, default=None, help='For DMFM. If omitted, DMFM script auto/needs its default behavior.')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

    root = find_project_root()
    os.chdir(root)
    py = sys.executable

    data_dir = 'ml_core/src/data_processing/outputs/branchB/osm_edge_granger_series_like_branchA'
    results_dir = 'ml_core/src/models/ML_BranchB/results/06_branchB_gt_pipeline'
    plot_dir = 'ml_core/src/models/ML_BranchB/results/plots_branchB_gt_pipeline'

    prepare_cmd = [
        py, '-u', 'ml_core/src/data_processing/prepare_branchB_osm_edge_granger_series_like_branchA.py',
        '--granger-horizon', str(args.granger_horizon),
        '--granger-p', str(args.granger_p),
        '--bucket-minutes', str(args.bucket_minutes),
        '--max-candidates', str(args.max_candidates),
    ]
    if args.max_nodes > 0:
        prepare_cmd += ['--max-nodes', str(args.max_nodes)]
    if args.overwrite:
        prepare_cmd += ['--overwrite']
    run(prepare_cmd)

    check_cmd = [
        py, '-u', 'ml_core/src/models/ML_BranchB/scripts/00_check_branchB_prepared_data.py',
        '--data-dir', data_dir,
    ]
    run(check_cmd)

    run_cmd = [
        py, '-u', 'ml_core/src/models/ML_BranchB/scripts/06B_branchB_run_xt_forecast_topk_gt.py',
        '--data-dir', data_dir,
        '--results-dir', results_dir,
        '--methods', 'true_gt,persistence_gt,ewma_gt,sparse_tvpvar_gt,sparse_var_gt',
        '--topk', str(args.topk),
        '--lags', str(args.lags),
        '--parallel-level', 'method',
        '--n-jobs', str(args.n_jobs),
    ]
    run(run_cmd)

    if not args.skip_dmfm:
        dmfm_cmd = [
            py, '-u', 'ml_core/src/models/ML_BranchB/scripts/06D_dmfm_paper_xt_forecast.py',
            '--rank', str(args.dmfm_rank),
            '--horizons', str(args.lags),
            '--methods', 'all',
            '--output-dir', 'ml_core/src/models/ML_BranchB/results/06D_dmfm_paper_xt_forecast_gt_pipeline',
        ]
        if args.max_nodes > 0:
            dmfm_cmd += ['--max-nodes', str(args.max_nodes)]
        if args.matrix_shape:
            dmfm_cmd += ['--matrix-shape', str(args.matrix_shape)]
        if args.overwrite:
            dmfm_cmd += ['--overwrite']
        run(dmfm_cmd)

    plot_cmd = [
        py, '-u', 'ml_core/src/models/ML_BranchB/scripts/07_branchB_plot_xt_forecast_results.py',
        '--base-dir', results_dir,
        '--plot-dir', plot_dir,
        '--topk', str(args.topk),
        '--nogamma-only',
    ]
    run(plot_cmd)

    print('\nDONE.', flush=True)
    print('Gt method results:', root / results_dir, flush=True)
    print('Gt plots:', root / plot_dir, flush=True)


if __name__ == '__main__':
    main()
