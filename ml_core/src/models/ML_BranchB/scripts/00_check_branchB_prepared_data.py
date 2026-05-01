from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def find_project_root() -> Path:
    cwd = Path.cwd().resolve()
    for p in [cwd, *cwd.parents, Path('/kaggle/working/UTraffic-ML')]:
        if (p / 'ml_core').exists() and (p / 'dataset').exists():
            return p
        if p.name == 'UTraffic-ML':
            return p
    return cwd


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data-dir',
        type=str,
        default=None,
        help='Prepared Branch-B data dir. Default: outputs/branchB/osm_edge_gt_like_branchA',
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    project_root = find_project_root()
    if args.data_dir is None:
        common_dir = project_root / 'ml_core' / 'src' / 'data_processing' / 'outputs' / 'branchB' / 'osm_edge_gt_like_branchA'
    else:
        common_dir = Path(args.data_dir)
        if not common_dir.is_absolute():
            common_dir = project_root / common_dir

    print('PROJECT_ROOT:', project_root)
    print('COMMON_DIR  :', common_dir)

    required = []
    for split in ['train', 'val', 'test']:
        for name in [
            'G_weight_series.npy',
            'G_best_lag_series.npy',
            'z.npy',
            'segment_ids.npy',
            'timestamps.npy',
            'G_series_meta.csv',
            'raw_meta.csv',
        ]:
            required.append(common_dir / split / name)

    missing = [p for p in required if not p.exists()]
    if missing:
        print('\nMissing files:')
        for p in missing:
            print(' -', p)
        raise FileNotFoundError(
            'Branch B prepared data is incomplete. Check --data-dir or run the appropriate prepare script.'
        )

    print('\nAll required files exist.')

    for split in ['train', 'val', 'test']:
        split_dir = common_dir / split
        G = np.load(split_dir / 'G_weight_series.npy', mmap_mode='r')
        L = np.load(split_dir / 'G_best_lag_series.npy', mmap_mode='r')
        z = np.load(split_dir / 'z.npy', mmap_mode='r')
        seg = np.load(split_dir / 'segment_ids.npy', mmap_mode='r')
        meta = pd.read_csv(split_dir / 'G_series_meta.csv')
        print(f'\n[{split}]')
        print('  G:', G.shape, G.dtype)
        print('  L:', L.shape, L.dtype)
        print('  z:', z.shape, z.dtype)
        print('  segment_ids:', seg.shape)
        print('  meta rows:', len(meta))
        if G.ndim == 3 and G.shape[0] <= 32:
            print('  note: G first axis looks like horizon-indexed static graph, not time-indexed dynamic series.')


if __name__ == '__main__':
    main()
