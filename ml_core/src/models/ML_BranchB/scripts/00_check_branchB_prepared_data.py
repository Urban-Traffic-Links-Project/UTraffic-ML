from pathlib import Path
import argparse
import numpy as np
import pandas as pd


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default=None)
    args = parser.parse_args()

    project_root = find_project_root()
    if args.data_dir:
        common_dir = Path(args.data_dir)
        if not common_dir.is_absolute():
            common_dir = project_root / common_dir
    else:
        common_dir = project_root / 'ml_core' / 'src' / 'data_processing' / 'outputs' / 'branchB' / 'osm_edge_gt_like_branchA'

    print('PROJECT_ROOT:', project_root)
    print('COMMON_DIR  :', common_dir)

    is_dynamic = (common_dir / 'graphs' / 'bucket_table.csv').exists() and (common_dir / 'graphs' / 'available_horizons.npy').exists()
    print('FORMAT      :', 'dynamic_granger_bucket' if is_dynamic else 'standard_series')

    required = []
    for split in ['train', 'val', 'test']:
        d = common_dir / split
        base_names = ['z.npy', 'segment_ids.npy', 'timestamps.npy', 'G_series_meta.csv']
        if is_dynamic:
            base_names += ['origin_bucket_ids.npy']
        else:
            base_names += ['G_weight_series.npy', 'G_best_lag_series.npy']
        for name in base_names:
            required.append(d / name)

    if is_dynamic:
        required += [common_dir / 'graphs' / 'bucket_table.csv', common_dir / 'graphs' / 'available_horizons.npy']

    missing = [p for p in required if not p.exists()]
    if missing:
        print('\nMissing files:')
        for p in missing:
            print(' -', p)
        raise FileNotFoundError('Branch B prepared data is incomplete.')

    print('\nAll required files exist.')

    if is_dynamic:
        bucket_table = pd.read_csv(common_dir / 'graphs' / 'bucket_table.csv')
        horizons = np.asarray(np.load(common_dir / 'graphs' / 'available_horizons.npy'), dtype=np.int16)
        print('\n[graphs]')
        print('  bucket_table rows:', len(bucket_table))
        print(bucket_table)
        print('  available_horizons:', horizons.tolist())
        for h in horizons.tolist():
            Gp = common_dir / 'graphs' / f'G_bucket_h{int(h):03d}.npy'
            if Gp.exists():
                G = np.load(Gp, mmap_mode='r')
                print(f'  G h={int(h):>2}:', G.shape, G.dtype)
    
    for split in ['train', 'val', 'test']:
        d = common_dir / split
        z = np.load(d / 'z.npy', mmap_mode='r')
        seg = np.load(d / 'segment_ids.npy', mmap_mode='r')
        meta = pd.read_csv(d / 'G_series_meta.csv')
        print(f'\n[{split}]')
        print('  z:', z.shape, z.dtype)
        print('  segment_ids:', seg.shape)
        print('  meta rows:', len(meta))
        if is_dynamic:
            b = np.load(d / 'origin_bucket_ids.npy', mmap_mode='r')
            print('  origin_bucket_ids:', b.shape, b.dtype, 'unique=', sorted(set(map(int, np.asarray(b).tolist()))))
        else:
            G = np.load(d / 'G_weight_series.npy', mmap_mode='r')
            L = np.load(d / 'G_best_lag_series.npy', mmap_mode='r')
            print('  G:', G.shape, G.dtype)
            print('  L:', L.shape, L.dtype)


if __name__ == '__main__':
    main()
