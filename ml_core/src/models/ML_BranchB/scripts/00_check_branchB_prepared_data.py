# ml_core/src/models/ML_BranchB/scripts/00_check_branchB_prepared_data.py
from pathlib import Path
import numpy as np
import pandas as pd

def find_project_root() -> Path:
    cwd = Path.cwd().resolve()
    for p in [cwd, *cwd.parents, Path("/kaggle/working/UTraffic-ML")]:
        if (p / "ml_core").exists() and (p / "dataset").exists():
            return p
        if p.name == "UTraffic-ML":
            return p
    return cwd

PROJECT_ROOT = find_project_root()
COMMON_DIR = PROJECT_ROOT / "ml_core" / "src" / "data_processing" / "outputs" / "branchB" / "osm_edge_gt_like_branchA"

print("PROJECT_ROOT:", PROJECT_ROOT)
print("COMMON_DIR  :", COMMON_DIR)

required = []
for split in ["train", "val", "test"]:
    for name in [
        "G_weight_series.npy",
        "G_best_lag_series.npy",
        "z.npy",
        "segment_ids.npy",
        "timestamps.npy",
        "G_series_meta.csv",
        "raw_meta.csv",
    ]:
        required.append(COMMON_DIR / split / name)

missing = [p for p in required if not p.exists()]
if missing:
    print("\nMissing files:")
    for p in missing:
        print(" -", p)
    raise FileNotFoundError(
        "Branch B prepared data is incomplete. Run:\n"
        "python ml_core/src/data_processing/prepare_branchB_osm_edge_gt_like_branchA.py --overwrite"
    )

print("\nAll required files exist.")

for split in ["train", "val", "test"]:
    split_dir = COMMON_DIR / split
    G = np.load(split_dir / "G_weight_series.npy", mmap_mode="r")
    L = np.load(split_dir / "G_best_lag_series.npy", mmap_mode="r")
    z = np.load(split_dir / "z.npy", mmap_mode="r")
    seg = np.load(split_dir / "segment_ids.npy", mmap_mode="r")
    meta = pd.read_csv(split_dir / "G_series_meta.csv")
    print(f"\n[{split}]")
    print("  G:", G.shape, G.dtype)
    print("  L:", L.shape, L.dtype)
    print("  z:", z.shape, z.dtype)
    print("  segment_ids:", seg.shape)
    print("  meta rows:", len(meta))
