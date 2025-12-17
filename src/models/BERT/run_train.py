# run_train.py
from torch.utils.data import DataLoader

from .dataset_zones import TrafficZoneDataset, DatasetParams
from .collate import pad_collate_zones
from .model_sttransformer import STEncoderOnly, ModelParams
from .train import train_model, TrainParams
from pathlib import Path

from .matrix_correla import DBSCANParams, ZoneParams, CorrParams
import torch
import os

# ===== CPU tuning (AMD / non-CUDA) =====
torch.set_num_threads(8)        # số core CPU bạn muốn dùng
torch.set_num_interop_threads(2)

os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"

print("Torch threads:", torch.get_num_threads())

def main():
    THIS_FILE = os.getcwd()
    CWD = THIS_FILE

    DATA_DIR = os.path.join(CWD, "data", "processed", "tomtom_stats")

    traffic_npz = os.path.join(DATA_DIR, "traffic_tensor.npz")
    segments_csv = os.path.join(DATA_DIR, "segments.csv")
    nodes_csv = os.path.join(DATA_DIR, "nodes.csv")
    edges_csv = os.path.join(DATA_DIR, "edges.csv")
    segment_index_csv = os.path.join(DATA_DIR, "segment_index.csv")

    print("Project root:", CWD)
    print("Data dir:", DATA_DIR)
    print("Traffic:", traffic_npz)

    # Hyperparams
    ds = DatasetParams(L=11, delta=1, hist_len_corr=48)
    zone = ZoneParams(
        seed_congested_ratio=0.6,
        hops=2,
        R_max_m=1500.0,
        D_min_m=0.0,
        D_max_m=3000.0,
        top_k=64,
        corr=CorrParams(tau_max=3, tau_cut=3, W_min=0.2, eps=1e-6),  # tau_max in steps (15min)
        d_spa=16,
        laplacian_mode="edges",
        sigma_m=500.0,
        enable_cache=True,
        cache_dir="./cache_zone",
    )
    db = DBSCANParams(eps_m=250.0, min_samples=3)

    train_set = TrafficZoneDataset(traffic_npz, segments_csv, nodes_csv, edges_csv, segment_index_csv,
                                   split="train", ds=ds, dbscan=db, zone=zone, seed=13)
    val_set   = TrafficZoneDataset(traffic_npz, segments_csv, nodes_csv, edges_csv, segment_index_csv,
                                   split="val", ds=ds, dbscan=db, zone=zone, seed=13)

    train_loader = DataLoader(train_set, batch_size=8, shuffle=True, num_workers=0, collate_fn=pad_collate_zones)
    val_loader   = DataLoader(val_set, batch_size=8, shuffle=False, num_workers=0, collate_fn=pad_collate_zones)

    mp = ModelParams(d_in=1, d_model=128, nhead=4, num_layers=4, dropout=0.1, d_spa=zone.d_spa, L_max=64)
    model = STEncoderOnly(mp)

    tp = TrainParams(lr=1e-4, weight_decay=1e-4, epochs=10)

    model = train_model(model, train_loader, val_loader, tp)

if __name__ == "__main__":
    main()
