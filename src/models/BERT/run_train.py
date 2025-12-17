# run_train.py
from torch.utils.data import DataLoader
from .dataset_zones import TrafficZoneDataset, DatasetParams
from .collate import pad_collate_zones
from .model_sttransformer import STEncoderOnly, ModelParams
from .train import train_model, TrainParams
from .matrix_correla import DBSCANParams, ZoneParams, CorrParams

import torch
import os

import time
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

def main():
    # -------------------------
    # 0) Speed flags (GPU)
    # -------------------------
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # PyTorch >= 2.0
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    # -------------------------
    # 1) Paths
    # -------------------------
    CWD = os.getcwd()
    DATA_DIR = os.path.join(CWD, "data", "processed", "tomtom_stats")

    traffic_npz = os.path.join(DATA_DIR, "traffic_tensor.npz")
    segments_csv = os.path.join(DATA_DIR, "segments.csv")
    nodes_csv = os.path.join(DATA_DIR, "nodes.csv")
    edges_csv = os.path.join(DATA_DIR, "edges.csv")
    segment_index_csv = os.path.join(DATA_DIR, "segment_index.csv")

    print("Project root:", CWD)
    print("Data dir:", DATA_DIR)
    print("Traffic:", traffic_npz)

    # -------------------------
    # 2) Hyperparams
    # -------------------------
    # FAST PRESET (để debug tốc độ trước)
    # - hist_len_corr nhỏ hơn => xcorr nhanh hơn
    # - hops=1, R_max nhỏ hơn, top_k nhỏ hơn => candidates ít => giảm O(M^2)
    ds = DatasetParams(L=11, delta=1, hist_len_corr=24)  # 48 -> 24 (nhanh hơn)

    zone = ZoneParams(
        seed_congested_ratio=0.6,
        hops=1,              # 2 -> 1
        R_max_m=1000.0,      # 1500 -> 1000
        D_min_m=0.0,
        D_max_m=3000.0,
        top_k=32,            # 64 -> 32
        corr=CorrParams(tau_max=3, tau_cut=3, W_min=0.2, eps=1e-6),
        d_spa=16,
        laplacian_mode="edges",
        sigma_m=500.0,
        enable_cache=True,
        cache_dir="./cache_zone",
    )
    db = DBSCANParams(eps_m=250.0, min_samples=3)
    log.info("Init datasets...")
    train_set = TrafficZoneDataset(
        traffic_npz, segments_csv, nodes_csv, edges_csv, segment_index_csv,
        split="train", ds=ds, dbscan=db, zone=zone, seed=13
    )
    val_set = TrafficZoneDataset(
        traffic_npz, segments_csv, nodes_csv, edges_csv, segment_index_csv,
        split="val", ds=ds, dbscan=db, zone=zone, seed=13
    )

    # -------------------------
    # 3) DataLoader (tối ưu)
    # -------------------------
    # Kaggle thường 2–8 cores: thử 4 trước
    num_workers = 8
    log.info("Init dataloaders...")

    train_loader = DataLoader(
        train_set,
        batch_size=8,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=pad_collate_zones,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
        drop_last=True,  # ổn định batch / tăng tốc nhẹ
    )

    val_loader = DataLoader(
        val_set,
        batch_size=8,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=pad_collate_zones,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )

    # -------------------------
    # 4) Model
    # -------------------------
    log.info("Init model...")

    mp = ModelParams(d_in=1, d_model=128, nhead=4, num_layers=4, dropout=0.1, d_spa=zone.d_spa, L_max=64)
    model = STEncoderOnly(mp)

    # -------------------------
    # 5) Train
    # -------------------------
    tp = TrainParams(lr=1e-4, weight_decay=1e-4, epochs=10)
    model = train_model(model, train_loader, val_loader, tp)


if __name__ == "__main__":
    main()
