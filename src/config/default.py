# src/config/defaults.py
from dataclasses import dataclass
import os

@dataclass
class TrainConfig:
    # ====== Data ======
    dataset_root: str = os.path.join("data", "processed", "tomtom_stats")  # chứa train/val/test + edges.csv...
    train_split: str = "train"
    val_split: str = "val"
    test_split: str = "test"

    # ====== Windowing ======
    # Bạn lấy dữ liệu mỗi 15 phút => P=12 là 3h quá khứ; H=6 là 1.5h tương lai (tuỳ bạn chỉnh)
    P: int = 12
    H: int = 6

    # ====== Train ======
    batch: int = 16
    epochs: int = 50
    lr: float = 1e-3
    seed: int = 42
    weight_decay: float = 0.0
    grad_clip: float = 5.0
    dropout: float = 0.1
    device: str = "cuda"  # "cpu" nếu máy không có GPU

    # ====== Model ======
    in_dim: int = 1              # speed z-score
    gat_hidden: int = 32
    gat_heads: int = 4
    gat_layers: int = 2          # multi-hop
    gru_hidden: int = 64

    # ====== Route sampling (Random Walk) ======
    use_routes_in_train: bool = True
    routes_per_batch: int = 8
    route_len_min: int = 6
    route_len_max: int = 16
    route_restart_prob: float = 0.15  # random-walk restart

    # ====== Attention pooling ======
    route_pool_hidden: int = 64

    # ====== Checkpoint / output ======
    save_dir: str = os.path.join("saved_models", "GAT_GRU")
    save_best_name: str = "best.pt"
    export_attn_every_epoch: bool = True  # lưu attention weights theo epoch (hoặc chỉ best)
