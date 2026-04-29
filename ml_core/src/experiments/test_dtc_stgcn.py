"""
Quick Test Script - DTC-STGCN (5 Epochs)
Chạy test nhanh model DTC-STGCN với 5 epoch để kiểm tra pipeline.

Usage:
    python test_dtcstgcn.py
"""

import os
import sys
import torch
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# ── Path setup ──
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
sys.path.append(src_dir)

# ── Imports ──
from models.DTC_STGCN import DTCSTGCN, DTCSTGCNTrainer, count_parameters as dtc_count
from utils.data_loader import DataManager
from utils.metrics import evaluate_all_metrics, print_metrics

# ── Config ──
CONFIG = {
    "lr": 0.001,
    "weight_decay": 0.0001,
    "batch_size": 32,
    "epochs": 5,           # Chỉ 5 epoch để test nhanh
    "patience": 5,
    "resume": False,
    # DTC-STGCN hyperparams
    "gcn_hidden":     32,
    "gcn_out":        64,
    "lstm_hidden":    64,
    "hybrid_hidden1": 32,
    "hybrid_hidden2": 16,
    "dynamic_method": "FD",   # "FD" hoặc "FR"
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HORIZONS = [1, 2, 3, 4]  # days


# ═══════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════

def make_dummy_loaders(num_nodes=114, seq_len=12, pred_len=12, n_train=80,
                       n_val=20, n_test=20, batch_size=32):
    """Tạo dummy data khi không có file .npz thực."""
    from torch.utils.data import DataLoader, TensorDataset

    def _make(n):
        X = torch.FloatTensor(np.random.rand(n, seq_len, num_nodes, 1).astype(np.float32))
        y = torch.FloatTensor(np.random.rand(n, pred_len, num_nodes, 1).astype(np.float32))
        return DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=True)

    adj = np.random.rand(num_nodes, num_nodes).astype(np.float32)
    adj = (adj + adj.T) / 2
    np.fill_diagonal(adj, 1.0)
    adj_t = torch.FloatTensor(adj)

    return _make(n_train), _make(n_val), _make(n_test), adj_t


def load_data():
    """Load dữ liệu thực; fallback sang dummy nếu không có file."""
    print("\n" + "─" * 60)
    print("Bước 1: Load dữ liệu")
    print("─" * 60)

    try:
        dm = DataManager()
        dm.load_all()

        if not hasattr(dm, "data") or dm.data is None:
            raise RuntimeError("DataManager.data is None")

        train_loader, val_loader, test_loader, adj = dm.prepare_for_training(
            batch_size=CONFIG["batch_size"], normalize=False  # NPZ đã normalize trong pipeline
        )

        X_tr = dm.data["X_train"]
        print(f"✓ Dữ liệu thực  —  train: {X_tr.shape}")
        return train_loader, val_loader, test_loader, adj, True

    except Exception as e:
        print(f"⚠  Không load được dữ liệu thực ({e})")
        print("   → Sử dụng dummy data (114 nodes, seq=12, pred=12)")
        train_l, val_l, test_l, adj = make_dummy_loaders()
        return train_l, val_l, test_l, adj, False


# ═══════════════════════════════════════════════════════════
# Build & Train
# ═══════════════════════════════════════════════════════════

def build_model(train_loader, adj):
    """Khởi tạo DTCSTGCN từ shape của data loader."""
    print("\n" + "─" * 60)
    print("Bước 2: Khởi tạo model DTC-STGCN")
    print("─" * 60)

    for X, y in train_loader:
        _, seq_len, num_nodes, input_dim = X.shape
        _, pred_len, _, _ = y.shape
        output_dim = 1  # predict speed scalar only; y.shape[-1]=40 gây metrics lệch
        break

    model = DTCSTGCN(
        num_nodes=num_nodes,
        input_dim=input_dim,
        hidden_dim=64,
        output_dim=output_dim,
        seq_len=seq_len,
        pred_len=pred_len,
        adj=adj.numpy() if isinstance(adj, torch.Tensor) else adj,
        gcn_hidden=CONFIG["gcn_hidden"],
        gcn_out=CONFIG["gcn_out"],
        lstm_hidden=CONFIG["lstm_hidden"],
        hybrid_hidden1=CONFIG["hybrid_hidden1"],
        hybrid_hidden2=CONFIG["hybrid_hidden2"],
        dynamic_method=CONFIG["dynamic_method"],
    )

    n_params = dtc_count(model)
    print(f"✓ Model: DTC-STGCN-{CONFIG['dynamic_method']}")
    print(f"  Nodes          : {num_nodes}")
    print(f"  Input dim      : {input_dim}")
    print(f"  Seq / Pred len : {seq_len} / {pred_len}")
    print(f"  Tổng parameters: {n_params:,}")
    print(f"  Device         : {DEVICE}")

    return model


def train_model(model, train_loader, val_loader, adj):
    """Train 5 epoch."""
    print("\n" + "─" * 60)
    print(f"Bước 3: Huấn luyện DTC-STGCN ({CONFIG['epochs']} epochs)")
    print("─" * 60)

    trainer_config = {
        "lr":            CONFIG["lr"],
        "weight_decay":  CONFIG["weight_decay"],
        "epochs":        CONFIG["epochs"],
        "patience":      CONFIG["patience"],
        "resume":        CONFIG["resume"],
    }

    trainer = DTCSTGCNTrainer(model, trainer_config, DEVICE)
    trainer.train(
        train_loader,
        val_loader,
        epochs=CONFIG["epochs"],
        early_stopping_patience=CONFIG["patience"],
        resume=CONFIG["resume"],
    )

    return trainer


def evaluate(trainer, test_loader):
    """Predict và tính metrics cho từng horizon."""
    print("\n" + "─" * 60)
    print("Bước 4: Đánh giá trên tập test")
    print("─" * 60)

    predictions, targets = trainer.predict(test_loader)

    print(f"  Prediction shape : {predictions.shape}")
    print(f"  Target shape     : {targets.shape}\n")

    for h in HORIZONS:
        if h > predictions.shape[1]:
            continue
        # Chỉ evaluate feature 0 (speed) — model học từ scalar speed signal
        pred_h   = predictions[:, :h, :, :1]
        target_h = targets[:, :h, :, :1]
        metrics  = evaluate_all_metrics(target_h, pred_h)

        print(f"  Horizon {h} day(s):")
        for k, v in metrics.items():
            print(f"    {k:<10}: {v:.4f}")
        print()

    return predictions, targets


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

def main():
    print("\n" + "═" * 60)
    print("  QUICK TEST — DTC-STGCN (5 EPOCHS)")
    print("═" * 60)
    print(f"  Dynamic method : {CONFIG['dynamic_method']}")
    print(f"  Epochs         : {CONFIG['epochs']}")
    print(f"  Batch size     : {CONFIG['batch_size']}")
    print(f"  Device         : {DEVICE}")

    # 1. Data
    train_loader, val_loader, test_loader, adj, real_data = load_data()

    # 2. Model
    model = build_model(train_loader, adj)

    # 3. Train
    trainer = train_model(model, train_loader, val_loader, adj)

    # 4. Evaluate
    predictions, targets = evaluate(trainer, test_loader)

    # ── Summary ──
    print("═" * 60)
    print("✅  Quick test DTC-STGCN hoàn thành!")
    print(f"   Data source  : {'thực' if real_data else 'dummy (giả lập)'}")
    print(f"   Epochs chạy  : {CONFIG['epochs']}")
    print(f"   Output shape : {predictions.shape}")
    print("═" * 60 + "\n")


if __name__ == "__main__":
    main()