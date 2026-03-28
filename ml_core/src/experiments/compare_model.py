"""
Train and Compare: T-GCN vs DTC-STGCN
Generates comparison tables like Tables 1-2 in DTC-STGCN paper.

Usage:
    python experiments/compare_model.py

Changes vs original:
    - normalize=False always (NPZ đã normalize trong pipeline)
    - adj raw binary → model tự normalize nội bộ 1 lần duy nhất
    - evaluate_model() chỉ slice speed dim [:, :h, :, :1]
    - GRU baseline nhận output_dim=1 đúng shape
    - Safe shape inference từ loader thay vì hardcode
    - Bỏ TH-GAT hoàn toàn
    - Tất cả except in traceback rõ ràng và tiếp tục
"""

import os
import sys
import json
import warnings
import traceback
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")

# ── Path setup ────────────────────────────────────────────────────────────────
current_dir = Path(__file__).resolve().parent
src_dir     = current_dir.parent
sys.path.insert(0, str(src_dir))

from models.T_GCN import TGCN, TGCNTrainer, count_parameters as tgcn_count
from models.DTC_STGCN import DTCSTGCN, DTCSTGCNTrainer, count_parameters as dtc_count
from models.DTC_STGCN.graph.correlation_matrix import TrafficCorrelationMatrix
from utils.data_loader import DataManager, normalize_data
from utils.metrics import evaluate_all_metrics
from utils.baselines import get_baseline_model, GRUOnly

# Config

CONFIG = {
    "lr":            0.001,
    "weight_decay":  0.0001,
    "batch_size":    32,
    "epochs":        50,
    "patience":      20,
    "tgcn": {
        "hidden_dim":     64,
        "gcn_hidden_dim": 64,
    },
    "dtcstgcn": {
        "gcn_hidden":      32,
        "gcn_out":         64,
        "lstm_hidden":     64,
        "hybrid_hidden1":  32,
        "hybrid_hidden2":  16,
    },
    # Thứ tự chạy — bỏ bất kỳ tên nào nếu không muốn train
    "models":  ["T-GCN", "DTC-STGCN-FD", "DTC-STGCN-FR", "HA", "SVR", "GRU"],
    "resume":  False,
}

device = "cuda" if torch.cuda.is_available() else "cpu"


# Shape helper

def _infer_shapes(loader):
    """Lấy (seq_len, num_nodes, input_dim, pred_len) từ 1 batch."""
    for X, y in loader:
        _, seq_len, num_nodes, input_dim = X.shape
        _, pred_len, _, _               = y.shape
        return seq_len, num_nodes, input_dim, pred_len
    raise RuntimeError("DataLoader rỗng — không thể infer shapes.")


# Training functions

def train_tgcn(train_loader, val_loader, test_loader, adj, config):
    print("\n" + "=" * 70)
    print("Training T-GCN")
    print("=" * 70)

    seq_len, num_nodes, input_dim, pred_len = _infer_shapes(train_loader)

    model = TGCN(
        num_nodes=num_nodes,
        input_dim=input_dim,
        hidden_dim=config["tgcn"]["hidden_dim"],
        output_dim=1,           # predict speed scalar only
        seq_len=seq_len,
        pred_len=pred_len,
        gcn_hidden_dim=config["tgcn"]["gcn_hidden_dim"],
    )
    print(f"T-GCN params: {tgcn_count(model):,}")

    trainer_cfg = {
        "lr":           config["lr"],
        "weight_decay": config["weight_decay"],
        "hidden_dim":   config["tgcn"]["hidden_dim"],
        "epochs":       config["epochs"],
        "patience":     config["patience"],
        "resume":       config["resume"],
    }
    trainer = TGCNTrainer(model, adj.numpy(), trainer_cfg, device)
    trainer.train(train_loader, val_loader,
                  epochs=config["epochs"],
                  early_stopping_patience=config["patience"],
                  resume=config["resume"])

    predictions, targets = trainer.predict(test_loader)
    return model, predictions, targets


def train_dtcstgcn(train_loader, val_loader, test_loader, adj, config, method="FD"):
    print("\n" + "=" * 70)
    print(f"Training DTC-STGCN-{method}")
    print("=" * 70)

    seq_len, num_nodes, input_dim, pred_len = _infer_shapes(train_loader)
    dc = config["dtcstgcn"]

    model = DTCSTGCN(
        num_nodes=num_nodes,
        input_dim=input_dim,
        hidden_dim=64,
        output_dim=1,           # predict speed scalar only
        seq_len=seq_len,
        pred_len=pred_len,
        adj=adj.numpy(),
        gcn_hidden=dc["gcn_hidden"],
        gcn_out=dc["gcn_out"],
        lstm_hidden=dc["lstm_hidden"],
        hybrid_hidden1=dc["hybrid_hidden1"],
        hybrid_hidden2=dc["hybrid_hidden2"],
        dynamic_method=method,
    )
    print(f"DTC-STGCN-{method} params: {dtc_count(model):,}")

    trainer_cfg = {
        "lr":           config["lr"],
        "weight_decay": config["weight_decay"],
        "epochs":       config["epochs"],
        "patience":     config["patience"],
        "resume":       config["resume"],
    }
    trainer = DTCSTGCNTrainer(model, trainer_cfg, device)
    trainer.train(train_loader, val_loader,
                  epochs=config["epochs"],
                  early_stopping_patience=config["patience"],
                  resume=config["resume"])

    predictions, targets = trainer.predict(test_loader)
    return model, predictions, targets


def train_gru(train_loader, val_loader, test_loader, config):
    """Train lightweight GRU baseline."""
    print("\n" + "─" * 50)
    print("Training GRU baseline")
    print("─" * 50)

    seq_len, num_nodes, input_dim, pred_len = _infer_shapes(train_loader)

    gru_model = GRUOnly(
        num_nodes=num_nodes,
        input_dim=input_dim,
        hidden_dim=64,
        output_dim=1,
        seq_len=seq_len,
        pred_len=pred_len,
    ).to(device)

    optimizer = torch.optim.Adam(gru_model.parameters(), lr=config["lr"])
    criterion = torch.nn.HuberLoss()

    best_val = float("inf")
    patience_count = 0

    for epoch in range(config["epochs"]):
        gru_model.train()
        for bx, by in train_loader:
            bx   = bx.to(device)
            by1  = by[:, :, :, :1].to(device)   # speed only
            optimizer.zero_grad()
            pred = gru_model(bx)
            loss = criterion(pred, by1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gru_model.parameters(), 5.0)
            optimizer.step()

        # Validation
        gru_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                pred = gru_model(bx.to(device))
                val_loss += criterion(pred, by[:, :, :, :1].to(device)).item()
        val_loss /= max(len(val_loader), 1)

        if val_loss < best_val:
            best_val = val_loss
            patience_count = 0
        else:
            patience_count += 1

        if patience_count >= config["patience"]:
            print(f"  GRU early stop at epoch {epoch+1}")
            break

    # Predict
    gru_model.eval()
    preds_list, tgts_list = [], []
    with torch.no_grad():
        for bx, by in test_loader:
            p = gru_model(bx.to(device)).cpu().numpy()
            t = by[:, :, :, :1].numpy()
            preds_list.append(p)
            tgts_list.append(t)

    return np.concatenate(preds_list), np.concatenate(tgts_list)


# Evaluate

def evaluate_model(predictions: np.ndarray, targets: np.ndarray,
                   horizons=(1, 2, 3, 4)) -> dict:
    """
    Evaluate trên speed (feature dim 0) only.
    predictions/targets shape: (samples, pred_len, nodes, output_dim)
    """
    results = {}
    for h in horizons:
        if h <= predictions.shape[1]:
            pred_h   = predictions[:, :h, :, :1]
            target_h = targets[:,    :h, :, :1]
            results[h] = evaluate_all_metrics(target_h, pred_h)
    return results


# Correlation matrices
def build_correlation_matrices(dm: DataManager, adj: torch.Tensor):
    print("\n" + "=" * 70)
    print("Building Traffic Correlation Matrices")
    print("=" * 70)

    X_train = dm.data["X_train"]
    N       = X_train.shape[2]
    speed_data = X_train[:, :, :, 0].reshape(-1, N)   # (T*samples, N)
    print(f"Speed data for correlation: {speed_data.shape}")

    coordinates = getattr(dm, "coordinates", None)
    cm = TrafficCorrelationMatrix(N, coordinates=coordinates)

    print("  Pearson correlation...")
    pearson = cm.compute_pearson_correlation(speed_data)

    print("  Dynamic correlation (FD)...")
    dynamic = cm.compute_dynamic_correlation(
        speed_data, method="FD", fixed_adj=adj.numpy())

    if coordinates is not None:
        print("  Spatial correlation...")
        cm.compute_spatial_correlation(sigma=0.15)

    summary = cm.get_summary()
    print("\n  Summary:")
    for name, stats in summary.items():
        print(f"    {name}: mean={stats['mean']:.4f}, "
              f"high_corr_pairs={stats['high_corr_pairs']}")

    # Save
    results_dir = src_dir.parent / "data" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    save_dict = {"pearson": pearson, "dynamic": dynamic}
    if coordinates is not None and cm.spatial_matrix is not None:
        save_dict["spatial"] = cm.spatial_matrix
    np.savez(results_dir / f"correlation_matrices_{ts}.npz", **save_dict)

    # Labels
    if "segment_ids" in dm.data and dm.data["segment_ids"] is not None:
        try:
            labels = [str(int(s)) for s in np.array(dm.data["segment_ids"]).reshape(-1)]
        except Exception:
            labels = [str(i) for i in range(N)]
    else:
        labels = [str(i) for i in range(N)]

    if len(labels) != N:
        labels = [str(i) for i in range(N)]

    for name, mat in save_dict.items():
        pd.DataFrame(mat, index=labels, columns=labels).to_csv(
            results_dir / f"correlation_{name}_{ts}.csv", float_format="%.6f")

    rows = [{"matrix": k, **v} for k, v in summary.items()]
    pd.DataFrame(rows).to_csv(results_dir / f"correlation_summary_{ts}.csv", index=False)
    print(f"  Saved correlation matrices to {results_dir}")
    return cm


# Save results

def compare_and_save(all_results: dict, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    horizons = sorted({h for r in all_results.values() for h in r})

    for h in horizons:
        rows = []
        for model_name, model_results in all_results.items():
            if h in model_results:
                row = {"Model": model_name}
                row.update({k: f"{v:.4f}" for k, v in model_results[h].items()})
                rows.append(row)
        df = pd.DataFrame(rows)
        print(f"\n{'='*90}")
        print(f"Horizon: {h} step(s) = {h*15} min")
        print(f"{'='*90}")
        print(df.to_string(index=False))
        df.to_csv(output_dir / f"comparison_horizon_{h}.csv", index=False)

    # Summary CSV
    summary_rows = []
    for model_name, model_results in all_results.items():
        for h, metrics in model_results.items():
            row = {"Model": model_name, "Horizon": f"{h} step ({h*15} min)"}
            row.update(metrics)
            summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(output_dir / "comparison_summary.csv", index=False)

    # JSON (numpy-safe)
    def _serial(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, dict):
            return {k: _serial(v) for k, v in obj.items()}
        return obj

    with open(output_dir / "comparison_results.json", "w") as f:
        json.dump(_serial(all_results), f, indent=2)

    print(f"\nSaved: comparison_summary.csv + comparison_results.json → {output_dir}")


# Main
def main():
    print("\n" + "=" * 70)
    print("TRAFFIC PREDICTION — T-GCN vs DTC-STGCN")
    print("=" * 70)
    print(f"Device: {device}")

    # ── Load data ─────────────────────────────────────────────────────────────
    print("\nLoading data...")
    dm = DataManager()
    dm.load_all()

    # normalize=False: NPZ đã normalize trong pipeline → không scale lại
    train_loader, val_loader, test_loader, adj = dm.prepare_for_training(
        batch_size=CONFIG["batch_size"], normalize=False)

    X_train = dm.data["X_train"]
    y_train = dm.data["y_train"]
    X_test  = dm.data["X_test"]
    y_test  = dm.data["y_test"]

    print(f"\nTrain: {X_train.shape}  Val: {dm.data['X_val'].shape}  Test: {X_test.shape}")
    print(f"Adj  : {adj.shape}  (raw binary — model normalizes internally)")

    # ── Run models ────────────────────────────────────────────────────────────
    all_results: dict = {}
    horizons = (1, 2, 3, 4)

    for model_name in CONFIG["models"]:
        print(f"\n{'─'*60}")
        print(f"  Model: {model_name}")
        print(f"{'─'*60}")

        try:
            # ── Neural models ──────────────────────────────────────────────
            if model_name == "T-GCN":
                _, preds, tgts = train_tgcn(
                    train_loader, val_loader, test_loader, adj, CONFIG)

            elif model_name == "DTC-STGCN-FD":
                _, preds, tgts = train_dtcstgcn(
                    train_loader, val_loader, test_loader, adj, CONFIG, "FD")

            elif model_name == "DTC-STGCN-FR":
                _, preds, tgts = train_dtcstgcn(
                    train_loader, val_loader, test_loader, adj, CONFIG, "FR")

            elif model_name == "GRU":
                preds, tgts = train_gru(
                    train_loader, val_loader, test_loader, CONFIG)

            # ── Statistical baselines ──────────────────────────────────────
            elif model_name in ("HA", "SVR", "ARIMA"):
                baseline = get_baseline_model(model_name)
                baseline.fit(X_train, y_train)
                preds = baseline.predict(X_test)
                tgts  = y_test

            else:
                print(f"Unknown model '{model_name}' — skipping.")
                continue

            # ── Evaluate ───────────────────────────────────────────────────
            results = evaluate_model(preds, tgts, horizons=horizons)
            all_results[model_name] = results

            for h, metrics in results.items():
                print(f"  H{h} ({h*15}min): "
                      f"RMSE={metrics['RMSE']:.4f}  "
                      f"MAE={metrics['MAE']:.4f}  "
                      f"Acc={metrics['Accuracy']:.4f}  "
                      f"R2={metrics['R2']:.4f}")

        except Exception as exc:
            print(f"\n[ERROR] {model_name} failed: {exc}")
            traceback.print_exc()
            print("Continuing with next model...\n")
            continue

    # ── Save results ──────────────────────────────────────────────────────────
    results_dir = src_dir.parent / "data" / "results"
    compare_and_save(all_results, results_dir)

    # ── Correlation matrices ───────────────────────────────────────────────────
    try:
        build_correlation_matrices(dm, adj)
    except Exception as exc:
        print(f"Warning: correlation matrix failed ({exc}) — skipping.")

    print("\n" + "=" * 70)
    print("Comparison Complete!")
    print(f"Results saved to: {results_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()