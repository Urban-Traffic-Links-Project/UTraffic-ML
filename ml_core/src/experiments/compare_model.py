"""
Train and Compare: T-GCN vs DTC-STGCN
Generates comparison tables like Tables 1-2 in DTC-STGCN paper.

Fixes vs original:
  - output_dim=1 cho mọi model (predict speed scalar, không phải 40 features)
  - evaluate_model() slice [:, :h, :, :1] — metrics chỉ trên speed
  - normalize=False (NPZ đã normalize trong pipeline)
  - TH-GAT đã xoá hoàn toàn
  - adj raw binary, mỗi model tự normalize nội bộ một lần

Usage:
    python experiments/compare_model.py
"""

import os
import sys
import json
import torch
import numpy as np
import pandas as pd
import warnings
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
sys.path.append(src_dir)

from models.T_GCN import TGCN, TGCNTrainer, count_parameters as tgcn_count
from models.DTC_STGCN import DTCSTGCN, DTCSTGCNTrainer, count_parameters as dtc_count
from models.DTC_STGCN.graph.correlation_matrix import TrafficCorrelationMatrix
from utils.data_loader import DataManager
from utils.metrics import evaluate_all_metrics, MetricsTracker
from utils.baselines import get_baseline_model

CONFIG = {
    "lr": 0.001,
    "weight_decay": 0.0001,
    "batch_size": 32,
    "epochs": 50,
    "patience": 20,
    "tgcn": {
        "hidden_dim": 64,
        "gcn_hidden_dim": 64,
    },
    "dtcstgcn": {
        "gcn_hidden": 32,
        "gcn_out": 64,
        "lstm_hidden": 64,
        "hybrid_hidden1": 32,
        "hybrid_hidden2": 16,
        "dynamic_method": "FD",
    },
    "models": ["T-GCN", "DTC-STGCN-FD", "DTC-STGCN-FR", "HA", "SVR", "GRU"],
    "resume": False,
}

device = "cuda" if torch.cuda.is_available() else "cpu"


def train_tgcn(train_loader, val_loader, test_loader, adj, dm, config):
    print("\n" + "=" * 70)
    print("Training T-GCN")
    print("=" * 70)
    for X, y in train_loader:
        _, seq_len, num_nodes, input_dim = X.shape
        _, pred_len, _, _ = y.shape
        break

    output_dim = 1  # speed scalar only

    model = TGCN(
        num_nodes=num_nodes,
        input_dim=input_dim,
        hidden_dim=config["tgcn"]["hidden_dim"],
        output_dim=output_dim,
        seq_len=seq_len,
        pred_len=pred_len,
        gcn_hidden_dim=config["tgcn"]["gcn_hidden_dim"],
    )
    print(f"Parameters: {tgcn_count(model):,}")

    trainer_config = {
        "lr": config["lr"],
        "weight_decay": config["weight_decay"],
        "hidden_dim": config["tgcn"]["hidden_dim"],
        "epochs": config["epochs"],
        "patience": config["patience"],
        "resume": config["resume"],
    }
    trainer = TGCNTrainer(model, adj.numpy(), trainer_config, device)
    trainer.train(train_loader, val_loader, config["epochs"],
                  early_stopping_patience=config["patience"],
                  resume=config["resume"])
    predictions, targets = trainer.predict(test_loader)
    return model, predictions, targets


def train_dtcstgcn(train_loader, val_loader, test_loader, adj, dm, config, method="FD"):
    print("\n" + "=" * 70)
    print(f"Training DTC-STGCN-{method}")
    print("=" * 70)
    for X, y in train_loader:
        _, seq_len, num_nodes, input_dim = X.shape
        _, pred_len, _, _ = y.shape
        break

    output_dim = 1  # speed scalar only

    dtccfg = config["dtcstgcn"]
    model = DTCSTGCN(
        num_nodes=num_nodes,
        input_dim=input_dim,
        hidden_dim=64,
        output_dim=output_dim,
        seq_len=seq_len,
        pred_len=pred_len,
        adj=adj.numpy(),
        gcn_hidden=dtccfg["gcn_hidden"],
        gcn_out=dtccfg["gcn_out"],
        lstm_hidden=dtccfg["lstm_hidden"],
        hybrid_hidden1=dtccfg["hybrid_hidden1"],
        hybrid_hidden2=dtccfg["hybrid_hidden2"],
        dynamic_method=method,
    )
    print(f"Parameters: {dtc_count(model):,}")

    trainer_config = {
        "lr": config["lr"],
        "weight_decay": config["weight_decay"],
        "epochs": config["epochs"],
        "patience": config["patience"],
        "resume": config["resume"],
    }
    trainer = DTCSTGCNTrainer(model, trainer_config, device)
    trainer.train(train_loader, val_loader, config["epochs"],
                  early_stopping_patience=config["patience"],
                  resume=config["resume"])
    predictions, targets = trainer.predict(test_loader)
    return model, predictions, targets


def build_correlation_matrices(dm, adj):
    print("\n" + "=" * 70)
    print("Building Traffic Correlation Matrices")
    print("=" * 70)

    X_train = dm.data["X_train"]
    _, _, N, _ = X_train.shape
    speed_data = X_train[:, :, :, 0].reshape(-1, N)
    print(f"Speed data: {speed_data.shape}")

    coordinates = getattr(dm, "coordinates", None)
    cm = TrafficCorrelationMatrix(N, coordinates=coordinates)

    print("  Pearson correlation...")
    pearson = cm.compute_pearson_correlation(speed_data)

    print("  Dynamic correlation (FD)...")
    dynamic = cm.compute_dynamic_correlation(speed_data, method="FD", fixed_adj=adj.numpy())

    if coordinates is not None:
        print("  Spatial correlation...")
        cm.compute_spatial_correlation(sigma=0.15)

    summary = cm.get_summary()
    print("\n  Summary:")
    for name, stats in summary.items():
        print(f"    {name}: mean={stats['mean']:.4f}, high_corr_pairs={stats['high_corr_pairs']}")

    results_dir = Path(src_dir).parent / "data" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    save_dict = {"pearson": pearson, "dynamic": dynamic}
    if coordinates is not None and cm.spatial_matrix is not None:
        save_dict["spatial"] = cm.spatial_matrix

    np.savez(results_dir / f"correlation_matrices_{ts}.npz", **save_dict)

    labels = None
    if "segment_ids" in dm.data and dm.data["segment_ids"] is not None:
        try:
            labels = [str(int(s)) for s in np.array(dm.data["segment_ids"]).reshape(-1)]
        except Exception:
            labels = None
    if labels is None or len(labels) != N:
        labels = [str(i) for i in range(N)]

    for name, mat in save_dict.items():
        pd.DataFrame(mat, index=labels, columns=labels).to_csv(
            results_dir / f"correlation_{name}_{ts}.csv", float_format="%.6f"
        )

    summary_rows = [{"matrix": k, **v} for k, v in summary.items()]
    pd.DataFrame(summary_rows).to_csv(results_dir / f"correlation_summary_{ts}.csv", index=False)
    print(f"  Saved to {results_dir}")
    return cm


def evaluate_model(predictions, targets, horizons=[1, 2, 3, 4]):
    """
    Evaluate trên speed (dim 0) only.
    output_dim=1 nen slice :1 van dung cho ca T-GCN va DTC-STGCN.
    Voi baselines (output_dim=40), slice :1 lay dung dim speed.
    """
    results = {}
    for h in horizons:
        if h <= predictions.shape[1]:
            pred_h   = predictions[:, :h, :, :1]
            target_h = targets[:, :h, :, :1]
            results[h] = evaluate_all_metrics(target_h, pred_h)
    return results


def compare_and_save(all_results, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    horizons = sorted(set(h for r in all_results.values() for h in r.keys()))

    for h in horizons:
        rows = []
        for model_name, model_results in all_results.items():
            if h in model_results:
                row = {"Model": model_name}
                row.update({k: f"{v:.4f}" for k, v in model_results[h].items()})
                rows.append(row)
        df = pd.DataFrame(rows)
        print(f"\n{'='*90}")
        print(f"Comparison — Horizon: {h} day(s)")
        print(f"{'='*90}")
        print(df.to_string(index=False))
        df.to_csv(output_dir / f"comparison_horizon_{h}days.csv", index=False)
        print(f"Saved: comparison_horizon_{h}days.csv")

    summary_rows = []
    for model_name, model_results in all_results.items():
        for h, metrics in model_results.items():
            row = {"Model": model_name, "Horizon": f"{h} day(s)"}
            row.update(metrics)
            summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(output_dir / "comparison_summary.csv", index=False)

    def _serialize(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, dict):
            return {k: _serialize(v) for k, v in obj.items()}
        return obj

    with open(output_dir / "comparison_results.json", "w") as f:
        json.dump(_serialize(all_results), f, indent=2)

    print(f"\nSaved: comparison_summary.csv, comparison_results.json")


def main():
    print("\n" + "=" * 70)
    print("TRAFFIC PREDICTION MODEL COMPARISON — T-GCN vs DTC-STGCN")
    print("=" * 70)
    print(f"Device: {device}")

    print("\nLoading data...")
    dm = DataManager()
    dm.load_all()
    # normalize=False: NPZ da co scaler tu pipeline.
    # adj raw binary — model tu normalize noi bo 1 lan.
    train_loader, val_loader, test_loader, adj = dm.prepare_for_training(
        batch_size=CONFIG["batch_size"], normalize=False
    )

    X_train = dm.data["X_train"]
    y_train = dm.data["y_train"]
    X_test  = dm.data["X_test"]
    y_test  = dm.data["y_test"]
    print(f"Train: {X_train.shape}, Val: {dm.data['X_val'].shape}, Test: {X_test.shape}")

    all_results = {}

    for model_name in CONFIG["models"]:
        print(f"\n{'─'*50}")
        print(f"  Model: {model_name}")
        print(f"{'─'*50}")

        try:
            if model_name == "T-GCN":
                _, preds, tgts = train_tgcn(
                    train_loader, val_loader, test_loader, adj, dm, CONFIG)

            elif model_name == "DTC-STGCN-FD":
                _, preds, tgts = train_dtcstgcn(
                    train_loader, val_loader, test_loader, adj, dm, CONFIG, "FD")

            elif model_name == "DTC-STGCN-FR":
                _, preds, tgts = train_dtcstgcn(
                    train_loader, val_loader, test_loader, adj, dm, CONFIG, "FR")

            elif model_name in ["HA", "SVR", "ARIMA"]:
                baseline = get_baseline_model(model_name)
                baseline.fit(X_train, y_train)
                preds = baseline.predict(X_test)
                tgts  = y_test

            elif model_name == "GRU":
                _, seq_len, num_nodes, input_dim = X_train.shape
                _, pred_len, _, _ = y_train.shape
                from utils.baselines import GRUOnly
                gru_model = GRUOnly(
                    num_nodes, input_dim, 64, 1, seq_len, pred_len).to(device)
                optimizer = torch.optim.Adam(gru_model.parameters(), lr=0.001)
                criterion = torch.nn.MSELoss()
                gru_model.train()
                for _ in range(30):
                    for bx, by in train_loader:
                        bx  = bx.to(device)
                        by1 = by[:, :, :, :1].to(device)
                        optimizer.zero_grad()
                        loss = criterion(gru_model(bx), by1)
                        loss.backward()
                        optimizer.step()
                gru_model.eval()
                preds_list, tgts_list = [], []
                with torch.no_grad():
                    for bx, by in test_loader:
                        preds_list.append(gru_model(bx.to(device)).cpu().numpy())
                        tgts_list.append(by[:, :, :, :1].numpy())
                preds = np.concatenate(preds_list)
                tgts  = np.concatenate(tgts_list)

            else:
                print(f"Unknown model: {model_name}, skipping.")
                continue

            results = evaluate_model(preds, tgts)
            all_results[model_name] = results

            for h, metrics in results.items():
                print(f"  H{h}: RMSE={metrics['RMSE']:.4f}  MAE={metrics['MAE']:.4f}"
                      f"  Acc={metrics['Accuracy']:.4f}  R2={metrics['R2']:.4f}")

        except Exception as e:
            print(f"Error with {model_name}: {e}")
            import traceback; traceback.print_exc()
            continue

    results_dir = Path(src_dir).parent / "data" / "results"
    compare_and_save(all_results, results_dir)
    build_correlation_matrices(dm, adj)

    print("\n" + "=" * 70)
    print("Comparison Complete!")
    print(f"Results saved to: {results_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()