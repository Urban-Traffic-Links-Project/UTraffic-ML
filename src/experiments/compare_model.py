"""
Train and Compare: T-GCN vs TH-GAT vs DTC-STGCN
Generates comparison tables like Tables 2-3 in TH-GAT paper and Tables 1-2 in DTC-STGCN paper.

Usage:
    python experiments/compare_models.py

Results saved to:
    data/results/comparison_*.csv
    data/results/comparison_results.json
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

# ── Path setup ──
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
sys.path.append(src_dir)

from models.T_GCN import TGCN, TGCNTrainer, count_parameters as tgcn_count
try:
    from models.TH_GAT import THGAT, THGATTrainer, count_parameters as thgat_count  # type: ignore
    _THGAT_AVAILABLE = True
except Exception:
    THGAT = None  # type: ignore
    THGATTrainer = None  # type: ignore
    thgat_count = None  # type: ignore
    _THGAT_AVAILABLE = False
from models.DTC_STGCN import DTCSTGCN, DTCSTGCNTrainer, count_parameters as dtc_count
from models.DTC_STGCN.graph.correlation_matrix import TrafficCorrelationMatrix
from utils.data_loader import DataManager
from utils.metrics import evaluate_all_metrics, MetricsTracker
from utils.baselines import get_baseline_model


# ═══════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════
CONFIG = {
    # Common
    "lr": 0.001,
    "weight_decay": 0.0001,
    "batch_size": 32,
    "epochs": 50,
    "patience": 20,
    # T-GCN
    "tgcn": {
        "hidden_dim": 64,
        "gcn_hidden_dim": 64,
    },
    # TH-GAT
    # "thgat": {
    #     "hidden_dim": 64,
    #     "n_regions_l1": 5,
    #     "n_regions_l2": 2,
    #     "gat_dropout": 0.1,
    #     "gru_dropout": 0.1,
    #     "l2_lambda": 0.0015,
    #     "batch_size": 4,    # Paper uses batch_size=4
    #     "epochs": 100,
    #     "patience": 100,
    # },
    # DTC-STGCN
    "dtcstgcn": {
        "gcn_hidden": 32,
        "gcn_out": 64,
        "lstm_hidden": 64,
        "hybrid_hidden1": 32,
        "hybrid_hidden2": 16,
        "dynamic_method": "FD",  # Best method from paper (Table 3-4)
    },
    # Models to run
    "models": ["T-GCN", "DTC-STGCN-FD", "DTC-STGCN-FR", "HA", "SVR", "GRU"],
    "resume": False,
}

device = "cuda" if torch.cuda.is_available() else "cpu"


# ═══════════════════════════════════════════════════════════
# Training functions
# ═══════════════════════════════════════════════════════════

def train_tgcn(train_loader, val_loader, test_loader, adj, dm, config):
    print("\n" + "═" * 70)
    print("Training T-GCN")
    print("═" * 70)
    for X, y in train_loader:
        _, seq_len, num_nodes, input_dim = X.shape
        _, pred_len, _, output_dim = y.shape
        break

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


def train_thgat(train_loader, val_loader, test_loader, adj, dm, config):
    if not _THGAT_AVAILABLE:
        raise RuntimeError("TH-GAT module not found in this repository. Remove 'TH-GAT' from CONFIG['models'] or add the TH-GAT implementation.")
    print("\n" + "═" * 70)
    print("Training TH-GAT")
    print("═" * 70)
    for X, y in train_loader:
        _, seq_len, num_nodes, input_dim = X.shape
        _, pred_len, _, output_dim = y.shape
        break

    thcfg = config["thgat"]
    coordinates = dm.coordinates if hasattr(dm, "coordinates") and dm.coordinates is not None else None

    model = THGAT(
        num_nodes=num_nodes,
        input_dim=input_dim,
        hidden_dim=thcfg["hidden_dim"],
        output_dim=output_dim,
        seq_len=seq_len,
        pred_len=pred_len,
        adj=adj.numpy(),
        coordinates=coordinates,
        n_regions_l1=thcfg["n_regions_l1"],
        n_regions_l2=thcfg["n_regions_l2"],
        gat_dropout=thcfg["gat_dropout"],
        gru_dropout=thcfg["gru_dropout"],
        l2_lambda=thcfg["l2_lambda"],
    )
    print(f"Parameters: {thgat_count(model):,}")
    print(f"Region-augmented nodes: {model.n_bar} (original: {num_nodes})")

    # Use TH-GAT specific batch_size and epochs (paper: batch=4, epochs=5000, patience=100)
    from torch.utils.data import DataLoader, TensorDataset
    # Rebuild loaders with TH-GAT batch size
    X_train = dm.data["X_train"]
    y_train = dm.data["y_train"]
    X_val = dm.data["X_val"]
    y_val = dm.data["y_val"]
    X_test = dm.data["X_test"]
    y_test = dm.data["y_test"]

    thgat_bs = thcfg.get("batch_size", 4)
    th_train = DataLoader(
        TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train)),
        batch_size=thgat_bs, shuffle=True
    )
    th_val = DataLoader(
        TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val)),
        batch_size=thgat_bs, shuffle=False
    )
    th_test = DataLoader(
        TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test)),
        batch_size=thgat_bs, shuffle=False
    )

    trainer_config = {
        "lr": config["lr"],
        "l2_lambda": thcfg["l2_lambda"],
        "epochs": thcfg["epochs"],
        "patience": thcfg["patience"],
        "resume": config["resume"],
    }
    trainer = THGATTrainer(model, trainer_config, device)
    trainer.train(th_train, th_val,
                  epochs=thcfg["epochs"],
                  early_stopping_patience=thcfg["patience"],
                  resume=config["resume"])

    predictions, targets = trainer.predict(th_test)
    return model, predictions, targets


def train_dtcstgcn(train_loader, val_loader, test_loader, adj, dm, config, method="FD"):
    print("\n" + "═" * 70)
    print(f"Training DTC-STGCN-{method}")
    print("═" * 70)
    for X, y in train_loader:
        _, seq_len, num_nodes, input_dim = X.shape
        _, pred_len, _, output_dim = y.shape
        break

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


# ═══════════════════════════════════════════════════════════
# Correlation Matrix Analysis
# ═══════════════════════════════════════════════════════════

def build_correlation_matrices(dm, adj, thgat_model=None, test_loader=None):
    """
    Build and save traffic node correlation matrices.
    """
    print("\n" + "═" * 70)
    print("Building Traffic Correlation Matrices")
    print("═" * 70)

    # Reconstruct speed time series from features
    # Use training data, first feature (speed)
    X_train = dm.data["X_train"]  # (samples, seq_len, N, features)
    samples, seq_len, N, _ = X_train.shape

    # Reshape to (T, N): flatten samples x seq_len
    speed_data = X_train[:, :, :, 0].reshape(-1, N)  # (T, N)
    print(f"Speed data for correlation: {speed_data.shape}")

    coordinates = dm.coordinates if hasattr(dm, "coordinates") and dm.coordinates is not None else None
    cm = TrafficCorrelationMatrix(N, coordinates=coordinates)

    # 1. Pearson correlation
    print("  Computing Pearson correlation...")
    pearson = cm.compute_pearson_correlation(speed_data)

    # 2. Dynamic correlation (DTC-STGCN method)
    print("  Computing dynamic correlation (FD method)...")
    dynamic = cm.compute_dynamic_correlation(speed_data, method="FD", fixed_adj=adj.numpy())

    # 3. Spatial correlation (if coordinates available)
    if coordinates is not None:
        print("  Computing spatial correlation...")
        spatial = cm.compute_spatial_correlation(sigma=0.15)
    else:
        print("  Skipping spatial correlation (no coordinates)")

    # 4. Attention-based correlation (if TH-GAT model provided)
    if thgat_model is not None and test_loader is not None:
        print("  Extracting attention-based correlation from TH-GAT...")
        cm.extract_attention_correlation(thgat_model, test_loader, device=device)

    # 5. Summary
    summary = cm.get_summary()
    print("\n  Correlation Matrix Summary:")
    for name, stats in summary.items():
        print(f"    {name}: mean={stats['mean']:.4f}, std={stats['std']:.4f}, "
              f"high_corr_pairs={stats['high_corr_pairs']}")

    # Save matrices
    results_dir = Path(src_dir).parent / "data" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = results_dir / f"correlation_matrices_{ts}.npz"

    save_dict = {"pearson": pearson, "dynamic": dynamic}
    if coordinates is not None and cm.spatial_matrix is not None:
        save_dict["spatial"] = cm.spatial_matrix
    if cm.attention_matrix is not None:
        save_dict["attention"] = cm.attention_matrix

    np.savez(save_path, **save_dict)
    print(f"\n  ✓ Correlation matrices saved: {save_path}")

    # Also export to CSV for easier tracking (Excel-friendly).
    # Note: this can be large (N x N). With N~500 it's still manageable.
    labels = None
    if hasattr(dm, "data") and isinstance(dm.data, dict) and "segment_ids" in dm.data and dm.data["segment_ids"] is not None:
        try:
            labels = [str(int(s)) for s in np.array(dm.data["segment_ids"]).reshape(-1)]
        except Exception:
            labels = [str(s) for s in np.array(dm.data["segment_ids"]).reshape(-1)]
    if labels is None or len(labels) != N:
        labels = [str(i) for i in range(N)]

    exported = {}
    for name, mat in save_dict.items():
        csv_path = results_dir / f"correlation_{name}_{ts}.csv"
        pd.DataFrame(mat, index=labels, columns=labels).to_csv(csv_path, float_format="%.6f")
        exported[name] = str(csv_path)

    summary_csv = results_dir / f"correlation_summary_{ts}.csv"
    summary_rows = []
    for name, stats in summary.items():
        summary_rows.append({"matrix": name, **stats})
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)

    print("  ✓ Correlation CSV exported:")
    for name, path in exported.items():
        print(f"    - {name}: {path}")
    print(f"    - summary: {summary_csv}")

    return cm


# ═══════════════════════════════════════════════════════════
# Evaluation & Comparison
# ═══════════════════════════════════════════════════════════

def evaluate_model(predictions, targets, horizons=[1, 2, 3, 4]):
    """Evaluate model predictions across different horizons."""
    results = {}
    for h in horizons:
        if h <= predictions.shape[1]:
            pred_h = predictions[:, :h, :, :]
            target_h = targets[:, :h, :, :]
            results[h] = evaluate_all_metrics(target_h, pred_h)
    return results


def compare_and_save(all_results, output_dir):
    """Create comparison tables (like Tables 2-3 in TH-GAT paper)."""
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
        print(f"\n{'═'*90}")
        print(f"Comparison Table — Prediction Horizon: {h} day(s)")
        print(f"{'═'*90}")
        print(df.to_string(index=False))

        csv_path = output_dir / f"comparison_horizon_{h}days.csv"
        df.to_csv(csv_path, index=False)
        print(f"✓ Saved: {csv_path}")

    # Summary CSV
    summary_rows = []
    for model_name, model_results in all_results.items():
        for h, metrics in model_results.items():
            row = {"Model": model_name, "Horizon": f"{h} day(s)"}
            row.update(metrics)
            summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = output_dir / "comparison_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    # JSON
    def _serialize(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, dict):
            return {k: _serialize(v) for k, v in obj.items()}
        return obj

    json_path = output_dir / "comparison_results.json"
    with open(json_path, "w") as f:
        json.dump(_serialize(all_results), f, indent=2)

    print(f"\n✓ Summary saved: {summary_path}")
    print(f"✓ JSON saved: {json_path}")


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

def main():
    print("\n" + "═" * 70)
    print("TRAFFIC PREDICTION MODEL COMPARISON")
    print("T-GCN vs TH-GAT vs DTC-STGCN")
    print("═" * 70)
    print(f"Device: {device}")

    # ── Load Data ──
    print("\nLoading data...")
    dm = DataManager()
    dm.load_all()
    train_loader, val_loader, test_loader, adj = dm.prepare_for_training(
        batch_size=CONFIG["batch_size"], normalize=True
    )

    X_train = dm.data["X_train"]
    y_train = dm.data["y_train"]
    X_test = dm.data["X_test"]
    y_test = dm.data["y_test"]
    adj_numpy = adj.numpy()

    print(f"Train: {X_train.shape}, Val: {dm.data['X_val'].shape}, Test: {X_test.shape}")

    all_results = {}
    thgat_model_ref = None

    # Auto-skip TH-GAT if not available in the codebase
    if "TH-GAT" in CONFIG["models"] and not _THGAT_AVAILABLE:
        print("\n⚠ TH-GAT is listed in CONFIG but its code is not available. Skipping TH-GAT.\n")
        CONFIG["models"] = [m for m in CONFIG["models"] if m != "TH-GAT"]

    for model_name in CONFIG["models"]:
        print(f"\n{'─'*50}")
        print(f"  Model: {model_name}")
        print(f"{'─'*50}")

        try:
            if model_name == "T-GCN":
                _, preds, tgts = train_tgcn(train_loader, val_loader, test_loader, adj, dm, CONFIG)

            elif model_name == "TH-GAT":
                model, preds, tgts = train_thgat(train_loader, val_loader, test_loader, adj, dm, CONFIG)
                thgat_model_ref = model

            elif model_name == "DTC-STGCN-FD":
                _, preds, tgts = train_dtcstgcn(train_loader, val_loader, test_loader, adj, dm, CONFIG, "FD")

            elif model_name == "DTC-STGCN-FR":
                _, preds, tgts = train_dtcstgcn(train_loader, val_loader, test_loader, adj, dm, CONFIG, "FR")

            elif model_name in ["HA", "SVR", "ARIMA"]:
                baseline = get_baseline_model(model_name)
                baseline.fit(X_train, y_train)
                preds = baseline.predict(X_test)
                tgts = y_test

            elif model_name == "GRU":
                _, seq_len, num_nodes, input_dim = X_train.shape
                _, pred_len, _, output_dim = y_train.shape
                from utils.baselines import GRUOnly
                gru_model = GRUOnly(num_nodes, input_dim, 64, output_dim, seq_len, pred_len).to(device)
                optimizer = torch.optim.Adam(gru_model.parameters(), lr=0.001)
                criterion = torch.nn.MSELoss()
                gru_model.train()
                for _ in range(30):
                    for bx, by in train_loader:
                        bx, by = bx.to(device), by.to(device)
                        optimizer.zero_grad()
                        loss = criterion(gru_model(bx), by)
                        loss.backward()
                        optimizer.step()
                gru_model.eval()
                preds_list, tgts_list = [], []
                with torch.no_grad():
                    for bx, by in test_loader:
                        preds_list.append(gru_model(bx.to(device)).cpu().numpy())
                        tgts_list.append(by.numpy())
                preds = np.concatenate(preds_list)
                tgts = np.concatenate(tgts_list)

            else:
                print(f"Unknown model: {model_name}, skipping.")
                continue

            # Evaluate
            results = evaluate_model(preds, tgts)
            all_results[model_name] = results

            for h, metrics in results.items():
                print(f"  Horizon {h}: RMSE={metrics['RMSE']:.4f}, "
                      f"MAE={metrics['MAE']:.4f}, "
                      f"Acc={metrics['Accuracy']:.4f}, "
                      f"R²={metrics['R2']:.4f}")

        except Exception as e:
            print(f"⚠ Error with {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # ── Save comparison results ──
    results_dir = Path(src_dir).parent / "data" / "results"
    compare_and_save(all_results, results_dir)

    # ── Build correlation matrices ──
    cm = build_correlation_matrices(
        dm, adj,
        thgat_model=thgat_model_ref,
        test_loader=test_loader if thgat_model_ref else None
    )

    print("\n" + "═" * 70)
    print("✓ Comparison Complete!")
    print("═" * 70)
    print(f"\nResults saved to: {results_dir}")


if __name__ == "__main__":
    main()