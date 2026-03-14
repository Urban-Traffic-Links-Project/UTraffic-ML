"""
Main Training Script for T-GCN and Baseline Models
Train multiple models and compare results like TH-Hierarchical paper
"""

import os
from pathlib import Path
import sys
import torch
import numpy as np
import pandas as pd
import json
from datetime import datetime

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
sys.path.append(src_dir)

from models.T_GCN import TGCN, count_parameters, TGCNTrainer
from utils.data_loader import DataManager
from utils.metrics import evaluate_all_metrics, MetricsTracker
from utils.baselines import get_baseline_model
import warnings
warnings.filterwarnings('ignore')


def train_tgcn(train_loader, val_loader, test_loader, adj, config, device):
    """
    Train T-GCN model
    
    Returns:
        trained model, predictions, targets
    """
    print("\n" + "="*80)
    print("Training T-GCN Model")
    print("="*80)
    
    # Get data dimensions
    for X, y in train_loader:
        _, seq_len, num_nodes, input_dim = X.shape
        _, pred_len, _, output_dim = y.shape
        break
    
    # Create model
    model = TGCN(
        num_nodes=num_nodes,
        input_dim=input_dim,
        hidden_dim=config['hidden_dim'],
        output_dim=output_dim,
        seq_len=seq_len,
        pred_len=pred_len,
        gcn_hidden_dim=config.get('gcn_hidden_dim', 64)
    )
    
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Create trainer
    trainer = TGCNTrainer(model, adj.numpy(), config, device)
    
    # Train
    trainer.train(
        train_loader, 
        val_loader, 
        epochs=config['epochs'],
        early_stopping_patience=config.get('patience', 20),
        resume=config.get('resume', False)
    )
    
    # Test
    predictions, targets = trainer.predict(test_loader)
    
    return model, predictions, targets


def train_baseline(model_name, X_train, y_train, X_test, y_test, adj=None, device='cpu'):
    """
    Train baseline model
    
    Returns:
        predictions, targets
    """
    print(f"\n{'='*80}")
    print(f"Training {model_name} Model")
    print(f"{'='*80}")

    if model_name in ['GCN', 'GRU']:
        # PyTorch models
        _, seq_len, num_nodes, input_dim = X_train.shape
        _, pred_len, _, output_dim = y_train.shape
        
        model = get_baseline_model(
            model_name,
            num_nodes=num_nodes,
            input_dim=input_dim,
            hidden_dim=64,
            output_dim=output_dim,
            seq_len=seq_len,
            pred_len=pred_len
        )
        
        model = model.to(device)
        
        # Simple training
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.MSELoss()
        
        # Convert to tensors
        X_train_t = torch.FloatTensor(X_train).to(device)
        y_train_t = torch.FloatTensor(y_train).to(device)
        X_test_t = torch.FloatTensor(X_test).to(device)
        
        if adj is not None:
            adj_t = torch.FloatTensor(adj).to(device)
        else:
            adj_t = None
        
        # Train
        model.train()
        epochs = 50
        for epoch in range(epochs):
            optimizer.zero_grad()
            output = model(X_train_t, adj_t)
            loss = criterion(output, y_train_t)
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")
        
        # Predict
        model.eval()
        with torch.no_grad():
            predictions = model(X_test_t, adj_t).cpu().numpy()
        targets = y_test
        
    else:
        # Sklearn models (HA, ARIMA, SVR)
        model = get_baseline_model(model_name)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        targets = y_test
    
    print(f"✓ {model_name} training completed")
    
    return predictions, targets


def compare_models(results_dict, output_dir='results'):
    """
    Create comparison table like TH-Hierarchical paper Tables 2-5
    
    Args:
        results_dict: {model_name: {horizon: metrics}}
        output_dir: Directory to save results
    """
    output_dir = Path(__file__).resolve().parents[2] / "data" / "results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all horizons
    horizons = set()
    for model_results in results_dict.values():
        horizons.update(model_results.keys())
    horizons = sorted(list(horizons))
    
    # Create comparison for each horizon
    for horizon in horizons:
        print(f"\n{'='*100}")
        print(f"Comparison Table - Prediction Horizon: {horizon} day(s)")
        print(f"{'='*100}")
        
        # Collect data
        data = []
        for model_name, model_results in results_dict.items():
            if horizon in model_results:
                row = {'Model': model_name}
                row.update(model_results[horizon])
                data.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Reorder columns
        cols = ['Model', 'RMSE', 'MAE', 'Accuracy', 'R2', 'VAR']
        df = df[cols]
        
        # Format numbers
        for col in ['RMSE', 'MAE', 'Accuracy', 'R2', 'VAR']:
            df[col] = df[col].apply(lambda x: f"{x:.4f}")
        
        # Print table
        print(df.to_string(index=False))
        
        # Save to CSV
        csv_file = os.path.join(output_dir, f'comparison_horizon_{horizon}days.csv')
        df.to_csv(csv_file, index=False)
        print(f"\n✓ Saved to: {csv_file}")
    
    # Create summary table (all horizons)
    print(f"\n{'='*100}")
    print("Summary Table - All Models and Horizons")
    print(f"{'='*100}")
    
    summary_data = []
    for model_name, model_results in results_dict.items():
        for horizon, metrics in model_results.items():
            row = {'Model': model_name, 'Horizon': f'{horizon} day(s)'}
            row.update(metrics)
            summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    
    # Save summary
    summary_file = os.path.join(output_dir, 'comparison_summary.csv')
    summary_df.to_csv(summary_file, index=False)
    print(f"\n✓ Summary saved to: {summary_file}")
    
    # Save as JSON (convert numpy types to native Python types)
    def _to_serializable(obj):
        import numpy as _np
        if isinstance(obj, dict):
            return {k: _to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_to_serializable(v) for v in obj]
        if isinstance(obj, (_np.floating, _np.integer)):
            return float(obj)
        return obj
    
    json_file = os.path.join(output_dir, 'comparison_results.json')
    with open(json_file, 'w') as f:
        json.dump(_to_serializable(results_dict), f, indent=2)
    print(f"✓ JSON saved to: {json_file}")


def main():
    """Main training and comparison workflow"""
    
    # Configuration
    config = {
        'lr': 0.001,
        'weight_decay': 0.0001,
        'batch_size': 32,
        'epochs': 50,
        'patience': 20,
        'hidden_dim': 64,
        'gcn_hidden_dim': 64,
        'resume': False  # Set to True to resume from checkpoint
    }
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    # Load data
    print("\n" + "="*80)
    print("Loading Data")
    print("="*80)
    
    dm = DataManager()
    dm.load_all()
    
    train_loader, val_loader, test_loader, adj = dm.prepare_for_training(
        batch_size=config['batch_size'],
        normalize=True
    )
    
    adj_numpy = adj.numpy()
    
    # Get numpy arrays for baseline models
    X_train_np = dm.data['X_train']
    y_train_np = dm.data['y_train']
    X_test_np = dm.data['X_test']
    y_test_np = dm.data['y_test']
    
    print(f"\nData loaded successfully!")
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Models to compare
    models_to_compare = ['T-GCN', 'HA', 'SVR', 'GRU', 'GCN']
    
    # Storage for results
    results = {}
    
    # Train each model
    for model_name in models_to_compare:
        try:
            if model_name == 'T-GCN':
                model, predictions, targets = train_tgcn(
                    train_loader, val_loader, test_loader, adj, config, device
                )
            else:
                predictions, targets = train_baseline(
                    model_name, X_train_np, y_train_np, X_test_np, y_test_np,
                    adj=adj_numpy, device=device
                )
            
            # Evaluate for different horizons
            # Assuming predictions: (samples, pred_len, nodes, features)
            horizons = [1, 2, 3, 4]  # days
            results[model_name] = {}
            
            for horizon in horizons:
                if horizon <= predictions.shape[1]:
                    # Take only first 'horizon' time steps
                    pred_h = predictions[:, :horizon, :, :]
                    target_h = targets[:, :horizon, :, :]
                    
                    # Calculate metrics
                    metrics = evaluate_all_metrics(target_h, pred_h)
                    results[model_name][horizon] = metrics
                    
                    print(f"\n{model_name} - Horizon {horizon} day(s):")
                    for metric, value in metrics.items():
                        print(f"  {metric}: {value:.4f}")
        
        except Exception as e:
            print(f"\n⚠ Error training {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Compare results
    print("\n" + "="*80)
    print("MODEL COMPARISON")
    print("="*80)
    
    compare_models(results, output_dir='results')
    
    print("\n" + "="*80)
    print("✓ Training and Comparison Completed!")
    print("="*80)


if __name__ == "__main__":
    main()