"""
Evaluation Metrics for Traffic Prediction
Based on T-GCN paper (Zhao et al., 2019) - Equations 8-12
"""

import numpy as np
import torch
from sklearn.metrics import r2_score, explained_variance_score

def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x

def calculate_rmse(y_true, y_pred):
    """
    Root Mean Squared Error (Equation 8)
    
    RMSE = sqrt(1/n * sum((Y_t - Y_pred)^2))
    """
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def calculate_mae(y_true, y_pred):
    """
    Mean Absolute Error (Equation 9)
    
    MAE = 1/n * sum(|Y_t - Y_pred|)
    """
    return np.mean(np.abs(y_true - y_pred))

    
def calculate_accuracy(y_true, y_pred, eps=1e-8):
    """
    Accuracy (Equation 10)
    
    Accuracy = 1 - ||Y - Y_pred||_F / ||Y||_F
    
    where ||·||_F is the Frobenius norm
    """
    y_true = to_numpy(y_true)
    y_pred = to_numpy(y_pred)

    diff = y_true - y_pred

    numerator = np.linalg.norm(diff.ravel(), 2)
    denominator = np.linalg.norm(y_true.ravel(), 2)

    return 1 - numerator / (denominator + eps)



def calculate_r2(y_true, y_pred):
    """
    Coefficient of Determination (Equation 11)
    
    R² = 1 - sum((Y_t - Y_pred)^2) / sum((Y_t - Y_mean)^2)
    """
    # Flatten arrays
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    # sklearn implementation
    return r2_score(y_true_flat, y_pred_flat)


def calculate_var(y_true, y_pred):
    """
    Explained Variance Score (Equation 12)
    
    VAR = 1 - Var{Y - Y_pred} / Var{Y}
    """
    # Flatten arrays
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    # sklearn implementation
    return explained_variance_score(y_true_flat, y_pred_flat)


def evaluate_all_metrics(y_true, y_pred):
    """
    Calculate all 5 metrics from T-GCN paper
    
    Args:
        y_true: Ground truth values (numpy array)
        y_pred: Predicted values (numpy array)
    
    Returns:
        dict: Dictionary containing all metrics
    """
    metrics = {
        'RMSE': calculate_rmse(y_true, y_pred),
        'MAE': calculate_mae(y_true, y_pred),
        'Accuracy': calculate_accuracy(y_true, y_pred),
        'R2': calculate_r2(y_true, y_pred),
        'VAR': calculate_var(y_true, y_pred)
    }
    
    return metrics


def print_metrics(metrics, model_name='Model'):
    """Pretty print metrics"""
    print(f"\n{'='*60}")
    print(f"{model_name} Evaluation Metrics")
    print(f"{'='*60}")
    print(f"RMSE      : {metrics['RMSE']:.4f}")
    print(f"MAE       : {metrics['MAE']:.4f}")
    print(f"Accuracy  : {metrics['Accuracy']:.4f}")
    print(f"R²        : {metrics['R2']:.4f}")
    print(f"VAR       : {metrics['VAR']:.4f}")
    print(f"{'='*60}\n")


class MetricsTracker:
    """Track metrics across different prediction horizons"""
    
    def __init__(self, horizons=[1, 2, 3, 4]):
        """
        Args:
            horizons: List of prediction horizons (e.g., days)
        """
        self.horizons = horizons
        self.results = {h: {} for h in horizons}
    
    def add_result(self, horizon, metrics):
        """Add metrics for a specific horizon"""
        if horizon in self.horizons:
            self.results[horizon] = metrics
    
    def get_comparison_table(self):
        """
        Get comparison table like Table 1 in T-GCN paper
        
        Returns:
            pandas DataFrame with metrics for each horizon
        """
        import pandas as pd
        
        data = []
        for horizon in self.horizons:
            if horizon in self.results and self.results[horizon]:
                row = {'Horizon': f'{horizon * 15} min'}
                row.update(self.results[horizon])
                data.append(row)
        
        df = pd.DataFrame(data)
        return df
        
    def print_comparison(self, model_name='T-GCN'):
        """Print comparison table"""
        df = self.get_comparison_table()
        print(f"\n{model_name} Results Across Different Horizons")
        print("="*80)
        print(df.to_string(index=False))
        print("="*80 + "\n")


if __name__ == "__main__":
    # Test metrics
    np.random.seed(42)
    
    # Simulate predictions
    y_true = np.random.rand(100, 10, 1) * 50 + 20  # Speed 20-70
    y_pred = y_true + np.random.randn(100, 10, 1) * 3  # Add noise
    
    # Calculate metrics
    metrics = evaluate_all_metrics(y_true, y_pred)
    print_metrics(metrics, 'Test Model')
    
    # Test tracker
    tracker = MetricsTracker(horizons=[1, 2, 3, 4])
    
    for horizon in [1, 2, 3, 4]:
        # Simulate different performance for different horizons
        noise_level = horizon * 1.5
        y_pred_h = y_true + np.random.randn(100, 10, 1) * noise_level
        metrics_h = evaluate_all_metrics(y_true, y_pred_h)
        tracker.add_result(horizon, metrics_h)
    
    tracker.print_comparison('Test T-GCN')