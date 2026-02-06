# File: src/evaluate/metrics.py
import numpy as np

def mae(yhat, y):
    return float(np.mean(np.abs(yhat - y)))

def rmse(yhat, y):
    return float(np.sqrt(np.mean((yhat - y) ** 2)))

def mape(yhat, y, eps=1e-6):
    denom = np.maximum(np.abs(y), eps)
    return float(np.mean(np.abs((yhat - y) / denom)) * 100.0)

def horizon_metrics(yhat, y):
    """
    yhat,y: [B,H,N]
    returns overall + per_horizon
    """
    B,H,N = y.shape
    out = {"overall": {}, "per_horizon": []}
    out["overall"]["MAE"] = mae(yhat, y)
    out["overall"]["RMSE"] = rmse(yhat, y)
    out["overall"]["MAPE"] = mape(yhat, y)

    for h in range(H):
        out["per_horizon"].append({
            "h": h+1,
            "MAE": mae(yhat[:,h,:], y[:,h,:]),
            "RMSE": rmse(yhat[:,h,:], y[:,h,:]),
            "MAPE": mape(yhat[:,h,:], y[:,h,:]),
        })
    return out
