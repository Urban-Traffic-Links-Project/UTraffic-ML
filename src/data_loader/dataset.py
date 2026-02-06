# File: src/data/dataset.py
from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import Dataset

class TrafficWindowDataset(Dataset):
    """
    X: [T,N,F]
    return:
      xb: [P,N,F]
      yb: [H,N]   (target feature = 0 by default)
    trong đó
    T – Time steps (tổng thời gian)
    N – Number of nodes (số tuyến )
    F – Number of features (số đặc trưng)
    P – Past window length (độ dài cửa sổ quá khứ)
    H – Horizon (độ dài dự đoán tương lai)
    """
    def __init__(self, X: np.ndarray, P: int, H: int, target_feature_idx: int = 0):
        assert X.ndim == 3
        self.X = X.astype(np.float32)
        self.P = P
        self.H = H
        self.tidx = target_feature_idx
        self.T = X.shape[0]
        self.max_i = self.T - (P + H) + 1
        if self.max_i <= 0:
            raise ValueError(f"Not enough timesteps: T={self.T}, need >= P+H={P+H}")

    def __len__(self):
        return self.max_i

    def __getitem__(self, i: int):
        x = self.X[i:i+self.P]                 # [P,N,F]
        y = self.X[i+self.P:i+self.P+self.H]   # [H,N,F]
        y = y[..., self.tidx]                  # [H,N]
        return torch.from_numpy(x), torch.from_numpy(y)
