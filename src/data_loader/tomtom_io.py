# File: src/data/tomtom_io.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import pickle

def load_npz(npz_path: str) -> Dict[str, np.ndarray]:
    data = np.load(npz_path, allow_pickle=True)
    return {k: data[k] for k in data.files}

def get_values_3d(npz: Dict[str, np.ndarray], values_key: str = "values") -> np.ndarray:
    X = npz[values_key].astype(np.float32)
    # expect [T,N] or [T,N,F]
    if X.ndim == 2:
        X = X[..., None]  # [T,N,1]
    if X.ndim != 3:
        raise ValueError(f"values must be [T,N] or [T,N,F], got {X.shape}")
    return X

def load_zscore_stats(pkl_path: str) -> Dict:
    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)
    # your notebook saved {"mu": Series, "sigma": Series}
    if isinstance(obj, dict) and "mu" in obj and "sigma" in obj:
        return {"mu": obj["mu"], "sigma": obj["sigma"]}
    if isinstance(obj, dict) and "mean" in obj and "std" in obj:
        return {"mu": obj["mean"], "sigma": obj["std"]}
    raise ValueError("Unknown zscore stats format.")

def load_segment_index(csv_path: str) -> Tuple[Dict[int,int], Dict[int,int]]:
    df = pd.read_csv(csv_path)
    if not {"idx","segment_id"}.issubset(df.columns):
        raise ValueError(f"segment_index.csv requires idx,segment_id. Got {df.columns.tolist()}")
    segid_to_idx = {int(r.segment_id): int(r.idx) for r in df.itertuples(index=False)}
    idx_to_segid = {int(r.idx): int(r.segment_id) for r in df.itertuples(index=False)}
    return segid_to_idx, idx_to_segid

def inverse_z(z: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    # z: [...,N]
    return z * sigma + mu
