import numpy as np

class StandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, x: np.ndarray):
        # x: [T, N, F] or flattened
        self.mean = x.mean(axis=0, keepdims=True)
        self.std = x.std(axis=0, keepdims=True) + 1e-8
        return self

    def transform(self, x: np.ndarray):
        return (x - self.mean) / self.std

    def inverse_transform(self, x: np.ndarray):
        return x * self.std + self.mean
