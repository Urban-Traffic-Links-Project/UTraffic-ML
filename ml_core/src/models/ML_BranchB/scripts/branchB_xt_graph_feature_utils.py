import numpy as np

EPS = 1e-8


def topk_filter_G(G: np.ndarray, k: int = 4, normalize: bool = True) -> np.ndarray:
    """
    Keep top-k strongest edges per row (target node).
    G shape: (N, N) = target x source
    """
    G = np.asarray(G, dtype=np.float32)
    N = G.shape[0]

    G_new = np.zeros_like(G, dtype=np.float32)

    for j in range(N):
        row = G[j].copy()

        # bỏ self-loop
        row[j] = 0.0

        if np.all(row == 0):
            continue

        # lấy top-k theo |weight|
        idx = np.argpartition(np.abs(row), -k)[-k:]

        G_new[j, idx] = row[idx]

        if normalize:
            norm = np.sum(np.abs(G_new[j])) + EPS
            G_new[j] = G_new[j] / norm

    return G_new


def build_graph_signal_topk(G_used: np.ndarray, x_t: np.ndarray, k: int = 4, gamma: float = 1.0):
    """
    gx = gamma * (TopK(G) @ x_t)
    """
    G_filtered = topk_filter_G(G_used, k=k, normalize=True)
    gx = G_filtered @ x_t
    return gamma * gx