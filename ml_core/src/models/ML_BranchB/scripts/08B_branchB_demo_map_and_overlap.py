import numpy as np
import pandas as pd
import folium
from pathlib import Path


# =========================
# LOAD
# =========================
def load_split(common_dir, split):
    d = common_dir / split
    return {
        "G": np.load(d / "G_weight_series.npy", mmap_mode="r"),
        "seg": np.load(d / "segment_ids.npy"),
    }


# =========================
# TOP-K EDGES
# =========================
def get_topk_edges(G, k=10):
    edges = []

    N = G.shape[0]

    for j in range(N):
        row = G[j].copy()
        row[j] = 0

        idx = np.argpartition(np.abs(row), -k)[-k:]

        for i in idx:
            if row[i] != 0:
                edges.append((i, j, row[i]))

    return edges


# =========================
# OVERLAP
# =========================
def compute_overlap(edges_true, edges_pred):
    set_true = {(i, j) for i, j, _ in edges_true}
    set_pred = {(i, j) for i, j, _ in edges_pred}

    inter = len(set_true & set_pred)
    union = len(set_true)

    return inter / (union + 1e-8)


# =========================
# MAP
# =========================
def plot_map(edges, save_path):
    m = folium.Map(location=[10.77, 106.66], zoom_start=12)

    for i, j, w in edges:
        folium.CircleMarker(
            location=[10.77 + i * 1e-4, 106.66 + j * 1e-4],
            radius=2,
            color="red" if w > 0 else "blue"
        ).add_to(m)

    m.save(save_path)


# =========================
# MAIN
# =========================
def main():
    root = Path("ml_core/src/data_processing/outputs/branchB/osm_edge_gt_like_branchA")

    val = load_split(root, "val")

    t = 100  # demo time

    G_true = val["G"][t]
    G_pred = val["G"][t-1]  # persistence demo

    edges_true = get_topk_edges(G_true, k=20)
    edges_pred = get_topk_edges(G_pred, k=20)

    overlap = compute_overlap(edges_true, edges_pred)

    print("Overlap ratio:", overlap)

    plot_map(edges_true, "true_map.html")
    plot_map(edges_pred, "pred_map.html")


if __name__ == "__main__":
    main()