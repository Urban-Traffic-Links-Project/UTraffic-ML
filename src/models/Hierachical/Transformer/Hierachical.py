# -*- coding: utf-8 -*-
"""
Hierachical.py
So sánh Q modularity giữa Louvain/Leiden/Infomap/Spectral trên segment graph.

Input (mặc định):
  data/processed/tomtom_stats/edges.csv          (segment_u, segment_v)
  data/processed/tomtom_stats/segment_index.csv  (idx, segment_id)

Output:
  output/Hierachical_Compare/result/
    - summary.csv
    - best_partition.npz (labels, Tran, A_R, method, Q, gamma, seed)
    - partitions/<method>_gammaX_seedY_labels.npy
    - partitions/<method>_gammaX_seedY_meta.json
"""

import os
import json
import math
import random
import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import networkx as nx


# =========================
# Utils: modularity + export
# =========================

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)

def labels_to_tran(labels: np.ndarray) -> np.ndarray:
    """labels: [N] -> Tran [N,R] onehot float32"""
    N = labels.shape[0]
    R = int(labels.max()) + 1 if N > 0 else 0
    Tran = np.zeros((N, R), dtype=np.float32)
    Tran[np.arange(N), labels] = 1.0
    return Tran

def region_adjacency_from_edges(N: int, edges_idx: np.ndarray, labels: np.ndarray, binarize=True) -> np.ndarray:
    """
    edges_idx: [E,2] undirected edge list in idx space
    Return A_R [R,R]
    """
    R = int(labels.max()) + 1
    A_R = np.zeros((R, R), dtype=np.float32)
    for u, v in edges_idx:
        ru, rv = labels[u], labels[v]
        if ru != rv:
            A_R[ru, rv] += 1.0
            A_R[rv, ru] += 1.0
    np.fill_diagonal(A_R, 0.0)
    if binarize:
        A_R = (A_R > 0).astype(np.float32)
    return A_R

def modularity_Q_nx(G: nx.Graph, labels: np.ndarray) -> float:
    """
    Compute modularity using networkx community modularity.
    labels: [N] community id
    """
    # Convert labels -> list of sets
    comms: Dict[int, List[int]] = {}
    for i, c in enumerate(labels.tolist()):
        comms.setdefault(int(c), []).append(i)
    communities = [set(v) for v in comms.values()]
    if len(communities) == 0:
        return 0.0
    try:
        from networkx.algorithms.community.quality import modularity
        return float(modularity(G, communities, weight="weight"))
    except Exception:
        # fallback: 0 if something weird
        return 0.0


# =========================
# Load graph from your CSVs
# =========================

def load_segment_graph(edges_csv: str, segment_index_csv: str, undirected: bool = True) -> Tuple[nx.Graph, int, np.ndarray]:
    """
    edges.csv: columns (segment_u, segment_v) are segment_id (NOT idx)
    segment_index.csv: columns (idx, segment_id) map segment_id -> idx

    Returns:
      G (nx.Graph), N, edges_idx [E,2]
    """
    edges_df = pd.read_csv(edges_csv)
    idx_df = pd.read_csv(segment_index_csv)

    # Build map segment_id -> idx
    seg2idx = dict(zip(idx_df["segment_id"].astype(np.int64).tolist(),
                       idx_df["idx"].astype(np.int64).tolist()))
    N = int(idx_df["idx"].max()) + 1

    # Map edges
    su = edges_df["segment_u"].astype(np.int64).map(seg2idx)
    sv = edges_df["segment_v"].astype(np.int64).map(seg2idx)
    mapped = pd.DataFrame({"u": su, "v": sv}).dropna()
    u = mapped["u"].astype(np.int64).to_numpy()
    v = mapped["v"].astype(np.int64).to_numpy()

    edges_idx = np.stack([u, v], axis=1)  # [E,2]

    # Build undirected graph
    G = nx.Graph() if undirected else nx.DiGraph()
    G.add_nodes_from(range(N))
    for a, b in edges_idx:
        if a == b:
            continue
        if G.has_edge(int(a), int(b)):
            # weight++ if repeated
            G[int(a)][int(b)]["weight"] += 1.0
        else:
            G.add_edge(int(a), int(b), weight=1.0)

    if undirected and isinstance(G, nx.Graph):
        # already undirected
        pass

    return G, N, edges_idx


# =========================
# Methods
# =========================

@dataclass
class PartitionResult:
    method: str
    labels: np.ndarray        # [N]
    Q: float
    gamma: Optional[float] = None
    seed: Optional[int] = None
    extra: Optional[dict] = None


def run_louvain(G: nx.Graph, seed: int, gamma: float) -> Optional[np.ndarray]:
    """
    python-louvain package: pip install python-louvain
    """
    try:
        import community as community_louvain  # python-louvain
    except Exception:
        return None

    part = community_louvain.best_partition(G, random_state=seed, resolution=gamma, weight="weight")
    labels = np.zeros((G.number_of_nodes(),), dtype=np.int64)
    for n, c in part.items():
        labels[int(n)] = int(c)

    # Re-index community ids to 0..R-1
    _, new_labels = np.unique(labels, return_inverse=True)
    return new_labels.astype(np.int64)


def run_leiden(G: nx.Graph, seed: int, gamma: float) -> Optional[np.ndarray]:
    """
    Leiden via igraph + leidenalg:
      pip install igraph leidenalg
    """
    try:
        import igraph as ig
        import leidenalg
    except Exception:
        return None

    # Convert nx -> igraph
    # igraph expects edges list
    N = G.number_of_nodes()
    edges = list(G.edges())
    weights = [G[u][v].get("weight", 1.0) for u, v in edges]

    g = ig.Graph(n=N, edges=edges, directed=False)
    g.es["weight"] = weights

    # RBConfigurationVertexPartition supports resolution_parameter
    part = leidenalg.find_partition(
        g,
        leidenalg.RBConfigurationVertexPartition,
        weights="weight",
        resolution_parameter=gamma,
        seed=seed
    )
    labels = np.array(part.membership, dtype=np.int64)
    _, new_labels = np.unique(labels, return_inverse=True)
    return new_labels.astype(np.int64)


def run_infomap(G: nx.Graph, seed: int) -> Optional[np.ndarray]:
    """
    Infomap:
      pip install infomap
    Note: Infomap is more natural for directed/flow graphs; still can run undirected.
    """
    try:
        from infomap import Infomap
    except Exception:
        return None

    im = Infomap(f"--seed {seed} --two-level")  # keep simple
    # Add links
    for u, v, d in G.edges(data=True):
        w = d.get("weight", 1.0)
        im.addLink(int(u), int(v), float(w))
        im.addLink(int(v), int(u), float(w))

    im.run()

    labels = np.zeros((G.number_of_nodes(),), dtype=np.int64)
    for node in im.tree:
        if node.isLeaf:
            labels[int(node.physicalId)] = int(node.moduleIndex())

    _, new_labels = np.unique(labels, return_inverse=True)
    return new_labels.astype(np.int64)


def run_spectral(G: nx.Graph, n_clusters: int, seed: int) -> Optional[np.ndarray]:
    """
    Spectral clustering on adjacency (normalized cut style approximation).
      pip install scikit-learn
    """
    try:
        from sklearn.cluster import SpectralClustering
    except Exception:
        return None

    N = G.number_of_nodes()
    # adjacency matrix (dense for simplicity)
    A = np.zeros((N, N), dtype=np.float32)
    for u, v, d in G.edges(data=True):
        w = float(d.get("weight", 1.0))
        A[int(u), int(v)] = w
        A[int(v), int(u)] = w

    # SpectralClustering expects affinity matrix
    sc = SpectralClustering(
        n_clusters=n_clusters,
        affinity="precomputed",
        assign_labels="kmeans",
        random_state=seed
    )
    labels = sc.fit_predict(A).astype(np.int64)
    _, new_labels = np.unique(labels, return_inverse=True)
    return new_labels.astype(np.int64)


# =========================
# Main compare
# =========================

def compare_partitions(
    G: nx.Graph,
    edges_idx: np.ndarray,
    out_dir: str,
    seeds: List[int],
    gammas: List[float],
    spectral_clusters: List[int],
    run_methods: List[str]
) -> Tuple[List[PartitionResult], PartitionResult]:

    ensure_dir(out_dir)
    part_dir = os.path.join(out_dir, "partitions")
    ensure_dir(part_dir)

    all_results: List[PartitionResult] = []

    N = G.number_of_nodes()

    def save_one(res: PartitionResult):
        tag = res.method
        if res.gamma is not None:
            tag += f"_gamma{res.gamma}"
        if res.seed is not None:
            tag += f"_seed{res.seed}"
        np.save(os.path.join(part_dir, f"{tag}_labels.npy"), res.labels)

        meta = {
            "method": res.method,
            "Q": res.Q,
            "gamma": res.gamma,
            "seed": res.seed,
            "R": int(res.labels.max()) + 1,
            "N": int(N),
            "extra": res.extra or {}
        }
        with open(os.path.join(part_dir, f"{tag}_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    for seed in seeds:
        set_seed(seed)

        # Louvain/Leiden sweep gamma
        if "louvain" in run_methods:
            for gamma in gammas:
                labels = run_louvain(G, seed=seed, gamma=gamma)
                if labels is not None:
                    Q = modularity_Q_nx(G, labels)
                    res = PartitionResult("louvain", labels, Q, gamma=gamma, seed=seed)
                    all_results.append(res)
                    save_one(res)

        if "leiden" in run_methods:
            for gamma in gammas:
                labels = run_leiden(G, seed=seed, gamma=gamma)
                if labels is not None:
                    Q = modularity_Q_nx(G, labels)
                    res = PartitionResult("leiden", labels, Q, gamma=gamma, seed=seed)
                    all_results.append(res)
                    save_one(res)

        # Infomap (no gamma)
        if "infomap" in run_methods:
            labels = run_infomap(G, seed=seed)
            if labels is not None:
                Q = modularity_Q_nx(G, labels)
                res = PartitionResult("infomap", labels, Q, gamma=None, seed=seed)
                all_results.append(res)
                save_one(res)

        # Spectral needs n_clusters
        if "spectral" in run_methods:
            for k in spectral_clusters:
                labels = run_spectral(G, n_clusters=k, seed=seed)
                if labels is not None:
                    Q = modularity_Q_nx(G, labels)
                    res = PartitionResult("spectral", labels, Q, gamma=None, seed=seed, extra={"k": int(k)})
                    all_results.append(res)
                    save_one(res)

    if len(all_results) == 0:
        raise RuntimeError(
            "Không chạy được method nào. "
            "Bạn cần cài ít nhất 1 trong các package: python-louvain, igraph+leidenalg, infomap, scikit-learn."
        )

    best = max(all_results, key=lambda r: r.Q)
    return all_results, best


def export_best(out_dir: str, G: nx.Graph, edges_idx: np.ndarray, best: PartitionResult):
    """
    Export:
      - summary.csv
      - best_partition.npz: labels, Tran, A_R, method, Q, gamma, seed
    """
    labels = best.labels
    Tran = labels_to_tran(labels)  # [N,R]
    A_R = region_adjacency_from_edges(G.number_of_nodes(), edges_idx, labels, binarize=True)  # [R,R]

    payload = {
        "labels": labels.astype(np.int64),
        "Tran": Tran.astype(np.float32),
        "A_R": A_R.astype(np.float32),
        "method": np.array([best.method], dtype=object),
        "Q": np.array([best.Q], dtype=np.float32),
        "gamma": np.array([best.gamma if best.gamma is not None else -1.0], dtype=np.float32),
        "seed": np.array([best.seed if best.seed is not None else -1], dtype=np.int64),
    }
    np.savez(os.path.join(out_dir, "best_partition.npz"), **payload)

    # Also save friendly JSON
    best_json = {
        "method": best.method,
        "Q": best.Q,
        "gamma": best.gamma,
        "seed": best.seed,
        "N": int(G.number_of_nodes()),
        "R": int(labels.max()) + 1
    }
    with open(os.path.join(out_dir, "best_partition.json"), "w", encoding="utf-8") as f:
        json.dump(best_json, f, ensure_ascii=False, indent=2)

    # quick check prints
    print(f"[BEST] method={best.method} Q={best.Q:.6f} gamma={best.gamma} seed={best.seed} R={int(labels.max())+1}")
    print(f"Saved: {os.path.join(out_dir, 'best_partition.npz')}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default=".",
                        help="Project root (Urban-Traffic-Links). Default='.' if you run from root.")
    parser.add_argument("--edges_csv", type=str, default="data/processed/tomtom_stats/edges.csv")
    parser.add_argument("--segment_index_csv", type=str, default="data/processed/tomtom_stats/segment_index.csv")
    parser.add_argument("--out_dir", type=str, default="output/Hierachical_Compare/result")
    parser.add_argument("--methods", type=str, default="louvain,leiden,infomap,spectral",
                        help="Comma-separated: louvain,leiden,infomap,spectral")
    parser.add_argument("--seeds", type=str, default="0,1,2",
                        help="Comma-separated seeds, e.g. 0,1,2")
    parser.add_argument("--gammas", type=str, default="0.5,1.0,1.5,2.0",
                        help="Resolution sweep for Louvain/Leiden")
    parser.add_argument("--spectral_k", type=str, default="10,20,30",
                        help="Spectral clustering k list")
    args = parser.parse_args()

    root = args.root
    edges_csv = os.path.join(root, args.edges_csv)
    segment_index_csv = os.path.join(root, args.segment_index_csv)
    out_dir = os.path.join(root, args.out_dir)

    run_methods = [m.strip().lower() for m in args.methods.split(",") if m.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    gammas = [float(x) for x in args.gammas.split(",") if x.strip()]
    spectral_k = [int(x) for x in args.spectral_k.split(",") if x.strip()]

    ensure_dir(out_dir)

    print("Loading graph...")
    G, N, edges_idx = load_segment_graph(edges_csv, segment_index_csv, undirected=True)
    print(f"Graph: N={N}, E={G.number_of_edges()}")

    print("Running community detection + modularity compare...")
    all_results, best = compare_partitions(
        G=G,
        edges_idx=edges_idx,
        out_dir=out_dir,
        seeds=seeds,
        gammas=gammas,
        spectral_clusters=spectral_k,
        run_methods=run_methods
    )

    # Export summary.csv
    rows = []
    for r in all_results:
        rows.append({
            "method": r.method,
            "Q": r.Q,
            "gamma": r.gamma if r.gamma is not None else "",
            "seed": r.seed if r.seed is not None else "",
            "R": int(r.labels.max()) + 1,
            "extra": json.dumps(r.extra or {}, ensure_ascii=False)
        })
    summary = pd.DataFrame(rows).sort_values(by="Q", ascending=False)
    summary_path = os.path.join(out_dir, "summary.csv")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"Saved: {summary_path}")

    # Export best partition artifacts
    export_best(out_dir, G, edges_idx, best)

    print("DONE.")


if __name__ == "__main__":
    main()
