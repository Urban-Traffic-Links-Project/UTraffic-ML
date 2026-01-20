# -*- coding: utf-8 -*-
"""
Hierachical_Compare.py
Compare modularity Q between Louvain/Leiden/Infomap/Spectral (+ optional DBSCAN/HDBSCAN)
on segment graph. Export BEST partition + SOFT Tran by distance-to-centroid.

Inputs (default):
  data/processed/tomtom_stats/edges.csv          (segment_u, segment_v)  # segment_id
  data/processed/tomtom_stats/segment_index.csv  (idx, segment_id)       # map segment_id->idx
  data/processed/tomtom_stats/segments.csv       (segment_id, node_u, node_v, ...)  # endpoint nodes
  data/processed/tomtom_stats/nodes.csv          (node_id, lat, lon, ...)

Outputs (default):
  src/models/Hierachical/Transformer/output_compaer_Hierachical/
    - summary.csv
    - best_partition.npz: labels_hard, Tran_soft, A_R, method, Q, gamma, seed, R, N, tau_km, topk
    - best_partition.json
    - partitions/<tag>_labels.npy
    - partitions/<tag>_meta.json

Run (from project root Urban-Traffic-Links):
  python src/models/Hierachical/Transformer/Hierachical_Compare.py
"""

import os
import json
import random
import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import networkx as nx


# =========================
# Utils
# =========================

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)

def modularity_Q_nx(G: nx.Graph, labels: np.ndarray) -> float:
    """Compute modularity using networkx. labels: [N]"""
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
        return 0.0

def reindex_labels(labels: np.ndarray) -> np.ndarray:
    """Map arbitrary community ids -> 0..R-1"""
    _, new_labels = np.unique(labels, return_inverse=True)
    return new_labels.astype(np.int64)

# ---- geo distance ----
def haversine_km(lat1, lon1, lat2, lon2):
    """
    Vectorized haversine distance in km.
    lat1, lon1: [...], lat2, lon2: [...]
    Broadcastable.
    """
    R = 6371.0
    lat1 = np.deg2rad(lat1); lon1 = np.deg2rad(lon1)
    lat2 = np.deg2rad(lat2); lon2 = np.deg2rad(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2.0)**2
    c = 2*np.arcsin(np.sqrt(a))
    return R*c


# =========================
# Load graph + coords
# =========================

def load_segment_graph(edges_csv: str, segment_index_csv: str, undirected: bool = True) -> Tuple[nx.Graph, int, np.ndarray]:
    """
    edges.csv: (segment_u, segment_v) are segment_id
    segment_index.csv: (idx, segment_id) map segment_id -> idx
    Returns: G, N, edges_idx [E,2] in idx space
    """
    edges_df = pd.read_csv(edges_csv)
    idx_df = pd.read_csv(segment_index_csv)

    seg2idx = dict(zip(
        idx_df["segment_id"].astype(np.int64).tolist(),
        idx_df["idx"].astype(np.int64).tolist()
    ))

    idx_min = int(idx_df["idx"].min())
    idx_max = int(idx_df["idx"].max())
    N = idx_max + 1
    if idx_min != 0 or idx_max != len(idx_df) - 1:
        print(f"[WARN] idx không liên tục hoặc không bắt đầu từ 0. Dùng N=max(idx)+1={N} (có thể thừa node rỗng).")

    su = edges_df["segment_u"].astype(np.int64).map(seg2idx)
    sv = edges_df["segment_v"].astype(np.int64).map(seg2idx)
    mapped = pd.DataFrame({"u": su, "v": sv}).dropna()
    print(f"Mapped edges: {len(mapped)} / raw edges: {len(edges_df)}")

    u = mapped["u"].astype(np.int64).to_numpy()
    v = mapped["v"].astype(np.int64).to_numpy()
    edges_idx = np.stack([u, v], axis=1)

    G = nx.Graph() if undirected else nx.DiGraph()
    G.add_nodes_from(range(N))

    for a, b in edges_idx:
        a, b = int(a), int(b)
        if a == b:
            continue
        if G.has_edge(a, b):
            G[a][b]["weight"] += 1.0
        else:
            G.add_edge(a, b, weight=1.0)

    return G, N, edges_idx


def load_segment_latlon(
    segments_csv: str,
    nodes_csv: str,
    segment_index_csv: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build segment centroid coords (lat, lon) for each segment idx [0..N-1].
    Need:
      segments.csv: segment_id, node_u, node_v
      nodes.csv: node_id, lat, lon
      segment_index.csv: idx, segment_id

    Returns:
      seg_lat: [N]
      seg_lon: [N]
    """
    seg_df = pd.read_csv(segments_csv)
    node_df = pd.read_csv(nodes_csv)
    idx_df = pd.read_csv(segment_index_csv)

    seg2idx = dict(zip(
        idx_df["segment_id"].astype(np.int64).tolist(),
        idx_df["idx"].astype(np.int64).tolist()
    ))
    N = int(idx_df["idx"].max()) + 1

    node_lat = dict(zip(node_df["node_id"].astype(np.int64).tolist(),
                        node_df["lat"].astype(np.float64).tolist()))
    node_lon = dict(zip(node_df["node_id"].astype(np.int64).tolist(),
                        node_df["lon"].astype(np.float64).tolist()))

    seg_lat = np.full((N,), np.nan, dtype=np.float64)
    seg_lon = np.full((N,), np.nan, dtype=np.float64)

    # only keep rows with segment_id in mapping
    seg_df = seg_df.copy()
    seg_df["segment_id"] = seg_df["segment_id"].astype(np.int64)
    seg_df["idx"] = seg_df["segment_id"].map(seg2idx)
    seg_df = seg_df.dropna(subset=["idx"])
    seg_df["idx"] = seg_df["idx"].astype(np.int64)

    missing_node = 0
    for _, row in seg_df.iterrows():
        i = int(row["idx"])
        nu = int(row["node_u"])
        nv = int(row["node_v"])
        if (nu not in node_lat) or (nv not in node_lat):
            missing_node += 1
            continue
        lat = 0.5*(node_lat[nu] + node_lat[nv])
        lon = 0.5*(node_lon[nu] + node_lon[nv])
        seg_lat[i] = lat
        seg_lon[i] = lon

    nan_cnt = int(np.isnan(seg_lat).sum())
    if nan_cnt > 0:
        print(f"[WARN] segment centroid missing: {nan_cnt}/{N} (thiếu node_u/node_v trong nodes.csv?).")
    if missing_node > 0:
        print(f"[WARN] missing endpoint nodes in nodes.csv rows: {missing_node}")

    # fill NaN (fallback): use global mean so code still runs
    mean_lat = np.nanmean(seg_lat)
    mean_lon = np.nanmean(seg_lon)
    seg_lat = np.where(np.isnan(seg_lat), mean_lat, seg_lat)
    seg_lon = np.where(np.isnan(seg_lon), mean_lon, seg_lon)

    return seg_lat.astype(np.float64), seg_lon.astype(np.float64)


# =========================
# Methods
# =========================

@dataclass
class PartitionResult:
    method: str
    labels: np.ndarray        # [N] hard
    Q: float
    gamma: Optional[float] = None
    seed: Optional[int] = None
    extra: Optional[dict] = None


def run_louvain(G: nx.Graph, seed: int, gamma: float) -> Optional[np.ndarray]:
    """pip install python-louvain"""
    try:
        import community.community_louvain as community_louvain
    except Exception:
        return None

    part = community_louvain.best_partition(G, random_state=seed, resolution=gamma, weight="weight")
    labels = np.zeros((G.number_of_nodes(),), dtype=np.int64)
    for n, c in part.items():
        labels[int(n)] = int(c)
    return reindex_labels(labels)


def run_leiden(G: nx.Graph, seed: int, gamma: float) -> Optional[np.ndarray]:
    """pip install igraph leidenalg"""
    try:
        import igraph as ig
        import leidenalg
    except Exception:
        return None

    N = G.number_of_nodes()
    edges = list(G.edges())
    weights = [float(G[u][v].get("weight", 1.0)) for u, v in edges]

    g = ig.Graph(n=N, edges=edges, directed=False)
    g.es["weight"] = weights

    part = leidenalg.find_partition(
        g,
        leidenalg.RBConfigurationVertexPartition,
        weights="weight",
        resolution_parameter=gamma,
        seed=seed
    )
    return reindex_labels(np.array(part.membership, dtype=np.int64))


def run_infomap(G: nx.Graph, seed: int) -> Optional[np.ndarray]:
    """pip install infomap"""
    try:
        from infomap import Infomap
    except Exception:
        return None

    im = Infomap(f"--seed {seed} --two-level --silent")
    for u, v, d in G.edges(data=True):
        w = float(d.get("weight", 1.0))
        im.addLink(int(u), int(v), w)
        im.addLink(int(v), int(u), w)

    im.run()

    labels = np.zeros((G.number_of_nodes(),), dtype=np.int64)
    for node in im.tree:
        if node.isLeaf:
            labels[int(node.physicalId)] = int(node.moduleIndex())
    return reindex_labels(labels)


def run_spectral(G: nx.Graph, n_clusters: int, seed: int, max_dense_n: int = 5000) -> Optional[np.ndarray]:
    """pip install scikit-learn  (dense affinity -> auto-skip if N too large)"""
    try:
        from sklearn.cluster import SpectralClustering
    except Exception:
        return None

    N = G.number_of_nodes()
    if N > max_dense_n:
        print(f"[SKIP] spectral k={n_clusters} vì N={N} > {max_dense_n} (dense NxN quá nặng).")
        return None

    A = np.zeros((N, N), dtype=np.float32)
    for u, v, d in G.edges(data=True):
        w = float(d.get("weight", 1.0))
        A[int(u), int(v)] = w
        A[int(v), int(u)] = w

    sc = SpectralClustering(
        n_clusters=n_clusters,
        affinity="precomputed",
        assign_labels="kmeans",
        random_state=seed
    )
    labels = sc.fit_predict(A).astype(np.int64)
    return reindex_labels(labels)


def run_dbscan_coords(seg_lat: np.ndarray, seg_lon: np.ndarray, eps_km: float, min_samples: int) -> Optional[np.ndarray]:
    """
    Optional baseline if you want distance clustering.
    pip install scikit-learn
    """
    try:
        from sklearn.cluster import DBSCAN
    except Exception:
        return None

    # Use haversine metric in radians for DBSCAN
    coords = np.vstack([np.deg2rad(seg_lat), np.deg2rad(seg_lon)]).T  # [N,2]
    eps = eps_km / 6371.0  # km -> radians
    db = DBSCAN(eps=eps, min_samples=min_samples, metric="haversine")
    labels = db.fit_predict(coords).astype(np.int64)  # -1 is noise

    # handle noise: make each noise point its own cluster (so we still can compute modularity)
    if np.any(labels == -1):
        noise_idx = np.where(labels == -1)[0]
        next_id = labels.max() + 1
        for i in noise_idx:
            labels[i] = next_id
            next_id += 1

    return reindex_labels(labels)


# =========================
# Soft Tran by distance-to-centroid
# =========================

def compute_region_centroids(labels: np.ndarray, seg_lat: np.ndarray, seg_lon: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return centroid_lat [R], centroid_lon [R]"""
    R = int(labels.max()) + 1
    c_lat = np.zeros((R,), dtype=np.float64)
    c_lon = np.zeros((R,), dtype=np.float64)
    for r in range(R):
        idx = np.where(labels == r)[0]
        if idx.size == 0:
            c_lat[r] = np.mean(seg_lat)
            c_lon[r] = np.mean(seg_lon)
        else:
            c_lat[r] = float(np.mean(seg_lat[idx]))
            c_lon[r] = float(np.mean(seg_lon[idx]))
    return c_lat, c_lon


def soft_tran_by_distance(
    labels_hard: np.ndarray,
    seg_lat: np.ndarray,
    seg_lon: np.ndarray,
    tau_km: Optional[float] = None,
    topk: int = 8,
    chunk: int = 4096,
) -> Tuple[np.ndarray, float]:
    """
    Create Tran_soft [N,R] in [0,1], row-sum=1 using distance-to-centroid.

    If tau_km is None: auto set tau = median distance(node->own centroid).
    Use topk nearest centroids per node for sparsity.
    """
    N = labels_hard.shape[0]
    R = int(labels_hard.max()) + 1
    c_lat, c_lon = compute_region_centroids(labels_hard, seg_lat, seg_lon)

    # compute tau if auto
    if tau_km is None:
        own = labels_hard
        d_own = haversine_km(seg_lat, seg_lon, c_lat[own], c_lon[own])
        tau = float(np.median(d_own) + 1e-6)
    else:
        tau = float(tau_km)

    Tran = np.zeros((N, R), dtype=np.float32)

    # chunked compute distances to all centroids to avoid huge RAM spike
    for s in range(0, N, chunk):
        e = min(N, s + chunk)
        lat = seg_lat[s:e][:, None]  # [B,1]
        lon = seg_lon[s:e][:, None]
        # [B,R]
        D = haversine_km(lat, lon, c_lat[None, :], c_lon[None, :])

        if topk is not None and topk > 0 and topk < R:
            # keep only topk nearest
            idx = np.argpartition(D, kth=topk-1, axis=1)[:, :topk]  # [B,topk]
            d_top = np.take_along_axis(D, idx, axis=1)              # [B,topk]
            logits = -d_top / tau
            # stable softmax
            logits = logits - np.max(logits, axis=1, keepdims=True)
            w = np.exp(logits).astype(np.float32)                   # [B,topk]
            w = w / (np.sum(w, axis=1, keepdims=True) + 1e-12)

            # scatter to full Tran
            rows = np.arange(e - s)[:, None]
            Tran[s:e, :] = 0.0
            Tran[s:e, idx] = w
        else:
            logits = -D / tau
            logits = logits - np.max(logits, axis=1, keepdims=True)
            w = np.exp(logits).astype(np.float32)
            w = w / (np.sum(w, axis=1, keepdims=True) + 1e-12)
            Tran[s:e] = w

    # final renorm safety
    Tran = Tran / (Tran.sum(axis=1, keepdims=True) + 1e-12)
    return Tran.astype(np.float32), tau


def region_adjacency_soft_from_edges(edges_idx: np.ndarray, edge_w: np.ndarray, Tran_soft: np.ndarray, binarize: bool = False) -> np.ndarray:
    """
    Build A_R from node edges using Tran_soft:
      For each edge (u,v) with weight w:
        A_R += w * outer(Tran[u], Tran[v]) + w * outer(Tran[v], Tran[u])  (undirected)
    Output: [R,R]
    """
    N, R = Tran_soft.shape
    A_R = np.zeros((R, R), dtype=np.float32)

    for (u, v), w in zip(edges_idx, edge_w):
        tu = Tran_soft[u]  # [R]
        tv = Tran_soft[v]  # [R]
        A_R += (w * np.outer(tu, tv)).astype(np.float32)
        A_R += (w * np.outer(tv, tu)).astype(np.float32)

    np.fill_diagonal(A_R, 0.0)
    if binarize:
        A_R = (A_R > 0).astype(np.float32)
    return A_R


# =========================
# Compare + Export
# =========================

def compare_partitions(
    G: nx.Graph,
    out_dir: str,
    edges_idx: np.ndarray,
    seeds: List[int],
    gammas: List[float],
    spectral_clusters: List[int],
    run_methods: List[str],
    # dbscan optional
    seg_lat: Optional[np.ndarray] = None,
    seg_lon: Optional[np.ndarray] = None,
    dbscan_eps_km: float = 0.5,
    dbscan_min_samples: int = 10,
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
        if res.extra:
            for k, v in res.extra.items():
                tag += f"_{k}{v}"

        np.save(os.path.join(part_dir, f"{tag}_labels.npy"), res.labels)

        meta = {
            "method": res.method,
            "Q": float(res.Q),
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

        if "infomap" in run_methods:
            labels = run_infomap(G, seed=seed)
            if labels is not None:
                Q = modularity_Q_nx(G, labels)
                res = PartitionResult("infomap", labels, Q, seed=seed)
                all_results.append(res)
                save_one(res)

        if "spectral" in run_methods:
            for k in spectral_clusters:
                labels = run_spectral(G, n_clusters=k, seed=seed)
                if labels is not None:
                    Q = modularity_Q_nx(G, labels)
                    res = PartitionResult("spectral", labels, Q, seed=seed, extra={"k": int(k)})
                    all_results.append(res)
                    save_one(res)

        if "dbscan" in run_methods:
            if seg_lat is None or seg_lon is None:
                print("[SKIP] dbscan vì thiếu seg_lat/seg_lon")
            else:
                labels = run_dbscan_coords(seg_lat, seg_lon, eps_km=dbscan_eps_km, min_samples=dbscan_min_samples)
                if labels is not None:
                    Q = modularity_Q_nx(G, labels)
                    res = PartitionResult("dbscan", labels, Q, seed=seed, extra={"eps_km": dbscan_eps_km, "min": dbscan_min_samples})
                    all_results.append(res)
                    save_one(res)

    if len(all_results) == 0:
        raise RuntimeError(
            "Không chạy được method nào. Bạn cần cài ít nhất 1 trong các package:\n"
            "  pip install python-louvain\n"
            "  pip install igraph leidenalg\n"
            "  pip install infomap\n"
            "  pip install scikit-learn\n"
        )

    best = max(all_results, key=lambda r: r.Q)
    return all_results, best


def export_best_soft(
    out_dir: str,
    G: nx.Graph,
    edges_idx: np.ndarray,
    best: PartitionResult,
    seg_lat: np.ndarray,
    seg_lon: np.ndarray,
    tau_km: Optional[float],
    topk: int,
    binarize_AR: bool = False,
):
    """
    Export best partition + SOFT Tran (distance to centroid) + A_R built from Tran_soft.
    """
    labels_hard = best.labels.astype(np.int64)
    N = int(G.number_of_nodes())
    R = int(labels_hard.max()) + 1

    Tran_soft, tau_used = soft_tran_by_distance(
        labels_hard=labels_hard,
        seg_lat=seg_lat,
        seg_lon=seg_lon,
        tau_km=tau_km,
        topk=topk
    )

    # edge weights array aligned with edges_idx
    edge_w = np.ones((edges_idx.shape[0],), dtype=np.float32)
    # If you want true weights from nx, you can rebuild edge list weights; but edges_idx came from CSV (unweighted).
    # For now, keep 1.0. (Graph weights were built by duplicate edges anyway.)

    A_R = region_adjacency_soft_from_edges(edges_idx, edge_w=edge_w, Tran_soft=Tran_soft, binarize=binarize_AR)

    payload = {
        "labels_hard": labels_hard,
        "Tran_soft": Tran_soft.astype(np.float32),
        "A_R": A_R.astype(np.float32),
        "method": np.array([best.method], dtype=object),
        "Q": np.array([best.Q], dtype=np.float32),
        "gamma": np.array([best.gamma if best.gamma is not None else -1.0], dtype=np.float32),
        "seed": np.array([best.seed if best.seed is not None else -1], dtype=np.int64),
        "R": np.array([R], dtype=np.int64),
        "N": np.array([N], dtype=np.int64),
        "tau_km": np.array([tau_used], dtype=np.float32),
        "topk": np.array([topk], dtype=np.int64),
    }
    np.savez(os.path.join(out_dir, "best_partition.npz"), **payload)

    best_json = {
        "method": best.method,
        "Q": float(best.Q),
        "gamma": best.gamma,
        "seed": best.seed,
        "N": N,
        "R": R,
        "tau_km": float(tau_used),
        "topk": int(topk),
        "A_R_binarize": bool(binarize_AR),
    }
    with open(os.path.join(out_dir, "best_partition.json"), "w", encoding="utf-8") as f:
        json.dump(best_json, f, ensure_ascii=False, indent=2)

    print(f"[BEST] method={best.method} Q={best.Q:.6f} gamma={best.gamma} seed={best.seed} R={R}")
    print(f"[SOFT] tau_km={tau_used:.6f} topk={topk}")
    print(f"Saved: {os.path.join(out_dir, 'best_partition.npz')}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--root", type=str, default=".",
                        help="Project root (Urban-Traffic-Links). Default='.' if you run from root.")

    parser.add_argument("--edges_csv", type=str, default="data/processed/tomtom_stats/edges.csv")
    parser.add_argument("--segment_index_csv", type=str, default="data/processed/tomtom_stats/segment_index.csv")

    # for distance-based Tran
    parser.add_argument("--segments_csv", type=str, default="data/processed/tomtom_stats/segments.csv")
    parser.add_argument("--nodes_csv", type=str, default="data/processed/tomtom_stats/nodes.csv")

    # OUTPUT: đúng tên bạn yêu cầu (output_compaer_Hierachical)
    parser.add_argument("--out_dir", type=str, default="src/models/Hierachical/Transformer/output_compaer_Hierachical")

    parser.add_argument("--methods", type=str, default="louvain,leiden,infomap,spectral",
                        help="Comma-separated: louvain,leiden,infomap,spectral,dbscan")

    parser.add_argument("--seeds", type=str, default="0,1,2", help="Comma-separated seeds")
    parser.add_argument("--gammas", type=str, default="0.5,1.0,1.5,2.0",
                        help="Resolution sweep gamma for Louvain/Leiden")
    parser.add_argument("--spectral_k", type=str, default="10,20,30",
                        help="Spectral clustering k list")

    # soft Tran controls
    parser.add_argument("--tau_km", type=float, default=-1.0,
                        help="tau in km for exp(-d/tau). If -1 => auto median dist to own centroid.")
    parser.add_argument("--topk", type=int, default=8,
                        help="Keep only top-k nearest regions for each node. Use <=0 to disable.")
    parser.add_argument("--binarize_AR", action="store_true",
                        help="If set, A_R will be binarized (0/1). Otherwise weighted.")

    # dbscan optional
    parser.add_argument("--dbscan_eps_km", type=float, default=0.5)
    parser.add_argument("--dbscan_min_samples", type=int, default=10)

    args = parser.parse_args()

    root = args.root
    edges_csv = os.path.join(root, args.edges_csv)
    segment_index_csv = os.path.join(root, args.segment_index_csv)
    segments_csv = os.path.join(root, args.segments_csv)
    nodes_csv = os.path.join(root, args.nodes_csv)
    out_dir = os.path.join(root, args.out_dir)

    run_methods = [m.strip().lower() for m in args.methods.split(",") if m.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    gammas = [float(x) for x in args.gammas.split(",") if x.strip()]
    spectral_k = [int(x) for x in args.spectral_k.split(",") if x.strip()]

    tau_km = None if args.tau_km is None or args.tau_km < 0 else float(args.tau_km)
    topk = int(args.topk)

    ensure_dir(out_dir)

    print("Loading graph...")
    G, N, edges_idx = load_segment_graph(edges_csv, segment_index_csv, undirected=True)
    print(f"Graph: N={N}, E={G.number_of_edges()}")

    print("Loading segment lat/lon for soft Tran...")
    seg_lat, seg_lon = load_segment_latlon(segments_csv, nodes_csv, segment_index_csv)
    print(f"Coords loaded: seg_lat/seg_lon shape = {seg_lat.shape}")

    print("Running community detection + modularity compare...")
    all_results, best = compare_partitions(
        G=G,
        out_dir=out_dir,
        edges_idx=edges_idx,
        seeds=seeds,
        gammas=gammas,
        spectral_clusters=spectral_k,
        run_methods=run_methods,
        seg_lat=seg_lat,
        seg_lon=seg_lon,
        dbscan_eps_km=float(args.dbscan_eps_km),
        dbscan_min_samples=int(args.dbscan_min_samples),
    )

    # summary.csv
    rows = []
    for r in all_results:
        rows.append({
            "method": r.method,
            "Q": float(r.Q),
            "gamma": r.gamma if r.gamma is not None else "",
            "seed": r.seed if r.seed is not None else "",
            "R": int(r.labels.max()) + 1,
            "extra": json.dumps(r.extra or {}, ensure_ascii=False)
        })
    summary = pd.DataFrame(rows).sort_values(by="Q", ascending=False)
    summary_path = os.path.join(out_dir, "summary.csv")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"Saved: {summary_path}")

    # export best + soft Tran
    export_best_soft(
        out_dir=out_dir,
        G=G,
        edges_idx=edges_idx,
        best=best,
        seg_lat=seg_lat,
        seg_lon=seg_lon,
        tau_km=tau_km,
        topk=topk if topk > 0 else 0,
        binarize_AR=bool(args.binarize_AR),
    )

    print("DONE.")


if __name__ == "__main__":
    main()
