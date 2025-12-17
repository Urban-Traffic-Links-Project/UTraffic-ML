
from __future__ import annotations

import os
import math
import json
import pickle
import hashlib
from dataclasses import dataclass,field
from typing import Dict, Tuple, List, Optional, Iterable, Any, Set
import time
import numpy as np
import pandas as pd
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)
try:
    from sklearn.cluster import DBSCAN
except Exception as e:
    DBSCAN = None

try:
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
except Exception:
    sp = None

EARTH_RADIUS_M = 6371000.0

def haversine_distance_m(lat1, lon1, lat2, lon2) -> float:
    """Great-circle distance in meters."""
    lat1, lon1, lat2, lon2 = map(np.deg2rad, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(a))
    return float(EARTH_RADIUS_M * c)

def pairwise_haversine_m(latlon: np.ndarray) -> np.ndarray:
    """
    latlon: (M,2) [lat,lon]
    returns: (M,M) distance matrix meters
    """
    lat = np.deg2rad(latlon[:, 0].astype(np.float64))
    lon = np.deg2rad(latlon[:, 1].astype(np.float64))
    dlat = lat[:, None] - lat[None, :]
    dlon = lon[:, None] - lon[None, :]
    a = np.sin(dlat / 2) ** 2 + np.cos(lat)[:, None] * np.cos(lat)[None, :] * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    return (EARTH_RADIUS_M * c).astype(np.float32)

def stable_hash(obj: Any) -> str:
    """Hash for caching keys (stable across runs)."""
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    return hashlib.md5(data).hexdigest()

# ============================================================
# Data IO
# ============================================================

@dataclass
class TrafficTensor:
    values: np.ndarray          # (T,N) float32 z-speed
    is_congested: np.ndarray    # (T,N) int8 0/1
    segment_ids: np.ndarray     # (N,) int64
    time_of_day: np.ndarray     # (T,) float32 [0,1]
    day_of_week: np.ndarray     # (T,) int64 0..6
    dates: Optional[np.ndarray] = None        # (T,) str 'YYYY-MM-DD'
    time_set_id: Optional[np.ndarray] = None  # (T,) int32

def load_traffic_tensor(npz_path: str) -> TrafficTensor:
    z = np.load(npz_path, allow_pickle=True)
    tt = TrafficTensor(
        values=z["values"].astype(np.float32),
        is_congested=z["is_congested"].astype(np.int8),
        segment_ids=z["segment_ids"].astype(np.int64),
        time_of_day=z["time_of_day"].astype(np.float32) if "time_of_day" in z else None,
        day_of_week=z["day_of_week"].astype(np.int64) if "day_of_week" in z else None,
        dates=z["dates"].astype(str) if "dates" in z else None,
        time_set_id=z["time_set_id"].astype(np.int32) if "time_set_id" in z else None,
    )
    return tt

@dataclass
class GraphTopology:
    # global index mapping
    segid_to_idx: Dict[int, int]
    idx_to_segid: np.ndarray  # (N,)
    # segment centers lat/lon in global index order
    centers_latlon: np.ndarray  # (N,2) float32
    # adjacency list by global index
    neighbors: List[List[int]]

def load_graph_topology(
    segments_csv_path: str,
    nodes_csv_path: str,
    edges_csv_path: str,
    segment_index_csv_path: str,
) -> GraphTopology:
    """
    Expect:
      segments.csv: segment_id, node_u, node_v, ...
      nodes.csv: node_id, lat, lon, ...
      edges.csv: segment_u, segment_v   (segment IDs, undirected)
      segment_index.csv: idx, segment_id  (mapping)
    """
    seg_df = pd.read_csv(segments_csv_path)
    node_df = pd.read_csv(nodes_csv_path)
    edge_df = pd.read_csv(edges_csv_path)
    map_df = pd.read_csv(segment_index_csv_path)

    # mapping
    idx_to_segid = map_df.sort_values("idx")["segment_id"].astype(np.int64).values
    segid_to_idx = {int(s): int(i) for i, s in enumerate(idx_to_segid)}

    # node coords
    node_df = node_df.copy()
    node_df["node_id"] = node_df["node_id"].astype(int)
    node_lat = dict(zip(node_df["node_id"].values, node_df["lat"].values))
    node_lon = dict(zip(node_df["node_id"].values, node_df["lon"].values))

    # segment centers
    seg_df = seg_df.copy()
    seg_df["segment_id"] = seg_df["segment_id"].astype(np.int64)
    seg_df["node_u"] = seg_df["node_u"].astype(int)
    seg_df["node_v"] = seg_df["node_v"].astype(int)

    centers = np.zeros((len(idx_to_segid), 2), dtype=np.float32)
    missing = 0
    for seg_id, u, v in zip(seg_df["segment_id"].values, seg_df["node_u"].values, seg_df["node_v"].values):
        if int(seg_id) not in segid_to_idx:
            continue
        ui = int(u); vi = int(v)
        if ui not in node_lat or vi not in node_lat:
            missing += 1
            continue
        latc = 0.5 * (float(node_lat[ui]) + float(node_lat[vi]))
        lonc = 0.5 * (float(node_lon[ui]) + float(node_lon[vi]))
        centers[segid_to_idx[int(seg_id)], 0] = latc
        centers[segid_to_idx[int(seg_id)], 1] = lonc
    print("Số missing:", missing)
    # adjacency list (global idx)
    N = len(idx_to_segid)
    neighbors: List[List[int]] = [[] for _ in range(N)]
    # edges.csv may store segment ids or indices; assume segment_id
    for su, sv in zip(edge_df.iloc[:, 0].values, edge_df.iloc[:, 1].values):
        su = int(su); sv = int(sv)
        if su in segid_to_idx and sv in segid_to_idx:
            iu = segid_to_idx[su]; iv = segid_to_idx[sv]
            if iv not in neighbors[iu]:
                neighbors[iu].append(iv)
            if iu not in neighbors[iv]:
                neighbors[iv].append(iu)

    return GraphTopology(
        segid_to_idx=segid_to_idx,
        idx_to_segid=idx_to_segid,
        centers_latlon=centers,
        neighbors=neighbors,
    )

# ============================================================
# Correlation (lead-lag) utilities
# ============================================================

@dataclass
class CorrParams:
    tau_max: int = 3          # max lag (+/-) in steps
    tau_cut: int = 3          # filter by |tau_pos| <= tau_cut
    W_min: float = 0.2        # threshold for Wpos
    # optional normalization
    eps: float = 1e-6

def _xcorr_peak_wpos_tau_paper(x: np.ndarray, y: np.ndarray, tau_max: int, eps: float = 1e-6) -> Tuple[float, int]:
    """
    Paper-style:
      - Compute X_{i,j}(tau) Pearson correlation for tau in [-tau_max, +tau_max]
      - tau_pos = argmax X(tau)
      - Wpos = (max(X) - mean(X)) / std(X)
      - Return (Wpos_clipped, tau_pos)
    Notes:
      - If std(X) is tiny -> protect with eps
      - Optionally clip Wpos at >=0 (paper focuses on positive links)
    """
    x = x.astype(np.float32)
    y = y.astype(np.float32)
    T = x.shape[0]
    if T < 4:
        print("time series < 4", x)
        return 0.0, 0

    Xtaus = []
    taus = list(range(-tau_max, tau_max + 1))

    for tau in taus:
        if tau < 0:
            # X_{i,j}(tau) = X_{j,i}(-tau)
            # Equivalent compute corr between x[-tau:] and y[:T+tau]
            xa = x[-tau:]
            ya = y[: T + tau]
        elif tau > 0:
            xa = x[: T - tau]
            ya = y[tau:]
        else:
            xa = x
            ya = y

        if xa.size < 4:
            Xtaus.append(0.0)
            continue

        # Pearson corr for this overlap window
        xa0 = xa - xa.mean()
        ya0 = ya - ya.mean()
        denom = (np.sqrt((xa0 * xa0).sum()) * np.sqrt((ya0 * ya0).sum())) + eps
        corr = float((xa0 * ya0).sum() / denom)
        Xtaus.append(corr)

    Xarr = np.array(Xtaus, dtype=np.float32)
    k = int(np.argmax(Xarr))
    maxX = float(Xarr[k])
    meanX = float(Xarr.mean())
    stdX = float(Xarr.std()) + eps

    Wpos = (maxX - meanX) / stdX
    tau_pos = int(taus[k])

    # Paper later filters "strong positive" links, bạn có thể clip âm về 0
    if Wpos < 0:
        Wpos = 0.0

    return float(Wpos), tau_pos

# ============================================================
# Zone building
# ============================================================

@dataclass
class DBSCANParams:
    eps_m: float = 250.0      # DBSCAN eps in meters
    min_samples: int = 3

@dataclass
class ZoneParams:
    # seed sampling
    seed_congested_ratio: float = 0.6  # 60% congested, 40% non-congested
    # candidate expansion
    hops: int = 2
    # radius filter
    R_max_m: float = 2000.0
    # link filters
    D_min_m: float = 0.0
    D_max_m: float = 3000.0
    top_k: int = 64
    # correlation params
    corr: CorrParams = field(default_factory=CorrParams)
    # laplacian PE
    d_spa: int = 16
    laplacian_mode: str = "edges"  # "edges" or "kernel"
    sigma_m: float = 500.0         # for kernel mode exp(-D/sigma)
    # caching
    enable_cache: bool = True
    cache_dir: str = "./cache_zone"
    max_candidates: int = 256
    max_zone_size: int = 512

class ZoneBuilder:
    def __init__(
        self,
        traffic: TrafficTensor,
        topo: GraphTopology,
        dbscan: DBSCANParams = DBSCANParams(),
        zone: ZoneParams = ZoneParams(),
        random_seed: int = 13,
    ):
        if DBSCAN is None:
            raise ImportError("scikit-learn is required for DBSCAN. Please install scikit-learn.")
        if sp is None or spla is None:
            # still can build zone, but laplacian PE optional
            pass

        self.traffic = traffic
        self.topo = topo
        self.dbscan = dbscan
        self.zone = zone
        self.rng = np.random.default_rng(random_seed)

        if zone.enable_cache:
            os.makedirs(zone.cache_dir, exist_ok=True)

        # ------------------------
        # Snapshot DBSCAN clusters
        # ------------------------
    def _dbscan_clusters_at_t(self, t: int, congested: bool) -> Tuple[np.ndarray, Dict[int, List[int]]]:
            """
            Return (labels, clusters_dict) for either congested or non-congested segments at time t.
            labels: for selected indices only (same order as selected list)
            clusters_dict: cluster_id -> list of global indices
              cluster_id uses DBSCAN labels (>=0)
            """
            y = self.traffic.is_congested[t].astype(np.int8)
            if congested:
                sel = np.where(y == 1)[0]
            else:
                sel = np.where(y == 0)[0]

            if sel.size == 0:
                return np.array([], dtype=np.int32), {}

            latlon = self.topo.centers_latlon[sel]
            # DBSCAN uses euclidean distance; here we approximate by projecting meters
            # Better: precompute distance matrix with haversine for selected points when sel is small
            # We'll do haversine distance matrix to be correct.
            D = pairwise_haversine_m(latlon)
            clustering = DBSCAN(eps=self.dbscan.eps_m, min_samples=self.dbscan.min_samples, metric="precomputed")
            labels = clustering.fit_predict(D).astype(np.int32)

            clusters: Dict[int, List[int]] = {}
            for idx_local, lab in enumerate(labels):
                if lab < 0:
                    continue
                gi = int(sel[idx_local])
                clusters.setdefault(int(lab), []).append(gi)

            return labels, clusters

    def _cap_by_distance(
            self,
            indices: List[int],
            seed_center_latlon: Tuple[float, float],
            keep: Optional[Iterable[int]] = None,
            cap: int = 512,
    ) -> List[int]:
        """
        Keep 'keep' first, then fill remaining slots by nearest-to-seed.
        """
        if cap is None or cap <= 0 or len(indices) <= cap:
            return sorted(list(dict.fromkeys(indices)))

        keep_set = set(int(x) for x in (keep or []))
        # Always keep only those that exist in indices ∪ keep
        pool_set = set(int(x) for x in indices) | keep_set

        # compute dist to seed for pool
        lat0, lon0 = seed_center_latlon
        dists = []
        for idx in pool_set:
            lat, lon = self.topo.centers_latlon[idx]  # bạn đã có center cho segment global idx
            d = haversine_distance_m(lat0, lon0, lat, lon)
            dists.append((d, idx))
        dists.sort(key=lambda x: x[0])

        out = []
        # 1) keep first (preserve)
        for k in keep_set:
            out.append(k)
        out = list(dict.fromkeys(out))

        # 2) fill by nearest
        for _, idx in dists:
            if len(out) >= cap:
                break
            if idx not in out:
                out.append(idx)

        return out

    # ------------------------
    # Seed selection 60/40
    # ------------------------

    def sample_seed_cluster(self, t: int) -> Dict[str, Any]:
        """
        Fast seed sampling:
          - 60%: pick a DBSCAN cluster from congested segments (y==1)
          - 40%: pick a random non-congested segment (y==0) as a "singleton seed cluster"
        Avoids DBSCAN on non-congested set (usually huge), which causes O(M^2) distance matrix blow-up.
        """
        p = float(self.zone.seed_congested_ratio)
        choose_cong = (self.rng.random() < p)

        # --- only DBSCAN congested ---
        labels_c, clusters_c = self._dbscan_clusters_at_t(t, congested=True)

        y = self.traffic.is_congested[t].astype(np.int8)
        noncong = np.where(y == 0)[0]
        cong = np.where(y == 1)[0]

        def pick_from_clusters(clusters: Dict[int, List[int]]) -> Optional[List[int]]:
            if not clusters:
                return None
            keys = list(clusters.keys())
            sizes = np.array([len(clusters[k]) for k in keys], dtype=np.float32)
            probs = sizes / (sizes.sum() + 1e-6)
            k = int(self.rng.choice(keys, p=probs))
            return clusters[k]

        seed_indices: Optional[List[int]] = None
        seed_type: str = "congested" if choose_cong else "non_congested"

        if choose_cong:
            # try congested DBSCAN cluster
            seed_indices = pick_from_clusters(clusters_c)
            if seed_indices is None:
                # fallback: pick any congested segment as singleton
                if cong.size > 0:
                    seed_indices = [int(self.rng.choice(cong))]
                    seed_type = "congested_singleton"
        else:
            # non-congested: pick random segment as singleton (NO DBSCAN)
            if noncong.size > 0:
                seed_indices = [int(self.rng.choice(noncong))]
                seed_type = "non_congested_singleton"
            else:
                # fallback to congested cluster/singleton
                seed_indices = pick_from_clusters(clusters_c)
                if seed_indices is not None:
                    seed_type = "congested"
                elif cong.size > 0:
                    seed_indices = [int(self.rng.choice(cong))]
                    seed_type = "congested_singleton"

        if seed_indices is None or len(seed_indices) == 0:
            # last-resort fallback: random segment
            gi = int(self.rng.integers(0, self.traffic.values.shape[1]))
            seed_indices = [gi]
            seed_type = "fallback_random"

        latlon = self.topo.centers_latlon[np.array(seed_indices, dtype=np.int64)]
        seed_center = latlon.mean(axis=0).astype(np.float32)

        return {
            "t": int(t),
            "seed_type": seed_type,
            "seed_cluster_indices": [int(x) for x in seed_indices],
            "seed_center_latlon": seed_center,
            # stats (optional)
            "num_congested_clusters": int(len(clusters_c)),
            "congested_cluster_sizes": [len(v) for v in clusters_c.values()],
            "num_noncongested_clusters": -1,  # not computed anymore
            "noncongested_cluster_sizes": [],
        }

    # ------------------------
    # Candidate expansion
    # ------------------------
    def _k_hop_expand(self, seeds: Iterable[int], hops: int) -> List[int]:
        visited: Set[int] = set(int(s) for s in seeds)
        frontier: Set[int] = set(int(s) for s in seeds)
        for _ in range(hops):
            nxt: Set[int] = set()
            for u in frontier:
                for v in self.topo.neighbors[u]:
                    if v not in visited:
                        visited.add(v)
                        nxt.add(v)
            frontier = nxt
            if not frontier:
                break
        return sorted(list(visited))

    def _radius_filter(self, candidates: List[int], seed_center_latlon: np.ndarray, R_max_m: float) -> List[int]:
        c = np.array(candidates, dtype=np.int64)
        latlon = self.topo.centers_latlon[c]
        lat0, lon0 = float(seed_center_latlon[0]), float(seed_center_latlon[1])

        # vectorized approximate: compute haversine per point
        # For simplicity: loop (candidates size usually moderate)
        keep = []
        for gi, (la, lo) in zip(c.tolist(), latlon):
            d = haversine_distance_m(lat0, lon0, float(la), float(lo))
            if d <= R_max_m:
                keep.append(int(gi))
        return keep

    # ------------------------
    # Candidate-vs-candidate correlation + link filtering
    # ------------------------
    def _compute_links_candidate_pairwise(
            self,
            candidate_indices: List[int],
            t: int,
            hist_len: int,
            corr: CorrParams,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute paper-style Wpos and tau_pos for all pairs (i<j) in candidates
        using history window ending at t.

        Return:
          pairs: (E,2) global indices
          attrs: (E,2) columns [Wpos, tau_pos]
        """

        # -------------------------------------------------
        # 1. Prepare candidate data
        # -------------------------------------------------
        c = np.array(candidate_indices, dtype=np.int64)
        M = c.size
        if M < 2:
            return (
                np.zeros((0, 2), dtype=np.int64),
                np.zeros((0, 2), dtype=np.float32),
            )

        t0 = t - hist_len + 1
        if t0 < 0:
            return (
                np.zeros((0, 2), dtype=np.int64),
                np.zeros((0, 2), dtype=np.float32),
            )

        # history window (L, N)
        X = self.traffic.values[t0: t + 1, :]
        Xc = X[:, c]  # (L, M)
        L = Xc.shape[0]

        # guard: history too short for correlation
        if L < max(4, corr.tau_max + 3):
            return (
                np.zeros((0, 2), dtype=np.int64),
                np.zeros((0, 2), dtype=np.float32),
            )

        # -------------------------------------------------
        # 2. Pairwise correlation
        # -------------------------------------------------
        E_pairs: List[Tuple[int, int]] = []
        E_attr: List[Tuple[float, int]] = []

        for a in range(M):
            xa = Xc[:, a]
            for b in range(a + 1, M):
                xb = Xc[:, b]

                wpos, tau_pos = _xcorr_peak_wpos_tau_paper(
                    xa, xb,
                    tau_max=corr.tau_max,
                    eps=corr.eps,
                )

                # chỉ giữ link có tín hiệu dương
                if wpos <= 0.0:
                    continue

                E_pairs.append((int(c[a]), int(c[b])))
                E_attr.append((float(wpos), int(tau_pos)))

        if not E_pairs:
            return (
                np.zeros((0, 2), dtype=np.int64),
                np.zeros((0, 2), dtype=np.float32),
            )

        pairs = np.array(E_pairs, dtype=np.int64)
        attrs = np.array(E_attr, dtype=np.float32)  # [Wpos, tau_pos]

        return pairs, attrs

    def _filter_links(
            self,
            pairs: np.ndarray,
            attrs: np.ndarray,
            D_min_m: float,
            D_max_m: float,
            W_min: float,
            tau_cut: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Filter by Wpos, distance range, |tau| <= tau_cut.
        Return:
          kept_pairs (E',2), kept_attrs (E',2), kept_dist (E',)
        """
        if pairs.shape[0] == 0:
            return pairs, attrs, np.zeros((0,), dtype=np.float32)

        gi = pairs[:, 0]
        gj = pairs[:, 1]
        latlon_i = self.topo.centers_latlon[gi]
        latlon_j = self.topo.centers_latlon[gj]

        # compute distances per edge (loop ok)
        dist = np.zeros((pairs.shape[0],), dtype=np.float32)
        for k in range(pairs.shape[0]):
            dist[k] = haversine_distance_m(
                float(latlon_i[k, 0]), float(latlon_i[k, 1]),
                float(latlon_j[k, 0]), float(latlon_j[k, 1]),
            )

        Wpos = attrs[:, 0]
        tau = attrs[:, 1]
        m = (
                (Wpos >= float(W_min)) &
                (dist >= float(D_min_m)) &
                (dist <= float(D_max_m)) &
                (np.abs(tau) <= float(tau_cut))
        )

        kp = pairs[m]
        ka = attrs[m]
        kd = dist[m]
        return kp, ka, kd

    def _topk_targets_from_links(
            self,
            candidate_indices: List[int],
            pairs: np.ndarray,
            attrs: np.ndarray,
            top_k: int,
    ) -> Tuple[List[int], np.ndarray]:
        """
        Define targets as all nodes that appear in kept links; if too many -> top_k by max incident Wpos.
        Return:
          targets_global_indices (list),
          target_mask_zone (len M_zone,)
        """
        c = np.array(candidate_indices, dtype=np.int64)
        M = c.size
        if pairs.shape[0] == 0:
            return [], np.zeros((M,), dtype=np.int8)

        # incident max weight for each node
        score = np.zeros((M,), dtype=np.float32)
        pos_in_zone = {int(g): i for i, g in enumerate(c.tolist())}

        Wpos = attrs[:, 0]
        for (u, v), w in zip(pairs.tolist(), Wpos.tolist()):
            if u in pos_in_zone:
                score[pos_in_zone[u]] = max(score[pos_in_zone[u]], float(w))
            if v in pos_in_zone:
                score[pos_in_zone[v]] = max(score[pos_in_zone[v]], float(w))

        # nodes that appear in at least one link
        appear = score > 0
        idxs = np.where(appear)[0]
        if idxs.size == 0:
            return [], np.zeros((M,), dtype=np.int8)

        # if too many -> top_k
        if idxs.size > int(top_k):
            order = np.argsort(score[idxs])[::-1]
            idxs = idxs[order[: int(top_k)]]

        target_mask = np.zeros((M,), dtype=np.int8)
        target_mask[idxs] = 1
        targets = c[idxs].astype(np.int64).tolist()
        return targets, target_mask

    # ------------------------
    # Laplacian PE (optional)
    # ------------------------
    def build_laplacian_eigvec(
            self,
            zone_indices: List[int],
            d_spa: int,
            mode: str = "edges",
            sigma_m: float = 500.0,
    ) -> np.ndarray:
        """
        Return lap_eigvec: (N_zone, d_spa) float32
        mode:
          - "edges": adjacency from restricted topology edges; weight=1
          - "kernel": fully-connected A_ij = exp(-D/sigma)
        """
        N = len(zone_indices)
        if N == 0:
            return np.zeros((0, d_spa), dtype=np.float32)

        if sp is None or spla is None:
            # scipy not available: return zeros
            return np.zeros((N, d_spa), dtype=np.float32)

        z = np.array(zone_indices, dtype=np.int64)
        zpos = {int(g): i for i, g in enumerate(z.tolist())}

        if mode == "edges":
            rows = []
            cols = []
            data = []
            for g in z.tolist():
                i = zpos[int(g)]
                for nb in self.topo.neighbors[int(g)]:
                    if nb in zpos:
                        j = zpos[int(nb)]
                        rows.append(i);
                        cols.append(j);
                        data.append(1.0)
            if not rows:
                # isolated nodes => identity laplacian => eigvec ambiguous; return zeros
                return np.zeros((N, d_spa), dtype=np.float32)

            A = sp.coo_matrix((data, (rows, cols)), shape=(N, N), dtype=np.float32).tocsr()
            # symmetrize
            A = 0.5 * (A + A.T)

        elif mode == "kernel":
            latlon = self.topo.centers_latlon[z]
            D = pairwise_haversine_m(latlon)  # (N,N)
            A_dense = np.exp(-D / max(1e-6, float(sigma_m))).astype(np.float32)
            np.fill_diagonal(A_dense, 0.0)
            A = sp.csr_matrix(A_dense)
        else:
            raise ValueError(f"Unknown laplacian mode: {mode}")

        deg = np.array(A.sum(axis=1)).reshape(-1).astype(np.float32)
        L = sp.diags(deg) - A

        # Compute smallest eigenvectors
        k = min(int(d_spa), max(1, N - 1))
        try:
            # sigma=0 targets smallest magnitude eigenvalues for symmetric L
            vals, vecs = spla.eigsh(L, k=k, which="SM")
            vecs = vecs.astype(np.float32)
        except Exception:
            # fallback
            return np.zeros((N, d_spa), dtype=np.float32)

        # If need fixed d_spa, pad
        if vecs.shape[1] < int(d_spa):
            pad = np.zeros((N, int(d_spa) - vecs.shape[1]), dtype=np.float32)
            vecs = np.concatenate([vecs, pad], axis=1)

        return vecs[:, : int(d_spa)].astype(np.float32)

    # ------------------------
    # Cache helpers
    # ------------------------
    def _cache_path(self, key: Dict[str, Any]) -> str:
        h = stable_hash(key)
        return os.path.join(self.zone.cache_dir, f"zone_{h}.pkl")

    def _try_load_cache(self, key: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not self.zone.enable_cache:
            return None
        path = self._cache_path(key)
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    return pickle.load(f)
            except Exception:
                return None
        return None

    def _save_cache(self, key: Dict[str, Any], obj: Dict[str, Any]) -> None:
        if not self.zone.enable_cache:
            return
        path = self._cache_path(key)
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)

    # ------------------------
    # Main entry: build zone
    # ------------------------
    def build_zone(
            self,
            t: int,
            hist_len: int = 48,
            return_lap: bool = True,
    ) -> Dict[str, Any]:
        """
        Build a zone at time t.

        Inputs:
          t: time index
          hist_len: how many past steps used to compute correlation (ending at t)
          return_lap: compute Laplacian PE or not

        Outputs dict:
          - zone_indices: (N_zone,) global indices list
          - target_mask_zone: (N_zone,) int8
          - targets_global: list global indices
          - links: dict with kept pairs/attrs/dist
          - lap_eigvec: (N_zone,d_spa) float32 if return_lap else None
          - meta: seed + stats
        """
        t = int(t)
        T = self.traffic.values.shape[0]
        if t < 0 or t >= T:
            raise IndexError(f"t out of range: {t} (T={T})")
        tA = time.time()
        meta = self.sample_seed_cluster(t)
        log.info(f"[zone] sample_seed_cluster: {time.time() - tA:.2f}s")

        seed_indices = meta["seed_cluster_indices"]
        seed_center = meta["seed_center_latlon"]

        # candidate from k-hop
        candidates = self._k_hop_expand(seed_indices, hops=int(self.zone.hops))
        # radius filter
        candidates = self._radius_filter(candidates, seed_center, float(self.zone.R_max_m))

        # ensure seed included
        seed_set = set(int(x) for x in seed_indices)
        cand_set = set(int(x) for x in candidates)
        if not seed_set.issubset(cand_set):
            candidates = sorted(list(cand_set.union(seed_set)))

        if getattr(self.zone, "max_candidates", 0) and len(candidates) > int(self.zone.max_candidates):
            candidates = self._cap_by_distance(
                indices=candidates,
                seed_center_latlon=seed_center,
                keep=seed_indices,  # keep seeds
                cap=int(self.zone.max_candidates),
            )

        # cache key (correlation heavy part)
        cache_key = {
            "t": t,
            "hist_len": int(hist_len),
            "seed_type": meta["seed_type"],
            "seed_cluster": tuple(sorted(seed_indices)),
            "candidates": tuple(candidates),
            "tau_max": int(self.zone.corr.tau_max),
            "tau_cut": int(self.zone.corr.tau_cut),
            "W_min": float(self.zone.corr.W_min),
            "D_min_m": float(self.zone.D_min_m),
            "D_max_m": float(self.zone.D_max_m),
            "top_k": int(self.zone.top_k),
        }

        cached = self._try_load_cache(cache_key)
        if cached is not None:
            # optionally recompute lap if requested
            if return_lap and cached.get("lap_eigvec") is None:
                cached["lap_eigvec"] = self.build_laplacian_eigvec(
                    cached["zone_indices"],
                    d_spa=int(self.zone.d_spa),
                    mode=str(self.zone.laplacian_mode),
                    sigma_m=float(self.zone.sigma_m),
                )
            return cached

        # compute candidate-vs-candidate correlations
        tB = time.time()
        pairs, attrs = self._compute_links_candidate_pairwise(
            candidate_indices=candidates,
            t=t,
            hist_len=int(hist_len),
            corr=self.zone.corr,
        )
        log.info(f"[zone] xcorr pairs: {time.time() - tB:.2f}s | M={len(candidates)}")

        # filter links
        kp, ka, kd = self._filter_links(
            pairs=pairs,
            attrs=attrs,
            D_min_m=float(self.zone.D_min_m),
            D_max_m=float(self.zone.D_max_m),
            W_min=float(self.zone.corr.W_min),
            tau_cut=int(self.zone.corr.tau_cut),
        )

        # select targets from links
        targets_global, _ = self._topk_targets_from_links(
            candidate_indices=candidates,
            pairs=kp,
            attrs=ka,
            top_k=int(self.zone.top_k),
        )

        zone_indices = candidates
        # rebuild target_mask_zone theo zone_indices
        pos = {int(g): i for i, g in enumerate(zone_indices)}
        target_mask_zone = np.zeros((len(zone_indices),), dtype=np.int8)
        for g in targets_global:
            if g in pos:
                target_mask_zone[pos[g]] = 1

        lap = None
        tC = time.time()

        if return_lap:
            lap = self.build_laplacian_eigvec(
                zone_indices=zone_indices,
                d_spa=int(self.zone.d_spa),
                mode=str(self.zone.laplacian_mode),
                sigma_m=float(self.zone.sigma_m),
            )
        log.info(f"[zone] lap eigsh: {time.time() - tC:.2f}s | N={len(zone_indices)}")

        out = {
            "zone_indices": [int(x) for x in zone_indices],
            "target_mask_zone": target_mask_zone.astype(np.int8),
            "targets_global": [int(x) for x in targets_global],
            "links": {
                "pairs": kp.astype(np.int64),  # (E,2)
                "attrs": ka.astype(np.float32),  # (E,2) [Wpos, tau]
                "dist_m": kd.astype(np.float32)  # (E,)
            },
            "lap_eigvec": lap,  # (N_zone, d_spa) or None
            "meta": meta | {
                "num_candidates": int(len(zone_indices)),
                "num_links_kept": int(kp.shape[0]),
                "num_targets": int(int(target_mask_zone.sum())),
                "params": {
                    "dbscan_eps_m": float(self.dbscan.eps_m),
                    "dbscan_min_samples": int(self.dbscan.min_samples),
                    "hops": int(self.zone.hops),
                    "R_max_m": float(self.zone.R_max_m),
                    "tau_max": int(self.zone.corr.tau_max),
                    "tau_cut": int(self.zone.corr.tau_cut),
                    "W_min": float(self.zone.corr.W_min),
                    "D_min_m": float(self.zone.D_min_m),
                    "D_max_m": float(self.zone.D_max_m),
                    "top_k": int(self.zone.top_k),
                    "lap_mode": str(self.zone.laplacian_mode),
                    "d_spa": int(self.zone.d_spa),
                }
            }
        }

        self._save_cache(cache_key, out)
        return out

