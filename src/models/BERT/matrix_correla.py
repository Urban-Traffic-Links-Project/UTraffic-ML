#%%

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Union, Tuple
from math import radians, sin, cos, asin, sqrt

#%%

def load_traffic_tensor(output_dir: Union[str, Path]) -> Dict[str, Any]:
    tensor_path = output_dir / "traffic_tensor.npz"
    data = np.load(tensor_path)

    values = data["values"].astype("float32")
    segment_ids = data["segment_ids"]
    time_of_day = data["time_of_day"]
    day_of_week = data["day_of_week"]
    is_congested = data["is_congested"]
    return {
        "values": values,
        "segment_ids": segment_ids,
        "time_of_day": time_of_day,
        "day_of_week": day_of_week,
        "is_congested": is_congested,
    }
# ============================================================
# 1) Pearson time-lag cross-correlation cho một cặp (i, j)
# ============================================================
def cross_corr_time_lag_pair(
    x: np.ndarray,y:np.ndarray,tau_max:int
) -> Tuple[np.ndarray,np.ndarray]:
    assert x.ndim == 1 and y.ndim == 1
    assert len(x) == len(y)
    L = len(x)
    taus = np.arange(-tau_max, tau_max + 1, dtype=int)
    corrs = np.zeros_like(taus,dtype=np.float32)

    def _pearson(a: np.ndarray, b: np.ndarray) -> float:
        if a.size < 2:
            return 0.0
        a_mean = a.mean()
        b_mean = b.mean()
        a_centered = a - a_mean
        b_centered = b - b_mean

        num = np.sum(a_centered * b_centered)
        den = np.sqrt(np.sum(a_centered**2) * np.sum(b_centered**2))
        if den <= 1e-12:
            return 0.0
        return float(num / den)
    for k, tau in enumerate(taus):
        if tau == 0:
            a = x
            b = y
        elif tau > 0:
            a = x[: L - tau]
            b = y[tau:]
        else:
            lag = -tau
            a = y[: L - lag]
            b = x[lag:]
        corrs[k] = _pearson(a, b)
    return taus, corrs

# ============================================================
# 2) Link weight W^{pos}_{i,j} như paper
# ============================================================

def link_weight_pos(x: np.ndarray,y: np.ndarray,tau_max: int) -> Tuple[float, int, float]:
    taus, corrs = cross_corr_time_lag_pair(x, y, tau_max)
    max_corr = float(np.max(corrs))
    tau_peak = int(taus[np.argmax(corrs)])
    mean_corr = float(np.mean(corrs))
    std_corr = float(np.std(corrs))
    if std_corr < 1e-6:
        std_corr = 1e-6
    Wpos = (max_corr - mean_corr) / std_corr
    return float(Wpos), tau_peak, max_corr

def compute_anchor_correlations(
    output_dir: Union[str, Path],
    anchor_segment_id: Union[int, str],
    tau_max: int = 3,
) -> Dict[str, Any]:
    """
    Tính link weight W^{pos}_{anchor, j} giữa 1 anchor và toàn bộ segments.
    """
    data = load_traffic_tensor(output_dir)
    values = data["values"]            # (T, N)
    segment_ids = data["segment_ids"]  # (N,)

    # tìm index anchor trong segment_ids
    mask = (segment_ids == anchor_segment_id)
    idx_arr = np.where(mask)[0]
    if len(idx_arr) == 0:
        raise ValueError(f"Anchor segment_id {anchor_segment_id} not found in segment_ids.")
    anchor_idx = int(idx_arr[0])

    x = values[:, anchor_idx]  # (T,)

    T, N = values.shape
    weights = np.zeros(N, dtype=np.float32)
    tau_peaks = np.zeros(N, dtype=np.int32)
    max_corrs = np.zeros(N, dtype=np.float32)

    for j in range(N):
        if j == anchor_idx:
            # self-link
            weights[j] = 0.0
            tau_peaks[j] = 0
            max_corrs[j] = 1.0
            continue

        y = values[:, j]  # (T,)
        Wpos, tau_peak, max_corr = link_weight_pos(x, y, tau_max=tau_max)

        weights[j] = Wpos
        tau_peaks[j] = tau_peak
        max_corrs[j] = max_corr

    return {
        "anchor_segment_id": anchor_segment_id,
        "segment_ids": segment_ids,
        "anchor_index": anchor_idx,
        "weights": weights,
        "tau_peaks": tau_peaks,
        "max_corrs": max_corrs,
    }

# ============================================================
# 3) Haversine distance (m) giữa 2 điểm (lat, lon)
# ============================================================
def haversine_distance(lat1, lon1, lat2, lon2) -> float:
    """
    lat, lon tính bằng độ. Kết quả trả về mét.
    """
    R = 6371000.0  # bán kính Trái Đất (m)

    phi1 = radians(lat1)
    phi2 = radians(lat2)
    dphi = radians(lat2 - lat1)
    dlambda = radians(lon2 - lon1)

    a = sin(dphi / 2.0) ** 2 + cos(phi1) * cos(phi2) * sin(dlambda / 2.0) ** 2
    c = 2 * asin(sqrt(a))
    return float(R * c)


def load_node_positions(output_dir: Union[str, Path]) -> Dict[int, Tuple[float, float]]:
    """
    Đọc nodes.csv → map node_id -> (lat, lon)
    Giả sử file nằm trong: output_dir / "tomtom_stats" / "nodes.csv"
    """
    nodes_path = output_dir  / "nodes.csv"
    df_nodes = pd.read_csv(nodes_path)

    node2pos: Dict[int, Tuple[float, float]] = {}
    for _, row in df_nodes.iterrows():
        node_id = int(row["node_id"])
        lat = float(row["lat"])
        lon = float(row["lon"])
        node2pos[node_id] = (lat, lon)
    return node2pos


def load_segment_centers(output_dir: Union[str, Path]) -> Dict[int, Tuple[float, float]]:
    """
    Đọc segments.csv + nodes.csv để tính center (lat, lon)
    của mỗi segment_id.

    center = midpoint(node_u, node_v) theo lat/lon.
    """
    seg_path = output_dir  / "segments.csv"

    df_segments = pd.read_csv(seg_path)
    node2pos = load_node_positions(output_dir)

    seg2center: Dict[int, Tuple[float, float]] = {}
    for _, row in df_segments.iterrows():
        seg_id = int(row["segment_id"])
        u = int(row["node_u"])
        v = int(row["node_v"])

        if (u not in node2pos) or (v not in node2pos):
            # thiếu thông tin node → bỏ qua (sẽ set NaN sau)
            continue

        lat_u, lon_u = node2pos[u]
        lat_v, lon_v = node2pos[v]
        lat_c = (lat_u + lat_v) / 2.0
        lon_c = (lon_u + lon_v) / 2.0
        seg2center[seg_id] = (lat_c, lon_c)

    return seg2center

def compute_dist_to_anchor_from_centers(
    output_dir: Union[str, Path],
    segment_ids_tensor: np.ndarray,
    anchor_index: int,
) -> np.ndarray:
    """
    Từ segment_ids trong traffic_tensor (segment_ids_tensor),
    dùng segments.csv + nodes.csv để tính center (lat, lon),
    sau đó tính khoảng cách Haversine từ anchor đến mọi segment.

    return:
        dist_to_anchor: (N,) mét. Seg nào thiếu center sẽ có dist = +inf.
    """
    seg2center = load_segment_centers(output_dir)

    N = len(segment_ids_tensor)
    lat_all = np.full(N, np.nan, dtype="float32")
    lon_all = np.full(N, np.nan, dtype="float32")

    # map theo đúng thứ tự segment_ids_tensor
    for i, sid in enumerate(segment_ids_tensor):
        sid_int = int(sid)
        if sid_int in seg2center:
            lat_all[i] = seg2center[sid_int][0]
            lon_all[i] = seg2center[sid_int][1]
        else:
            # thiếu center → giữ NaN
            pass

    lat_anchor = float(lat_all[anchor_index])
    lon_anchor = float(lon_all[anchor_index])

    dist_to_anchor = np.full(N, np.inf, dtype="float32")

    for j in range(N):
        if j == anchor_index:
            dist_to_anchor[j] = 0.0
            continue
        if np.isnan(lat_all[j]) or np.isnan(lon_all[j]):
            # không có center → coi như rất xa
            dist_to_anchor[j] = np.inf
            continue

        d = haversine_distance(
            lat_anchor, lon_anchor,
            float(lat_all[j]), float(lon_all[j])
        )
        dist_to_anchor[j] = d

    return dist_to_anchor  # (N,) mét

def build_influenced_subgraph(
    output_dir: Union[str, Path],
    corr_info: Dict[str, Any],
    tau_cut: int = 40,
    Wmin: float = None,
    Dmax: float = None,
    top_k: int = 30,
) -> Dict[str, Any]:
    """
    Lọc các tuyến có khả năng bị anchor (segment tâm vùng) ảnh hưởng, dùng:
      - |tau_peak| <= tau_cut
      - Wpos >= Wmin  (nếu Wmin=None thì lấy theo percentile)
      - dist_to_anchor <= Dmax (mét; nếu Dmax=None thì lấy theo percentile)

    Sau đó sort theo Wpos giảm dần và lấy top_k neighbor.
    """
    output_dir = Path(output_dir)

    anchor_segment_id = corr_info["anchor_segment_id"]
    segment_ids = corr_info["segment_ids"]
    anchor_idx = int(corr_info["anchor_index"])
    weights = corr_info["weights"]
    tau_peaks = corr_info["tau_peaks"]
    max_corrs = corr_info["max_corrs"]

    # 1) khoảng cách (Haversine) tới anchor, theo thứ tự segment_ids
    dist_to_anchor = compute_dist_to_anchor_from_centers(
        output_dir, segment_ids, anchor_idx
    )  # (N,) mét

    # 2) lọc theo |tau_peak|
    mask = np.abs(tau_peaks) <= tau_cut

    W_candidates   = weights[mask]
    tau_candidates = tau_peaks[mask]
    corr_candidates = max_corrs[mask]
    dist_candidates = dist_to_anchor[mask]
    idx_candidates  = np.arange(len(segment_ids))[mask]

    if W_candidates.size == 0:
        raise RuntimeError("Không có candidate nào sau khi lọc theo tau_cut.")

    # 3) nếu chưa set Wmin/Dmax thì lấy theo percentile
    if Wmin is None:
        Wmin = float(np.percentile(W_candidates, 70))   # top 30% mạnh nhất
    if Dmax is None:
        Dmax = float(np.percentile(dist_candidates[np.isfinite(dist_candidates)], 70))

    mask2 = (W_candidates >= Wmin) & (dist_candidates <= Dmax)

    W_sel   = W_candidates[mask2]
    tau_sel = tau_candidates[mask2]
    corr_sel = corr_candidates[mask2]
    dist_sel = dist_candidates[mask2]
    idx_sel  = idx_candidates[mask2]

    # --- bỏ anchor trên TẤT CẢ các mảng ---
    mask_not_anchor = (idx_sel != anchor_idx)
    W_sel   = W_sel[mask_not_anchor]
    tau_sel = tau_sel[mask_not_anchor]
    corr_sel = corr_sel[mask_not_anchor]
    dist_sel = dist_sel[mask_not_anchor]
    idx_sel  = idx_sel[mask_not_anchor]

    if idx_sel.size == 0:
        raise RuntimeError("Không còn neighbor nào sau khi áp Wmin/Dmax, hãy giảm ngưỡng.")

    # 4) sort theo Wpos giảm dần
    order = np.argsort(-W_sel)
    W_sel   = W_sel[order]
    tau_sel = tau_sel[order]
    corr_sel = corr_sel[order]
    dist_sel = dist_sel[order]
    idx_sel  = idx_sel[order]

    # 5) lấy top_k
    if top_k is not None and idx_sel.size > top_k:
        W_sel   = W_sel[:top_k]
        tau_sel = tau_sel[:top_k]
        corr_sel = corr_sel[:top_k]
        dist_sel = dist_sel[:top_k]
        idx_sel  = idx_sel[:top_k]

    neighbor_indices = idx_sel
    neighbor_segment_ids = segment_ids[neighbor_indices]
    selected_indices = np.concatenate([[anchor_idx], neighbor_indices])

    return {
        "anchor_segment_id": anchor_segment_id,
        "anchor_index": anchor_idx,
        "segment_ids": segment_ids,
        "selected_indices": selected_indices,
        "neighbor_segment_ids": neighbor_segment_ids,
        "neighbor_indices": neighbor_indices,
        "Wpos": W_sel,
        "tau_peaks": tau_sel,
        "max_corrs": corr_sel,
        "dist_to_anchor": dist_sel,      # mét
        "Wmin_used": Wmin,
        "Dmax_used": Dmax,
    }


def build_laplacian_from_distance(
    subgraph_info: Dict[str, Any],
    d_spa: int = 16,
    sigma: float = 500.0,
) -> Dict[str, Any]:
    """
    Xây adjacency + Laplacian eigenvectors cho subgraph
    dùng khoảng cách Haversine tới anchor (mét):

      - node 0: anchor
      - node 1..k: neighbors
      - A[0,i] = A[i,0] = exp(-dist_i / sigma)

    sigma: tham số scale (m). Ví dụ:
      - 500 → edge weight ~0.37 tại 500m
      - 1000 → edge weight ~0.37 tại 1km
    """
    dist_to_anchor = subgraph_info["dist_to_anchor"]  # (k,) mét
    k = dist_to_anchor.shape[0]
    n_nodes = k + 1

    A = np.zeros((n_nodes, n_nodes), dtype="float32")
    for i in range(1, n_nodes):
        d = float(dist_to_anchor[i - 1])  # mét
        w = float(np.exp(-d / sigma))
        A[0, i] = w
        A[i, 0] = w

    d_vec = A.sum(axis=1)
    D = np.diag(d_vec)
    L = D - A

    evals, evecs = np.linalg.eigh(L)
    idx = np.argsort(evals)
    if d_spa < n_nodes:
        idx = idx[:d_spa]
    lap_eigvec = evecs[:, idx].astype("float32")  # (n_nodes, d_spa)

    out = dict(subgraph_info)
    out.update({
        "adjacency": A,
        "lap_eigvec": lap_eigvec,
    })
    return out

def load_topology(output_dir: Union[str, Path]):
    edges = pd.read_csv(output_dir / "edges.csv")   # (segment_u, segment_v)
    idx_map = pd.read_csv(output_dir / "segment_index.csv")  # (idx, segment_id)

    segid2idx = {int(r.segment_id): int(r.idx) for _, r in idx_map.iterrows()}

    # Build adjacency list (undirected)
    graph = {}
    for _, row in edges.iterrows():
        u = int(row["segment_u"])
        v = int(row["segment_v"])
        graph.setdefault(u, []).append(v)
        graph.setdefault(v, []).append(u)
    return graph, segid2idx

def get_two_hop_zone(seed_seg_id: int, graph: Dict[int, list]):
    hop1 = set(graph.get(seed_seg_id, []))
    hop2 = set()

    for n1 in hop1:
        for n2 in graph.get(n1, []):
            if n2 != seed_seg_id and n2 not in hop1:
                hop2.add(n2)

    return {seed_seg_id} | hop1 | hop2

def spatial_radius_filter(candidate_seg_ids, seg2center, seed_seg_id, R_max):
    lat0, lon0 = seg2center[seed_seg_id]
    out = []
    for sid in candidate_seg_ids:
        if sid not in seg2center:
            continue
        lat, lon = seg2center[sid]
        d = haversine_distance(lat0, lon0, lat, lon)
        if d <= R_max:
            out.append(sid)
    return out

def spatial_radius_filter(candidate_seg_ids, seg2center, seed_seg_id, R_max):
    lat0, lon0 = seg2center[seed_seg_id]
    out = []
    for sid in candidate_seg_ids:
        if sid not in seg2center:
            continue
        lat, lon = seg2center[sid]
        d = haversine_distance(lat0, lon0, lat, lon)
        if d <= R_max:
            out.append(sid)
    return out

def compute_zone_correlations(values, segment_ids, seed_idx, candidate_indices, tau_max):
    W = {}
    Tau = {}
    MaxCorr = {}

    x = values[:, seed_idx]

    for j in candidate_indices:
        if j == seed_idx:
            continue
        y = values[:, j]
        w, tau, mc = link_weight_pos(x, y, tau_max)
        W[j] = w
        Tau[j] = tau
        MaxCorr[j] = mc
    return W, Tau, MaxCorr

def filter_links(seed_idx, candidate_indices, seg2center, segment_ids, W, Tau,
                 Wmin, Dmin, Dmax, tau_cut):
    keep = []
    lat0, lon0 = seg2center[int(segment_ids[seed_idx])]

    for j in candidate_indices:
        sid = int(segment_ids[j])
        if sid not in seg2center:
            continue
        lat, lon = seg2center[sid]
        d = haversine_distance(lat0, lon0, lat, lon)

        if abs(Tau[j]) > tau_cut:
            continue
        if W[j] < Wmin:
            continue
        if not (Dmin <= d <= Dmax):
            continue
        keep.append(j)

    return keep

def build_zone(output_dir, seed_seg_id, R_max=2000, tau_max=10,
               Wmin=0.5, Dmin=0, Dmax=1500, tau_cut=10, top_k=60, d_spa=16,
               sigma=500.0):
    output_dir = Path(output_dir)

    # --------- load data + topo ----------
    graph, segid2idx = load_topology(output_dir)          # graph: seg_id -> list neighbors
    seg2center = load_segment_centers(output_dir)         # seg_id -> (lat,lon)
    data = load_traffic_tensor(output_dir)

    values = data["values"]          # (T, N)
    segment_ids = data["segment_ids"]# (N,)

    seed_seg_id = int(seed_seg_id)
    seed_idx = segid2idx[seed_seg_id]

    # 1) 2-hop zone quanh seed (theo topology)
    zone_segids = get_two_hop_zone(seed_seg_id, graph)    # set(seg_id)

    # 2) Spatial radius filter quanh seed
    zone_segids = spatial_radius_filter(zone_segids, seg2center, seed_seg_id, R_max)
    zone_segids = set(zone_segids)  # đảm bảo là set

    # 3) Convert segment_id -> global idx trong tensor
    zone_indices = [segid2idx[sid] for sid in zone_segids if sid in segid2idx]

    # 4) Tìm *tuyến liên quan* cho từng segment trong zone
    #
    #   Với mỗi seg_i trong zone_segids:
    #       - Lấy 2-hop topology quanh seg_i
    #       - (optional) radius filter quanh seg_i nếu muốn
    #       - Chỉ trong tập candidates này mới tính cross-correlation với seg_i
    #
    correlated_segids = set()            # các seg_id liên quan tìm được
    link_weights = {}                    # (i_idx, j_idx) -> Wpos, dùng sau để sort / top_k

    for sid_i in zone_segids:
        if sid_i not in segid2idx:
            continue
        idx_i = segid2idx[sid_i]
        x_i = values[:, idx_i]           # (T,)

        # 4.1) 2-hop topology quanh sid_i
        local_neis = get_two_hop_zone(sid_i, graph)   # set(seg_id)
        if sid_i in local_neis:
            local_neis.remove(sid_i)

        # 4.2) giới hạn nhỏ hơn R_max
        local_neis = spatial_radius_filter(local_neis, seg2center, sid_i, R_max)

        # 4.3) Lặp qua các candidate trong 2-hop của sid_i
        for sid_j in local_neis:
            if sid_j not in segid2idx:
                continue
            idx_j = segid2idx[sid_j]

            # khoảng cách địa lý giữa i và j
            if (sid_i not in seg2center) or (sid_j not in seg2center):
                continue
            lat_i, lon_i = seg2center[sid_i]
            lat_j, lon_j = seg2center[sid_j]
            d_ij = haversine_distance(lat_i, lon_i, lat_j, lon_j)
            if d_ij < Dmin or d_ij > Dmax:
                continue

            # cross-correlation (lead–lag)
            Wpos, tau_peak, _ = link_weight_pos(x_i, values[:, idx_j], tau_max=tau_max)
            if abs(tau_peak) > tau_cut:
                continue
            if Wpos < Wmin:
                continue

            correlated_segids.add(sid_j)
            link_weights[(idx_i, idx_j)] = float(Wpos)

    # 5) Zone cuối cùng = zone ban đầu ∪ các tuyến liên quan
    final_segids = set(zone_segids) | correlated_segids

    # 6) Nếu quá nhiều node => giới hạn top_k nhưng luôn giữ toàn bộ zone ban đầu
    if top_k is not None and len(final_segids) > top_k:
        base_segids = set(zone_segids)
        extra_segids = list(final_segids - base_segids)

        # score mỗi node = max Wpos của các link liên quan tới node đó
        node_score = {sid: 0.0 for sid in extra_segids}
        for (gi, gj), w in link_weights.items():
            sid_gi = int(segment_ids[gi])
            sid_gj = int(segment_ids[gj])
            if sid_gi in node_score:
                node_score[sid_gi] = max(node_score[sid_gi], w)
            if sid_gj in node_score:
                node_score[sid_gj] = max(node_score[sid_gj], w)

        extra_sorted = sorted(extra_segids, key=lambda s: node_score.get(s, 0.0), reverse=True)
        remain_slots = max(0, top_k - len(base_segids))
        extra_kept = extra_sorted[:remain_slots]
        final_segids = base_segids | set(extra_kept)

    # 7) Convert final seg_id -> global idx
    final_segids = list(final_segids)
    # có thể ưu tiên seed đứng đầu cho dễ debug
    final_segids_sorted = [seed_seg_id] + [s for s in final_segids if s != seed_seg_id]
    final_indices = np.array([segid2idx[s] for s in final_segids_sorted], dtype=np.int64)

    # 8) Build adjacency + Laplacian CHỈ TỪ EDGES NỘI BỘ TRONG ZONE
    #    → Spatial PE thể hiện quan hệ không gian giữa tất cả tuyến trong vùng
    edges = pd.read_csv(output_dir / "edges.csv")

    zone_set = set(final_segids_sorted)
    N_zone = len(final_segids_sorted)
    global2local = {g_idx: i for i, g_idx in enumerate(final_indices)}

    A = np.zeros((N_zone, N_zone), dtype="float32")

    for _, row in edges.iterrows():
        su = int(row["segment_u"])
        sv = int(row["segment_v"])
        # nếu 2 segment này đều nằm trong zone_final thì thêm cạnh
        if (su not in zone_set) or (sv not in zone_set):
            continue
        iu = global2local[segid2idx[su]]
        iv = global2local[segid2idx[sv]]

        if (su in seg2center) and (sv in seg2center):
            lat_u, lon_u = seg2center[su]
            lat_v, lon_v = seg2center[sv]
            d_uv = haversine_distance(lat_u, lon_u, lat_v, lon_v)
            w = float(np.exp(-d_uv / sigma))   # weight theo distance
        else:
            w = 1.0

        A[iu, iv] += w
        A[iv, iu] += w

    # Không thêm self-loop
    np.fill_diagonal(A, 0.0)

    # 9) Laplacian eigenvectors
    d_vec = A.sum(axis=1)
    D = np.diag(d_vec)
    L = D - A

    evals, evecs = np.linalg.eigh(L)
    order = np.argsort(evals)
    if d_spa < N_zone:
        order = order[:d_spa]
    lap_eigvec = evecs[:, order].astype("float32")  # (N_zone, d_spa)

    return {
        "seed_segment_id": seed_seg_id,
        "zone_indices": final_indices,                 # index vào traffic_tensor
        "zone_segment_ids": segment_ids[final_indices],
        "adjacency": A,                                # (N_zone, N_zone)
        "lap_eigvec": lap_eigvec,                      # (N_zone, d_spa)
    }

#%%
