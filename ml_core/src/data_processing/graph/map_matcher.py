# src/data_processing/graph/map_matcher.py
"""
Map Matcher — gán TomTom traffic features vào OSM edges (Option B: subgraph).

Design: chỉ giữ lại OSM edges nào có TomTom segment match vào.
OSM edges không có TomTom data → bị loại bỏ hoàn toàn (không dùng default giả).

Logic (shortest-path matching):
    1. Với mỗi TomTom segment, snap điểm ĐẦU và CUỐI của shape lên OSM node gần nhất.
    2. Tính shortest_path(start_node → end_node) trên OSM graph.
    3. Tất cả OSM edges trên path đó đều nhận features của TomTom segment này.
       → Giải quyết vấn đề 1 TomTom segment dài bị chia thành nhiều OSM edges nhỏ.
    4. Aggregate nếu nhiều TomTom segments cover cùng 1 OSM edge → mean.
    5. Build subgraph từ tập OSM edges đã được match.
    6. Edge features = TomTom features thực (100% real, không có default).
    7. Temporal: edge_features_temporal [E', T, F_edge].
"""

from __future__ import annotations

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

try:
    import osmnx as ox
    import networkx as nx
except ImportError:
    raise ImportError("Cài đặt: pip install osmnx networkx")

from utils.config import config
from utils.logger import LoggerMixin


# Features TomTom dùng để aggregate vào edges
TOMTOM_FEATURE_COLS = [
    "average_speed",
    "harmonic_average_speed",
    "median_speed",
    "std_speed",
    "average_travel_time",
    "travel_time_ratio",
    "congestion_index",
    "speed_limit_ratio",
    "sample_size",
]

# Edge features = TomTom dynamic + OSM static
# Mỗi edge trong output graph = 1 OSM road segment đã được TomTom match
EDGE_FEATURE_COLS = [
    # TomTom dynamic features (100% real data, không có default)
    "average_speed",
    "harmonic_average_speed",
    "std_speed",
    "travel_time_ratio",
    "congestion_index",
    "speed_limit_ratio",
    "sample_size",
    # OSM static features
    "osm_length_m",
    "osm_maxspeed",
    "osm_lanes",
    "osm_highway_type",
]

# Node features = topology của subgraph (ngã tư thuộc edges được match)
NODE_FEATURE_COLS = [
    "degree",           # số edges liền kề trong subgraph
    "betweenness_norm", # betweenness centrality trong subgraph
    "lat_norm",         # tọa độ đã chuẩn hóa
    "lon_norm",
]


class TomTomOSMMapMatcher(LoggerMixin):
    """
    Map-match TomTom probe data segments vào OSM road graph.

    Sử dụng OSMnx spatial index để tìm nearest edge, hiệu quả hơn brute-force.

    Args:
        osm_graph_data  : Dict từ OSMGraphBuilder.load_latest() hoặc build_from_polygon().
        match_threshold_m : Khoảng cách tối đa (m) để accept một match.
                            Segments xa hơn được bỏ qua (không đủ confidence).
        aggregate_method  : Cách aggregate khi nhiều TomTom segments → 1 OSM edge.
                            "mean" (mặc định) hoặc "median".
    """

    def __init__(
        self,
        osm_graph_data: Dict[str, np.ndarray],
        match_threshold_m: float = 100.0,
        aggregate_method: str = "mean",
    ):
        self.osm_data = osm_graph_data
        self.match_threshold_m = match_threshold_m
        self.aggregate_method = aggregate_method

        # Giải nén dữ liệu OSM
        self.osm_node_ids = osm_graph_data["osm_node_ids"]           # [N]
        self.coordinates = osm_graph_data["coordinates"]             # [N, 2] lat/lon
        self.edge_index = osm_graph_data["edge_index"]               # [2, E]
        self.edge_lengths = osm_graph_data["edge_lengths"]           # [E]
        self.edge_maxspeed = osm_graph_data["edge_maxspeed"]         # [E] km/h
        self.edge_lanes = osm_graph_data["edge_lanes"]               # [E]
        self.edge_highway_type = osm_graph_data["edge_highway_type"] # [E]
        self.adjacency_matrix = osm_graph_data["adjacency_matrix"]   # [N, N]

        N = len(self.osm_node_ids)
        self.node_id_to_idx: Dict[int, int] = {
            int(nid): idx for idx, nid in enumerate(self.osm_node_ids)
        }

        # Sẽ được populated sau match
        self._matched_features: Optional[np.ndarray] = None  # [N, F]
        self._coverage_ratio: float = 0.0

        self.logger.info(
            f"MapMatcher initialized: {N} OSM nodes, "
            f"{self.edge_index.shape[1]} directed edges, "
            f"threshold={match_threshold_m}m"
        )

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def match_and_build_node_features(
        self,
        traffic_df: pd.DataFrame,
        osm_networkx_graph: "nx.MultiDiGraph",
        time_slot: Optional[str] = None,
        output_name: str = "graph_structure",
        output_dir: Optional[Path] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Match TomTom data → OSM edges, sau đó aggregate lên OSM nodes.

        Args:
            traffic_df          : DataFrame từ NPZReader.read_features().
                                  Cần cột: segment_id, latitude/longitude (raw),
                                  average_speed, travel_time_ratio, ...
            osm_networkx_graph  : NetworkX graph từ OSMGraphBuilder.G (để dùng ox.nearest_edges).
            time_slot           : Filter theo time_slot cụ thể (vd "Slot_0700").
                                  None = aggregate tất cả time slots → mean.
            output_name         : Tên file NPZ output.
            output_dir          : Thư mục output.

        Returns:
            Dict chứa node_features, edge_index, adjacency_matrix,
            edge_features, feature_names, và metadata.
        """
        self.logger.info("=" * 60)
        self.logger.info("MAP MATCHER — TomTom → OSM")
        self.logger.info("=" * 60)

        # ── Chuẩn bị TomTom data ─────────────────────────────────────────────
        df = self._prepare_traffic_df(traffic_df, time_slot)
        if df.empty:
            self.logger.error("Không có dữ liệu TomTom sau khi filter.")
            return {}

        self.logger.info(
            f"TomTom data: {len(df)} records, "
            f"{df['segment_id'].nunique()} unique segments"
        )

        # ── Map matching ──────────────────────────────────────────────────────
        matched_df = self._map_match_segments(df, osm_networkx_graph)
        n_unique = df['segment_id'].nunique()
        n_matched = matched_df['segment_id'].nunique() if not matched_df.empty else 0
        self._coverage_ratio = n_matched / max(n_unique, 1)
        self.logger.info(
            f"Matched: {n_matched} / {n_unique} segments "
            f"(coverage = {self._coverage_ratio:.1%})"
        )

        if matched_df.empty:
            self.logger.error("Không có segment nào được match. Tăng match_threshold_m?")
            return {}

        # ── Build matched subgraph (chỉ OSM edges có TomTom data) ────────────
        result = self._build_matched_subgraph(matched_df)

        # ── Save ──────────────────────────────────────────────────────────────
        if output_dir:
            self._save_npz(result, output_name, output_dir, time_slot)

        return result

    def match_and_build_temporal_features(
        self,
        traffic_df: pd.DataFrame,
        osm_networkx_graph: "nx.MultiDiGraph",
        output_name: str = "graph_structure_temporal",
        output_dir: Optional[Path] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Build temporal edge features: [E', T, F_edge] cho T-GCN / DTC-STGCN.

        Design (Option B): chỉ giữ OSM edges có TomTom data.
            1. Chạy map matching trên TOÀN BỘ data (tất cả time slots) để xác định
               tập edges cố định E' = subgraph edges (stable across time).
            2. Với mỗi time slot t, aggregate TomTom features → edge features [E', F_edge].
            3. Output tensor: edge_features_temporal [E', T, F_edge].
               Node features cũng được tính nhưng chỉ là topology (không đổi theo t).

        Args:
            traffic_df   : DataFrame đầy đủ 31 ngày × 24 time slots.
            output_name  : Tên NPZ output.
        """
        self.logger.info("=" * 60)
        self.logger.info("MAP MATCHER — Temporal edge features [E', T, F]")
        self.logger.info("=" * 60)

        df = self._prepare_traffic_df(traffic_df, time_slot=None)
        if df.empty:
            self.logger.error("DataFrame rỗng sau khi prepare.")
            return {}

        date_col = next(
            (c for c in ["date_from", "date_range", "date"] if c in df.columns),
            None,
        )
        if date_col is None:
            self.logger.error("Không tìm thấy cột ngày trong DataFrame.")
            return {}

        # ── Bước 1: Xác định subgraph cố định từ toàn bộ data ────────────────
        # Match tất cả segments (không filter time slot) để có tập edges ổn định.
        self.logger.info("Bước 1: Xác định subgraph cố định từ toàn bộ data...")
        all_matched_df = self._map_match_segments(df, osm_networkx_graph)

        if all_matched_df.empty:
            self.logger.error("Không match được segment nào. Tăng match_threshold_m?")
            return {}

        total_segments = df["segment_id"].nunique()
        matched_segments = all_matched_df["segment_id"].nunique()
        self._coverage_ratio = matched_segments / max(total_segments, 1)
        self.logger.info(
            f"Coverage: {matched_segments}/{total_segments} segments "
            f"({self._coverage_ratio:.1%})"
        )

        # Lấy tập edge pairs cố định (u_old_idx, v_old_idx) đã được match ít nhất 1 lần
        edge_pairs = (
            all_matched_df
            .groupby(["osm_u_idx", "osm_v_idx"])
            .size()
            .reset_index()[["osm_u_idx", "osm_v_idx"]]
        )
        edge_pairs["osm_u_idx"] = edge_pairs["osm_u_idx"].astype(int)
        edge_pairs["osm_v_idx"] = edge_pairs["osm_v_idx"].astype(int)

        # Re-index nodes của subgraph
        old_node_idxs = np.unique(
            np.concatenate([
                edge_pairs["osm_u_idx"].values,
                edge_pairs["osm_v_idx"].values,
            ])
        )
        N_sub = len(old_node_idxs)
        old_to_new: Dict[int, int] = {
            int(old_idx): new_idx for new_idx, old_idx in enumerate(old_node_idxs)
        }

        sub_node_ids = self.osm_node_ids[old_node_idxs]
        sub_coords   = self.coordinates[old_node_idxs]

        new_u_arr = np.array([old_to_new[u] for u in edge_pairs["osm_u_idx"].values], dtype=np.int64)
        new_v_arr = np.array([old_to_new[v] for v in edge_pairs["osm_v_idx"].values], dtype=np.int64)
        sub_edge_index = np.stack([new_u_arr, new_v_arr], axis=0)  # [2, E_sub]
        E_sub = sub_edge_index.shape[1]

        sub_adj = np.zeros((N_sub, N_sub), dtype=np.float32)
        sub_adj[new_u_arr, new_v_arr] = 1.0

        self.logger.info(f"Subgraph cố định: {N_sub} nodes, {E_sub} edges")

        # ── Bước 2: Build temporal edge feature tensor [E_sub, T, F_edge] ─────
        df["_ts_key"] = df[date_col].astype(str) + "__" + df["time_set"].astype(str)
        ts_keys = sorted(df["_ts_key"].unique())
        T = len(ts_keys)
        F_edge = len(EDGE_FEATURE_COLS)

        self.logger.info(f"Building temporal tensor: E={E_sub}, T={T}, F={F_edge}")

        # edge key → index trong sub_edge_index
        edge_key_to_idx: Dict[Tuple[int, int], int] = {
            (int(edge_pairs.iloc[i]["osm_u_idx"]), int(edge_pairs.iloc[i]["osm_v_idx"])): i
            for i in range(E_sub)
        }

        # Tensor [E_sub, T, F_edge] — khởi tạo NaN
        edge_tensor = np.full((E_sub, T, F_edge), np.nan, dtype=np.float32)

        tomtom_cols = [c for c in TOMTOM_FEATURE_COLS if c in df.columns]

        for t_idx, ts_key in enumerate(ts_keys):
            slot_df = df[df["_ts_key"] == ts_key].copy()
            if slot_df.empty:
                continue

            try:
                matched_slot = self._map_match_segments(slot_df, osm_networkx_graph)
                if matched_slot.empty:
                    continue

                # Aggregate per (u_old, v_old) cho slot này
                agg_funcs = {c: self.aggregate_method for c in tomtom_cols if c in matched_slot.columns}
                slot_agg = (
                    matched_slot
                    .groupby(["osm_u_idx", "osm_v_idx"], sort=False)
                    .agg(agg_funcs)
                    .reset_index()
                )

                # Thêm OSM static features
                global_src = self.edge_index[0]
                global_dst = self.edge_index[1]
                for col_name, arr in [
                    ("osm_length_m",    self.edge_lengths),
                    ("osm_maxspeed",    self.edge_maxspeed),
                    ("osm_lanes",       self.edge_lanes),
                    ("osm_highway_type", self.edge_highway_type),
                ]:
                    vals = []
                    for u_old, v_old in zip(
                        slot_agg["osm_u_idx"].values, slot_agg["osm_v_idx"].values
                    ):
                        mask = (global_src == u_old) & (global_dst == v_old)
                        vals.append(float(arr[mask][0]) if mask.any() else 0.0)
                    slot_agg[col_name] = vals

                # Điền vào tensor
                for _, row in slot_agg.iterrows():
                    key = (int(row["osm_u_idx"]), int(row["osm_v_idx"]))
                    e_idx = edge_key_to_idx.get(key)
                    if e_idx is None:
                        continue
                    for f_idx, col in enumerate(EDGE_FEATURE_COLS):
                        val = row.get(col, np.nan)
                        if pd.notna(val):
                            edge_tensor[e_idx, t_idx, f_idx] = float(val)

            except Exception as e:
                self.logger.warning(f"Slot {ts_key} failed: {e}")
                continue

            if (t_idx + 1) % 50 == 0:
                self.logger.info(f"Progress: {t_idx + 1}/{T} time slots")

        # Forward-fill NaN theo chiều thời gian cho mỗi edge
        edge_tensor = self._fill_nan_temporal_edges(edge_tensor)

        # ── Bước 3: Node features (topology — không đổi theo thời gian) ───────
        degrees = sub_adj.sum(axis=1).astype(np.float32)
        betweenness = np.zeros(N_sub, dtype=np.float32)
        try:
            import networkx as nx
            G_sub = nx.DiGraph()
            G_sub.add_nodes_from(range(N_sub))
            G_sub.add_edges_from(zip(new_u_arr.tolist(), new_v_arr.tolist()))
            bt_dict = nx.betweenness_centrality(G_sub, normalized=True)
            for nidx, bt_val in bt_dict.items():
                betweenness[nidx] = float(bt_val)
        except Exception as e:
            self.logger.warning(f"Betweenness failed: {e}")

        lats = sub_coords[:, 0]
        lons = sub_coords[:, 1]
        lat_norm = ((lats - lats.min()) / (lats.max() - lats.min() + 1e-8)).astype(np.float32)
        lon_norm = ((lons - lons.min()) / (lons.max() - lons.min() + 1e-8)).astype(np.float32)
        deg_norm = (degrees / (degrees.max() + 1e-8)).astype(np.float32)

        node_features = np.stack([deg_norm, betweenness, lat_norm, lon_norm], axis=1)  # [N_sub, 4]

        timestamps = np.array(ts_keys)

        output = {
            "edge_features_temporal": edge_tensor,      # [E_sub, T, F_edge]
            "node_features":          node_features,     # [N_sub, F_node]
            "timestamps":             timestamps,         # [T]
            "osm_node_ids":           sub_node_ids,       # [N_sub]
            "coordinates":            sub_coords,         # [N_sub, 2]
            "edge_index":             sub_edge_index,     # [2, E_sub]
            "adjacency_matrix":       sub_adj,            # [N_sub, N_sub]
            "node_feature_names":     np.array(NODE_FEATURE_COLS),
            "edge_feature_names":     np.array(EDGE_FEATURE_COLS),
        }

        if output_dir:
            self._save_npz(output, output_name, output_dir, time_slot=None)

        self.logger.info(
            f"✅ Temporal edge features built: {edge_tensor.shape} "
            f"(E={E_sub}, T={T}, F={F_edge})"
        )
        return output

    # =========================================================================
    # INTERNAL — DATA PREPARATION
    # =========================================================================

    def _prepare_traffic_df(
        self, df: pd.DataFrame, time_slot: Optional[str]
    ) -> pd.DataFrame:
        """
        Chuẩn bị DataFrame: chọn đúng cột, dùng raw coordinates, filter time slot.

        Giữ lại:
            _lat/_lon       : centroid (mean) của segment — fallback nếu không có start/end
            _lat_start/_lon_start : tọa độ điểm ĐẦU của TomTom segment shape
            _lat_end/_lon_end     : tọa độ điểm CUỐI của TomTom segment shape
        """
        df = df.copy()

        # Dùng raw coordinates (không bị normalize)
        if "raw_latitude" in df.columns and "raw_longitude" in df.columns:
            df["_lat"] = df["raw_latitude"]
            df["_lon"] = df["raw_longitude"]
        elif "latitude" in df.columns and "longitude" in df.columns:
            lat_max = df["latitude"].max()
            if lat_max <= 1.0:
                self.logger.warning(
                    "latitude có vẻ đã normalize ([0,1]). "
                    "Cần raw coordinates để map match chính xác."
                )
            df["_lat"] = df["latitude"]
            df["_lon"] = df["longitude"]
        else:
            self.logger.error("Không tìm thấy cột tọa độ.")
            return pd.DataFrame()

        # Tọa độ start/end của segment (nếu pipeline lưu raw_lat_start/end)
        has_endpoints = (
            "raw_lat_start" in df.columns and "raw_lon_start" in df.columns
            and "raw_lat_end" in df.columns and "raw_lon_end" in df.columns
        )
        if has_endpoints:
            df["_lat_start"] = df["raw_lat_start"]
            df["_lon_start"] = df["raw_lon_start"]
            df["_lat_end"]   = df["raw_lat_end"]
            df["_lon_end"]   = df["raw_lon_end"]

        # Filter time slot
        if time_slot and "time_set" in df.columns:
            df = df[df["time_set"] == time_slot]

        # Drop rows thiếu tọa độ
        df = df.dropna(subset=["_lat", "_lon", "segment_id"])

        # Ép kiểu TOMTOM_FEATURE_COLS về float64
        for col in TOMTOM_FEATURE_COLS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype(np.float64)

        # Aggregate per segment (+ time cols nếu temporal)
        agg_dict: Dict = {"_lat": "mean", "_lon": "mean"}

        if has_endpoints:
            agg_dict["_lat_start"] = "first"
            agg_dict["_lon_start"] = "first"
            agg_dict["_lat_end"]   = "last"
            agg_dict["_lon_end"]   = "last"

        for col in TOMTOM_FEATURE_COLS:
            if col in df.columns and df[col].notna().any():
                agg_dict[col] = self.aggregate_method

        for opt_col in ["street_name", "frc", "speed_limit", "distance"]:
            if opt_col in df.columns:
                agg_dict[opt_col] = "first"

        group_cols = ["segment_id"]
        if time_slot is None:
            for t_col in ["date_from", "date_range", "date", "time_set"]:
                if t_col in df.columns:
                    group_cols.append(t_col)

        try:
            df_seg = df.groupby(group_cols, dropna=False).agg(agg_dict).reset_index()
        except KeyError as e:
            missing_col = str(e).strip("'")
            self.logger.warning(f"Cột '{missing_col}' không tìm thấy khi aggregate, bỏ qua.")
            agg_dict.pop(missing_col, None)
            df_seg = df.groupby(group_cols, dropna=False).agg(agg_dict).reset_index()

        for col in TOMTOM_FEATURE_COLS:
            if col not in df_seg.columns:
                df_seg[col] = np.nan

        # Nếu không có start/end riêng, dùng centroid làm cả hai
        if not has_endpoints:
            df_seg["_lat_start"] = df_seg["_lat"]
            df_seg["_lon_start"] = df_seg["_lon"]
            df_seg["_lat_end"]   = df_seg["_lat"]
            df_seg["_lon_end"]   = df_seg["_lon"]

        return df_seg

    # =========================================================================
    # INTERNAL — MAP MATCHING (shortest-path approach)
    # =========================================================================

    def _map_match_segments(
        self,
        seg_df: pd.DataFrame,
        G: "nx.MultiDiGraph",
    ) -> pd.DataFrame:
        """
        Với mỗi TomTom segment, snap 2 đầu mút lên OSM node gần nhất,
        sau đó tính shortest_path → lấy TẤT CẢ OSM edges trên path.

        Tại sao:
            TomTom segment = đoạn đường dài (ví dụ 500m Nguyễn Huệ).
            OSM chia đoạn đó thành nhiều edges nhỏ (~50-100m mỗi cái).
            nearest_edge chỉ lấy 1 edge → bỏ sót phần còn lại.
            shortest_path lấy toàn bộ → graph đầy đủ như hình màu tím.

        Returns:
            DataFrame với 1 row per (segment_id, osm_u_idx, osm_v_idx) —
            mỗi OSM edge trên path đều có features của TomTom segment đó.
        """
        seg_df = seg_df.reset_index(drop=True)

        # Snap start/end points lên OSM nodes gần nhất
        start_lons = seg_df["_lon_start"].values
        start_lats = seg_df["_lat_start"].values
        end_lons   = seg_df["_lon_end"].values
        end_lats   = seg_df["_lat_end"].values

        try:
            start_osm_nodes = ox.distance.nearest_nodes(G, X=start_lons, Y=start_lats)
            end_osm_nodes   = ox.distance.nearest_nodes(G, X=end_lons,   Y=end_lats)
        except Exception as e:
            self.logger.error(f"ox.distance.nearest_nodes failed: {e}")
            return pd.DataFrame()

        results = []
        n_no_path = 0

        for idx in range(len(seg_df)):
            osm_start = start_osm_nodes[idx] if hasattr(start_osm_nodes, '__len__') else start_osm_nodes
            osm_end   = end_osm_nodes[idx]   if hasattr(end_osm_nodes,   '__len__') else end_osm_nodes

            # Nếu start == end (segment quá ngắn), snap vào 1 node
            if osm_start == osm_end:
                u_idx = self.node_id_to_idx.get(int(osm_start))
                if u_idx is None:
                    continue
                # Lấy edges liền kề node đó thay vì bỏ qua
                edge_mask = self.edge_index[0] == u_idx
                for e_pos in np.where(edge_mask)[0]:
                    v_idx = int(self.edge_index[1, e_pos])
                    row = seg_df.iloc[idx].to_dict()
                    row["osm_u"]      = int(self.osm_node_ids[u_idx])
                    row["osm_v"]      = int(self.osm_node_ids[v_idx])
                    row["osm_u_idx"]  = u_idx
                    row["osm_v_idx"]  = v_idx
                    row["match_dist_m"] = 0.0
                    results.append(row)
                continue

            # Tìm shortest path giữa 2 đầu mút
            try:
                path = nx.shortest_path(G, osm_start, osm_end, weight="length")
            except nx.NetworkXNoPath:
                # Thử chiều ngược lại (đường 1 chiều)
                try:
                    path = nx.shortest_path(G, osm_end, osm_start, weight="length")
                except nx.NetworkXNoPath:
                    n_no_path += 1
                    continue
            except nx.NodeNotFound:
                n_no_path += 1
                continue

            # Mỗi cặp node liên tiếp trên path = 1 OSM edge
            for u_osm, v_osm in zip(path[:-1], path[1:]):
                u_idx = self.node_id_to_idx.get(int(u_osm))
                v_idx = self.node_id_to_idx.get(int(v_osm))

                if u_idx is None or v_idx is None:
                    continue

                # Tính khoảng cách snap (centroid TomTom → edge trung điểm)
                u_coord = self.coordinates[u_idx]
                v_coord = self.coordinates[v_idx]
                mid_lat = (u_coord[0] + v_coord[0]) / 2
                mid_lon = (u_coord[1] + v_coord[1]) / 2
                dist_m = self._haversine_m(
                    seg_df.iloc[idx]["_lat"], seg_df.iloc[idx]["_lon"],
                    mid_lat, mid_lon
                )

                row = seg_df.iloc[idx].to_dict()
                row["osm_u"]       = int(u_osm)
                row["osm_v"]       = int(v_osm)
                row["osm_u_idx"]   = u_idx
                row["osm_v_idx"]   = v_idx
                row["match_dist_m"] = dist_m
                results.append(row)

        if n_no_path > 0:
            self.logger.info(
                f"  {n_no_path}/{len(seg_df)} segments không tìm được path "
                f"(có thể do đường 1 chiều hoặc segment nằm ngoài OSM graph)"
            )

        if not results:
            return pd.DataFrame()

        matched_df = pd.DataFrame(results)
        self.logger.debug(
            f"Matched {seg_df['segment_id'].nunique()} segments "
            f"→ {matched_df.groupby(['osm_u_idx','osm_v_idx']).ngroups} unique OSM edges"
        )
        return matched_df

    # =========================================================================
    # INTERNAL — SUBGRAPH BUILDING (Option B: chỉ giữ edges có TomTom data)
    # =========================================================================

    def _build_matched_subgraph(
        self, matched_df: pd.DataFrame
    ) -> Dict[str, np.ndarray]:
        """
        Xây dựng subgraph CHỈ gồm các OSM edges đã được match với TomTom data.

        Strategy:
            1. Lấy tập các (osm_u_idx, osm_v_idx) từ matched_df.
            2. Aggregate nhiều TomTom segments → 1 OSM edge (mean/median).
            3. Lấy tập nodes = union của tất cả u_idx, v_idx.
            4. Re-index nodes 0..N'-1, rebuild edge_index, adjacency_matrix.
            5. Node features = topology (degree, betweenness, lat/lon) trong subgraph.
            6. Edge features = TomTom features thực (không có default giả).

        Returns:
            Dict với node_features [N', F_node], edge_features [E', F_edge],
            edge_index [2, E'], adjacency_matrix [N', N'], coordinates [N', 2],
            osm_node_ids [N'], v.v.
        """
        if matched_df.empty:
            self.logger.warning("matched_df rỗng — subgraph trống.")
            return {}

        # ── 1. Aggregate TomTom features per OSM edge ─────────────────────────
        # Mỗi OSM edge (u_idx, v_idx) có thể được map bởi nhiều TomTom segments
        # → lấy mean của tất cả.
        tomtom_cols = [
            c for c in TOMTOM_FEATURE_COLS if c in matched_df.columns
        ]

        agg_funcs = {c: self.aggregate_method for c in tomtom_cols}
        # OSM edge attrs lấy từ arrays gốc (không aggregate TomTom)
        # → thêm vào sau khi groupby

        edge_df = (
            matched_df
            .groupby(["osm_u_idx", "osm_v_idx"], sort=False)
            .agg({**agg_funcs, "match_dist_m": "mean"})
            .reset_index()
        )

        # ── 2. Lấy tập nodes trong subgraph, re-index ─────────────────────────
        u_idxs = edge_df["osm_u_idx"].values.astype(int)
        v_idxs = edge_df["osm_v_idx"].values.astype(int)

        # Unique node indices từ OSM global index (đã được lưu trong osm_node_ids)
        old_node_idxs = np.unique(np.concatenate([u_idxs, v_idxs]))
        N_sub = len(old_node_idxs)

        # Map: old_global_idx → new_subgraph_idx (0..N_sub-1)
        old_to_new: Dict[int, int] = {
            int(old_idx): new_idx
            for new_idx, old_idx in enumerate(old_node_idxs)
        }

        # ── 3. Subgraph node arrays ────────────────────────────────────────────
        sub_node_ids  = self.osm_node_ids[old_node_idxs]      # [N_sub]
        sub_coords    = self.coordinates[old_node_idxs]        # [N_sub, 2]

        # ── 4. Rebuild edge_index với new indices ─────────────────────────────
        new_u = np.array([old_to_new[u] for u in u_idxs], dtype=np.int64)
        new_v = np.array([old_to_new[v] for v in v_idxs], dtype=np.int64)
        sub_edge_index = np.stack([new_u, new_v], axis=0)      # [2, E_sub]
        E_sub = sub_edge_index.shape[1]

        # ── 5. Subgraph adjacency matrix ──────────────────────────────────────
        sub_adj = np.zeros((N_sub, N_sub), dtype=np.float32)
        sub_adj[new_u, new_v] = 1.0

        # ── 6. OSM static edge features (từ arrays gốc theo old_edge_idx) ─────
        # Tìm vị trí của mỗi (u_old, v_old) trong edge_index gốc
        # để lấy length, maxspeed, lanes, highway_type.
        osm_length_list    = []
        osm_maxspeed_list  = []
        osm_lanes_list     = []
        osm_highway_list   = []

        global_edge_src = self.edge_index[0]
        global_edge_dst = self.edge_index[1]

        for u_old, v_old in zip(u_idxs, v_idxs):
            # Tìm edge (u_old→v_old) trong global edge_index
            mask = (global_edge_src == u_old) & (global_edge_dst == v_old)
            if mask.any():
                e_pos = np.where(mask)[0][0]
                osm_length_list.append(float(self.edge_lengths[e_pos]))
                osm_maxspeed_list.append(float(self.edge_maxspeed[e_pos]))
                osm_lanes_list.append(float(self.edge_lanes[e_pos]))
                osm_highway_list.append(float(self.edge_highway_type[e_pos]))
            else:
                # Edge reverse hoặc không tìm thấy → fallback
                osm_length_list.append(0.0)
                osm_maxspeed_list.append(0.0)
                osm_lanes_list.append(1.0)
                osm_highway_list.append(14.0)  # "other"

        edge_df["osm_length_m"]    = osm_length_list
        edge_df["osm_maxspeed"]    = osm_maxspeed_list
        edge_df["osm_lanes"]       = osm_lanes_list
        edge_df["osm_highway_type"] = osm_highway_list

        # ── 7. Build edge feature matrix [E_sub, F_edge] ──────────────────────
        F_edge = len(EDGE_FEATURE_COLS)
        edge_features = np.zeros((E_sub, F_edge), dtype=np.float32)
        for f_idx, col in enumerate(EDGE_FEATURE_COLS):
            if col in edge_df.columns:
                vals = pd.to_numeric(edge_df[col], errors="coerce").values
                edge_features[:, f_idx] = np.nan_to_num(
                    vals.astype(np.float32), nan=0.0
                )

        # ── 8. Node features: degree + betweenness + lat/lon trong subgraph ───
        degrees = sub_adj.sum(axis=1).astype(np.float32)  # [N_sub]

        # Betweenness trong subgraph (tính nhanh bằng networkx)
        betweenness = np.zeros(N_sub, dtype=np.float32)
        try:
            import networkx as nx
            G_sub = nx.DiGraph()
            G_sub.add_nodes_from(range(N_sub))
            G_sub.add_edges_from(zip(new_u.tolist(), new_v.tolist()))
            bt_dict = nx.betweenness_centrality(G_sub, normalized=True)
            for nidx, bt_val in bt_dict.items():
                betweenness[nidx] = float(bt_val)
        except Exception as e:
            self.logger.warning(f"Betweenness failed for subgraph: {e}")

        # Normalize lat/lon trong subgraph
        lats = sub_coords[:, 0]
        lons = sub_coords[:, 1]
        lat_norm = (lats - lats.min()) / (lats.max() - lats.min() + 1e-8)
        lon_norm = (lons - lons.min()) / (lons.max() - lons.min() + 1e-8)
        deg_norm = degrees / (degrees.max() + 1e-8)

        node_features = np.stack(
            [deg_norm, betweenness, lat_norm.astype(np.float32), lon_norm.astype(np.float32)],
            axis=1,
        ).astype(np.float32)   # [N_sub, 4]

        self.logger.info(
            f"Subgraph built: {N_sub} nodes, {E_sub} directed edges "
            f"(từ {len(self.osm_node_ids)} OSM nodes gốc)"
        )

        return {
            "node_features":     node_features,        # [N_sub, F_node]
            "edge_features":     edge_features,         # [E_sub, F_edge]
            "edge_index":        sub_edge_index,        # [2, E_sub]
            "adjacency_matrix":  sub_adj,               # [N_sub, N_sub]
            "osm_node_ids":      sub_node_ids,          # [N_sub]
            "coordinates":       sub_coords,            # [N_sub, 2]
            "node_feature_names": np.array(NODE_FEATURE_COLS),
            "edge_feature_names": np.array(EDGE_FEATURE_COLS),
            # Giữ lại arrays OSM chi tiết để debug/viz
            "edge_lengths":      np.array(osm_length_list,   dtype=np.float32),
            "edge_maxspeed":     np.array(osm_maxspeed_list,  dtype=np.float32),
            "edge_lanes":        np.array(osm_lanes_list,     dtype=np.float32),
            "edge_highway_type": np.array(osm_highway_list,   dtype=np.int32),
        }

    def _build_feature_arrays(
        self, matched_df: pd.DataFrame
    ) -> Dict[str, np.ndarray]:
        """
        Wrapper gọi _build_matched_subgraph.
        Giữ lại tên method này để tương thích với match_and_build_temporal_features.
        """
        return self._build_matched_subgraph(matched_df)

    # =========================================================================
    # INTERNAL — UTILITIES
    # =========================================================================

    def _fill_nan_temporal(self, tensor: np.ndarray) -> np.ndarray:
        """
        Forward-fill NaN trong temporal tensor [N, T, F].
        Sau đó backward-fill phần đầu còn NaN.
        Cuối cùng fill 0 cho bất kỳ NaN còn lại.
        """
        N, T, F = tensor.shape
        for n in range(N):
            for f in range(F):
                series = tensor[n, :, f]
                mask = np.isnan(series)
                if not mask.any():
                    continue
                last_valid = None
                for t in range(T):
                    if not mask[t]:
                        last_valid = series[t]
                    elif last_valid is not None:
                        series[t] = last_valid
                first_valid = None
                for t in range(T - 1, -1, -1):
                    if not np.isnan(series[t]):
                        first_valid = series[t]
                    elif first_valid is not None:
                        series[t] = first_valid
                tensor[n, :, f] = series
        tensor = np.nan_to_num(tensor, nan=0.0)
        return tensor

    def _fill_nan_temporal_edges(self, tensor: np.ndarray) -> np.ndarray:
        """
        Forward-fill NaN trong edge temporal tensor [E, T, F].
        Cùng logic với _fill_nan_temporal nhưng chiều đầu là số edges.
        """
        E, T, F = tensor.shape
        for e in range(E):
            for f in range(F):
                series = tensor[e, :, f]
                mask = np.isnan(series)
                if not mask.any():
                    continue
                last_valid = None
                for t in range(T):
                    if not mask[t]:
                        last_valid = series[t]
                    elif last_valid is not None:
                        series[t] = last_valid
                first_valid = None
                for t in range(T - 1, -1, -1):
                    if not np.isnan(series[t]):
                        first_valid = series[t]
                    elif first_valid is not None:
                        series[t] = first_valid
                tensor[e, :, f] = series
        tensor = np.nan_to_num(tensor, nan=0.0)
        return tensor

    def _save_npz(
        self,
        data: Dict[str, np.ndarray],
        output_name: str,
        output_dir: Path,
        time_slot: Optional[str],
    ):
        """Lưu kết quả map matching ra NPZ."""
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = output_dir / f"{output_name}_{timestamp}.npz"

        # Support cả static (node_features) và temporal (edge_features_temporal)
        N = len(data.get("osm_node_ids", []))
        E = data.get("edge_index", np.zeros((2, 0))).shape[1]

        temporal_shape = None
        if "edge_features_temporal" in data:
            temporal_shape = list(data["edge_features_temporal"].shape)
        elif "node_features_temporal" in data:
            temporal_shape = list(data["node_features_temporal"].shape)

        metadata = {
            "num_nodes":          N,
            "num_edges_directed": E,
            "match_threshold_m":  self.match_threshold_m,
            "aggregate_method":   self.aggregate_method,
            "time_slot":          time_slot or "all",
            "edge_feature_cols":  EDGE_FEATURE_COLS,
            "node_feature_cols":  NODE_FEATURE_COLS,
            "topology_source":    "OpenStreetMap (matched subgraph only)",
            "feature_source":     "TomTom real data only (no defaults)",
            "design":             "Option B: only OSM edges with TomTom match",
            "temporal_shape":     temporal_shape,
            "created_at":         timestamp,
            "coverage_ratio":     self._coverage_ratio,
        }

        save_dict = {k: v for k, v in data.items()}
        save_dict["_metadata"] = np.array([json.dumps(metadata)])
        np.savez_compressed(str(file_path), **save_dict)

        size_mb = file_path.stat().st_size / (1024 * 1024)
        self.logger.info(
            f"✅ Map-matched subgraph saved: {file_path.name} ({size_mb:.2f} MB)"
        )

    @staticmethod
    def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Haversine distance in meters."""
        R = 6_371_000.0
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        return R * 2 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))