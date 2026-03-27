# src/data_processing/graph/map_matcher.py
"""
Map Matcher — gán TomTom traffic features vào OSM edges.

Đây là module cốt lõi thay thế hoàn toàn logic build_graph_structure()
dùng distance_threshold.

Logic:
    1. Với mỗi TomTom segment (có lat/lon trung điểm), tìm OSM edge gần nhất.
    2. Nếu khoảng cách ≤ match_threshold_m, assign features TomTom vào OSM edge đó.
    3. OSM edges không có TomTom data → dùng free_flow_speed (từ maxspeed OSM) và congestion=0.
    4. Aggregate nhiều TomTom segments map tới cùng 1 OSM edge → mean.
    5. Output: node_features [N, F] và edge_features [E, F] sẵn sàng cho T-GCN / DTC-STGCN.

Tại sao không dùng GPS snap library phức tạp:
    - Với dữ liệu Quận 1 (515 segments), nearest-edge search đủ nhanh (< 5 giây).
    - OSMnx cung cấp ox.nearest_edges() dùng spatial index (STRtree), O(N log M).
    - Không cần HMM map matching vì segments TomTom đã là đoạn đường, không phải GPS trace liên tục.
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


# Features TomTom sẽ được gán vào OSM nodes/edges
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

# Features sẽ giữ lại sau map matching (cho T-GCN node features)
NODE_FEATURE_COLS = [
    # TomTom dynamic features
    "average_speed",
    "harmonic_average_speed",
    "std_speed",
    "travel_time_ratio",
    "congestion_index",
    "speed_limit_ratio",
    # OSM static features
    "osm_length_m",
    "osm_maxspeed",
    "osm_lanes",
    "osm_highway_type",
    # Graph topology features
    "degree",
    "betweenness_norm",
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
        match_threshold_m: float = 50.0,
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
        self.logger.info(
            f"Matched: {len(matched_df)} / {df['segment_id'].nunique()} segments "
            f"(coverage = {len(matched_df) / max(df['segment_id'].nunique(), 1):.1%})"
        )

        # ── Aggregate TomTom features lên OSM nodes ───────────────────────────
        node_features_df = self._aggregate_to_nodes(matched_df)

        # ── Build final feature matrix ────────────────────────────────────────
        result = self._build_feature_arrays(node_features_df)

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
        Build temporal node features: [N, T, F] cho T-GCN / DTC-STGCN.

        Với mỗi time slot T, chạy map matching và aggregate features.
        Output shape: node_features_temporal [N, T, F].

        Args:
            traffic_df   : DataFrame đầy đủ 31 ngày × 24 time slots.
            output_name  : Tên NPZ output.
        """
        self.logger.info("=" * 60)
        self.logger.info("MAP MATCHER — Temporal features [N, T, F]")
        self.logger.info("=" * 60)

        # Xác định time slots và dates
        df = self._prepare_traffic_df(traffic_df, time_slot=None)
        if df.empty:
            self.logger.error("DataFrame rỗng sau khi prepare.")
            return {}

        # Tạo unique timestamps theo (date, time_slot)
        date_col = "date_from"
        if "date_from" not in df.columns:
            date_col = "date_range" if "date_range" in df.columns else "date"

        df["_ts_key"] = df[date_col].astype(str) + "__" + df["time_set"].astype(str)
        ts_keys = sorted(df["_ts_key"].unique())
        T = len(ts_keys)
        N = len(self.osm_node_ids)
        F = len(NODE_FEATURE_COLS)

        self.logger.info(f"Building temporal tensor: N={N}, T={T}, F={F}")

        # [N, T, F] — khởi tạo với NaN, fill sau
        temporal_tensor = np.full((N, T, F), np.nan, dtype=np.float32)

        for t_idx, ts_key in enumerate(ts_keys):
            slot_df = df[df["_ts_key"] == ts_key].copy()
            if slot_df.empty:
                continue

            try:
                matched_df = self._map_match_segments(slot_df, osm_networkx_graph)
                node_features_df = self._aggregate_to_nodes(matched_df)
                result = self._build_feature_arrays(node_features_df)
                node_feat = result["node_features"]  # [N, F]
                if node_feat.shape == (N, F):
                    temporal_tensor[:, t_idx, :] = node_feat
            except KeyError as e:
                self.logger.warning(
                    f"Slot {ts_key} failed: cột {e} không tồn tại trong data. "
                    f"Slot này sẽ được fill bằng forward-fill."
                )
                continue
            except Exception as e:
                self.logger.warning(f"Slot {ts_key} failed: {e}")
                continue

            if (t_idx + 1) % 50 == 0:
                self.logger.info(f"Progress: {t_idx + 1}/{T} time slots")

        # Fill NaN bằng forward-fill theo thời gian (per node)
        temporal_tensor = self._fill_nan_temporal(temporal_tensor)

        # Parse timestamps
        timestamps = np.array(ts_keys)

        output = {
            "node_features_temporal": temporal_tensor,   # [N, T, F]
            "timestamps":             timestamps,          # [T]
            "osm_node_ids":           self.osm_node_ids,  # [N]
            "coordinates":            self.coordinates,    # [N, 2]
            "edge_index":             self.edge_index,     # [2, E]
            "adjacency_matrix":       self.adjacency_matrix,
            "feature_names":          np.array(NODE_FEATURE_COLS),
        }

        if output_dir:
            self._save_npz(output, output_name, output_dir, time_slot=None)

        self.logger.info(
            f"✅ Temporal features built: {temporal_tensor.shape}"
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
        """
        df = df.copy()

        # Dùng raw coordinates (không bị normalize)
        if "raw_latitude" in df.columns and "raw_longitude" in df.columns:
            df["_lat"] = df["raw_latitude"]
            df["_lon"] = df["raw_longitude"]
        elif "latitude" in df.columns and "longitude" in df.columns:
            # Kiểm tra xem có phải normalized không
            lat_max = df["latitude"].max()
            if lat_max <= 1.0:
                self.logger.warning(
                    "latitude có vẻ đã normalize ([0,1]). "
                    "Cần raw coordinates để map match chính xác. "
                    "Đảm bảo pipeline lưu raw_latitude/raw_longitude."
                )
            df["_lat"] = df["latitude"]
            df["_lon"] = df["longitude"]
        else:
            self.logger.error("Không tìm thấy cột tọa độ.")
            return pd.DataFrame()

        # Filter time slot
        if time_slot and "time_set" in df.columns:
            df = df[df["time_set"] == time_slot]

        # Drop rows thiếu tọa độ
        df = df.dropna(subset=["_lat", "_lon", "segment_id"])

        # ── Ép kiểu tất cả TOMTOM_FEATURE_COLS về float64 TRƯỚC khi aggregate ──
        # Lý do: khi đọc từ NPZ, các cột có NaN có thể bị lưu dưới dạng object dtype
        # (ví dụ median_speed khi một số records TomTom không trả về medianSpeed).
        # pandas groupby.agg() sẽ ném KeyError hoặc TypeError với cột object dtype.
        for col in TOMTOM_FEATURE_COLS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype(np.float64)

        # Tính trung điểm tọa độ per segment
        agg_dict: Dict = {
            "_lat": "mean",
            "_lon": "mean",
        }

        # Chỉ aggregate các cột có ít nhất 1 giá trị không phải NaN
        # (đã force-cast về float64 ở trên nên không cần check dtype nữa)
        for col in TOMTOM_FEATURE_COLS:
            if col in df.columns and df[col].notna().any():
                agg_dict[col] = self.aggregate_method

        for opt_col in ["street_name", "frc", "speed_limit", "distance"]:
            if opt_col in df.columns:
                agg_dict[opt_col] = "first"

        group_cols = ["segment_id"]
        # Nếu đang build Temporal (time_slot=None), BẮT BUỘC phải giữ lại các cột thời gian
        if time_slot is None:
            for t_col in ["date_from", "date_range", "date", "time_set"]:
                if t_col in df.columns:
                    group_cols.append(t_col)

        try:
            df_seg = df.groupby(group_cols, dropna=False).agg(agg_dict).reset_index()
        except KeyError as e:
            # Một số cột có thể biến mất sau khi normalize — loại bỏ và thử lại
            missing_col = str(e).strip("'")
            self.logger.warning(
                f"Cột '{missing_col}' không tìm thấy khi aggregate, bỏ qua cột này."
            )
            agg_dict.pop(missing_col, None)
            df_seg = df.groupby(group_cols, dropna=False).agg(agg_dict).reset_index()

        # Đảm bảo các cột TOMTOM_FEATURE_COLS bị thiếu được thêm vào với NaN
        # để downstream code không bị KeyError
        for col in TOMTOM_FEATURE_COLS:
            if col not in df_seg.columns:
                df_seg[col] = np.nan

        return df_seg

    # =========================================================================
    # INTERNAL — MAP MATCHING
    # =========================================================================

    def _map_match_segments(
        self,
        seg_df: pd.DataFrame,
        G: "nx.MultiDiGraph",
    ) -> pd.DataFrame:
        """
        Với mỗi TomTom segment (lat/lon), tìm nearest OSM edge dùng ox.nearest_edges.

        Returns:
            DataFrame với cột bổ sung:
                osm_u, osm_v    — OSM node IDs của edge được match
                osm_u_idx, osm_v_idx — index trong node array
                match_dist_m    — khoảng cách match (m)
        """
        lats = seg_df["_lat"].values
        lons = seg_df["_lon"].values

        # ox.nearest_edges nhận (X=lon, Y=lat)
        try:
            if len(lats) == 1:
                # THÊM return_dist=True ĐỂ OSMNX TRẢ VỀ KHOẢNG CÁCH CHÍNH XÁC NHẤT ĐẾN ĐƯỜNG CONG
                u, v, k, dist = ox.nearest_edges(G, X=lons[0], Y=lats[0], return_dist=True)
                nearest_edges = [(u, v, k)]
                distances = [dist]
            else:nearest_edges, distances = ox.nearest_edges(G, X=lons, Y=lats, return_dist=True)
        except Exception as e:
            self.logger.error(f"ox.nearest_edges failed: {e}")
            return pd.DataFrame()

        results = []
        for idx, (u, v, k) in enumerate(nearest_edges):
            u_idx = self.node_id_to_idx.get(int(u))
            v_idx = self.node_id_to_idx.get(int(v))

            if u_idx is None or v_idx is None:
                continue

            # Lấy khoảng cách chuẩn từ OSMnx (mét)
            dist_m = distances[idx]

            # Dùng khoảng cách chuẩn này để filter
            if dist_m > self.match_threshold_m:
                continue  # Bỏ qua nếu điểm centroid nằm quá xa bất kỳ con đường nào

            row = seg_df.iloc[idx].to_dict()
            row["osm_u"] = int(u)
            row["osm_v"] = int(v)
            row["osm_u_idx"] = u_idx
            row["osm_v_idx"] = v_idx
            row["match_dist_m"] = dist_m
            results.append(row)

        if not results:
            return pd.DataFrame()

        matched_df = pd.DataFrame(results)
        self.logger.debug(
            f"Matched {len(matched_df)}/{len(seg_df)} segments "
            f"within {self.match_threshold_m}m"
        )
        return matched_df

    # =========================================================================
    # INTERNAL — AGGREGATION
    # =========================================================================

    def _aggregate_to_nodes(self, matched_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate TomTom features từ matched edges lên OSM nodes.

        Strategy:
            - Mỗi OSM node nhận features trung bình của tất cả edges liền kề
              đã được match với TomTom data.
            - Nodes không có TomTom data → dùng free_flow defaults.
        """
        N = len(self.osm_node_ids)

        # Khởi tạo node feature dict với defaults
        node_features_dict: Dict[str, List] = {col: [np.nan] * N for col in NODE_FEATURE_COLS}

        # Điền OSM static features vào tất cả nodes
        for node_idx in range(N):
            node_id = int(self.osm_node_ids[node_idx])
            degree = int(self.adjacency_matrix[node_idx].sum())

            # Tìm edges liền kề node này
            edge_mask_u = self.edge_index[0] == node_idx
            edge_mask_v = self.edge_index[1] == node_idx
            adj_edge_mask = edge_mask_u | edge_mask_v

            if adj_edge_mask.any():
                mean_length = float(self.edge_lengths[adj_edge_mask].mean())
                mean_maxspeed = float(self.edge_maxspeed[adj_edge_mask].mean())
                mean_lanes = float(self.edge_lanes[adj_edge_mask].mean())
                mean_highway = float(self.edge_highway_type[adj_edge_mask].mean())
            else:
                mean_length = 0.0
                mean_maxspeed = 0.0
                mean_lanes = 1.0
                mean_highway = 14.0

            node_features_dict["osm_length_m"][node_idx] = mean_length
            node_features_dict["osm_maxspeed"][node_idx] = mean_maxspeed
            node_features_dict["osm_lanes"][node_idx] = mean_lanes
            node_features_dict["osm_highway_type"][node_idx] = mean_highway
            node_features_dict["degree"][node_idx] = float(degree)

        # Betweenness (từ osm_data node_features nếu có)
        if "node_features" in self.osm_data:
            # FIX: tra cứu index qua feature_names thay vì hardcode bt_idx = 3.
            # Nếu OSMGraphBuilder thay đổi thứ tự feature, code này vẫn đúng.
            feature_names_arr = self.osm_data.get("feature_names", np.array([]))
            feature_names_list = list(feature_names_arr.astype(str))
            if "betweenness_norm" in feature_names_list:
                bt_idx = feature_names_list.index("betweenness_norm")
                bt_col = self.osm_data["node_features"][:, bt_idx]
                node_features_dict["betweenness_norm"] = bt_col.tolist()
            else:
                self.logger.warning(
                    "'betweenness_norm' không tìm thấy trong feature_names. "
                    "Sẽ dùng zeros. Kiểm tra OSMGraphBuilder.feature_names."
                )
                node_features_dict["betweenness_norm"] = [0.0] * N
        else:
            node_features_dict["betweenness_norm"] = [0.0] * N

        # Điền TomTom dynamic features nếu có match
        if not matched_df.empty:
            # Chỉ lấy các cột có trong CẢ HAI: matched_df VÀ node_features_dict
            # (TOMTOM_FEATURE_COLS có thể chứa cột như median_speed không có trong NODE_FEATURE_COLS)
            tomtom_cols_present = [
                c for c in TOMTOM_FEATURE_COLS
                if c in matched_df.columns and c in node_features_dict
            ]

            # Aggregate per (osm_u_idx, osm_v_idx)
            for _, row in matched_df.iterrows():
                u_idx = int(row["osm_u_idx"])
                v_idx = int(row["osm_v_idx"])

                for node_idx in [u_idx, v_idx]:
                    if node_idx >= N:
                        continue
                    for col in tomtom_cols_present:
                        val = row.get(col, np.nan)
                        if pd.notna(val):
                            current = node_features_dict[col][node_idx]
                            if np.isnan(current):
                                node_features_dict[col][node_idx] = float(val)
                            else:
                                # Running mean (simple 2-way merge)
                                node_features_dict[col][node_idx] = (
                                    current + float(val)
                                ) / 2

        # Fill NaN TomTom features với free-flow defaults
        # (speed = maxspeed OSM, congestion = 0)
        for node_idx in range(N):
            free_flow_speed = node_features_dict["osm_maxspeed"][node_idx] or 40.0

            if np.isnan(node_features_dict["average_speed"][node_idx]):
                node_features_dict["average_speed"][node_idx] = free_flow_speed
                node_features_dict["harmonic_average_speed"][node_idx] = free_flow_speed
                node_features_dict["std_speed"][node_idx] = 0.0
                node_features_dict["travel_time_ratio"][node_idx] = 1.0
                node_features_dict["congestion_index"][node_idx] = 0.0
                node_features_dict["speed_limit_ratio"][node_idx] = 1.0

        # Convert → DataFrame
        node_df = pd.DataFrame(node_features_dict)
        node_df["osm_node_id"] = self.osm_node_ids
        return node_df

    def _build_feature_arrays(
        self, node_df: pd.DataFrame
    ) -> Dict[str, np.ndarray]:
        """Convert node DataFrame → numpy arrays."""
        N = len(self.osm_node_ids)
        F = len(NODE_FEATURE_COLS)

        node_features = np.zeros((N, F), dtype=np.float32)
        for f_idx, col in enumerate(NODE_FEATURE_COLS):
            if col in node_df.columns:
                vals = node_df[col].values.astype(np.float32)
                node_features[:, f_idx] = np.nan_to_num(vals, nan=0.0)

        return {
            "node_features":    node_features,        # [N, F]
            "edge_index":       self.edge_index,       # [2, E]
            "adjacency_matrix": self.adjacency_matrix, # [N, N]
            "osm_node_ids":     self.osm_node_ids,     # [N]
            "coordinates":      self.coordinates,      # [N, 2]
            "feature_names":    np.array(NODE_FEATURE_COLS),
            "edge_lengths":     self.edge_lengths,
            "edge_maxspeed":    self.edge_maxspeed,
            "edge_lanes":       self.edge_lanes,
            "edge_highway_type": self.edge_highway_type,
        }

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
                # Forward fill
                last_valid = None
                for t in range(T):
                    if not mask[t]:
                        last_valid = series[t]
                    elif last_valid is not None:
                        series[t] = last_valid
                # Backward fill
                first_valid = None
                for t in range(T - 1, -1, -1):
                    if not np.isnan(series[t]):
                        first_valid = series[t]
                    elif first_valid is not None:
                        series[t] = first_valid
                tensor[n, :, f] = series
        # Zero-fill remaining NaN
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

        N = len(self.osm_node_ids)
        E = self.edge_index.shape[1]
        tomtom_matched = data.get("node_features", np.array([])).shape

        metadata = {
            "num_nodes":          N,
            "num_edges_directed": E,
            "match_threshold_m":  self.match_threshold_m,
            "aggregate_method":   self.aggregate_method,
            "time_slot":          time_slot or "all",
            "feature_cols":       NODE_FEATURE_COLS,
            "topology_source":    "OpenStreetMap",
            "feature_source":     "TomTom + OSM hybrid",
            "created_at":         timestamp,
            "coverage_ratio":     self._coverage_ratio,
        }

        save_dict = {k: v for k, v in data.items()}
        save_dict["_metadata"] = np.array([json.dumps(metadata)])
        np.savez_compressed(str(file_path), **save_dict)

        size_mb = file_path.stat().st_size / (1024 * 1024)
        self.logger.info(
            f"✅ Map-matched graph saved: {file_path.name} ({size_mb:.2f} MB)"
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