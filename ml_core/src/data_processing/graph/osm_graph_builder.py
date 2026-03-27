# src/data_processing/graph/osm_graph_builder.py
"""
OSM Graph Builder — xây dựng "bộ xương" đường bộ từ OpenStreetMap.

Thay thế hoàn toàn build_graph_structure() cũ (dùng TomTom distance_threshold).
Nodes = ngã tư thực tế (OSM intersections).
Edges = đoạn đường thực tế (OSM road segments).
Adjacency matrix = topology đường bộ chuẩn, không còn lỗi cắt qua nhà dân.
"""

from __future__ import annotations

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from datetime import datetime

try:
    import osmnx as ox
    import networkx as nx
    from shapely.geometry import Polygon, mapping
except ImportError as e:
    raise ImportError(
        f"Thiếu dependency: {e}. "
        "Cài đặt: pip install osmnx networkx shapely"
    )

from utils.config import config
from utils.logger import LoggerMixin


class OSMGraphBuilder(LoggerMixin):
    """
    Xây dựng đồ thị đường bộ từ OpenStreetMap cho một vùng địa lý.

    Workflow:
        1. Tải đồ thị OSM từ polygon (bbox của Quận 1, HCMC).
        2. Simplify, project, rồi convert về WGS84 (lat/lon).
        3. Xây adjacency matrix từ topology thực tế (không dùng distance threshold).
        4. Lưu ra NPZ với đầy đủ metadata để map_matcher dùng sau.

    Output NPZ chứa:
        - osm_node_ids      : [N]         — OSM node IDs (int64)
        - coordinates       : [N, 2]      — (lat, lon) của từng node
        - edge_index        : [2, E]      — directed edges (PyG / T-GCN format)
        - edge_osmids       : [E]         — OSM edge IDs (u, v pair encoded)
        - edge_lengths      : [E]         — độ dài từng edge (m)
        - edge_maxspeed     : [E]         — tốc độ tối đa (km/h), 0 nếu không có
        - edge_lanes        : [E]         — số làn đường, 1 nếu không có
        - edge_highway_type : [E]         — mã hóa loại đường (int)
        - adjacency_matrix  : [N, N]      — ma trận kề nhị phân
        - node_features     : [N, 4]      — placeholder features (lat, lon, degree, betweenness)
        - feature_names     : [4]         — tên của node features
    """

    # Map OSM highway type → integer code (dùng làm edge feature)
    HIGHWAY_TYPE_MAP: Dict[str, int] = {
        "motorway":        0,
        "trunk":           1,
        "primary":         2,
        "secondary":       3,
        "tertiary":        4,
        "residential":     5,
        "service":         6,
        "unclassified":    7,
        "living_street":   8,
        "motorway_link":   9,
        "trunk_link":      10,
        "primary_link":    11,
        "secondary_link":  12,
        "tertiary_link":   13,
        "other":           14,
    }

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        cache_graph: bool = True,
    ):
        """
        Args:
            output_dir  : Thư mục lưu NPZ, mặc định config.data.processed_dir.
            cache_graph : Lưu cache đồ thị OSM dưới dạng GraphML để tránh tải lại.
        """
        self.output_dir = output_dir or config.data.processed_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.cache_graph = cache_graph
        self.cache_dir = self.output_dir / "osm_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Sẽ được populate sau khi build
        self.G: Optional[nx.MultiDiGraph] = None
        self.node_ids: Optional[np.ndarray] = None
        self.coordinates: Optional[np.ndarray] = None
        self.node_id_to_idx: Optional[Dict[int, int]] = None

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def build_from_polygon(
        self,
        polygon: Dict,
        simplify: bool = True,
        retain_all: bool = False,
        output_name: str = "osm_graph",
    ) -> Dict[str, np.ndarray]:
        """
        Xây đồ thị từ GeoJSON Polygon geometry (format TomTom).

        Args:
            polygon     : GeoJSON Polygon dict, ví dụ:
                          {"type": "Polygon", "coordinates": [[[lon, lat], ...]]}
            simplify    : Simplify đồ thị OSM (hợp nhất các đường thẳng, bỏ nodes giữa đường).
            retain_all  : Giữ lại tất cả edges kể cả đường 1 chiều cụt.
            output_name : Tên file NPZ output.

        Returns:
            Dict chứa tất cả arrays đã được lưu vào NPZ.
        """
        self.logger.info("=" * 60)
        self.logger.info("OSM GRAPH BUILDER — Build from polygon")
        self.logger.info("=" * 60)

        # Chuyển GeoJSON polygon → shapely Polygon
        coords_list = polygon["coordinates"][0]
        # GeoJSON: [lon, lat] → Shapely: (lon, lat)
        shapely_polygon = Polygon([(c[0], c[1]) for c in coords_list])

        # Tạo cache key từ bbox
        minx, miny, maxx, maxy = shapely_polygon.bounds
        cache_key = f"{miny:.4f}_{maxx:.4f}_{maxy:.4f}_{minx:.4f}"
        cache_file = self.cache_dir / f"osm_{cache_key}.graphml"

        # Load hoặc tải OSM graph
        if self.cache_graph and cache_file.exists():
            self.logger.info(f"Loading cached OSM graph: {cache_file.name}")
            self.G = ox.load_graphml(str(cache_file))
        else:
            self.logger.info(
                f"Downloading OSM graph for bbox: "
                f"lat=[{miny:.4f}, {maxy:.4f}], lon=[{minx:.4f}, {maxx:.4f}]"
            )
            self.G = ox.graph_from_polygon(
                shapely_polygon,
                network_type="drive",
                simplify=simplify,
                retain_all=retain_all,
            )
            if self.cache_graph:
                ox.save_graphml(self.G, str(cache_file))
                self.logger.info(f"Cached OSM graph to: {cache_file.name}")

        graph_data = self._build_arrays(output_name)
        self._save_npz(graph_data, output_name)
        return graph_data

    def build_from_bbox(
        self,
        north: float,
        south: float,
        east: float,
        west: float,
        simplify: bool = True,
        output_name: str = "osm_graph",
    ) -> Dict[str, np.ndarray]:
        """
        Xây đồ thị từ bounding box (lat/lon).

        Args:
            north, south, east, west : Tọa độ bbox (degrees).
        """
        self.logger.info(
            f"Building OSM graph from bbox: N={north}, S={south}, E={east}, W={west}"
        )

        cache_key = f"{south:.4f}_{east:.4f}_{north:.4f}_{west:.4f}"
        cache_file = self.cache_dir / f"osm_{cache_key}.graphml"

        if self.cache_graph and cache_file.exists():
            self.logger.info(f"Loading cached OSM graph: {cache_file.name}")
            self.G = ox.load_graphml(str(cache_file))
        else:
            try:
                # osmnx >= 1.3: bbox là positional tuple (north, south, east, west)
                self.G = ox.graph_from_bbox(
                    bbox=(north, south, east, west),
                    network_type="drive",
                    simplify=simplify,
                )
            except TypeError:
                # osmnx < 1.3: positional args
                self.G = ox.graph_from_bbox(
                    north, south, east, west,
                    network_type="drive",
                    simplify=simplify,
                )
            if self.cache_graph:
                ox.save_graphml(self.G, str(cache_file))

        graph_data = self._build_arrays(output_name)
        self._save_npz(graph_data, output_name)
        return graph_data

    # =========================================================================
    # INTERNAL METHODS
    # =========================================================================

    def _build_arrays(self, output_name: str) -> Dict[str, np.ndarray]:
        """
        Convert NetworkX graph → numpy arrays cho T-GCN / DTC-STGCN.

        Node indexing: nodes được index 0..N-1 theo thứ tự ổn định.
        Edge: cả 2 chiều (directed) để T-GCN xử lý đúng.
        """
        assert self.G is not None, "Graph chưa được tải."

        G = self.G
        self.logger.info(
            f"OSM graph loaded: {G.number_of_nodes()} nodes, "
            f"{G.number_of_edges()} edges"
        )

        # ── Node indexing ─────────────────────────────────────────────────────
        osm_node_ids = np.array(sorted(G.nodes()), dtype=np.int64)
        N = len(osm_node_ids)
        self.node_id_to_idx = {nid: idx for idx, nid in enumerate(osm_node_ids)}
        self.node_ids = osm_node_ids

        # ── Node coordinates (lat, lon) ───────────────────────────────────────
        node_data = G.nodes(data=True)
        lats = np.array([node_data[nid]["y"] for nid in osm_node_ids], dtype=np.float64)
        lons = np.array([node_data[nid]["x"] for nid in osm_node_ids], dtype=np.float64)
        self.coordinates = np.stack([lats, lons], axis=1)  # [N, 2]

        self.logger.info(
            f"Coordinate range: lat=[{lats.min():.5f}, {lats.max():.5f}], "
            f"lon=[{lons.min():.5f}, {lons.max():.5f}]"
        )

        # ── Edge arrays ───────────────────────────────────────────────────────
        src_list, dst_list = [], []
        edge_osmid_list = []
        edge_length_list = []
        edge_maxspeed_list = []
        edge_lanes_list = []
        edge_highway_list = []

        for u, v, data in G.edges(data=True):
            if u not in self.node_id_to_idx or v not in self.node_id_to_idx:
                continue
            u_idx = self.node_id_to_idx[u]
            v_idx = self.node_id_to_idx[v]

            src_list.append(u_idx)
            dst_list.append(v_idx)

            # Edge osmid: dùng index thay vì encode node ID (tránh overflow int64)
            # u_idx * N + v_idx đảm bảo unique và luôn < N^2 < 2^63
            edge_osmid_list.append(u_idx * N + v_idx)

            # Length (m)
            length = data.get("length", 0.0)
            edge_length_list.append(float(length) if length else 0.0)

            # Max speed (km/h) — OSM có thể lưu string "50" hoặc list ["50", "60"]
            maxspeed_raw = data.get("maxspeed", 0)
            edge_maxspeed_list.append(self._parse_speed(maxspeed_raw))

            # Lanes
            lanes_raw = data.get("lanes", 1)
            edge_lanes_list.append(self._parse_lanes(lanes_raw))

            # Highway type → integer code
            highway_raw = data.get("highway", "other")
            edge_highway_list.append(self._encode_highway(highway_raw))

        E = len(src_list)
        edge_index = np.array([src_list, dst_list], dtype=np.int64)  # [2, E]
        edge_osmids = np.array(edge_osmid_list, dtype=np.int64)
        edge_lengths = np.array(edge_length_list, dtype=np.float32)
        edge_maxspeed = np.array(edge_maxspeed_list, dtype=np.float32)
        edge_lanes = np.array(edge_lanes_list, dtype=np.float32)
        edge_highway_type = np.array(edge_highway_list, dtype=np.int32)

        self.logger.info(f"Edges built: {E} directed ({E // 2} undirected)")

        # ── Adjacency matrix ─────────────────────────────────────────────────
        adj = np.zeros((N, N), dtype=np.float32)
        if E > 0:
            adj[edge_index[0], edge_index[1]] = 1.0

        degrees = adj.sum(axis=1).astype(int)
        self.logger.info(
            f"Degree stats: min={degrees.min()}, max={degrees.max()}, "
            f"mean={degrees.mean():.2f}, median={np.median(degrees):.1f}"
        )

        # ── Node features (placeholder cho khi chưa có TomTom features) ──────
        # [lat_norm, lon_norm, degree_norm, betweenness_norm]
        lat_norm = (lats - lats.min()) / (lats.max() - lats.min() + 1e-8)
        lon_norm = (lons - lons.min()) / (lons.max() - lons.min() + 1e-8)
        degree_norm = degrees / (degrees.max() + 1e-8)

        # Betweenness centrality (xấp xỉ nhanh cho đồ thị lớn)
        self.logger.info("Computing betweenness centrality (approximation)...")
        try:
            G_undirected = G.to_undirected()
            k_samples = max(100, int(N * 0.1))
            betweenness = nx.betweenness_centrality(G_undirected, k=k_samples, normalized=True)
            betweenness_arr = np.array(
                [betweenness.get(nid, 0.0) for nid in osm_node_ids], dtype=np.float32
            )
        except Exception as e:
            self.logger.warning(f"Betweenness computation failed: {e}. Using zeros.")
            betweenness_arr = np.zeros(N, dtype=np.float32)

        node_features = np.stack(
            [lat_norm, lon_norm, degree_norm, betweenness_arr], axis=1
        ).astype(np.float32)  # [N, 4]

        feature_names = np.array(
            ["lat_norm", "lon_norm", "degree_norm", "betweenness_norm"]
        )

        # ── Isolated nodes warning ────────────────────────────────────────────
        isolated = np.where(degrees == 0)[0]
        if len(isolated) > 0:
            self.logger.warning(
                f"{len(isolated)} isolated nodes found (degree=0). "
                "Xem xét dùng retain_all=False."
            )

        graph_data = {
            "osm_node_ids":      osm_node_ids,
            "coordinates":       self.coordinates,
            "edge_index":        edge_index,
            "edge_osmids":       edge_osmids,
            "edge_lengths":      edge_lengths,
            "edge_maxspeed":     edge_maxspeed,
            "edge_lanes":        edge_lanes,
            "edge_highway_type": edge_highway_type,
            "adjacency_matrix":  adj,
            "node_features":     node_features,
            "feature_names":     feature_names,
        }

        return graph_data

    def _save_npz(self, graph_data: Dict[str, np.ndarray], output_name: str):
        """Lưu graph data ra NPZ file với metadata."""
        output_path = self.output_dir / output_name
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = output_path / f"{output_name}_{timestamp}.npz"

        N = len(graph_data["osm_node_ids"])
        E = graph_data["edge_index"].shape[1]

        metadata = {
            "num_nodes":         N,
            "num_edges_directed": E,
            "num_edges_undirected": E // 2,
            "avg_degree":        float(graph_data["adjacency_matrix"].sum() / N),
            "source":            "OpenStreetMap via OSMnx",
            "network_type":      "drive",
            "topology":          "OSM_road_graph",
            "created_at":        timestamp,
        }

        save_dict = {k: v for k, v in graph_data.items()}
        save_dict["_metadata"] = np.array([json.dumps(metadata)])

        np.savez_compressed(str(file_path), **save_dict)

        size_mb = file_path.stat().st_size / (1024 * 1024)
        self.logger.info(f"✅ OSM graph saved: {file_path.name} ({size_mb:.2f} MB)")
        self.logger.info(f"   Nodes: {N} | Directed edges: {E} | Undirected: {E // 2}")

    # =========================================================================
    # STATIC HELPERS
    # =========================================================================

    @staticmethod
    def _parse_speed(raw) -> float:
        """Parse OSM maxspeed field về float km/h."""
        if raw is None or raw == 0:
            return 0.0
        if isinstance(raw, (int, float)):
            return float(raw)
        if isinstance(raw, list):
            raw = raw[0]
        try:
            s = str(raw).strip().lower().replace("mph", "").replace("km/h", "").strip()
            val = float(s)
            if "mph" in str(raw).lower():
                val *= 1.60934
            return val
        except (ValueError, TypeError):
            return 0.0

    @staticmethod
    def _parse_lanes(raw) -> float:
        """Parse OSM lanes field về float."""
        if raw is None:
            return 1.0
        if isinstance(raw, list):
            raw = raw[0]
        try:
            return float(str(raw).strip())
        except (ValueError, TypeError):
            return 1.0

    def _encode_highway(self, raw) -> int:
        """Encode OSM highway type về integer."""
        if isinstance(raw, list):
            raw = raw[0]
        highway_str = str(raw).lower().strip() if raw else "other"
        return self.HIGHWAY_TYPE_MAP.get(highway_str, self.HIGHWAY_TYPE_MAP["other"])

    # =========================================================================
    # UTILITY: Load saved OSM graph NPZ
    # =========================================================================

    @classmethod
    def load_latest(
        cls, output_name: str = "osm_graph", base_dir: Optional[Path] = None
    ) -> Optional[Dict[str, np.ndarray]]:
        """
        Load file NPZ mới nhất của osm_graph.

        Returns:
            Dict với arrays, hoặc None nếu không tìm thấy.
        """
        base = base_dir or config.data.processed_dir
        graph_dir = base / output_name

        if not graph_dir.exists():
            return None

        npz_files = sorted(graph_dir.glob(f"{output_name}_*.npz"))
        if not npz_files:
            return None

        data = np.load(str(npz_files[-1]), allow_pickle=True)
        result = {}
        for key in data.files:
            if key == "_metadata":
                result["_metadata"] = json.loads(str(data[key][0]))
            else:
                result[key] = data[key]
        return result