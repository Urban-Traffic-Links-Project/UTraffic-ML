# src/modeling/graph/graph_builder.py
import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Tuple, Optional
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr

from utils.config import config
from utils.logger import LoggerMixin

class TrafficGraphBuilder(LoggerMixin):
    """
    Xây dựng đồ thị giao thông từ dữ liệu segments
    """
    
    def __init__(self):
        self.graph = None
        self.adjacency_matrix = None
        self.node_features = None
        self.node_mapping = {}  # segment_id -> node_index
    
    def build_spatial_graph(
        self,
        df: pd.DataFrame,
        distance_threshold: float = 5.0,
        use_road_network: bool = True
    ) -> Tuple[np.ndarray, Dict]:
        """
        Xây dựng đồ thị dựa trên khoảng cách không gian
        
        Args:
            df: DataFrame với columns [segment_id, latitude, longitude]
            distance_threshold: Ngưỡng khoảng cách (km)
            use_road_network: Sử dụng road network connectivity
            
        Returns:
            adjacency_matrix, node_mapping
        """
        self.logger.info("Building spatial graph...")
        
        # Get unique segments với coordinates
        segments = df[['segment_id', 'latitude', 'longitude']].drop_duplicates()
        segments = segments.dropna()
        
        n_nodes = len(segments)
        self.logger.info(f"Number of nodes: {n_nodes}")
        
        # Create node mapping
        self.node_mapping = {
            seg_id: idx for idx, seg_id in enumerate(segments['segment_id'])
        }
        
        # Calculate distance matrix
        coords = segments[['latitude', 'longitude']].values
        
        # Haversine distance (approximation)
        distances = self._calculate_haversine_distances(coords)
        
        # Create adjacency matrix
        adjacency = (distances <= distance_threshold).astype(float)
        
        # Remove self-loops
        np.fill_diagonal(adjacency, 0)
        
        # Add self-loops back (for GCN)
        adjacency = adjacency + np.eye(n_nodes)
        
        self.adjacency_matrix = adjacency
        
        self.logger.info(
            f"Graph built: {n_nodes} nodes, "
            f"{np.sum(adjacency > 0) - n_nodes} edges"
        )
        
        return adjacency, self.node_mapping
    
    def build_correlation_graph(
        self,
        df: pd.DataFrame,
        value_col: str = 'average_speed',
        correlation_threshold: float = 0.5,
        max_lag: int = 12
    ) -> np.ndarray:
        """
        Xây dựng đồ thị dựa trên correlation giữa time series
        
        Args:
            df: DataFrame với [segment_id, timestamp, value_col]
            value_col: Cột giá trị để tính correlation
            correlation_threshold: Ngưỡng correlation
            max_lag: Maximum lag để tính cross-correlation
            
        Returns:
            correlation adjacency matrix
        """
        self.logger.info("Building correlation graph...")
        
        # Pivot to time series format
        df_pivot = df.pivot_table(
            index='timestamp',
            columns='segment_id',
            values=value_col,
            aggfunc='mean'
        )
        
        n_nodes = len(df_pivot.columns)
        correlation_matrix = np.zeros((n_nodes, n_nodes))
        
        segments = df_pivot.columns.tolist()
        
        # Calculate pairwise correlations
        for i, seg_i in enumerate(segments):
            for j, seg_j in enumerate(segments):
                if i == j:
                    correlation_matrix[i, j] = 1.0
                elif i < j:
                    series_i = df_pivot[seg_i].dropna()
                    series_j = df_pivot[seg_j].dropna()
                    
                    # Find common timestamps
                    common_idx = series_i.index.intersection(series_j.index)
                    
                    if len(common_idx) > 10:
                        # Pearson correlation
                        corr, _ = pearsonr(
                            series_i.loc[common_idx],
                            series_j.loc[common_idx]
                        )
                        correlation_matrix[i, j] = abs(corr)
                        correlation_matrix[j, i] = abs(corr)
        
        # Threshold
        adjacency = (correlation_matrix >= correlation_threshold).astype(float)
        
        # Add self-loops
        np.fill_diagonal(adjacency, 1.0)
        
        self.logger.info(
            f"Correlation graph: {n_nodes} nodes, "
            f"{np.sum(adjacency > 0) - n_nodes} edges"
        )
        
        return adjacency
    
    def build_hybrid_graph(
        self,
        df: pd.DataFrame,
        spatial_weight: float = 0.5,
        correlation_weight: float = 0.5
    ) -> np.ndarray:
        """
        Kết hợp spatial và correlation graphs
        
        Returns:
            hybrid adjacency matrix
        """
        self.logger.info("Building hybrid graph...")
        
        # Build both graphs
        spatial_adj, _ = self.build_spatial_graph(df)
        corr_adj = self.build_correlation_graph(df)
        
        # Ensure same size
        min_size = min(spatial_adj.shape[0], corr_adj.shape[0])
        spatial_adj = spatial_adj[:min_size, :min_size]
        corr_adj = corr_adj[:min_size, :min_size]
        
        # Weighted combination
        hybrid_adj = (
            spatial_weight * spatial_adj +
            correlation_weight * corr_adj
        )
        
        # Normalize
        hybrid_adj = hybrid_adj / hybrid_adj.max()
        
        # Add self-loops
        np.fill_diagonal(hybrid_adj, 1.0)
        
        self.adjacency_matrix = hybrid_adj
        
        return hybrid_adj
    
    def normalize_adjacency(self, adjacency: np.ndarray) -> np.ndarray:
        """
        Normalize adjacency matrix (Symmetric normalization for GCN)
        
        A_norm = D^(-1/2) * A * D^(-1/2)
        """
        # Calculate degree matrix
        degree = np.sum(adjacency, axis=1)
        
        # D^(-1/2)
        degree_inv_sqrt = np.power(degree, -0.5)
        degree_inv_sqrt[np.isinf(degree_inv_sqrt)] = 0.
        
        D_inv_sqrt = np.diag(degree_inv_sqrt)
        
        # Normalize
        adjacency_norm = D_inv_sqrt @ adjacency @ D_inv_sqrt
        
        return adjacency_norm
    
    def _calculate_haversine_distances(self, coords: np.ndarray) -> np.ndarray:
        """
        Calculate haversine distances between coordinates (km)
        
        Args:
            coords: Array of shape (n, 2) with [lat, lon]
            
        Returns:
            Distance matrix (km)
        """
        lat = np.radians(coords[:, 0])
        lon = np.radians(coords[:, 1])
        
        n = len(coords)
        distances = np.zeros((n, n))
        
        for i in range(n):
            # Vectorized haversine
            dlat = lat - lat[i]
            dlon = lon - lon[i]
            
            a = np.sin(dlat/2)**2 + np.cos(lat[i]) * np.cos(lat) * np.sin(dlon/2)**2
            c = 2 * np.arcsin(np.sqrt(a))
            
            # Earth radius in km
            distances[i] = 6371 * c
        
        return distances
    
    def to_networkx(self) -> nx.DiGraph:
        """Convert to NetworkX graph for visualization"""
        if self.adjacency_matrix is None:
            raise ValueError("No adjacency matrix built yet")
        
        G = nx.DiGraph()
        
        n_nodes = self.adjacency_matrix.shape[0]
        
        # Add nodes
        for idx in range(n_nodes):
            G.add_node(idx)
        
        # Add edges
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j and self.adjacency_matrix[i, j] > 0:
                    G.add_edge(i, j, weight=self.adjacency_matrix[i, j])
        
        return G
    
    def get_node_statistics(self) -> pd.DataFrame:
        """
        Tính toán các thống kê của nodes trong graph
        """
        if self.adjacency_matrix is None:
            raise ValueError("No graph built yet")
        
        G = self.to_networkx()
        
        stats = {
            'node_id': list(range(len(G.nodes()))),
            'in_degree': [G.in_degree(n) for n in G.nodes()],
            'out_degree': [G.out_degree(n) for n in G.nodes()],
            'degree': [G.degree(n) for n in G.nodes()],
        }
        
        # Centrality measures
        try:
            betweenness = nx.betweenness_centrality(G)
            stats['betweenness'] = [betweenness[n] for n in G.nodes()]
        except:
            stats['betweenness'] = [0] * len(G.nodes())
        
        try:
            pagerank = nx.pagerank(G)
            stats['pagerank'] = [pagerank[n] for n in G.nodes()]
        except:
            stats['pagerank'] = [0] * len(G.nodes())
        
        return pd.DataFrame(stats)