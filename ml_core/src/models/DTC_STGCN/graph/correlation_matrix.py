"""
Traffic Node Correlation Matrix Builder
Inspired by TH-GAT and DTC-STGCN papers.

Computes multiple correlation measures between traffic nodes:
1. Pearson correlation of speed time series
2. Dynamic correlation (feature ratio method from DTC-STGCN)
3. Attention-based correlation (from trained TH-GAT attention weights)
4. Spatial proximity correlation (based on node coordinates)

Used for:
- Visualizing traffic relationship patterns
- Building region-augmented network structure
- Validating model's learned spatial dependencies
"""

import numpy as np
import torch
from scipy.stats import pearsonr
from sklearn.preprocessing import MinMaxScaler
import warnings

warnings.filterwarnings("ignore")


class TrafficCorrelationMatrix:
    """
    Build and analyze correlation matrices between traffic nodes.
    
    Supports both static (Pearson) and dynamic correlations.
    """

    def __init__(self, num_nodes, node_ids=None, coordinates=None):
        """
        Args:
            num_nodes: Number of traffic nodes (road segments)
            node_ids: Optional list of node identifiers
            coordinates: (num_nodes, 2) node coordinates [lat, lon]
        """
        self.num_nodes = num_nodes
        self.node_ids = node_ids or list(range(num_nodes))
        self.coordinates = coordinates

        self.pearson_matrix = None
        self.dynamic_matrix = None
        self.spatial_matrix = None
        self.attention_matrix = None

    # ──────────────────────────────────────────
    # 1. Pearson Correlation
    # ──────────────────────────────────────────

    def compute_pearson_correlation(self, speed_data):
        """
        Compute pairwise Pearson correlation of speed time series.
        
        Args:
            speed_data: (num_timesteps, num_nodes) or (num_nodes, num_timesteps)
                        speed time series for all nodes
        Returns:
            pearson_matrix: (num_nodes, num_nodes) correlation matrix ∈ [-1, 1]
        """
        if speed_data.shape[0] == self.num_nodes:
            speed_data = speed_data.T  # → (T, N)

        T, N = speed_data.shape
        assert N == self.num_nodes, f"Expected {self.num_nodes} nodes, got {N}"

        corr_matrix = np.zeros((N, N), dtype=np.float32)

        for i in range(N):
            for j in range(i, N):
                series_i = speed_data[:, i]
                series_j = speed_data[:, j]

                # Handle NaN or constant series
                valid = ~(np.isnan(series_i) | np.isnan(series_j))
                if valid.sum() < 3 or np.std(series_i[valid]) < 1e-8 or np.std(series_j[valid]) < 1e-8:
                    corr = 0.0
                else:
                    corr, _ = pearsonr(series_i[valid], series_j[valid])
                    corr = float(corr) if not np.isnan(corr) else 0.0

                corr_matrix[i, j] = corr
                corr_matrix[j, i] = corr

        np.fill_diagonal(corr_matrix, 1.0)
        self.pearson_matrix = corr_matrix
        return corr_matrix

    # ──────────────────────────────────────────
    # 2. Dynamic Correlation (DTC-STGCN method)
    # ──────────────────────────────────────────

    def compute_dynamic_correlation(self, speed_data, method="FD", fixed_adj=None):
        """
        Compute time-averaged dynamic correlation matrices.
        Based on Equations 3-4 of DTC-STGCN paper.
        
        Args:
            speed_data: (T, N) speed time series
            method: "FR" (feature ratio) | "FD" (feature difference)
            fixed_adj: (N, N) fixed connectivity matrix (optional mask)
        Returns:
            dynamic_matrix: (N, N) averaged dynamic correlation
        """
        if speed_data.shape[0] == self.num_nodes:
            speed_data = speed_data.T

        T, N = speed_data.shape

        # Min-max scale to [0,1] to avoid negative speeds causing issues
        scaler = MinMaxScaler()
        speed_scaled = scaler.fit_transform(speed_data)  # (T, N)

        cumulative = np.zeros((N, N), dtype=np.float64)
        count = 0

        for t in range(T):
            F_t = speed_scaled[t]  # (N,)

            if method == "FR":
                # St_ij = F_i / (F_i + F_j + eps)
                Fi = F_t[:, None]   # (N, 1)
                Fj = F_t[None, :]   # (1, N)
                St = Fi / (Fi + Fj + 1e-8)

            elif method == "FD":
                # St_ij = |F_i - F_j|
                Fi = F_t[:, None]
                Fj = F_t[None, :]
                St = np.abs(Fi - Fj)

            # Apply fixed adjacency mask if provided
            if fixed_adj is not None:
                St = St * fixed_adj

            cumulative += St
            count += 1

        dynamic_matrix = (cumulative / count).astype(np.float32)
        self.dynamic_matrix = dynamic_matrix
        return dynamic_matrix

    # ──────────────────────────────────────────
    # 3. Spatial Proximity Correlation
    # ──────────────────────────────────────────

    def compute_spatial_correlation(self, sigma=0.1):
        """
        Compute spatial correlation based on node coordinates.
        Uses Gaussian kernel: w_ij = exp(-d_ij^2 / (2*sigma^2))
        
        Args:
            sigma: Bandwidth for Gaussian kernel
        Returns:
            spatial_matrix: (N, N) spatial correlation matrix
        """
        if self.coordinates is None:
            raise ValueError("Coordinates not provided.")

        N = self.num_nodes
        coords = np.array(self.coordinates)

        # Pairwise Euclidean distances
        diff = coords[:, None, :] - coords[None, :, :]  # (N, N, 2)
        dist_sq = np.sum(diff ** 2, axis=-1)  # (N, N)

        # Gaussian kernel
        spatial_matrix = np.exp(-dist_sq / (2 * sigma ** 2)).astype(np.float32)
        np.fill_diagonal(spatial_matrix, 1.0)

        self.spatial_matrix = spatial_matrix
        return spatial_matrix

    # ──────────────────────────────────────────
    # 4. Attention-Based Correlation (from TH-GAT)
    # ──────────────────────────────────────────

    def extract_attention_correlation(self, model, data_loader, device="cpu", n_batches=20):
        """
        Extract attention weight-based correlation from trained TH-GAT model.
        
        Args:
            model: Trained THGAT model
            data_loader: DataLoader with traffic data
            device: Computation device
            n_batches: Number of batches to average over
        Returns:
            attention_matrix: (num_nodes, num_nodes) averaged attention weights
        """
        model.eval()
        attention_accumulate = None
        count = 0

        with torch.no_grad():
            for i, (x, _) in enumerate(data_loader):
                if i >= n_batches:
                    break

                x = x.to(device)
                attn = model.get_attention_weights(x)  # (batch, n_bar, n_bar)

                # Extract only original node-to-node part (top-left block)
                n = model.num_nodes
                node_attn = attn[:, :n, :n]  # (batch, N, N)
                mean_attn = node_attn.mean(dim=0).cpu().numpy()  # (N, N)

                if attention_accumulate is None:
                    attention_accumulate = mean_attn
                else:
                    attention_accumulate += mean_attn
                count += 1

        if count > 0:
            self.attention_matrix = (attention_accumulate / count).astype(np.float32)

        return self.attention_matrix

    # ──────────────────────────────────────────
    # 5. Combined Correlation
    # ──────────────────────────────────────────

    def compute_combined_correlation(self, weights=None):
        """
        Combine available correlation matrices into a unified matrix.
        
        Args:
            weights: Dict with keys "pearson", "dynamic", "spatial", "attention"
                     and float values summing to 1.0
        Returns:
            combined: (N, N) combined correlation matrix
        """
        if weights is None:
            weights = {
                "pearson": 0.4,
                "dynamic": 0.3,
                "spatial": 0.3,
            }

        available = {}
        if self.pearson_matrix is not None:
            available["pearson"] = self._normalize_matrix(np.abs(self.pearson_matrix))
        if self.dynamic_matrix is not None:
            available["dynamic"] = self._normalize_matrix(self.dynamic_matrix)
        if self.spatial_matrix is not None:
            available["spatial"] = self._normalize_matrix(self.spatial_matrix)
        if self.attention_matrix is not None:
            available["attention"] = self._normalize_matrix(self.attention_matrix)

        if not available:
            raise ValueError("No correlation matrices computed yet.")

        # Re-normalize weights for available matrices
        total_w = sum(weights.get(k, 0.0) for k in available)
        if total_w <= 0:
            total_w = len(available)
            adj_weights = {k: 1.0 / len(available) for k in available}
        else:
            adj_weights = {k: weights.get(k, 0.0) / total_w for k in available}

        combined = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float32)
        for name, mat in available.items():
            combined += adj_weights[name] * mat

        return combined

    def _normalize_matrix(self, mat):
        """Normalize matrix to [0, 1] range."""
        min_v, max_v = mat.min(), mat.max()
        if max_v - min_v < 1e-8:
            return np.zeros_like(mat)
        return (mat - min_v) / (max_v - min_v)

    # ──────────────────────────────────────────
    # 6. Region Detection (for TH-GAT)
    # ──────────────────────────────────────────

    def detect_regions_from_correlation(self, correlation_matrix, n_regions=5, threshold=0.7):
        """
        Detect communities/regions based on correlation matrix.
        Nodes with high pairwise correlation form a region.
        
        Args:
            correlation_matrix: (N, N) correlation matrix
            n_regions: Target number of regions
            threshold: Correlation threshold for connectivity
        Returns:
            region_labels: (N,) array of region assignments
        """
        from sklearn.cluster import SpectralClustering

        # Threshold correlation to create similarity graph
        similarity = np.where(np.abs(correlation_matrix) >= threshold,
                              np.abs(correlation_matrix), 0.0)
        np.fill_diagonal(similarity, 1.0)

        # Ensure connectivity
        min_conn = similarity.sum(axis=1).min()
        if min_conn < 1e-8:
            # Fallback: use full correlation as similarity
            similarity = np.abs(correlation_matrix)
            np.fill_diagonal(similarity, 1.0)

        clustering = SpectralClustering(
            n_clusters=n_regions,
            affinity="precomputed",
            random_state=42,
            n_init=10,
        )
        region_labels = clustering.fit_predict(similarity)

        return region_labels

    # ──────────────────────────────────────────
    # 7. Summary Statistics
    # ──────────────────────────────────────────

    def get_summary(self):
        """Get summary statistics of all computed correlation matrices."""
        summary = {}
        matrices = {
            "pearson": self.pearson_matrix,
            "dynamic": self.dynamic_matrix,
            "spatial": self.spatial_matrix,
            "attention": self.attention_matrix,
        }
        for name, mat in matrices.items():
            if mat is not None:
                upper = mat[np.triu_indices(self.num_nodes, k=1)]
                summary[name] = {
                    "mean": float(np.mean(upper)),
                    "std": float(np.std(upper)),
                    "max": float(np.max(upper)),
                    "min": float(np.min(upper)),
                    "high_corr_pairs": int(np.sum(np.abs(upper) > 0.7)),
                }
        return summary


if __name__ == "__main__":
    N = 20
    T = 100
    
    # Dummy speed data
    speed_data = np.random.rand(T, N) * 50 + 20  # 20-70 km/h
    coords = np.random.rand(N, 2)
    adj = (np.random.rand(N, N) > 0.7).astype(float)
    adj = np.clip(adj + adj.T, 0, 1)

    cm = TrafficCorrelationMatrix(N, coordinates=coords)

    pearson = cm.compute_pearson_correlation(speed_data)
    print(f"Pearson matrix: {pearson.shape}, mean={pearson.mean():.3f}")

    dynamic = cm.compute_dynamic_correlation(speed_data, method="FD", fixed_adj=adj)
    print(f"Dynamic matrix: {dynamic.shape}, mean={dynamic.mean():.3f}")

    spatial = cm.compute_spatial_correlation(sigma=0.2)
    print(f"Spatial matrix: {spatial.shape}, mean={spatial.mean():.3f}")

    combined = cm.compute_combined_correlation()
    print(f"Combined matrix: {combined.shape}, mean={combined.mean():.3f}")

    summary = cm.get_summary()
    for name, stats in summary.items():
        print(f"  {name}: {stats}")

    print("✓ TrafficCorrelationMatrix OK")