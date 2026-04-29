"""
Data Loader for DTC-STGCN
Load and prepare data from .npz files produced by the pipeline.

Normalize contract (single source of truth):
    - NPZ từ export_for_model_training(normalize=True) → data ĐÃ normalize + có scaler_mean/scale
    - NPZ từ export_for_model_training(normalize=False) → data CHƯA normalize, không có scaler
    - prepare_for_training(normalize=False) → KHÔNG scale thêm, dùng dữ liệu nguyên bản
    - prepare_for_training(normalize=True)  → chỉ scale khi NPZ chưa normalize (fallback)
    => Không bao giờ normalize 2 lần.
"""

import os
import sys
import json
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

# Dataset
class TrafficDataset(Dataset):
    """PyTorch Dataset for traffic data."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        Args:
            X: (samples, seq_len, nodes, features)
            y: (samples, pred_len, nodes, features)
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Low-level loaders
def load_graph_structure(filepath: str):
    """
    Load graph structure from .npz (graph_structure_*.npz từ map_matcher).

    Returns:
        adj            : (N, N) binary adjacency — raw, chưa normalize
        node_features  : (N, F) hoặc None
        coordinates    : (N, 2) hoặc None
        segment_ids    : (N,) hoặc None
    """
    data = np.load(filepath, allow_pickle=True)
    print(f"\n=== Loading Graph Structure: {os.path.basename(filepath)} ===")

    # ── Adjacency matrix ──
    if 'adjacency_matrix' in data:
        adj = data['adjacency_matrix'].astype(np.float32)
        print(f"Adjacency matrix: {adj.shape}")
    elif 'edge_index' in data:
        edge_index = data['edge_index']
        if 'num_nodes' in data:
            num_nodes = int(data['num_nodes'])
        elif 'node_features' in data:
            num_nodes = int(data['node_features'].shape[0])
        elif 'coordinates' in data:
            num_nodes = int(data['coordinates'].shape[0])
        else:
            num_nodes = int(edge_index.max()) + 1
        adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        if edge_index.ndim == 2 and edge_index.shape[1] > 0:
            adj[edge_index[0], edge_index[1]] = 1.0
        print(f"Built adjacency from edge_index: {adj.shape}, edges={int(adj.sum())}")
    else:
        adj = None
        print("Warning: No adjacency data found in graph file")

    node_features = data['node_features'] if 'node_features' in data else None
    coordinates   = data['coordinates']   if 'coordinates'   in data else None
    segment_ids   = data['osm_node_ids']  if 'osm_node_ids'  in data else (
                    data['segment_ids']   if 'segment_ids'   in data else None)

    if node_features is not None:
        print(f"Node features: {node_features.shape}")
    if coordinates is not None:
        print(f"Coordinates: {coordinates.shape}")
    if adj is not None:
        n_edges = int(adj.sum())
        avg_deg = n_edges / adj.shape[0] if adj.shape[0] > 0 else 0
        print(f"Graph stats: {n_edges} edges, avg degree={avg_deg:.2f}")

    return adj, node_features, coordinates, segment_ids


def _coerce_to_float32(arr: np.ndarray, key: str) -> np.ndarray:
    """
    Chuyển array (kể cả object dtype) về float32 một cách an toàn.
    Raise ValueError nếu không thể.
    """
    if arr.dtype in (np.float32, np.float64, np.int32, np.int64):
        return arr.astype(np.float32)

    # object array — thử cast trực tiếp trước
    try:
        return arr.astype(np.float32)
    except (ValueError, TypeError):
        pass

    # fallback: chuyển từng phần tử, bỏ qua non-numeric
    flat = arr.flatten()
    numeric = []
    for v in flat:
        try:
            numeric.append(float(v))
        except (ValueError, TypeError):
            numeric.append(np.nan)

    result = np.array(numeric, dtype=np.float32).reshape(arr.shape)
    n_nan = np.isnan(result).sum()
    if n_nan > 0:
        print(f"  Warning: {key} had {n_nan} non-numeric values replaced with NaN")
    return result


def load_model_ready_data(filepath: str) -> dict:
    """
    Load model_ready_data_*.npz được tạo bởi pipeline.

    Returns:
        dict với keys: X_train, y_train, X_val, y_val, X_test, y_test
                       (optional) segment_ids, feature_names, scaler_mean, scaler_scale
                       data_normalized: bool — True nếu data trong file đã scale
    """
    data = np.load(filepath, allow_pickle=True)
    print(f"\n=== Loading Model Ready Data: {os.path.basename(filepath)} ===")
    print(f"Keys in file: {data.files}")

    result = {}

    # ── Load X/y splits ──
    for key in ['X_train', 'y_train', 'X_val', 'y_val', 'X_test', 'y_test']:
        if key not in data:
            print(f"Warning: '{key}' not found in npz")
            continue
        arr = _coerce_to_float32(data[key], key)
        result[key] = arr
        print(f"{key}: {arr.shape}  dtype={arr.dtype}")

    # ── Optional arrays ──
    for key in ('segment_ids', 'feature_names', 'scaler_mean', 'scaler_scale'):
        if key in data:
            result[key] = data[key]

    # ── Metadata: biết data đã normalize chưa ──
    data_normalized = True  # mặc định an toàn: coi là đã normalize
    if '_metadata' in data:
        try:
            meta = json.loads(str(data['_metadata'].flat[0]))
            data_normalized = bool(meta.get('normalized', True))
        except Exception:
            pass
    result['data_normalized'] = data_normalized
    print(f"data_normalized (from metadata): {data_normalized}")

    # Kiểm tra đủ keys
    missing = [k for k in ('X_train', 'y_train', 'X_val', 'y_val', 'X_test', 'y_test')
               if k not in result]
    if missing:
        raise ValueError(f"Missing required keys in NPZ: {missing}")

    return result

# Helpers
def create_sequences(data: np.ndarray, seq_len: int = 12,
                     pred_len: int = 12, stride: int = 1):
    """
    Sliding-window sequences từ time series.

    Args:
        data: (timesteps, nodes, features)
    Returns:
        X: (samples, seq_len, nodes, features)
        y: (samples, pred_len, nodes, features)
    """
    X, y = [], []
    for i in range(0, len(data) - seq_len - pred_len + 1, stride):
        X.append(data[i:i + seq_len])
        y.append(data[i + seq_len:i + seq_len + pred_len])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def prepare_data_loaders(X_train, y_train, X_val, y_val,
                         X_test, y_test, batch_size: int = 32):
    """Tạo PyTorch DataLoaders."""
    train_loader = DataLoader(TrafficDataset(X_train, y_train),
                              batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=False)
    val_loader   = DataLoader(TrafficDataset(X_val,   y_val),
                              batch_size=batch_size, shuffle=False,
                              num_workers=0, pin_memory=False)
    test_loader  = DataLoader(TrafficDataset(X_test,  y_test),
                              batch_size=batch_size, shuffle=False,
                              num_workers=0, pin_memory=False)
    return train_loader, val_loader, test_loader


def normalize_adj(adj: np.ndarray) -> np.ndarray:
    """
    Symmetric normalization: D^(-1/2) * (A + I) * D^(-1/2)

    Dùng nội bộ trong các model. DataManager KHÔNG gọi hàm này —
    adj raw binary được trả về để model tự normalize một lần duy nhất.
    """
    A = adj + np.eye(adj.shape[0], dtype=np.float32)
    rowsum = A.sum(axis=1)
    d_inv_sqrt = np.where(rowsum > 0, rowsum ** -0.5, 0.0)
    D = np.diag(d_inv_sqrt)
    return (D @ A @ D).astype(np.float32)


def normalize_data(X_train, X_val, X_test):
    """
    StandardScaler fit trên train, transform val/test.
    Chỉ gọi khi NPZ không có scaler và normalize=True.
    """
    sh_train, sh_val, sh_test = X_train.shape, X_val.shape, X_test.shape
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train.reshape(-1, sh_train[-1])).reshape(sh_train).astype(np.float32)
    Xva = scaler.transform(X_val.reshape(-1, sh_val[-1])).reshape(sh_val).astype(np.float32)
    Xte = scaler.transform(X_test.reshape(-1, sh_test[-1])).reshape(sh_test).astype(np.float32)
    return Xtr, Xva, Xte, scaler

# DataManager
class DataManager:
    """
    Quản lý toàn bộ vòng đời data: load → align → (optional scale) → DataLoader.

    Normalize contract — không bao giờ scale 2 lần:
        prepare_for_training(normalize=False)  → dùng nguyên data từ NPZ (recommended)
        prepare_for_training(normalize=True)   → chỉ scale khi NPZ chưa normalize
    """

    def __init__(self, data_dir: str = None):
        if data_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__))))
            data_dir = os.path.join(base_dir, 'data', 'processed')
        self.data_dir = data_dir
        self.adj: np.ndarray = None
        self.node_features = None
        self.coordinates   = None
        self.graph_segment_ids = None
        self.data: dict    = {}
        self.scaler        = None
        self.scaler_mean   = None
        self.scaler_scale  = None

    # ── Load ──────────────────────────────────────────────────────────────────

    def load_all(self):
        """Load graph_structure và model_ready_data (file mới nhất của mỗi loại)."""
        graph_dir = os.path.join(self.data_dir, 'graph_structure')
        model_dir = os.path.join(self.data_dir, 'model_ready_data')

        graph_files = sorted(glob.glob(os.path.join(graph_dir, 'graph_structure_*.npz')))
        model_files = sorted(glob.glob(os.path.join(model_dir, 'model_ready_data_*.npz')))

        if not graph_files:
            print("Warning: No graph_structure_*.npz found")
        if not model_files:
            raise FileNotFoundError(
                f"No model_ready_data_*.npz found in {model_dir}")

        if graph_files:
            self.adj, self.node_features, self.coordinates, self.graph_segment_ids = \
                load_graph_structure(graph_files[-1])

        self.data = load_model_ready_data(model_files[-1])
        return self

    # ── Align nodes ───────────────────────────────────────────────────────────

    def _align_nodes(self):
        """
        Đồng bộ node dimension giữa X/y (từ model_ready_data) và adj (từ graph_structure).

        Trường hợp thường gặp: map_matcher trả về N_graph nodes, nhưng
        model_ready_data chỉ có N_model nodes (một subset đã được match).
        Hàm này lọc adj theo đúng tập nodes có trong X/y.
        """
        if self.adj is None:
            return
        if not (hasattr(self, 'graph_segment_ids') and self.graph_segment_ids is not None):
            return
        if 'segment_ids' not in self.data or self.data['segment_ids'] is None:
            return

        graph_ids = np.array(self.graph_segment_ids).reshape(-1).astype(int)
        model_ids = np.array(self.data['segment_ids']).reshape(-1).astype(int)

        id_to_graph_idx = {int(sid): i for i, sid in enumerate(graph_ids)}

        keep_model_pos, graph_pos_for_kept, missing = [], [], []
        for pos, sid in enumerate(model_ids):
            gi = id_to_graph_idx.get(int(sid))
            if gi is None:
                missing.append(int(sid))
            else:
                keep_model_pos.append(pos)
                graph_pos_for_kept.append(gi)

        if missing:
            print(f"Warning: {len(missing)} segment_ids in model data not found in graph; "
                  f"dropping them from X/y.")

        if not keep_model_pos:
            print("Warning: No overlapping segment_ids — using original data without alignment.")
            return

        kp = np.array(keep_model_pos,   dtype=int)
        gp = np.array(graph_pos_for_kept, dtype=int)

        # Slice node dimension (dim 2) of X/y
        for key in ('X_train', 'y_train', 'X_val', 'y_val', 'X_test', 'y_test'):
            if key in self.data:
                self.data[key] = self.data[key][:, :, kp, :]

        self.data['segment_ids'] = model_ids[kp]

        # Slice adjacency consistently
        self.adj = self.adj[np.ix_(gp, gp)]
        print(f"Aligned: {len(kp)} nodes kept, adjacency → {self.adj.shape}")

        if self.coordinates is not None and len(self.coordinates) == len(graph_ids):
            self.coordinates = self.coordinates[gp]
        if self.node_features is not None and len(self.node_features) == len(graph_ids):
            self.node_features = self.node_features[gp]

    # ── Prepare ───────────────────────────────────────────────────────────────

    def prepare_for_training(self, batch_size: int = 32, normalize: bool = False):
        """
        Chuẩn bị DataLoaders và adjacency tensor cho training.

        Args:
            batch_size: Kích thước batch.
            normalize : False (default & recommended) — dùng dữ liệu nguyên bản từ NPZ.
                        True — chỉ scale nếu NPZ chưa normalize; không scale lại nếu đã có.

        Returns:
            train_loader, val_loader, test_loader, adj_tensor (raw binary, float32)

        Lưu ý về adj_tensor:
            Trả về adjacency RAW BINARY (0/1). Normalize D^-1/2·A·D^-1/2 được thực hiện
            MỘT LẦN duy nhất BÊN TRONG mỗi model (DTCSTGCN._normalize_adj, v.v.).
            Không normalize ở đây để tránh double-normalize.
        """
        if not self.data:
            raise RuntimeError("Data not loaded. Call load_all() first.")

        required = ['X_train', 'y_train', 'X_val', 'y_val', 'X_test', 'y_test']
        missing  = [k for k in required if k not in self.data]
        if missing:
            raise ValueError(f"Missing data keys after loading: {missing}")

        # Align graph nodes với model data (an toàn nếu graph_segment_ids là None)
        self._align_nodes()

        X_train = self.data['X_train']
        y_train = self.data['y_train']
        X_val   = self.data['X_val']
        y_val   = self.data['y_val']
        X_test  = self.data['X_test']
        y_test  = self.data['y_test']

        print(f"\nData shapes after alignment:")
        print(f"  X_train: {X_train.shape}  y_train: {y_train.shape}")
        print(f"  X_val:   {X_val.shape}    y_val:   {y_val.shape}")
        print(f"  X_test:  {X_test.shape}   y_test:  {y_test.shape}")

        # ── Normalize contract ────────────────────────────────────────────────
        data_already_normalized = self.data.get('data_normalized', True)
        has_saved_scaler = ('scaler_mean' in self.data and 'scaler_scale' in self.data)

        if normalize:
            if data_already_normalized:
                # NPZ đã normalize → không làm gì thêm
                print("Data already normalized in NPZ — skipping extra scaling.")
                y_train_out = y_train
                y_val_out   = y_val
                y_test_out  = y_test

            elif has_saved_scaler:
                # NPZ chưa normalize nhưng có scaler → dùng scaler từ file
                mean  = np.asarray(self.data['scaler_mean'],  dtype=np.float32)
                scale = np.asarray(self.data['scaler_scale'], dtype=np.float32)
                scale_safe = np.where(scale != 0, scale, 1.0)

                def _apply(arr):
                    sh  = arr.shape
                    out = (arr.reshape(-1, sh[-1]) - mean) / scale_safe
                    return out.reshape(sh).astype(np.float32)

                X_train, X_val, X_test = _apply(X_train), _apply(X_val), _apply(X_test)
                y_train_out = _apply(y_train)
                y_val_out   = _apply(y_val)
                y_test_out  = _apply(y_test)
                print("Applied saved scaler to raw NPZ data.")

            else:
                # Không có scaler → fit trên train (fallback, thường không xảy ra)
                print("No saved scaler found — fitting StandardScaler on X_train.")
                X_train, X_val, X_test, self.scaler = normalize_data(X_train, X_val, X_test)
                sh_y = y_train.shape
                y_train_out = self.scaler.transform(
                    y_train.reshape(-1, sh_y[-1])).reshape(sh_y).astype(np.float32)
                y_val_out   = self.scaler.transform(
                    y_val.reshape(-1, sh_y[-1])).reshape(sh_y).astype(np.float32)
                y_test_out  = self.scaler.transform(
                    y_test.reshape(-1, sh_y[-1])).reshape(sh_y).astype(np.float32)
        else:
            # normalize=False: dùng nguyên data từ NPZ
            y_train_out = y_train
            y_val_out   = y_val
            y_test_out  = y_test

        # Expose scaler params để inverse_transform hoạt động sau training
        if has_saved_scaler:
            self.scaler_mean  = self.data['scaler_mean']
            self.scaler_scale = self.data['scaler_scale']

        # ── DataLoaders ───────────────────────────────────────────────────────
        train_loader, val_loader, test_loader = prepare_data_loaders(
            X_train, y_train_out,
            X_val,   y_val_out,
            X_test,  y_test_out,
            batch_size=batch_size,
        )

        # ── Adjacency tensor — RAW BINARY ─────────────────────────────────────
        if self.adj is not None:
            # Sanity check: N_adj == N_data
            N_data = X_train.shape[2]
            N_adj  = self.adj.shape[0]
            if N_adj != N_data:
                print(f"Warning: adj N={N_adj} != data N={N_data}. "
                      f"Creating identity adjacency as fallback.")
                adj_tensor = torch.eye(N_data, dtype=torch.float32)
            else:
                adj_tensor = torch.FloatTensor(self.adj)
        else:
            N_data = X_train.shape[2]
            print(f"Warning: No adjacency — creating fully-connected graph ({N_data} nodes).")
            adj_tensor = torch.ones(N_data, N_data, dtype=torch.float32)
            adj_tensor.fill_diagonal_(0.0)

        print(f"\nAdjacency tensor: {adj_tensor.shape}  "
              f"(raw binary, normalize=False here — model normalizes internally)")

        return train_loader, val_loader, test_loader, adj_tensor

    # ── Inverse transform ─────────────────────────────────────────────────────

    def inverse_transform(self, arr: np.ndarray) -> np.ndarray:
        """
        Đưa predictions / targets từ normalized space về đơn vị gốc.

        Dùng scaler lưu trong NPZ. Cần gọi prepare_for_training() trước.

        Args:
            arr: numpy array bất kỳ shape, dimension cuối = num_features
        Returns:
            arr_orig: cùng shape, đã inverse-transform
        """
        if self.scaler_mean is None or self.scaler_scale is None:
            # Thử dùng sklearn scaler nếu đã fit ở bước normalize fallback
            if self.scaler is not None:
                sh = arr.shape
                return self.scaler.inverse_transform(
                    arr.reshape(-1, sh[-1])).reshape(sh).astype(np.float32)
            raise RuntimeError(
                "Không có scaler để inverse transform. "
                "Đảm bảo NPZ được tạo với normalize=True hoặc đã gọi "
                "prepare_for_training(normalize=True).")

        mean  = np.asarray(self.scaler_mean,  dtype=np.float32)
        scale = np.asarray(self.scaler_scale, dtype=np.float32)
        sh    = arr.shape
        out   = arr.reshape(-1, sh[-1]) * scale + mean
        return out.reshape(sh).astype(np.float32)

# Quick smoke-test
if __name__ == "__main__":
    print("Testing DataManager...")
    dm = DataManager()
    dm.load_all()

    train_loader, val_loader, test_loader, adj = dm.prepare_for_training(
        batch_size=4, normalize=False)

    print(f"\nTrain batches : {len(train_loader)}")
    print(f"Val batches   : {len(val_loader)}")
    print(f"Test batches  : {len(test_loader)}")
    print(f"Adjacency     : {adj.shape}")

    for X, y in train_loader:
        print(f"X batch: {X.shape}  y batch: {y.shape}")
        break

    print("\n✓ DataManager OK")