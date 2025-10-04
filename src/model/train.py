import os, math, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import spearmanr
from collections import defaultdict


# ---------------------------
# Data
# ---------------------------
def default_paths(base = Path("C:/AI/Specialized_Project2_github/Urban-Traffic-Links/data/raw/kaggle")):
    base = Path(base)
    return dict(
        train=base / 'train.csv',
        nodes=base / 'nodes.csv',
        segments=base / 'segments.csv',
        streets=base / 'streets.csv',
        segment_status=base / 'segment_status.csv',
    )

def load_data(paths):
    train = pd.read_csv(paths['train'])
    nodes = pd.read_csv(paths['nodes'])
    segments = pd.read_csv(paths['segments'])
    streets = pd.read_csv(paths['streets'])
    segment_status = pd.read_csv(paths['segment_status'])
    return train, nodes, segments, streets, segment_status

# ---------------------------
# Features & targets
# ---------------------------

#Calculate D^-1/2.A.D^-1/2
def normalize_adj(A):
    D = np.diag(np.power(A.sum(1), -0.5).flatten())
    return D @ A @ D

#Tạo đặc trưng gán vào X

def prepare_feature_labels(train, seg_ids, snapshot_date, snapshot_period=None, task='classification'):
    d = pd.to_datetime(snapshot_date).floor('D')
    snap = train[train['date'].dt.floor('D') == d]

    if snapshot_period is not None:
        snap = snap[snap['period'] == snapshot_period]
    seg_first = (snap.sort_values('date')
                 .drop_duplicates('segment_id')
                 .set_index('segment_id'))
    X_num = pd.DataFrame({
        "length": seg_first["length"].astype(float),
        "street_level": seg_first["street_level"].astype(float).fillna(-1.0),
        "max_velocity": seg_first["max_velocity"].astype(float).fillna(-1.0),
        "weekday": seg_first["weekday"].astype(float) if "weekday" in seg_first.columns else 0.0
    })

    # categorical features
    X_cat_period = pd.get_dummies(seg_first["period"].fillna("unknown"),
                                  prefix="period") if "period" in seg_first.columns else pd.DataFrame()
    X_cat_type = pd.get_dummies(seg_first["street_type"].fillna("unknown"),
                                prefix="type") if "street_type" in seg_first.columns else pd.DataFrame()

    X_df = pd.concat([X_num, X_cat_period, X_cat_type], axis=1)
    X_df = X_df.reindex(seg_ids, fill_value=0.0)  # align order
    X = X_df.values.astype(np.float32)
    return X, {"snapshot_date": str(snapshot_date), "snapshot_period": snapshot_period}


def edges_from_adj(A):
    """Lấy tam giác trên từ ma trận kề."""
    u, v = np.where(np.triu(A, k=1) > 0.0)
    return np.stack([u, v], axis=1)

def edge_key(u, v):  # cho set()
    return (int(u), int(v)) if u <= v else (int(v), int(u))

def split_edges(pos_edges, val_ratio=0.1, test_ratio=0.1, seed=0):
    rng = np.random.default_rng(seed)

    #Init array
    idx = np.arange(len(pos_edges))

    #Mix elements
    rng.shuffle(idx)

    n_val  = int(len(idx) * val_ratio)
    n_test = int(len(idx) * test_ratio)
    val_idx  = idx[:n_val]
    test_idx = idx[n_val:n_val+n_test]
    train_idx= idx[n_val+n_test:]
    return pos_edges[train_idx], pos_edges[val_idx], pos_edges[test_idx]

def negative_sampling(n, num_samples, forbidden,seed=0):
    """
    Lấy num_samples cặp (u,v) không có cạnh.
    'forbidden' là set các cạnh dương + các âm đã chọn (để tránh trùng).
    """
    rng = np.random.default_rng(seed)
    neg = []
    tries = 0
    while len(neg) < num_samples and tries < num_samples*100:
        u = int(rng.integers(0,n))
        v = int(rng.integers(0,n))
        tries += 1
        if u == v:
            continue
        k = edge_key(u, v)
        if k in forbidden:
            continue
        forbidden.add(k)
        neg.append([u,v])
    return np.array(neg, dtype=np.int64)
# ---------------------------
# LOS utilities + temporal split
# ---------------------------
LOS_TO_ORD = {"A": 6, "B": 5, "C": 4, "D": 3, "E": 2, "F": 1}
def make_timestamp(df):
    if "date" not in df.columns:
        raise ValueError("Missing date column")
    d = pd.to_datetime(df["date"], errors="coerce")
    if "period" in df.columns and df["period"].notna().any():
        # chuyển 'period_h_m' -> 'h:m'
        per = df["period"].fillna("").astype(str).str.replace("period_","",regex=False)
        # pad giờ/phút (9_5 -> 09:05)
        per = per.str.replace("_",":",regex=False)
        per = pd.to_datetime(per, format="%H:%M", errors="coerce").dt.time
        df["ts"] = pd.to_datetime(d.dt.date.astype(str)+ " " + pd.Series(per).astype(str),errors="coerce")
    else:
        df["ts"] = d
    return df

def temporal_split_intervals(ts_series, train_ratio=0.7, mature_frac=0.8):
    """
        Chia timeline thành:
          Train:  [t0, tc]  với maturing [t0, tb], probing (train) (tb, tc]
          Test:   (tc, td]  với maturing (tc, td_m], probing (test) (td_m, td]
    """
    t_sorted = ts_series.dropna().sort_values()
    if len(t_sorted) < 6:
        raise ValueError("Không đủ mốc thời gian để temporal split.")
    t0 = t_sorted.iloc[0]
    td = t_sorted.iloc[-1]
    tc = t_sorted.quantile(train_ratio)
    tb = pd.to_datetime(t0 + mature_frac * (tc - t0))
    td_m = pd.to_datetime(tc + mature_frac * (td - tc))
    return t0, tb, tc, td_m, td

def build_spearman_adj_from_window(df, seg_ids, t_start, t_end, threshold=0.7, topk=None):
    """
        Tạo adjacency theo Spearman(LOS) trong cửa sổ [t_start, t_end].
        - df cần có cột: ts, segment_id, LOS (A..F)
        - seg_ids: list segment theo thứ tự cố định
        """
    sub = df[(df["ts"] >= t_start) & (df["ts"] <= t_end)][["ts", "segment_id", "LOS"]].copy()
    if sub.empty:
        raise ValueError("Empty dataframe")
    sub["los_ord"] = sub["LOS"].map(LOS_TO_ORD).astype("float32")
    piv = sub.pivot_table(index="ts", columns="segment_id", values="los_ord", aggfunc="mean")
    piv = piv.reindex(columns=seg_ids)
    piv = piv.sort_index().interpolate(limit_direction="both").fillna(piv.mean())
    rho, _ = spearmanr(piv.values, axis = 0)
    rho = np.nan_to_num(rho, nan=0.0)
    n = len(seg_ids)
    A = np.zeros((n,n), dtype=np.float32)
    if topk:
        idx = np.argsort(-rho, axis=1)[:, 1:topk + 1]  # bỏ self
        rows = np.arange(n)[:, None] #CHuyển thành cột
        A[rows, idx] = rho[rows, idx]
        A = np.maximum(A, A.T)  # đối xứng bằng max
    else:
        A = (rho >= float(threshold)).astype(np.float32)
        np.fill_diagonal(A, 1.0)
    u, v = np.where(np.triu(A, k=1) > 0)
    edges = np.stack([u, v], axis=1)
    return A, edges

def formed_edges(A_maturing, A_probing):
    U1,V1 = np.where(np.triu(A_maturing, k=1) > 0)
    U2,V2 = np.where(np.triu(A_probing, k=1) > 0)
    had = set(zip(U1.tolist(), V1.tolist()))
    formed = [(u, v) for u, v in zip(U2.tolist(), V2.tolist()) if (u, v) not in had]
    return np.array(formed, dtype=np.int64)


def build_topology_adj(segments_df, seg_ids):
    """
    A_topo: 2 segment có cạnh nếu CHIA SẺ cùng s_node_id hoặc e_node_id.
    """
    seg_df = segments_df.copy()
    # chuẩn hoá tên cột: có thể là '_id' hoặc 'segment_id'
    if "segment_id" not in seg_df.columns and "_id" in seg_df.columns:
        seg_df = seg_df.rename(columns={"_id": "segment_id"})
    needed = {"segment_id","s_node_id","e_node_id"}
    missing = needed - set(seg_df.columns)
    if missing:
        raise ValueError(f"Thiếu cột cho topo: {missing}")

    seg_df = seg_df[seg_df["segment_id"].isin(seg_ids)].copy()
    id2idx = {sid: i for i, sid in enumerate(seg_ids)}

    node_to_segs = defaultdict(list)
    for _, r in seg_df.iterrows():
        sid = int(r["segment_id"])
        if sid not in id2idx:
            continue
        i = id2idx[sid]
        node_to_segs[int(r["s_node_id"])].append(i)
        node_to_segs[int(r["e_node_id"])].append(i)

    n = len(seg_ids)
    A = np.zeros((n, n), dtype=np.float32)
    for seg_list in node_to_segs.values():
        for a in range(len(seg_list)):
            for b in range(a+1, len(seg_list)):
                i, j = seg_list[a], seg_list[b]
                A[i, j] = 1.0
                A[j, i] = 1.0
    np.fill_diagonal(A, 1.0)
    return A, seg_ids, id2idx

def combine_adj(A_topo, A_corr, alpha=0.7, thresh=0.5):
    """
    A_final = alpha*A_topo + (1-alpha)*A_corr  -> sau đó nhị phân hoá theo thresh.
    Trả về A_bin (0/1, có đường chéo 1.0)
    """
    Af = alpha * A_topo + (1.0 - alpha) * A_corr
    A_bin = (Af >= float(thresh)).astype(np.float32)
    np.fill_diagonal(A_bin, 1.0)
    return A_bin

# =========================
# GCN encoder + Dot decoder
# =========================



class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.lin3 = nn.Linear(hidden_dim, out_dim, bias=False)
    def forward(self,X,Ahat):
        H = Ahat @ self.lin1(X)
        H = torch.relu(H)
        H = Ahat @ self.lin2(H)
        H = torch.relu(H)
        Z = Ahat @ self.lin3(H)
        return Z

class DotDecoder(nn.Module):
    def forward(self,z,edges):
        u = z[edges[:,0]]
        v = z[edges[:,1]]
        return (u*v).sum(dim=1)

@torch.no_grad()
def eval_linkpred(encoder, decoder, Ahat_t, X_t, pos_edges, neg_edges):
    encoder.eval()
    z = encoder(X_t, Ahat_t)
    pos = torch.from_numpy(pos_edges).long()
    neg = torch.from_numpy(neg_edges).long()
    s_pos = torch.sigmoid(decoder(z, pos)).cpu().numpy()
    s_neg = torch.sigmoid(decoder(z, neg)).cpu().numpy()
    y = np.concatenate([np.ones_like(s_pos), np.zeros_like(s_neg)])
    s = np.concatenate([s_pos, s_neg])
    auc = roc_auc_score(y, s)
    ap  = average_precision_score(y, s)
    return auc, ap


def train_linkpred(Ahat, X, pos_tr, pos_val, pos_te, hidden=64, emb_dim=64, epochs=400, lr=1e-2, wd=5e-4,seed=0):
    device = torch.device("cpu")
    n = X.shape[0]
    # negative edges
    forb = set(edge_key(u,v) for u,v in np.vstack([pos_tr, pos_val, pos_te]))
    neg_tr = negative_sampling(n, len(pos_tr), set(forb),seed)
    neg_val= negative_sampling(n, len(pos_val), set(forb),seed+1)
    neg_te = negative_sampling(n, len(pos_te), set(forb),seed+2)

    X_t = torch.from_numpy(X).float().to(device)
    A_t = torch.from_numpy(Ahat).float().to(device)
    pos_tr_t = torch.from_numpy(pos_tr).long()
    neg_tr_t = torch.from_numpy(neg_tr).long()

    encoder = GCN(in_dim=X.shape[1], hidden_dim=min(hidden, X.shape[1]), out_dim=emb_dim)
    decoder = DotDecoder()
    opt = torch.optim.SGD(
        encoder.parameters(),
        lr=lr,  # tốc độ học
        momentum=0.9,  # thường thêm momentum để hội tụ nhanh hơn
        weight_decay=wd  # L2 regularization
    )
    #Tự bao gồm sigmoid nội bộ
    bce = nn.BCEWithLogitsLoss()

    for _ in range(epochs):
        encoder.train()
        opt.zero_grad()
        z = encoder(X_t, A_t)
        logits_pos = decoder(z, pos_tr_t)
        logits_neg = decoder(z, neg_tr_t)
        logits = torch.cat([logits_pos, logits_neg], dim=0)
        labels = torch.cat([torch.ones_like(logits_pos), torch.zeros_like(logits_neg)], dim=0).float()
        loss = bce(logits, labels)
        loss.backward()
        opt.step()

    # eval
    val_auc, val_ap = eval_linkpred(encoder, decoder, A_t, X_t, pos_val, neg_val)
    te_auc,  te_ap  = eval_linkpred(encoder, decoder, A_t, X_t, pos_te,  neg_te)
    return encoder, decoder, {"val_auc":val_auc, "val_ap":val_ap, "test_auc":te_auc, "test_ap":te_ap}






def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    try:
        import torch_directml
        return torch_directml.device()
    except:
        return torch.device("cpu")

device = get_device()
print("[INFO] Device:", device)


# =========================
# Main
# =========================
def main():
    dev = get_device()
    print(f"[INFO] Device: {dev}")
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=str(Path.cwd() / "Urban-Traffic-Links" / "outputs"))
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--emb_dim", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--alpha", type=float, default=0.7)
    ap.add_argument("--corr_threshold", type=float, default=0.7)
    ap.add_argument("--final_threshold", type=float, default=0.5)
    ap.add_argument("--corr_topk", type=int, default=0)
    args = ap.parse_args()

    # Load data
    print("[INFO] Loading data ...")
    paths = default_paths()
    train, nodes, segments, streets, segment_status = load_data(paths)

    # Preprocess timestamp
    train = make_timestamp(train)
    train["date"] = pd.to_datetime(train["date"], errors="coerce")
    if "LOS" not in train.columns:
        raise ValueError("Thiếu cột LOS (A..F) để tính Spearman!")

    # Temporal split
    print("[INFO] Temporal split ...")
    t0, tb, tc, td_m, td = temporal_split_intervals(train["ts"])
    print(f"  Train window: [{t0} → {tc}] | Test window: ({tc} → {td}]")

    # Build topo adjacency
    seg_ids = sorted(train["segment_id"].dropna().unique().astype(int).tolist())
    print("[INFO] Building topology adjacency ...")
    A_topo, seg_ids, id2idx = build_topology_adj(segments,seg_ids)
    print(f"  A_topo shape = {A_topo.shape}")

    # Spearman adjacency
    print("[INFO] Building Spearman adjacency ...")
    A_corr, _ = build_spearman_adj_from_window(
        train, seg_ids, t_start=t0, t_end=tc,
        threshold=args.corr_threshold,
        topk=args.corr_topk if args.corr_topk > 0 else None
    )
    # --------------------------
    # In ma trận Spearman + các cặp tương quan không kề
    # --------------------------
    print("[INFO] Checking Spearman correlation usefulness ...")

    # In ma trận Spearman (nếu nhỏ)
    if A_corr.shape[0] <= 20:
        print("\n[Spearman correlation matrix (A_corr)]:")
        np.set_printoptions(precision=2, suppress=True)
        print(A_corr)
    else:
        print(f"A_corr too large to print ({A_corr.shape}) -> skip full print")

    # Tìm các cặp có tương quan cao nhưng không kề trong topology
    diff_mask = (A_corr >= args.corr_threshold) & (A_topo == 0)
    u, v = np.where(np.triu(diff_mask, k=1))  # lấy tam giác trên (tránh trùng)

    if len(u) == 0:
        print("[INFO] No correlated but non-adjacent pairs found.")
    else:
        print(f"\n[INFO] Found {len(u)} correlated-but-non-adjacent pairs (potential hidden correlations):")
        for i in range(min(20, len(u))):  # chỉ in 20 cặp đầu
            print(f"  Segment {seg_ids[u[i]]} ↔ Segment {seg_ids[v[i]]} | corr={A_corr[u[i], v[i]]:.3f}")

    # Combine adjacency
    A_final = combine_adj(A_topo, A_corr, alpha=args.alpha, thresh=args.final_threshold)
    Ahat = normalize_adj(A_final)

    # Split edges
    pos_edges = edges_from_adj(A_final)
    pos_tr, pos_val, pos_te = split_edges(pos_edges, val_ratio=0.1, test_ratio=0.1, seed=args.seed)
    print(f"[INFO] Edges: train={len(pos_tr)}, val={len(pos_val)}, test={len(pos_te)}")

    # Prepare features
    X, meta = prepare_feature_labels(train, seg_ids, snapshot_date=tc)
    print(f"[INFO] X shape = {X.shape}")

    # Train + eval
    print("[INFO] Training GCN ...")
    _, _, metrics = train_linkpred(
        Ahat, X, pos_tr, pos_val, pos_te,
        hidden=args.hidden, emb_dim=args.emb_dim,
        epochs=args.epochs, lr=args.lr, wd=args.wd, seed=args.seed
    )

    print(f"[RESULT] LinkPred  val_AUC={metrics['val_auc']:.3f}  "
          f"val_AP={metrics['val_ap']:.3f}  "
          f"test_AUC={metrics['test_auc']:.3f}  "
          f"test_AP={metrics['test_ap']:.3f}")


if __name__ == "__main__":
    main()
