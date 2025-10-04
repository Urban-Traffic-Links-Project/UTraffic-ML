import os, math, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score


# ---------------------------
# Data
# ---------------------------
def default_paths(base = Path.cwd() / "Urban-Traffic-Links" / "data" / "raw" / "kaggle"):
    base = Path(base)
    return dict(
        train=base.joinpath('train.csv'),
        nodes=base.joinpath('nodes.csv'),
        segments=base.joinpath('segments.csv'),
        streets=base.joinpath('streets.csv'),
        segment_status=base.joinpath('segment_status.csv'),
    )

def load_data(paths):
    train = pd.read_csv(paths['train'])
    nodes = pd.read_csv(paths['nodes'])
    segments = pd.read_csv(paths['segments'])
    streets = pd.read_csv(paths['streets'])
    segment_status = pd.read_csv(paths['segment_status'])
    return train, nodes, segments, streets, segment_status

# ---------------------------
# Handle Graph
# ---------------------------
#Build matrix for route
def build_segment_graph(train):
    sub = train.drop_duplicates(subset=["segment_id"])[["segment_id", "s_node_id", "e_node_id"]].copy()

    # Map id -> index
    seg_ids = sub['segment_id'].astype(int).tolist()
    id2idx = {sid: i for i,sid in enumerate(seg_ids)}

    #Take length of segment
    n = len(seg_ids)

    from collections import defaultdict
    virtual_link = defaultdict(list)
    for _,s in sub.iterrows():
        idx = id2idx[int(s['segment_id'])]
        virtual_link[int(s['s_node_id'])].append(idx)
        virtual_link[int(s['e_node_id'])].append(idx)
    A = np.zeros((n, n), dtype=np.float32)
    for seg_list in virtual_link.values():
        for i in range(len(seg_list)):
            for j in range(i+1, len(seg_list)):
                a,b = seg_list[i], seg_list[j]
                A[a,b] = 1.0
                A[b,a] = 1.0
    # Add into main diagonal
    np.fill_diagonal(A, 1.0)
    return A, seg_ids, id2idx


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

# def roc_auc(y_true, y_score):
#     y_true = np.asarray(y_true).astype(int)
#     #yscore -> output from softmax
#     y_score = np.asarray(y_score).astype(float)
#     pos = y_score[y_true == 1]
#     neg = y_score[y_true == 0]
#     if len(pos) == 0 or len(neg) == 0:
#         return float('nan')
#     #Get rank
#     all_scores = np.concatenate([pos, neg])
#     ranks = all_scores.argsort().argsort() + 1
#     r_pos = ranks[:len(pos)]
#
#     u = r_pos.sum() - len(pos) * (len(pos) + 1) / 2
#     auc = u / (len(pos) * len(neg))
#     return float(auc)


# def roc_auc(y_true, y_score):
#     y_true  = np.asarray(y_true).astype(int)
#     y_score = np.asarray(y_score).astype(float)
#
#     pos = y_score[y_true == 1]
#     neg = y_score[y_true == 0]
#     if len(pos) == 0 or len(neg) == 0:
#         return float("nan")
#
#     all_scores = np.concatenate([pos, neg])              # [pos..., neg...]
#     order = np.argsort(all_scores)                       # chỉ số tăng dần theo score
#     ranks = np.empty_like(order, dtype=float)
#     ranks[order] = np.arange(1, len(all_scores) + 1)     # rank 1..N
#
#     # (tuỳ chọn) xử lý tie bằng average rank
#     # tìm các nhóm tie
#     s_sorted = all_scores[order]
#     i = 0
#     while i < len(s_sorted):
#         j = i + 1
#         while j < len(s_sorted) and s_sorted[j] == s_sorted[i]:
#             j += 1
#         if j - i > 1:
#             avg = (i + 1 + j) / 2.0
#             ranks[order[i:j]] = avg
#         i = j
#
#     r_pos = ranks[:len(pos)]                             # vì ta ghép pos trước
#     U = r_pos.sum() - len(pos) * (len(pos) + 1) / 2.0
#     auc = U / (len(pos) * len(neg))
#     return float(auc)

# def average_precision_numpy(y_true, y_score):
#     y_true = np.asarray(y_true).astype(int)
#     y_score= np.asarray(y_score).astype(float)
#
#     #Reorder y_true follow y_score
#     order = np.argsort(-y_score)
#     y_true = y_true[order]
#
#     #Đếm TP, FP
#     tp = np.cumsum(y_true)
#     fp = np.cumsum(1 - y_true)
#
#     #Calculate
#     precision = tp / np.maximum(tp + fp, 1)
#     recall = tp / np.maximum(tp[-1], 1)
#     # integrate precision over recall steps
#     ap = 0.0
#     prev_r = 0.0
#     for p, r in zip(precision, recall):
#         ap += p * (r - prev_r)
#         prev_r = r
#     return float(ap)

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



# def train_linkpred(
#     Ahat, X, pos_tr, pos_val, pos_te,
#     hidden=64, emb_dim=64, epochs=400, lr=1e-3, wd=5e-4, seed=0,
#     device=None, patience=30, eval_every=5, resample_neg_each_epoch=True
# ):
#     """
#     Huấn luyện với:
#       - Đánh giá validation định kỳ (eval_every epoch)
#       - Early stopping theo val_AUC (patience)
#       - Chọn model có val_AUC tốt nhất để final test
#     """
#     if device is None:
#         device = torch.device("cpu")
#     torch.manual_seed(seed); np.random.seed(seed)
#
#     n = X.shape[0]
#     # forbidden gồm toàn bộ cạnh dương để tránh lấy nhầm negative
#     forb_all = set(edge_key(u, v) for u, v in np.vstack([pos_tr, pos_val, pos_te]))
#
#     # Negative cố định cho val/test (để đánh giá nhất quán)
#     neg_val = negative_sampling(n, len(pos_val), set(forb_all), seed+1)
#     neg_te  = negative_sampling(n, len(pos_te),  set(forb_all), seed+2)
#
#     # Negatives cho train: có thể cố định hoặc resample mỗi epoch
#     def sample_train_negs():
#         return negative_sampling(n, len(pos_tr), set(forb_all), seed=np.random.randint(0, 10**9))
#
#     neg_tr = sample_train_negs()  # khởi tạo lần đầu
#
#     # Tensors
#     X_t = torch.from_numpy(X).float().to(device)
#     A_t = torch.from_numpy(Ahat).float().to(device)
#     pos_tr_t = torch.from_numpy(pos_tr).long().to(device)
#
#     # Model
#     encoder = GCN(in_dim=X.shape[1], hidden_dim=min(hidden, X.shape[1]), out_dim=emb_dim).to(device)
#     decoder = DotDecoder()
#     opt = torch.optim.Adam(encoder.parameters(), lr=lr, weight_decay=wd)
#     bce = nn.BCEWithLogitsLoss()
#
#     best_val_auc = -1.0
#     best_state = None
#     best_epoch = -1
#     bad_rounds = 0  # đếm số lần eval liên tiếp không cải thiện
#
#     for epoch in range(1, epochs + 1):
#         encoder.train()
#         opt.zero_grad()
#
#         # (tuỳ chọn) resample negatives cho train mỗi epoch
#         if resample_neg_each_epoch or epoch == 1:
#             neg_tr = sample_train_negs()
#         neg_tr_t = torch.from_numpy(neg_tr).long().to(device)
#
#         z = encoder(X_t, A_t)
#         logits_pos = decoder(z, pos_tr_t)
#         logits_neg = decoder(z, neg_tr_t)
#         logits = torch.cat([logits_pos, logits_neg], dim=0)
#         labels = torch.cat(
#             [torch.ones_like(logits_pos), torch.zeros_like(logits_neg)], dim=0
#         ).float()
#
#         loss = bce(logits, labels)
#         loss.backward()
#         opt.step()
#
#         # Đánh giá định kỳ trên validation
#         if epoch % eval_every == 0 or epoch == epochs:
#             val_auc, val_ap = eval_linkpred(encoder, decoder, A_t, X_t, pos_val, neg_val)
#
#             # Theo dõi model tốt nhất theo val_AUC
#             if val_auc > best_val_auc:
#                 best_val_auc = val_auc
#                 best_state = {k: v.detach().cpu().clone() for k, v in encoder.state_dict().items()}
#                 best_epoch = epoch
#                 bad_rounds = 0
#             else:
#                 bad_rounds += 1
#
#             # (tuỳ chọn) in log gọn nhẹ
#             # print(f"[Epoch {epoch:4d}] loss={loss.item():.4f} | valAUC={val_auc:.3f} valAP={val_ap:.3f} | bestAUC={best_val_auc:.3f}@{best_epoch}")
#
#             # Early stopping nếu không cải thiện sau 'patience' lần đánh giá
#             if bad_rounds >= patience:
#                 # print(f"[EarlyStop] No improvement in {patience} eval rounds. Stop at epoch {epoch}.")
#                 break
#
#     # Khôi phục tham số tốt nhất theo validation
#     if best_state is not None:
#         encoder.load_state_dict(best_state)
#     else:
#         best_epoch = epochs  # phòng khi không có lần eval nào tốt hơn
#
#     # Đánh giá cuối: dùng model tốt nhất theo validation
#     val_auc, val_ap = eval_linkpred(encoder, decoder, A_t, X_t, pos_val, neg_val)
#     te_auc,  te_ap  = eval_linkpred(encoder, decoder, A_t, X_t, pos_te,  neg_te)
#
#     metrics = {
#         "best_epoch": best_epoch,
#         "val_auc": float(val_auc), "val_ap": float(val_ap),
#         "test_auc": float(te_auc), "test_ap": float(te_ap),
#     }
#     return encoder, decoder, metrics


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
    ap.add_argument("--outdir", default=str(Path.cwd() /"Urban-Traffic-Links" / "outputs"))
    ap.add_argument("--snapshot_date", default="2020-08-02", help="yyy-mm-dd; nếu None sẽ dùng ngày sớm nhất trong train")
    ap.add_argument("--snapshot_period", default="period_23_30", help="ví dụ: 'AM_peak' (nếu cột period tồn tại)")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--emb_dim", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    # optional hparams
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--wd", type=float, default=5e-4)

    args = ap.parse_args()

    # Paths & outdir
    paths = default_paths()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("[INFO] Loading data ...")
    train, nodes, segments, streets, segment_status = load_data(paths)

    # Ensure datetime for 'date' if cột tồn tại
    if "date" in train.columns:
        train["date"] = pd.to_datetime(train["date"], errors="coerce")

    # Resolve snapshot_date if None -> earliest date in train
    if args.snapshot_date is None and "date" in train.columns:
        earliest = train["date"].dropna().min()
        if pd.isna(earliest):
            raise ValueError("Không tìm được 'snapshot_date' vì cột 'date' trống.")
        args.snapshot_date = earliest.date().isoformat()

    # Build full graph from topology
    print("[INFO] Building graph ...")
    A_full, seg_ids, id2idx = build_segment_graph(train)

    # Split positive edges
    pos_edges = edges_from_adj(A_full)  # full topo edges (no self-loops)
    pos_tr, pos_val, pos_te = split_edges(
        pos_edges, val_ratio=0.1, test_ratio=0.1, seed=args.seed
    )

    # Build train adjacency only with train edges (avoid leakage)
    A_train = np.zeros_like(A_full, dtype=np.float32)
    for u, v in pos_tr:
        A_train[u, v] = 1.0
        A_train[v, u] = 1.0
    np.fill_diagonal(A_train, 1.0)
    Ahat = normalize_adj(A_train)

    # Features snapshot
    print("[INFO] Preparing node features (snapshot) ...")
    X, meta = prepare_feature_labels(
        train, seg_ids,
        snapshot_date=args.snapshot_date,
        snapshot_period=args.snapshot_period,
        task="classification"
    )
    print(f"[INFO] Snapshot -> date={meta['snapshot_date']} "
          f"period={meta['snapshot_period']} | X shape={X.shape}")

    # Train link prediction
    print("[INFO] Training link prediction (GCN encoder + dot decoder) ...")
    _, _, metrics = train_linkpred(
        Ahat, X, pos_tr, pos_val, pos_te,
        hidden=args.hidden, emb_dim=args.emb_dim, epochs=args.epochs,
        lr=args.lr, wd=args.wd, seed=args.seed
    )
    print(f"[RESULT] LinkPred  val_AUC={metrics['val_auc']:.3f}  "
          f"val_AP={metrics['val_ap']:.3f}  "
          f"test_AUC={metrics['test_auc']:.3f}  "
          f"test_AP={metrics['test_ap']:.3f}")


if __name__ == "__main__":
    main()