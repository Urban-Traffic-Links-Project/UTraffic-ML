# src/train_gat_gru.py
from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config.default import TrainConfig  # :contentReference[oaicite:3]{index=3}
from src.data_loader.dataset import TrafficWindowDataset
from src.data_loader.graph import load_edge_index
from src.data_loader.tomtom_io import load_npz, load_zscore_stats
from src.utils.seed import set_seed
from src.utils.metrics import mae, rmse, mape
from src.models.GAT_GRU.model import GATGRU


def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


def inverse_z(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    # y: [..., N]
    shape = (1,) * (y.ndim - 1) + (mu.shape[0],)
    return y * sigma.reshape(shape) + mu.reshape(shape)



def build_adj_list(edge_index: torch.Tensor, num_nodes: int) -> List[List[int]]:
    # edge_index: [2,E] src->dst
    src = edge_index[0].cpu().numpy()
    dst = edge_index[1].cpu().numpy()
    adj = [[] for _ in range(num_nodes)]
    for s, d in zip(src, dst):
        if d != s:
            adj[s].append(d)
    # fallback: nếu node cô lập, tự nối với chính nó
    for i in range(num_nodes):
        if len(adj[i]) == 0:
            adj[i] = [i]
    return adj


def sample_random_walk_routes(
    adj: List[List[int]],
    batch_size: int,
    routes_per_batch: int,
    len_min: int,
    len_max: int,
    restart_prob: float,
    rng: np.random.Generator,
) -> torch.Tensor:
    """
    return routes: [B, R, Lmax] padded -1
    route is sequence of node indices (segments)
    """
    N = len(adj)
    Lmax = len_max
    routes = np.full((batch_size, routes_per_batch, Lmax), -1, dtype=np.int64)

    for b in range(batch_size):
        for r in range(routes_per_batch):
            L = int(rng.integers(len_min, len_max + 1))
            # start anywhere
            cur = int(rng.integers(0, N))
            visited = {cur}
            routes[b, r, 0] = cur
            for t in range(1, L):
                if rng.random() < restart_prob:
                    cur = int(rng.integers(0, N))
                    visited = {cur}
                    routes[b, r, t] = cur
                    continue
                neigh = adj[cur]
                nxt = int(neigh[int(rng.integers(0, len(neigh)))])
                # tránh vòng lặp quá chặt (nhưng không bắt buộc)
                if nxt in visited and len(neigh) > 1:
                    # thử vài lần
                    for _ in range(3):
                        cand = int(neigh[int(rng.integers(0, len(neigh)))])
                        if cand not in visited:
                            nxt = cand
                            break
                visited.add(nxt)
                routes[b, r, t] = nxt
                cur = nxt

    return torch.from_numpy(routes)


@torch.no_grad()
def evaluate(
    model: GATGRU,
    loader: DataLoader,
    device: str,
    mu: np.ndarray,
    sigma: np.ndarray,
    cfg, adj, rng
) -> Dict[str, float]:
    model.eval()
    maes, rmses, mapes = [], [], []

    for x, y in loader:
        x = x.to(device)  # [B,P,N,1]
        y = y.to(device)  # [B,H,N]

        out = model(x, routes=None, return_attn=False)
        yhat = out.y_hat  # [B,H,N]

        # inverse zscore để metric theo km/h thực (hoặc đơn vị speed gốc)
        y_np = y.detach().cpu().numpy()
        yhat_np = yhat.detach().cpu().numpy()

        y_inv = inverse_z(y_np, mu, sigma)
        yhat_inv = inverse_z(yhat_np, mu, sigma)

        maes.append(mae(yhat_inv, y_inv))
        rmses.append(rmse(yhat_inv, y_inv))
        mapes.append(mape(yhat_inv, y_inv))

    return {
        "MAE": float(np.mean(maes)),
        "RMSE": float(np.mean(rmses)),
        "MAPE": float(np.mean(mapes)),
    }


def main():
    cfg = TrainConfig()
    set_seed(cfg.seed)

    device = cfg.device if torch.cuda.is_available() and cfg.device == "cuda" else "cpu"

    data_root = Path(cfg.dataset_root)
    train_dir = data_root / cfg.train_split
    val_dir = data_root / cfg.val_split
    test_dir = data_root / cfg.test_split

    # load tensors (npz)
    train_npz = train_dir / "traffic_tensor.npz"
    val_npz = val_dir / "traffic_tensor.npz"
    test_npz = test_dir / "traffic_tensor.npz"

    train_pack = load_npz(train_npz)
    val_pack = load_npz(val_npz)
    test_pack = load_npz(test_npz)

    Xtr = train_pack["values"]  # shape [T,N]
    Xva = val_pack["values"]
    Xte = test_pack["values"]
    # ensure [T,N,1]
    if Xtr.ndim == 2:
        Xtr = Xtr[..., None]
        Xva = Xva[..., None]
        Xte = Xte[..., None]

    T, N, C = Xtr.shape

    # static graph from edges.csv
    edge_csv = data_root / "edges.csv"
    seg_index_csv = data_root / "segment_index.csv"
    edge_index = load_edge_index(edge_csv, segment_index_csv=seg_index_csv)  # [2,E] torch.Long
    edge_index = edge_index.to(torch.long)

    # z-score stats for inverse metrics
    stats_path = data_root / "zscore_stats_speed.pkl"
    stats = load_zscore_stats(stats_path)
    mu = stats["mu"].values.astype(np.float32)      # [N]
    sigma = stats["sigma"].values.astype(np.float32)# [N]

    # datasets
    ds_tr = TrafficWindowDataset(Xtr, P=cfg.P, H=cfg.H)
    ds_va = TrafficWindowDataset(Xva, P=cfg.P, H=cfg.H)
    ds_te = TrafficWindowDataset(Xte, P=cfg.P, H=cfg.H)

    dl_tr = DataLoader(ds_tr, batch_size=cfg.batch, shuffle=True, drop_last=True)
    dl_va = DataLoader(ds_va, batch_size=cfg.batch, shuffle=False)
    dl_te = DataLoader(ds_te, batch_size=cfg.batch, shuffle=False)

    # model
    model = GATGRU(
        num_nodes=N,
        edge_index=edge_index.to(device),
        in_dim=cfg.in_dim,
        gat_hidden=cfg.gat_hidden,
        gat_heads=cfg.gat_heads,
        gat_layers=cfg.gat_layers,
        gru_hidden=cfg.gru_hidden,
        horizon=cfg.H,
        dropout=cfg.dropout,
        route_pool_hidden=cfg.route_pool_hidden,
        routes_per_batch=cfg.routes_per_batch,
    ).to(device)

    optim = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    save_dir = Path(ensure_dir(cfg.save_dir))
    best_path = save_dir / cfg.save_best_name

    # random walk adjacency list (static)
    adj = build_adj_list(edge_index, N)
    rng = np.random.default_rng(cfg.seed + 123)

    best_val_mae = float("inf")

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        losses = []
        last_attn_dump = None  # store last batch attention for exporting

        for x, y in dl_tr:
            x = x.to(device)  # [B,P,N,1]
            y = y.to(device)  # [B,H,N]

            routes = None
            if cfg.use_routes_in_train:
                routes = sample_random_walk_routes(
                    adj=adj,
                    batch_size=x.size(0),
                    routes_per_batch=cfg.routes_per_batch,
                    len_min=cfg.route_len_min,
                    len_max=cfg.route_len_max,
                    restart_prob=cfg.route_restart_prob,
                    rng=rng,
                ).to(device)

            out = model(x, routes=routes, return_attn=False)
            yhat = out.y_hat

            # main loss: MAE on z-score space (ổn định train)
            loss = torch.mean(torch.abs(yhat - y))

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optim.step()

            losses.append(loss.item())

            # keep attention from last batch to export (nhẹ)
            if cfg.export_attn_every_epoch and out.gat_attn is not None:
                last_attn_dump = {
                    "gat_attn_layers": [a.detach().cpu().numpy() for a in out.gat_attn],  # each [B,E,H]
                    "route_pool_attn": None if out.route_pool_attn is None else out.route_pool_attn.detach().cpu().numpy()
                }

        # eval val
        val_metrics = evaluate(model, dl_va, device, mu, sigma, cfg, adj, rng)

        train_loss = float(np.mean(losses))

        print(f"[Epoch {epoch:03d}] loss={train_loss:.4f} | val MAE={val_metrics['MAE']:.4f} RMSE={val_metrics['RMSE']:.4f} MAPE={val_metrics['MAPE']:.2f}")

        # save best
        if val_metrics["MAE"] < best_val_mae:
            best_val_mae = val_metrics["MAE"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optim_state": optim.state_dict(),
                    "best_val_mae": best_val_mae,
                    "cfg": cfg.__dict__,
                },
                best_path
            )
            print(f"  ✅ Saved BEST checkpoint -> {best_path} (val MAE={best_val_mae:.4f})")

            # export attention weights for the best epoch (recommended)
            if last_attn_dump is not None:
                attn_dir = save_dir / "attn"
                attn_dir.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    attn_dir / f"best_epoch_{epoch:03d}_attn.npz",
                    **{f"gat_layer_{i}": last_attn_dump["gat_attn_layers"][i] for i in range(len(last_attn_dump["gat_attn_layers"]))},
                    route_pool_attn=last_attn_dump["route_pool_attn"] if last_attn_dump["route_pool_attn"] is not None else np.array([]),
                )
                print(f"  ✅ Exported attention -> {attn_dir / f'best_epoch_{epoch:03d}_attn.npz'}")

    # load best and test
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    test_metrics = evaluate(model, dl_te, device, mu, sigma)
    print(f"[TEST] MAE={test_metrics['MAE']:.4f} RMSE={test_metrics['RMSE']:.4f} MAPE={test_metrics['MAPE']:.2f}")
    print(f"[DONE] Best checkpoint: {best_path}")


if __name__ == "__main__":
    main()
