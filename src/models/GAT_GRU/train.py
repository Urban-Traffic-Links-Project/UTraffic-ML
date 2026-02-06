# File: src/models/GAT_GRU/train.py
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config.defaults import TrainConfig  # <-- đúng tên file defaults.py
from src.data.dataset import TrafficWindowDataset
from src.data.graph import load_edge_index
from src.data.tomtom_io import (
    load_npz, get_values_3d,
    load_zscore_stats, load_segment_index
)
from src.utils.seed import set_seed
from src.utils.metrics import mae, rmse, mape
from src.models.GAT_GRU.model import GATGRU


def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def inverse_z_safe(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """
    Fix broadcasting for y shape [B,H,N] or [...,N]
    mu,sigma: [N]
    """
    shape = (1,) * (y.ndim - 1) + (mu.shape[0],)
    return y * sigma.reshape(shape) + mu.reshape(shape)


def build_adj_list(edge_index: torch.Tensor, num_nodes: int) -> List[List[int]]:
    """
    edge_index: [2,E] src->dst
    adjacency for random-walk sampling: out-neighbors of each node
    """
    src = edge_index[0].cpu().numpy()
    dst = edge_index[1].cpu().numpy()
    adj = [[] for _ in range(num_nodes)]
    for s, d in zip(src, dst):
        if d != s:
            adj[int(s)].append(int(d))
    # fallback: isolated node -> self
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
    Return routes: [B, R, Lmax] padded -1
    """
    N = len(adj)
    Lmax = len_max
    routes = np.full((batch_size, routes_per_batch, Lmax), -1, dtype=np.int64)

    for b in range(batch_size):
        for r in range(routes_per_batch):
            L = int(rng.integers(len_min, len_max + 1))
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

                # avoid tight loops (optional)
                if nxt in visited and len(neigh) > 1:
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
def evaluate(model: GATGRU, loader: DataLoader, device: str, mu: np.ndarray, sigma: np.ndarray) -> Dict[str, float]:
    model.eval()
    maes, rmses, mapes = [], [], []

    for x, y in loader:
        x = x.to(device)  # [B,P,N,1]
        y = y.to(device)  # [B,H,N]

        out = model(x, routes=None, return_attn=False)
        yhat = out.y_hat  # [B,H,N]

        y_np = y.detach().cpu().numpy()
        yhat_np = yhat.detach().cpu().numpy()

        y_inv = inverse_z_safe(y_np, mu, sigma)
        yhat_inv = inverse_z_safe(yhat_np, mu, sigma)

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

    device = cfg.device if (cfg.device == "cuda" and torch.cuda.is_available()) else "cpu"
    print(f"[INFO] device={device}")

    data_root = Path(cfg.dataset_root)
    train_dir = data_root / cfg.train_split
    val_dir = data_root / cfg.val_split
    test_dir = data_root / cfg.test_split

    # ----------------------------
    # Load npz -> dict -> values
    # ----------------------------
    train_pack = load_npz(str(train_dir / "traffic_tensor.npz"))
    val_pack   = load_npz(str(val_dir / "traffic_tensor.npz"))
    test_pack  = load_npz(str(test_dir / "traffic_tensor.npz"))

    Xtr = get_values_3d(train_pack)  # [T,N,1] (auto add channel if needed)
    Xva = get_values_3d(val_pack)
    Xte = get_values_3d(test_pack)

    T, N, C = Xtr.shape
    print(f"[INFO] Xtr shape={Xtr.shape} (T,N,C)")

    # ----------------------------
    # Graph: edges.csv + segment_index.csv
    # load_edge_index requires num_nodes & segid_to_idx
    # ----------------------------
    edges_csv = data_root / "edges.csv"
    seg_index_csv = data_root / "segment_index.csv"
    segid_to_idx, _ = load_segment_index(str(seg_index_csv))

    edge_index = load_edge_index(
        edges_csv=str(edges_csv),
        num_nodes=N,
        segid_to_idx=segid_to_idx,
        add_self_loops=True,
    ).to(torch.long)

    # ----------------------------
    # Z-score stats (for inverse metrics)
    # stats saved as {"mu": Series, "sigma": Series}
    # ----------------------------
    stats = load_zscore_stats(str(data_root / "zscore_stats_speed.pkl"))
    mu = stats["mu"].values.astype(np.float32)       # [N]
    sigma = stats["sigma"].values.astype(np.float32) # [N]

    # ----------------------------
    # Datasets / Loaders
    # dataset returns:
    #   x: [P,N,F], y: [H,N]
    # so DataLoader gives:
    #   x: [B,P,N,F], y: [B,H,N]
    # ----------------------------
    ds_tr = TrafficWindowDataset(Xtr, P=cfg.P, H=cfg.H)
    ds_va = TrafficWindowDataset(Xva, P=cfg.P, H=cfg.H)
    ds_te = TrafficWindowDataset(Xte, P=cfg.P, H=cfg.H)

    dl_tr = DataLoader(ds_tr, batch_size=cfg.batch, shuffle=True, drop_last=True)
    dl_va = DataLoader(ds_va, batch_size=cfg.batch, shuffle=False)
    dl_te = DataLoader(ds_te, batch_size=cfg.batch, shuffle=False)

    # ----------------------------
    # Model
    # ----------------------------
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

    save_dir = ensure_dir(cfg.save_dir)
    best_path = save_dir / cfg.save_best_name

    # ----------------------------
    # Random-walk routes (static)
    # ----------------------------
    adj = build_adj_list(edge_index, N)
    rng = np.random.default_rng(cfg.seed + 123)

    best_val_mae = float("inf")

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        losses = []

        # dump attention from LAST batch of epoch (optional)
        last_attn_dump: Optional[Dict[str, np.ndarray]] = None

        for it, (x, y) in enumerate(dl_tr, start=1):
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

            need_attn = bool(cfg.export_attn_every_epoch) and (it == len(dl_tr))
            out = model(x, routes=routes, return_attn=need_attn)
            yhat = out.y_hat  # [B,H,N]

            loss = torch.mean(torch.abs(yhat - y))  # MAE in z-space
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optim.step()
            losses.append(float(loss.item()))

            if need_attn:
                last_attn_dump = {}
                if out.gat_attn is not None:
                    for li, a in enumerate(out.gat_attn):
                        last_attn_dump[f"gat_layer_{li}"] = a.detach().cpu().numpy()  # [B,E,heads]
                if out.route_pool_attn is not None:
                    last_attn_dump["route_pool_attn"] = out.route_pool_attn.detach().cpu().numpy()  # [B,R,L]

        train_loss = float(np.mean(losses))
        val_metrics = evaluate(model, dl_va, device, mu, sigma)

        print(
            f"[Epoch {epoch:03d}] "
            f"train_loss={train_loss:.4f} | "
            f"val MAE={val_metrics['MAE']:.4f} RMSE={val_metrics['RMSE']:.4f} MAPE={val_metrics['MAPE']:.2f}"
        )

        # ----------------------------
        # Save BEST checkpoint
        # ----------------------------
        if val_metrics["MAE"] < best_val_mae:
            best_val_mae = val_metrics["MAE"]
            torch.save(
                {
                    "epoch": epoch,
                    "best_val_mae": best_val_mae,
                    "model_state": model.state_dict(),
                    "optim_state": optim.state_dict(),
                    "cfg": cfg.__dict__,
                },
                best_path,
            )
            print(f"  ✅ Saved BEST -> {best_path} (val MAE={best_val_mae:.4f})")

        # ----------------------------
        # Export attention per epoch (optional)
        # ----------------------------
        if cfg.export_attn_every_epoch and last_attn_dump is not None:
            attn_dir = ensure_dir(save_dir / "attn")
            out_path = attn_dir / f"epoch_{epoch:03d}_attn.npz"
            np.savez_compressed(out_path, **last_attn_dump)
            print(f"  ✅ Exported attn -> {out_path}")

    # ----------------------------
    # TEST with best
    # ----------------------------
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    test_metrics = evaluate(model, dl_te, device, mu, sigma)
    print(f"[TEST] MAE={test_metrics['MAE']:.4f} RMSE={test_metrics['RMSE']:.4f} MAPE={test_metrics['MAPE']:.2f}")
    print(f"[DONE] Best checkpoint: {best_path}")


if __name__ == "__main__":
    main()
