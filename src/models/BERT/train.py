# src/train.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .model_sttransformer import STEncoderOnly, ModelParams
import time
import logging
log = logging.getLogger(__name__)


@dataclass
class TrainParams:
    lr: float = 1e-4
    weight_decay: float = 1e-4
    epochs: int = 10
    grad_clip: float = 1.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    thresh: float = 0.5  # for metrics


def masked_bce_loss(logits: torch.Tensor, y: torch.Tensor, final_mask: torch.Tensor) -> torch.Tensor:
    """
    logits: (B,N)
    y: (B,N)
    final_mask: (B,N) int {0,1}
    """
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    loss = loss_fn(logits, y)
    m = final_mask.to(loss.dtype)
    loss = loss * m
    denom = m.sum().clamp_min(1.0)
    return loss.sum() / denom


@torch.no_grad()
def masked_metrics(logits: torch.Tensor, y: torch.Tensor, final_mask: torch.Tensor, thresh: float = 0.5) -> Dict[str, float]:
    """
    Compute acc/prec/rec/f1 on masked positions only.
    """
    probs = torch.sigmoid(logits)
    pred = (probs >= thresh).to(torch.int64)
    yt = (y >= 0.5).to(torch.int64)
    m = final_mask.to(torch.bool)

    if m.sum() == 0:
        return {"acc": 0.0, "prec": 0.0, "rec": 0.0, "f1": 0.0}

    pred_m = pred[m]
    yt_m = yt[m]

    tp = ((pred_m == 1) & (yt_m == 1)).sum().item()
    tn = ((pred_m == 0) & (yt_m == 0)).sum().item()
    fp = ((pred_m == 1) & (yt_m == 0)).sum().item()
    fn = ((pred_m == 0) & (yt_m == 1)).sum().item()

    acc = (tp + tn) / max(1, (tp + tn + fp + fn))
    prec = tp / max(1, (tp + fp))
    rec = tp / max(1, (tp + fn))
    f1 = (2 * prec * rec) / max(1e-9, (prec + rec))
    return {"acc": acc, "prec": prec, "rec": rec, "f1": f1}


def run_one_epoch(model: STEncoderOnly, loader: DataLoader, optim: torch.optim.Optimizer,
                  tp: TrainParams, train: bool) -> Dict[str, float]:
    device = tp.device
    model.train(train)

    total_loss = 0.0
    n_batches = 0
    met_sum = {"acc": 0.0, "prec": 0.0, "rec": 0.0, "f1": 0.0}

    t_epoch0 = time.time()
    t_prev = time.time()

    for it, batch in enumerate(loader):
        # ---- (1) đo thời gian "load batch" (từ lần trước tới lúc batch xuất hiện)
        load_s = time.time() - t_prev

        if it == 0:
            log.info(f"[{'train' if train else 'val'}] First batch arrived | load_time={load_s:.2f}s")
            # in nhanh shape để chắc chắn data đúng
            log.info(
                f"Shapes: x={tuple(batch['x'].shape)} tod={tuple(batch['tod'].shape)} "
                f"dow={tuple(batch['dow'].shape)} lap={tuple(batch['lap'].shape)} "
                f"y={tuple(batch['y'].shape)} node_mask={tuple(batch['node_mask'].shape)} "
                f"final_mask={tuple(batch['final_mask'].shape)}"
            )

        t0 = time.time()

        # ---- (2) copy to device (đo riêng)
        x = batch["x"].to(device, non_blocking=True)                 # (B,L,N,d)
        tod = batch["tod"].to(device, non_blocking=True)
        dow = batch["dow"].to(device, non_blocking=True)
        lap = batch["lap"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        node_mask = batch["node_mask"].to(device, non_blocking=True)
        final_mask = batch["final_mask"].to(device, non_blocking=True)

        t1 = time.time()
        todev_s = t1 - t0

        # ---- (3) forward
        logits = model(x=x, tod=tod, dow=dow, lap=lap, node_mask=node_mask)
        loss = masked_bce_loss(logits, y, final_mask)
        t2 = time.time()
        fwd_s = t2 - t1

        # ---- (4) backward/step
        bwd_s = 0.0
        if train:
            optim.zero_grad(set_to_none=True)
            loss.backward()
            if tp.grad_clip is not None and tp.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), tp.grad_clip)
            optim.step()
            t3 = time.time()
            bwd_s = t3 - t2
        else:
            t3 = time.time()

        # ---- (5) metrics
        m = masked_metrics(logits, y, final_mask, thresh=tp.thresh)
        for k in met_sum:
            met_sum[k] += m[k]

        total_loss += float(loss.item())
        n_batches += 1

        # ---- (6) log định kỳ
        if it % 10 == 0:
            log.info(
                f"[{'train' if train else 'val'}] it={it:04d} "
                f"load={load_s:.2f}s todev={todev_s:.2f}s fwd={fwd_s:.2f}s bwd={bwd_s:.2f}s "
                f"loss={loss.item():.4f} mask_sum={final_mask.sum().item()}"
            )

        t_prev = time.time()

    if n_batches == 0:
        return {"loss": 0.0, **met_sum}

    out = {"loss": total_loss / n_batches}
    for k in met_sum:
        out[k] = met_sum[k] / n_batches

    log.info(f"[{'train' if train else 'val'}] epoch_done in {time.time() - t_epoch0:.1f}s")
    return out


def train_model(model: STEncoderOnly, train_loader: DataLoader, val_loader: DataLoader, tp: TrainParams):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
    log.info("Entered train_model()")

    device = tp.device
    model.to(device)
    log.info(f"Model moved to {device}")
    optim = torch.optim.AdamW(model.parameters(), lr=tp.lr, weight_decay=tp.weight_decay)

    best_f1 = -1.0
    best_state = None

    for ep in range(tp.epochs):
        tr = run_one_epoch(model, train_loader, optim, tp, train=True)
        va = run_one_epoch(model, val_loader, optim, tp, train=False)

        print(f"[Epoch {ep+1}/{tp.epochs}] "
              f"train loss={tr['loss']:.4f} f1={tr['f1']:.3f} | "
              f"val loss={va['loss']:.4f} f1={va['f1']:.3f}")

        if va["f1"] > best_f1:
            best_f1 = va["f1"]
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model
