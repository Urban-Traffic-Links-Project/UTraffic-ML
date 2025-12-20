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

@torch.no_grad()
def best_f1_over_thresholds_1d(probs_1d: torch.Tensor, y01_1d: torch.Tensor, thrs=None):
    """
    probs_1d: (K,) float in [0,1]
    y01_1d:   (K,) int64 {0,1}
    """
    if probs_1d.numel() == 0:
        return 0.5, 0.0, 0.0, 0.0

    if thrs is None:
        thrs = torch.linspace(0.05, 0.95, 19)

    best_thr, best_f1, best_prec, best_rec = 0.5, -1.0, 0.0, 0.0

    for thr in thrs:
        pred = (probs_1d >= thr).to(torch.int64)
        tp = ((pred == 1) & (y01_1d == 1)).sum().item()
        fp = ((pred == 1) & (y01_1d == 0)).sum().item()
        fn = ((pred == 0) & (y01_1d == 1)).sum().item()

        prec = tp / max(1, tp + fp)
        rec  = tp / max(1, tp + fn)
        f1   = (2 * prec * rec) / max(1e-9, prec + rec)

        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr.item())
            best_prec = float(prec)
            best_rec = float(rec)

    return best_thr, float(best_f1), float(best_prec), float(best_rec)

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

@torch.no_grad()
def best_f1_over_thresholds(
    logits: torch.Tensor,
    y: torch.Tensor,
    final_mask: torch.Tensor,
    thrs = None,
):
    """
    Scan thresholds and return (best_thr, best_f1, best_prec, best_rec).
    logits, y, final_mask: (B,N)
    """
    probs = torch.sigmoid(logits)
    m = final_mask.bool()

    if m.sum() == 0:
        return 0.5, 0.0, 0.0, 0.0

    probs_m = probs[m]
    yt = (y[m] >= 0.5).to(torch.int64)

    if thrs is None:
        thrs = torch.linspace(0.05, 0.95, 19, device=probs_m.device)

    best_thr = 0.5
    best_f1 = -1.0
    best_prec = 0.0
    best_rec = 0.0

    for thr in thrs:
        pred = (probs_m >= thr).to(torch.int64)
        tp = ((pred == 1) & (yt == 1)).sum().item()
        fp = ((pred == 1) & (yt == 0)).sum().item()
        fn = ((pred == 0) & (yt == 1)).sum().item()

        prec = tp / max(1, tp + fp)
        rec  = tp / max(1, tp + fn)
        f1   = (2 * prec * rec) / max(1e-9, prec + rec)

        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr.item())
            best_prec = prec
            best_rec = rec

    return best_thr, float(best_f1), float(best_prec), float(best_rec)

def run_one_epoch(model: STEncoderOnly, loader: DataLoader, optim: torch.optim.Optimizer,
                  tp: TrainParams, train: bool) -> Dict[str, float]:
    all_probs_m = []
    all_y_m = []
    all_mask = []
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
        attn_bias = batch["attn_bias"].to(device, non_blocking=True)

        t1 = time.time()
        todev_s = t1 - t0

        # ---- (3) forward
        logits = model(x=x, tod=tod, dow=dow, lap=lap, node_mask=node_mask,attn_bias=attn_bias)
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
            with torch.no_grad():
                probs = torch.sigmoid(logits)
                m = final_mask.bool()
                if m.any():
                    all_probs_m.append(probs[m].detach().cpu())  # (K,)
                    all_y_m.append((y[m] >= 0.5).detach().cpu())  # (K,) bool

            t3 = time.time()

        # ---- (5) metrics
        m = masked_metrics(logits, y, final_mask, thresh=tp.thresh)
        for k in met_sum:
            met_sum[k] += m[k]

        total_loss += float(loss.item())
        n_batches += 1

        # ---- (6) log định kỳ
        if it % 1 == 0:
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
    if not train:
        if len(all_probs_m) == 0:
            out["best_thr"] = 0.5
            out["best_f1"] = 0.0
            out["best_prec"] = 0.0
            out["best_rec"] = 0.0
        else:
            probs_all = torch.cat(all_probs_m, dim=0)  # (TotalK,)
            y_all = torch.cat(all_y_m, dim=0).to(torch.int64)  # (TotalK,)

            best_thr, best_f1, best_prec, best_rec = best_f1_over_thresholds_1d(probs_all, y_all)
            out["best_thr"] = best_thr
            out["best_f1"] = best_f1
            out["best_prec"] = best_prec
            out["best_rec"] = best_rec

    return out


def train_model(model: STEncoderOnly, train_loader: DataLoader, val_loader: DataLoader, tp: TrainParams):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
    log.info("Entered train_model()")

    device = tp.device
    model.to(device)
    log.info(f"Model moved to {device}")
    import threading
    import traceback

    def _fetch_first_batch(loader, out_dict):
        try:
            it = iter(loader)
            out_dict["batch"] = next(it)  # <-- chỗ hay kẹt
        except Exception as e:
            out_dict["exc"] = e
            out_dict["tb"] = traceback.format_exc()

    log.info("[DEBUG] Sanity: fetching FIRST train batch...")
    shared = {}
    th = threading.Thread(target=_fetch_first_batch, args=(train_loader, shared), daemon=True)
    t0 = time.time()
    th.start()

    # heartbeat mỗi 10s, báo đang chờ batch
    while th.is_alive():
        waited = time.time() - t0
        log.info(f"[DEBUG] Waiting for first batch... {waited:.1f}s")
        time.sleep(10.0)

    # thread kết thúc: hoặc lấy được batch, hoặc exception
    if "exc" in shared:
        log.error("[DEBUG] First batch FAILED with exception!")
        log.error(str(shared["exc"]))
        log.error(shared.get("tb", ""))
        raise shared["exc"]

    batch0 = shared["batch"]
    log.info(f"[DEBUG] First batch OK in {time.time() - t0:.1f}s")
    log.info(
        f"[DEBUG] batch0 shapes: x={tuple(batch0['x'].shape)} tod={tuple(batch0['tod'].shape)} "
        f"dow={tuple(batch0['dow'].shape)} lap={tuple(batch0['lap'].shape)} "
        f"y={tuple(batch0['y'].shape)} node_mask={tuple(batch0['node_mask'].shape)} "
        f"final_mask={tuple(batch0['final_mask'].shape)}"
    )

    optim = torch.optim.AdamW(model.parameters(), lr=tp.lr, weight_decay=tp.weight_decay)

    best_f1 = -1.0
    best_state = None

    for ep in range(tp.epochs):
        tr = run_one_epoch(model, train_loader, optim, tp, train=True)
        va = run_one_epoch(model, val_loader, optim, tp, train=False)

        print(
            f"[Epoch {ep + 1}/{tp.epochs}] "
            f"train loss={tr['loss']:.4f} prec={tr['prec']:.3f} rec={tr['rec']:.3f} f1={tr['f1']:.3f} | "
            f"val loss={va['loss']:.4f} prec={va['prec']:.3f} rec={va['rec']:.3f} f1={va['f1']:.3f} |"
            f"best_f1={va.get('best_f1', 0.0):.3f} thr={va.get('best_thr', 0.5):.2f}"
        )

        score = va.get("best_f1", va["f1"])
        if score > best_f1:
            best_f1 = score
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model
