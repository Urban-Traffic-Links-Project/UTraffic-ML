# # src/train.py
# from __future__ import annotations
# from dataclasses import dataclass
# from typing import Dict, Any
# import numpy as np
# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader
#
# from .model_sttransformer import STEncoderOnly, ModelParams
# import time
# import logging
# log = logging.getLogger(__name__)
#
#
# @dataclass
# class TrainParams:
#     lr: float = 1e-4
#     weight_decay: float = 1e-4
#     epochs: int = 10
#     grad_clip: float = 1.0
#     device: str = "cuda" if torch.cuda.is_available() else "cpu"
#     thresh: float = 0.5  # for metrics
#
# @torch.no_grad()
# def best_f1_over_thresholds_1d(probs_1d: torch.Tensor, y01_1d: torch.Tensor, thrs=None):
#     """
#     probs_1d: (K,) float in [0,1]
#     y01_1d:   (K,) int64 {0,1}
#     """
#     if probs_1d.numel() == 0:
#         return 0.5, 0.0, 0.0, 0.0
#
#     if thrs is None:
#         thrs = torch.linspace(0.05, 0.95, 19)
#
#     best_thr, best_f1, best_prec, best_rec = 0.5, -1.0, 0.0, 0.0
#
#     for thr in thrs:
#         pred = (probs_1d >= thr).to(torch.int64)
#         tp = ((pred == 1) & (y01_1d == 1)).sum().item()
#         fp = ((pred == 1) & (y01_1d == 0)).sum().item()
#         fn = ((pred == 0) & (y01_1d == 1)).sum().item()
#
#         prec = tp / max(1, tp + fp)
#         rec  = tp / max(1, tp + fn)
#         f1   = (2 * prec * rec) / max(1e-9, prec + rec)
#
#         if f1 > best_f1:
#             best_f1 = f1
#             best_thr = float(thr.item())
#             best_prec = float(prec)
#             best_rec = float(rec)
#
#     return best_thr, float(best_f1), float(best_prec), float(best_rec)
#
# def masked_bce_loss(logits: torch.Tensor, y: torch.Tensor, final_mask: torch.Tensor) -> torch.Tensor:
#     """
#     logits: (B,N)
#     y: (B,N)
#     final_mask: (B,N) int {0,1}
#     """
#     loss_fn = nn.BCEWithLogitsLoss(reduction="none")
#     loss = loss_fn(logits, y)
#     m = final_mask.to(loss.dtype)
#     loss = loss * m
#     denom = m.sum().clamp_min(1.0)
#     return loss.sum() / denom
#
#
# @torch.no_grad()
# def masked_metrics(logits: torch.Tensor, y: torch.Tensor, final_mask: torch.Tensor, thresh: float = 0.5) -> Dict[str, float]:
#     """
#     Compute acc/prec/rec/f1 on masked positions only.
#     """
#     probs = torch.sigmoid(logits)
#     pred = (probs >= thresh).to(torch.int64)
#     yt = (y >= 0.5).to(torch.int64)
#     m = final_mask.to(torch.bool)
#
#     if m.sum() == 0:
#         return {"acc": 0.0, "prec": 0.0, "rec": 0.0, "f1": 0.0}
#
#     pred_m = pred[m]
#     yt_m = yt[m]
#
#     tp = ((pred_m == 1) & (yt_m == 1)).sum().item()
#     tn = ((pred_m == 0) & (yt_m == 0)).sum().item()
#     fp = ((pred_m == 1) & (yt_m == 0)).sum().item()
#     fn = ((pred_m == 0) & (yt_m == 1)).sum().item()
#
#     acc = (tp + tn) / max(1, (tp + tn + fp + fn))
#     prec = tp / max(1, (tp + fp))
#     rec = tp / max(1, (tp + fn))
#     f1 = (2 * prec * rec) / max(1e-9, (prec + rec))
#     return {"acc": acc, "prec": prec, "rec": rec, "f1": f1}
#
# @torch.no_grad()
# def best_f1_over_thresholds(
#     logits: torch.Tensor,
#     y: torch.Tensor,
#     final_mask: torch.Tensor,
#     thrs = None,
# ):
#     """
#     Scan thresholds and return (best_thr, best_f1, best_prec, best_rec).
#     logits, y, final_mask: (B,N)
#     """
#     probs = torch.sigmoid(logits)
#     m = final_mask.bool()
#
#     if m.sum() == 0:
#         return 0.5, 0.0, 0.0, 0.0
#
#     probs_m = probs[m]
#     yt = (y[m] >= 0.5).to(torch.int64)
#
#     if thrs is None:
#         thrs = torch.linspace(0.05, 0.95, 19, device=probs_m.device)
#
#     best_thr = 0.5
#     best_f1 = -1.0
#     best_prec = 0.0
#     best_rec = 0.0
#
#     for thr in thrs:
#         pred = (probs_m >= thr).to(torch.int64)
#         tp = ((pred == 1) & (yt == 1)).sum().item()
#         fp = ((pred == 1) & (yt == 0)).sum().item()
#         fn = ((pred == 0) & (yt == 1)).sum().item()
#
#         prec = tp / max(1, tp + fp)
#         rec  = tp / max(1, tp + fn)
#         f1   = (2 * prec * rec) / max(1e-9, prec + rec)
#
#         if f1 > best_f1:
#             best_f1 = f1
#             best_thr = float(thr.item())
#             best_prec = prec
#             best_rec = rec
#
#     return best_thr, float(best_f1), float(best_prec), float(best_rec)
#
# def run_one_epoch(model: STEncoderOnly, loader: DataLoader, optim: torch.optim.Optimizer,
#                   tp: TrainParams, train: bool) -> Dict[str, float]:
#     all_probs_m = []
#     all_y_m = []
#     all_mask = []
#     device = tp.device
#     model.train(train)
#
#     total_loss = 0.0
#     n_batches = 0
#     met_sum = {"acc": 0.0, "prec": 0.0, "rec": 0.0, "f1": 0.0}
#
#     t_epoch0 = time.time()
#     t_prev = time.time()
#
#     for it, batch in enumerate(loader):
#         # ---- (1) đo thời gian "load batch" (từ lần trước tới lúc batch xuất hiện)
#         load_s = time.time() - t_prev
#
#         if it == 0:
#             log.info(f"[{'train' if train else 'val'}] First batch arrived | load_time={load_s:.2f}s")
#             # in nhanh shape để chắc chắn data đúng
#             log.info(
#                 f"Shapes: x={tuple(batch['x'].shape)} tod={tuple(batch['tod'].shape)} "
#                 f"dow={tuple(batch['dow'].shape)} lap={tuple(batch['lap'].shape)} "
#                 f"y={tuple(batch['y'].shape)} node_mask={tuple(batch['node_mask'].shape)} "
#                 f"final_mask={tuple(batch['final_mask'].shape)}"
#             )
#
#         t0 = time.time()
#
#         # ---- (2) copy to device (đo riêng)
#         x = batch["x"].to(device, non_blocking=True)                 # (B,L,N,d)
#         tod = batch["tod"].to(device, non_blocking=True)
#         dow = batch["dow"].to(device, non_blocking=True)
#         lap = batch["lap"].to(device, non_blocking=True)
#         y = batch["y"].to(device, non_blocking=True)
#         node_mask = batch["node_mask"].to(device, non_blocking=True)
#         final_mask = batch["final_mask"].to(device, non_blocking=True)
#         target_mask = batch["target_mask"].to(device, non_blocking=True)
#         attn_bias = batch["attn_bias"].to(device, non_blocking=True)
#
#         t1 = time.time()
#         todev_s = t1 - t0
#
#         # ---- (3) forward
#         logits = model(x=x, tod=tod, dow=dow, lap=lap, node_mask=node_mask,attn_bias=attn_bias)
#         loss = masked_bce_loss(logits, y, final_mask)
#         t2 = time.time()
#         fwd_s = t2 - t1
#
#         # ---- (4) backward/step
#         bwd_s = 0.0
#         if train:
#             optim.zero_grad(set_to_none=True)
#             loss.backward()
#             if tp.grad_clip is not None and tp.grad_clip > 0:
#                 torch.nn.utils.clip_grad_norm_(model.parameters(), tp.grad_clip)
#             optim.step()
#             t3 = time.time()
#             bwd_s = t3 - t2
#         else:
#             with torch.no_grad():
#                 probs = torch.sigmoid(logits)
#                 m = final_mask.bool()
#                 if m.any():
#                     all_probs_m.append(probs[m].detach().cpu())  # (K,)
#                     all_y_m.append((y[m] >= 0.5).detach().cpu())  # (K,) bool
#
#             t3 = time.time()
#
#         # ---- (5) metrics
#         m = masked_metrics(logits, y, final_mask, thresh=tp.thresh)
#         for k in met_sum:
#             met_sum[k] += m[k]
#
#         total_loss += float(loss.item())
#         n_batches += 1
#
#         # ---- (6) log định kỳ
#         if it % 1 == 0:
#             log.info(
#                 f"[{'train' if train else 'val'}] it={it:04d} "
#                 f"load={load_s:.2f}s todev={todev_s:.2f}s fwd={fwd_s:.2f}s bwd={bwd_s:.2f}s "
#                 f"loss={loss.item():.4f} mask_sum={final_mask.sum().item()}"
#             )
#
#         t_prev = time.time()
#
#     if n_batches == 0:
#         return {"loss": 0.0, **met_sum}
#
#     out = {"loss": total_loss / n_batches}
#     for k in met_sum:
#         out[k] = met_sum[k] / n_batches
#
#     log.info(f"[{'train' if train else 'val'}] epoch_done in {time.time() - t_epoch0:.1f}s")
#     if not train:
#         if len(all_probs_m) == 0:
#             out["best_thr"] = 0.5
#             out["best_f1"] = 0.0
#             out["best_prec"] = 0.0
#             out["best_rec"] = 0.0
#         else:
#             probs_all = torch.cat(all_probs_m, dim=0)  # (TotalK,)
#             y_all = torch.cat(all_y_m, dim=0).to(torch.int64)  # (TotalK,)
#
#             best_thr, best_f1, best_prec, best_rec = best_f1_over_thresholds_1d(probs_all, y_all)
#             out["best_thr"] = best_thr
#             out["best_f1"] = best_f1
#             out["best_prec"] = best_prec
#             out["best_rec"] = best_rec
#
#     return out
#
#
# def train_model(model: STEncoderOnly, train_loader: DataLoader, val_loader: DataLoader, tp: TrainParams):
#     logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
#     log.info("Entered train_model()")
#
#     device = tp.device
#     model.to(device)
#     log.info(f"Model moved to {device}")
#     import threading
#     import traceback
#
#     def _fetch_first_batch(loader, out_dict):
#         try:
#             it = iter(loader)
#             out_dict["batch"] = next(it)  # <-- chỗ hay kẹt
#         except Exception as e:
#             out_dict["exc"] = e
#             out_dict["tb"] = traceback.format_exc()
#
#     log.info("[DEBUG] Sanity: fetching FIRST train batch...")
#     shared = {}
#     th = threading.Thread(target=_fetch_first_batch, args=(train_loader, shared), daemon=True)
#     t0 = time.time()
#     th.start()
#
#     # heartbeat mỗi 10s, báo đang chờ batch
#     while th.is_alive():
#         waited = time.time() - t0
#         log.info(f"[DEBUG] Waiting for first batch... {waited:.1f}s")
#         time.sleep(10.0)
#
#     # thread kết thúc: hoặc lấy được batch, hoặc exception
#     if "exc" in shared:
#         log.error("[DEBUG] First batch FAILED with exception!")
#         log.error(str(shared["exc"]))
#         log.error(shared.get("tb", ""))
#         raise shared["exc"]
#
#     batch0 = shared["batch"]
#     log.info(f"[DEBUG] First batch OK in {time.time() - t0:.1f}s")
#     log.info(
#         f"[DEBUG] batch0 shapes: x={tuple(batch0['x'].shape)} tod={tuple(batch0['tod'].shape)} "
#         f"dow={tuple(batch0['dow'].shape)} lap={tuple(batch0['lap'].shape)} "
#         f"y={tuple(batch0['y'].shape)} node_mask={tuple(batch0['node_mask'].shape)} "
#         f"final_mask={tuple(batch0['final_mask'].shape)}"
#     )
#
#     optim = torch.optim.AdamW(model.parameters(), lr=tp.lr, weight_decay=tp.weight_decay)
#
#     best_f1 = -1.0
#     best_state = None
#
#     for ep in range(tp.epochs):
#         tr = run_one_epoch(model, train_loader, optim, tp, train=True)
#         va = run_one_epoch(model, val_loader, optim, tp, train=False)
#
#         print(
#             f"[Epoch {ep + 1}/{tp.epochs}] "
#             f"train loss={tr['loss']:.4f} prec={tr['prec']:.3f} rec={tr['rec']:.3f} f1={tr['f1']:.3f} | "
#             f"val loss={va['loss']:.4f} prec={va['prec']:.3f} rec={va['rec']:.3f} f1={va['f1']:.3f} |"
#             f"best_f1={va.get('best_f1', 0.0):.3f} thr={va.get('best_thr', 0.5):.2f}"
#         )
#
#         score = va.get("best_f1", va["f1"])
#         if score > best_f1:
#             best_f1 = score
#             best_state = {k: v.cpu() for k, v in model.state_dict().items()}
#
#     if best_state is not None:
#         model.load_state_dict(best_state)
#     return model
# src/train.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from .model_sttransformer import STEncoderOnly


# -------------------------
# Params
# -------------------------
@dataclass
class TrainParams:
    lr: float = 1e-4
    weight_decay: float = 1e-4
    epochs: int = 100
    grad_clip: float = 1.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    thresh: float = 0.3              # fixed threshold
    out_dir: str = "./outputs"
    save_name: str = "best_model.pt"
    best_metric: str = "f1"          # select best by valid f1 (recommended for imbalance)


# -------------------------
# Loss
# -------------------------
# loss trung bình trên mỗi node được học
def masked_bce_loss(logits: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    BCEWithLogitsLoss on masked entries only.
    logits, y, mask: (B, N)
    """
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    loss = loss_fn(logits, y)
    m = mask.to(loss.dtype)
    loss = loss * m
    denom = m.sum().clamp_min(1.0)
    return loss.sum() / denom


# -------------------------
# Metrics helpers
# -------------------------
@torch.no_grad()
def _conf_counts_from_logits(
    logits: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
    thresh: float,
) -> Tuple[int, int, int, int]:
    """
    Return (tp, tn, fp, fn) on masked entries only.
    """
    m = mask.to(torch.bool)
    if int(m.sum().item()) == 0:
        return 0, 0, 0, 0

    probs = torch.sigmoid(logits)
    pred = (probs >= thresh).to(torch.int64)
    yt = (y >= 0.5).to(torch.int64)

    pred_m = pred[m]
    yt_m = yt[m]

    tp = int(((pred_m == 1) & (yt_m == 1)).sum().item())
    tn = int(((pred_m == 0) & (yt_m == 0)).sum().item())
    fp = int(((pred_m == 1) & (yt_m == 0)).sum().item())
    fn = int(((pred_m == 0) & (yt_m == 1)).sum().item())
    return tp, tn, fp, fn


def _metrics_from_counts(tp: int, tn: int, fp: int, fn: int) -> Dict[str, float]:
    total = max(1, tp + tn + fp + fn)
    acc = (tp + tn) / total
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = (2 * prec * rec) / max(1e-9, prec + rec)
    return {"acc": float(acc), "prec": float(prec), "rec": float(rec), "f1": float(f1)}


# -------------------------
# ROC / AUC + PR helpers (no sklearn)
# -------------------------
def _roc_curve_and_auc(probs: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    probs: (M,), y: (M,) in {0,1}
    Returns: fpr, tpr, auc
    """
    if probs.size == 0:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32), 0.0

    order = np.argsort(-probs)
    y_sorted = y[order].astype(np.int64)

    P = int((y_sorted == 1).sum())
    N = int((y_sorted == 0).sum())
    if P == 0 or N == 0:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32), 0.0

    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)

    tpr = tp / P
    fpr = fp / N

    # prepend origin
    tpr = np.concatenate([[0.0], tpr.astype(np.float64)])
    fpr = np.concatenate([[0.0], fpr.astype(np.float64)])

    auc = float(np.trapz(tpr, fpr))
    return fpr.astype(np.float32), tpr.astype(np.float32), auc


def _pr_curve_points(probs: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns: recall, precision
    """
    if probs.size == 0:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    order = np.argsort(-probs)
    y_sorted = y[order].astype(np.int64)

    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)

    precision = tp / np.maximum(1, tp + fp)
    recall = tp / max(1, int((y_sorted == 1).sum()))

    return recall.astype(np.float32), precision.astype(np.float32)


# -------------------------
# Plot helpers
# -------------------------
def _plot_confusion_matrix(tp: int, tn: int, fp: int, fn: int, save_path: str, title: str) -> None:
    cm = np.array([[tn, fp],
                   [fn, tp]], dtype=np.int64)

    plt.figure()
    plt.imshow(cm)
    plt.title(title)
    plt.xlabel("Pred")
    plt.ylabel("True")
    plt.xticks([0, 1], ["0", "1"])
    plt.yticks([0, 1], ["0", "1"])
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def _plot_pr_curve(probs: np.ndarray, y: np.ndarray, save_path: str, title: str) -> None:
    rec, prec = _pr_curve_points(probs, y)
    plt.figure()
    if rec.size > 0:
        plt.plot(rec, prec)
    plt.title(title)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def _plot_roc_auc_curve(probs: np.ndarray, y: np.ndarray, save_path: str, title: str) -> float:
    fpr, tpr, auc = _roc_curve_and_auc(probs, y)
    plt.figure()
    if fpr.size > 0:
        plt.plot(fpr, tpr, label=f"AUC={auc:.4f}")
        plt.plot([0, 1], [0, 1], label="baseline")
        plt.legend()
    plt.title(title)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    return auc


def _plot_loss_curve(train_loss: List[float], val_loss: List[float], test_loss: List[float], save_path: str) -> None:
    plt.figure()
    plt.plot(train_loss, label="train")
    plt.plot(val_loss, label="valid")
    plt.plot(test_loss, label="test")
    plt.title("Loss curves")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def _plot_one_image_metrics(hist: Dict[str, List[float]], save_path: str, title: str) -> None:
    """
    One image contains 4 panels: Accuracy / Precision / Recall / F1
    for train/valid/test across epochs.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    ax = axes[0, 0]
    ax.plot(hist["train_acc"], label="train")
    ax.plot(hist["val_acc"], label="valid")
    ax.plot(hist["test_acc"], label="test")
    ax.set_title("Accuracy")
    ax.set_xlabel("Epoch")
    ax.legend()

    ax = axes[0, 1]
    ax.plot(hist["train_prec"], label="train")
    ax.plot(hist["val_prec"], label="valid")
    ax.plot(hist["test_prec"], label="test")
    ax.set_title("Precision")
    ax.set_xlabel("Epoch")
    ax.legend()

    ax = axes[1, 0]
    ax.plot(hist["train_rec"], label="train")
    ax.plot(hist["val_rec"], label="valid")
    ax.plot(hist["test_rec"], label="test")
    ax.set_title("Recall")
    ax.set_xlabel("Epoch")
    ax.legend()

    ax = axes[1, 1]
    ax.plot(hist["train_f1"], label="train")
    ax.plot(hist["val_f1"], label="valid")
    ax.plot(hist["test_f1"], label="test")
    ax.set_title("F1-score")
    ax.set_xlabel("Epoch")
    ax.legend()

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


# -------------------------
# Collect probabilities/labels on final_mask only (for PR/ROC)
# -------------------------
@torch.no_grad()
def _collect_probs_labels(
    model: STEncoderOnly,
    loader: DataLoader,
    device: str,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs_all: List[torch.Tensor] = []
    y_all: List[torch.Tensor] = []

    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        tod = batch["tod"].to(device, non_blocking=True)
        dow = batch["dow"].to(device, non_blocking=True)
        lap = batch["lap"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        node_mask = batch["node_mask"].to(device, non_blocking=True)
        final_mask = batch["final_mask"].to(device, non_blocking=True)
        attn_bias = batch["attn_bias"].to(device, non_blocking=True)

        logits = model(x=x, tod=tod, dow=dow, lap=lap, node_mask=node_mask, attn_bias=attn_bias)

        m = final_mask.to(torch.bool)
        if int(m.sum().item()) == 0:
            continue

        probs = torch.sigmoid(logits)[m].detach().cpu()
        yt = (y[m] >= 0.5).to(torch.int64).detach().cpu()
        probs_all.append(probs)
        y_all.append(yt)

    if len(probs_all) == 0:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.int64)

    probs_cat = torch.cat(probs_all, dim=0).numpy().astype(np.float32)
    y_cat = torch.cat(y_all, dim=0).numpy().astype(np.int64)
    return probs_cat, y_cat


# -------------------------
# One epoch runner
# -------------------------
def _run_epoch(
    model: STEncoderOnly,
    loader: DataLoader,
    optim: Optional[torch.optim.Optimizer],
    params: TrainParams,
    epoch: int,
    split_name: str,
    train: bool,
) -> Dict[str, Any]:
    device = params.device
    model.train(train)

    total_loss = 0.0
    n_batches = 0

    TP = TN = FP = FN = 0
    total_eval_nodes = 0       # sum(final_mask)
    total_valid_nodes = 0      # sum(node_mask)

    for it, batch in enumerate(loader):
        x = batch["x"].to(device, non_blocking=True)
        tod = batch["tod"].to(device, non_blocking=True)
        dow = batch["dow"].to(device, non_blocking=True)
        lap = batch["lap"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        node_mask = batch["node_mask"].to(device, non_blocking=True)
        final_mask = batch["final_mask"].to(device, non_blocking=True)
        attn_bias = batch["attn_bias"].to(device, non_blocking=True)

        logits = model(x=x, tod=tod, dow=dow, lap=lap, node_mask=node_mask, attn_bias=attn_bias)
        loss = masked_bce_loss(logits, y, final_mask)

        if train:
            assert optim is not None
            optim.zero_grad(set_to_none=True)
            loss.backward()
            if params.grad_clip is not None and params.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), params.grad_clip)
            optim.step()
        # Node target
        batch_eval_nodes = int(final_mask.sum().item())
        # Node tồn tại trong zone
        batch_valid_nodes = int(node_mask.sum().item())

        # each batch prints: loss + counts
        print(
            f"[E{epoch:03d}][{split_name}] it={it:04d} "
            f"loss={loss.item():.6f} eval_nodes={batch_eval_nodes} valid_nodes={batch_valid_nodes}"
        )

        total_loss += float(loss.item())
        n_batches += 1

        tp, tn, fp, fn = _conf_counts_from_logits(logits, y, final_mask, params.thresh)
        TP += tp; TN += tn; FP += fp; FN += fn

        total_eval_nodes += batch_eval_nodes
        total_valid_nodes += batch_valid_nodes

    avg_loss = total_loss / max(1, n_batches)
    metr = _metrics_from_counts(TP, TN, FP, FN)
    return {
        "loss": float(avg_loss),
        "metrics": metr,
        "counts": {"tp": TP, "tn": TN, "fp": FP, "fn": FN},
        "eval_nodes": int(total_eval_nodes),
        "valid_nodes": int(total_valid_nodes),
    }


# -------------------------
# Main train
# -------------------------
def train_model(
    model: STEncoderOnly,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    test_loader: DataLoader,
    params: TrainParams,
) -> Dict[str, Any]:
    os.makedirs(params.out_dir, exist_ok=True)
    device = params.device
    model.to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=params.lr, weight_decay=params.weight_decay)

    best_score = -1.0
    best_epoch = -1
    best_path = os.path.join(params.out_dir, params.save_name)

    hist: Dict[str, List[float]] = {
        "train_loss": [], "val_loss": [], "test_loss": [],
        "train_acc": [], "val_acc": [], "test_acc": [],
        "train_prec": [], "val_prec": [], "test_prec": [],
        "train_rec": [], "val_rec": [], "test_rec": [],
        "train_f1": [], "val_f1": [], "test_f1": [],
    }

    for ep in range(1, params.epochs + 1):
        print(f"\n========== Epoch {ep:03d}/{params.epochs} | threshold={params.thresh} ==========")

        tr = _run_epoch(model, train_loader, optim, params, ep, "train", train=True)
        va = _run_epoch(model, valid_loader, None, params, ep, "valid", train=False)
        te = _run_epoch(model, test_loader, None, params, ep, "test", train=False)

        # epoch summary with metrics + node counts
        def _fmt(s: Dict[str, Any]) -> str:
            m = s["metrics"]
            return (
                f"loss={s['loss']:.6f} "
                f"acc={m['acc']:.4f} prec={m['prec']:.4f} rec={m['rec']:.4f} f1={m['f1']:.4f} "
                f"eval_nodes={s['eval_nodes']} valid_nodes={s['valid_nodes']}"
            )

        print(f"[E{ep:03d}] train: {_fmt(tr)}")
        print(f"[E{ep:03d}] valid: {_fmt(va)}")
        print(f"[E{ep:03d}] test : {_fmt(te)}")

        # store history
        hist["train_loss"].append(tr["loss"])
        hist["val_loss"].append(va["loss"])
        hist["test_loss"].append(te["loss"])

        hist["train_acc"].append(tr["metrics"]["acc"])
        hist["val_acc"].append(va["metrics"]["acc"])
        hist["test_acc"].append(te["metrics"]["acc"])

        hist["train_prec"].append(tr["metrics"]["prec"])
        hist["val_prec"].append(va["metrics"]["prec"])
        hist["test_prec"].append(te["metrics"]["prec"])

        hist["train_rec"].append(tr["metrics"]["rec"])
        hist["val_rec"].append(va["metrics"]["rec"])
        hist["test_rec"].append(te["metrics"]["rec"])

        hist["train_f1"].append(tr["metrics"]["f1"])
        hist["val_f1"].append(va["metrics"]["f1"])
        hist["test_f1"].append(te["metrics"]["f1"])

        # select best based on valid metric (default f1)
        metric_key = params.best_metric.lower()
        score = float(va["metrics"].get(metric_key, va["metrics"]["f1"]))
        if score > best_score:
            best_score = score
            best_epoch = ep
            torch.save(
                {
                    "epoch": ep,
                    "model_state": model.state_dict(),
                    "best_valid_score": best_score,
                    "best_metric": metric_key,
                    "threshold": params.thresh,
                },
                best_path,
            )
            print(f"✅ Saved BEST checkpoint @epoch={ep} valid_{metric_key}={best_score:.4f} -> {best_path}")

    # ---------- load best and final test plots ----------
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    print(f"\n========== BEST checkpoint: epoch={ckpt['epoch']} valid_{ckpt['best_metric']}={ckpt['best_valid_score']:.4f} ==========")

    te_best = _run_epoch(model, test_loader, None, params, ckpt["epoch"], "test(best)", train=False)

    # curves
    _plot_loss_curve(
        hist["train_loss"], hist["val_loss"], hist["test_loss"],
        save_path=os.path.join(params.out_dir, "loss_curves.png"),
    )

    _plot_one_image_metrics(
        hist,
        save_path=os.path.join(params.out_dir, "metrics_train_valid_test.png"),
        title=f"Metrics (thresh={params.thresh}) | best_epoch={best_epoch}",
    )

    # test best: confusion matrix + PR + ROC/AUC
    TP = te_best["counts"]["tp"]
    TN = te_best["counts"]["tn"]
    FP = te_best["counts"]["fp"]
    FN = te_best["counts"]["fn"]

    _plot_confusion_matrix(
        tp=TP, tn=TN, fp=FP, fn=FN,
        save_path=os.path.join(params.out_dir, "test_confusion_matrix.png"),
        title=f"Test Confusion Matrix (best epoch={best_epoch}, thresh={params.thresh})",
    )

    probs, y = _collect_probs_labels(model, test_loader, device=device)
    _plot_pr_curve(probs, y, os.path.join(params.out_dir, "test_pr_curve.png"), "Test PR Curve (best)")
    test_auc = _plot_roc_auc_curve(probs, y, os.path.join(params.out_dir, "test_roc_auc_curve.png"), "Test ROC Curve (best)")
    print(f"[BEST] Test ROC-AUC = {test_auc:.6f}")

    summary = {
        "best_epoch": int(best_epoch),
        "best_valid_score": float(best_score),
        "best_metric": str(params.best_metric),
        "threshold": float(params.thresh),
        "checkpoint": best_path,
        "out_dir": params.out_dir,
        "test_best": te_best,
        "test_auc": float(test_auc),
        "plots": [
            "loss_curves.png",
            "metrics_train_valid_test.png",
            "test_confusion_matrix.png",
            "test_pr_curve.png",
            "test_roc_auc_curve.png",
        ],
    }
    return summary
