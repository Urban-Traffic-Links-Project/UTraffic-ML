"""
Trainer for T-GCN with checkpoint support and model comparison.

Fixes vs original:
    - Trainer nhận adj RAW binary (không normalize trước) — normalization
      được delegate vào TGCN.forward() để tránh double-normalize.
    - scheduler verbose kwarg removed (deprecated in newer PyTorch).
    - save_checkpoint / load_checkpoint: thêm adj device handling.
    - train(): early_stopping_patience default khớp với config key 'patience'.
    - Minor: type hints, cleaner tqdm desc per epoch.
"""

import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.logger import setup_logger, LoggerMixin


class TGCNTrainer(LoggerMixin):
    """
    Trainer with checkpoint/resume support for T-GCN.

    NOTE: `adj` should be passed as RAW (unnormalized) binary/weighted
    adjacency matrix. TGCN.forward() handles normalize_adj() internally
    with caching, so passing a pre-normalized matrix would double-normalize.
    """

    def __init__(
        self,
        model: nn.Module,
        adj: np.ndarray,
        config: dict,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model  = model.to(device)
        self.device = device
        self.config = config

        # Store adj as tensor on device (raw — not normalized)
        self.adj = torch.FloatTensor(adj).to(device)

        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.get("lr", 0.001),
            weight_decay=config.get("weight_decay", 1e-4),
        )

        # Loss — Huber is robust to speed outliers
        self.criterion = nn.HuberLoss(delta=1.0)

        # LR Scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=10,
        )

        # State
        self.train_losses: list   = []
        self.val_losses:   list   = []
        self.best_val_loss: float = float("inf")
        self.epochs_no_improve: int = 0
        self.current_epoch: int  = 0

        # Checkpoint directory
        base_dir = Path(__file__).resolve().parents[3] / "checkpoints" / "T-GCN"
        self.checkpoint_dir = Path(config.get("checkpoint_dir", base_dir))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Logger
        log_dir = self.checkpoint_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = str(
            log_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        self._logger = setup_logger(
            name="TGCNTrainer",
            log_file=log_file,
            level=config.get("log_level", "INFO"),
        )

    # ------------------------------------------------------------------
    # Train / Validate
    # ------------------------------------------------------------------

    def train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0

        pbar = tqdm(loader, desc=f"Epoch {self.current_epoch + 1} [Train]")
        for x, y in pbar:
            x = x.to(self.device)
            # Target: speed only (dim 0) — shape (B, pred, N, 1)
            y = y[:, :, :, :1].to(self.device)

            self.optimizer.zero_grad()
            out = self.model(x, self.adj)          # (B, pred, N, output_dim=1)
            loss = self.criterion(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.config.get("grad_clip", 5.0),
            )
            self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.6f}")

        return total_loss / max(len(loader), 1)

    def validate(self, loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0

        pbar = tqdm(loader, desc=f"Epoch {self.current_epoch + 1} [Val]")
        with torch.no_grad():
            for x, y in pbar:
                x = x.to(self.device)
                y = y[:, :, :, :1].to(self.device)
                out = self.model(x, self.adj)
                loss = self.criterion(out, y)
                total_loss += loss.item()
                pbar.set_postfix(loss=f"{loss.item():.6f}")

        return total_loss / max(len(loader), 1)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(
        self,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        epochs:       int,
        early_stopping_patience: int = 20,
        resume: bool = False,
    ):
        if resume:
            self.load_checkpoint("last_checkpoint.pth")

        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info("=" * 60)
        self.logger.info("T-GCN Training")
        self.logger.info(f"Device    : {self.device}")
        self.logger.info(f"Parameters: {n_params:,}")
        self.logger.info(f"Config    : {json.dumps(self.config, indent=2)}")
        self.logger.info("=" * 60)

        for epoch in range(self.current_epoch, epochs):
            self.current_epoch = epoch

            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)

            val_loss = self.validate(val_loader)
            self.val_losses.append(val_loss)

            self.scheduler.step(val_loss)
            lr_now = self.optimizer.param_groups[0]["lr"]

            self.logger.info(
                f"Epoch {epoch + 1}/{epochs} — "
                f"Train: {train_loss:.6f}  Val: {val_loss:.6f}  LR: {lr_now:.2e}"
            )

            self.save_checkpoint("last_checkpoint.pth")

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.epochs_no_improve = 0
                self.save_checkpoint("best_model.pth", is_best=True)
                self.logger.info(f"★ New best — Val Loss: {val_loss:.6f}")
            else:
                self.epochs_no_improve += 1

            if self.epochs_no_improve >= early_stopping_patience:
                self.logger.info(f"Early stopping at epoch {epoch + 1}")
                break

        self.logger.info("=" * 60)
        self.logger.info("Training complete!")
        self.logger.info(f"Best val loss: {self.best_val_loss:.6f}")
        self.logger.info("=" * 60)

        self.save_training_history()

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, filename: str = "checkpoint.pth", is_best: bool = False):
        ckpt = {
            "epoch":                self.current_epoch,
            "model_state_dict":     self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "train_losses":         self.train_losses,
            "val_losses":           self.val_losses,
            "best_val_loss":        self.best_val_loss,
            "config":               self.config,
            "adj":                  self.adj.cpu().numpy(),
        }
        path = self.checkpoint_dir / filename
        torch.save(ckpt, str(path))
        self.logger.info(f"✓ Checkpoint → {path.name}")

        if is_best:
            best_path = self.checkpoint_dir / "best_model.pth"
            torch.save(ckpt, str(best_path))
            self.logger.info(f"✓ Best model → {best_path.name}")

    def load_checkpoint(self, filename: str = "checkpoint.pth") -> bool:
        path = self.checkpoint_dir / filename
        if not path.exists():
            self.logger.warning(f"⚠ Checkpoint not found: {path}")
            return False

        ckpt = torch.load(str(path), map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.current_epoch     = ckpt["epoch"] + 1   # resume from NEXT epoch
        self.train_losses      = ckpt["train_losses"]
        self.val_losses        = ckpt["val_losses"]
        self.best_val_loss     = ckpt["best_val_loss"]

        self.logger.info(f"✓ Loaded {path.name} — resuming from epoch {self.current_epoch + 1}")
        return True

    def save_training_history(self):
        history = {
            "train_losses": self.train_losses,
            "val_losses":   self.val_losses,
            "best_val_loss": self.best_val_loss,
            "total_epochs": self.current_epoch + 1,
        }
        hist_path = self.checkpoint_dir / "training_history.json"
        with open(hist_path, "w") as f:
            json.dump(history, f, indent=2)
        self.logger.info(f"✓ Training history → {hist_path.name}")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, loader: DataLoader):
        """
        Returns:
            predictions : (N_samples, pred_len, num_nodes, output_dim)
            targets     : (N_samples, pred_len, num_nodes, output_dim)
        """
        self.model.eval()
        preds, tgts = [], []

        with torch.no_grad():
            for x, y in tqdm(loader, desc="Predicting"):
                x = x.to(self.device)
                y = y[:, :, :, :1].to(self.device)
                out = self.model(x, self.adj)
                preds.append(out.cpu().numpy())
                tgts.append(y.cpu().numpy())

        return np.concatenate(preds, axis=0), np.concatenate(tgts, axis=0)