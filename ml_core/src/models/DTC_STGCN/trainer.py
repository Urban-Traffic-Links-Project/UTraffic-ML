"""
Trainer for DTC-STGCN Model
Based on experiment settings in paper (Section 5.2):
- Adam optimizer, lr=0.001
- Training epoch=100, LSTM dropout: input=0.25, output=0.3
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from utils.logger import setup_logger, LoggerMixin


class DTCSTGCNTrainer(LoggerMixin):
    """Trainer for DTC-STGCN with checkpoint and early stopping."""

    def __init__(self, model, config, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.model = model.to(device)
        self.config = config
        self.device = device

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.get("lr", 0.001),
            weight_decay=config.get("weight_decay", 0.0001),
        )
        self.criterion = nn.HuberLoss()
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=10, verbose=True
        )

        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float("inf")
        self.epochs_no_improve = 0
        self.current_epoch = 0

        base_dir = Path(__file__).resolve().parents[3] / "checkpoints" / "DTC-STGCN"
        self.checkpoint_dir = config.get("checkpoint_dir", base_dir)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        log_file = os.path.join(
            self.checkpoint_dir,
            "logs",
            f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
        )
        self._logger = setup_logger("DTCSTGCNTrainer", log_file=log_file, level="INFO")

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch+1} [Train]")
        for x, y in pbar:
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            pred = self.model(x)
            loss = self.criterion(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.6f}"})
        return total_loss / len(train_loader)

    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {self.current_epoch+1} [Val]")
            for x, y in pbar:
                x, y = x.to(self.device), y.to(self.device)
                pred = self.model(x)
                loss = self.criterion(pred, y)
                total_loss += loss.item()
                pbar.set_postfix({"loss": f"{loss.item():.6f}"})
        return total_loss / len(val_loader)

    def save_checkpoint(self, filename="checkpoint.pth", is_best=False):
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "best_val_loss": self.best_val_loss,
            "config": self.config,
        }
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save(checkpoint, path)
        if is_best:
            torch.save(checkpoint, os.path.join(self.checkpoint_dir, "best_model.pth"))
            self.logger.info(f"★ Best model saved.")

    def load_checkpoint(self, filename="checkpoint.pth"):
        path = os.path.join(self.checkpoint_dir, filename)
        if not os.path.exists(path):
            return False
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.current_epoch = ckpt["epoch"]
        self.train_losses = ckpt["train_losses"]
        self.val_losses = ckpt["val_losses"]
        self.best_val_loss = ckpt["best_val_loss"]
        self.logger.info(f"✓ Resumed from epoch {self.current_epoch + 1}")
        return True

    def train(self, train_loader, val_loader, epochs, early_stopping_patience=20, resume=False):
        if resume:
            self.load_checkpoint("last_checkpoint.pth")

        self.logger.info("=" * 60)
        self.logger.info("Starting DTC-STGCN Training")
        self.logger.info("=" * 60)
        self.logger.info(f"Device: {self.device}")

        for epoch in range(self.current_epoch, epochs):
            self.current_epoch = epoch
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.scheduler.step(val_loss)
            self.logger.info(
                f"Epoch {epoch+1}/{epochs} | Train: {train_loss:.6f} | Val: {val_loss:.6f}"
            )
            self.save_checkpoint("last_checkpoint.pth")
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.epochs_no_improve = 0
                self.save_checkpoint("best_model.pth", is_best=True)
            else:
                self.epochs_no_improve += 1
            if self.epochs_no_improve >= early_stopping_patience:
                self.logger.info(f"Early stopping at epoch {epoch+1}")
                break

        self.logger.info(f"Best val loss: {self.best_val_loss:.6f}")
        history = {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "best_val_loss": self.best_val_loss,
        }
        with open(os.path.join(self.checkpoint_dir, "training_history.json"), "w") as f:
            json.dump(history, f, indent=2)

    def predict(self, test_loader):
        self.model.eval()
        predictions, targets = [], []
        with torch.no_grad():
            for x, y in tqdm(test_loader, desc="Predicting"):
                x, y = x.to(self.device), y.to(self.device)
                pred = self.model(x)
                predictions.append(pred.cpu().numpy())
                targets.append(y.cpu().numpy())
        return np.concatenate(predictions), np.concatenate(targets)