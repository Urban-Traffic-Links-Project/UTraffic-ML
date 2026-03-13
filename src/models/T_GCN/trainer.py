"""
Trainer for T-GCN with checkpoint support and model comparison
"""

from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import json
from datetime import datetime
from tqdm import tqdm
import logging
import pandas as pd
from utils.logger import setup_logger, LoggerMixin

class TGCNTrainer(LoggerMixin):
    """
    Trainer with checkpoint support for resuming interrupted training
    """
    def __init__(self, model, adj, config, device="cuda" if torch.cuda.is_available() else 'cpu'):
        """
        Args:
            model: T-GCN model
            adj: Adjacency matrix (num_nodes, num_nodes)
            config: Configuration dict
            device: 'cuda' or 'cpu'
        """
        self.model = model.to(device)
        self.adj = torch.FloatTensor(adj).to(device)
        self.config = config
        self.device = device

        # 1. Bộ tối ưu hóa (Optimizer): Adam là lựa chọn tiêu chuẩn
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.get('lr', 0.001),
            weight_decay=config.get('weight_decay', 0.0001)
        )

        # 2. Hàm mất mát (Loss Function): MSE (Sai số bình phương trung bình)
        # Dùng cho bài toán dự báo số (Regression)
        self.criterion = nn.HuberLoss()

        # 3. Trợ lý điều chỉnh tốc độ học (Scheduler)
        # Nếu loss không giảm sau 10 lần, giảm tốc độ học đi một nửa
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            verbose=True
        )

        # Các biến để theo dõi lịch sử
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.epochs_no_improve = 0
        self.current_epoch = 0

        # Checkpoint directory
        base_dir = Path(__file__).resolve().parents[3] / "checkpoints" / "T-GCN"
        self.checkpoint_dir = config.get('checkpoint_dir', base_dir)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Setup logger
        log_file = os.path.join(
            self.checkpoint_dir,
            "logs",
            f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )

        self._logger = setup_logger(
            name="TGCNTrainer",
            log_file=log_file,
            level=config.get("log_level", "INFO")
        )

    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0

        # Dùng tqdm để hiện thanh loading bar
        pbar = tqdm(train_loader, desc="Training")

        for x, y in pbar:
            x = x.to(self.device)
            y = y.to(self.device)

            # A. Xóa Gradient cũ
            self.optimizer.zero_grad()

            # B. Forward (Mô hình dự đoán)
            output = self.model(x, self.adj)

            # C. Tính sai số (Loss)
            loss = self.criterion(output, y)

            # D. Backward (Tìm lỗi sai)
            loss.backward()

            # E. Cắt ngọn Gradient (Clip Grad Norm) - CỰC KỲ QUAN TRỌNG VỚI GRU/RNN
            # Giúp tránh lỗi "Exploding Gradient" (số quá to gây lỗi)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)

            # F. Cập nhật trọng số
            self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.6f}'})

        return total_loss / len(train_loader) # Trả về loss trung bình

    def validate(self, val_loader):
        """Validate the model"""
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f'Epoch {self.current_epoch+1} [Val]')
            for batch_x, batch_y in pbar:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                output = self.model(batch_x, self.adj)
                loss = self.criterion(output, batch_y)
                total_loss += loss.item()
                
                pbar.set_postfix({'loss': f'{loss.item():.6f}'})
        
        avg_loss = total_loss / len(val_loader)
        return avg_loss

    def save_checkpoint(self, filename='checkpoint.pth', is_best=False):
        """
        Save checkpoint - CỰC KỲ QUAN TRỌNG để tiếp tục training khi bị ngắt quãng
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'config': self.config,
            'adj': self.adj.cpu().numpy()
        }
        
        filepath = os.path.join(self.checkpoint_dir, filename)
        torch.save(checkpoint, filepath)
        self.logger.info(f'✓ Checkpoint saved: {filepath}')
        
        if is_best:
            best_filepath = os.path.join(self.checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, best_filepath)
            self.logger.info(f'✓ Best model saved: {best_filepath}')
    
    def load_checkpoint(self, filename='checkpoint.pth'):
        """
        Load checkpoint - Để tiếp tục training từ điểm dừng
        """
        filepath = os.path.join(self.checkpoint_dir, filename)
        
        if not os.path.exists(filepath):
            self.logger.warning(f'⚠ Checkpoint not found: {filepath}')
            return False
            
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.best_val_loss = checkpoint['best_val_loss']
        
        self.logger.info(f'✓ Checkpoint loaded: {filepath}')
        self.logger.info(f'✓ Resuming from epoch {self.current_epoch + 1}')
        
        return True
    
    def train(self, train_loader, val_loader, epochs, early_stopping_patience=20, resume=False):
        """
        Train the model
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs
            early_stopping_patience: Patience for early stopping
            resume: Resume from last checkpoint
        """
        # Resume if requested
        if resume:
            self.load_checkpoint('last_checkpoint.pth')
        
        self.logger.info('=' * 60)
        self.logger.info('Starting T-GCN Training')
        self.logger.info('=' * 60)
        self.logger.info(f'Config: {json.dumps(self.config, indent=2)}')
        self.logger.info(f'Device: {self.device}')
        self.logger.info(f'Model parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}')
        
        for epoch in range(self.current_epoch, epochs):
            self.current_epoch = epoch
            
            # Train
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss = self.validate(val_loader)
            self.val_losses.append(val_loss)
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Log
            self.logger.info(
                f'Epoch {epoch+1}/{epochs} - '
                f'Train Loss: {train_loss:.6f}, '
                f'Val Loss: {val_loss:.6f}, '
                f'LR: {self.optimizer.param_groups[0]["lr"]:.6f}'
            )
            
            # Save checkpoint
            self.save_checkpoint('last_checkpoint.pth')
            
            # Check if best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.epochs_no_improve = 0
                self.save_checkpoint('best_model.pth', is_best=True)
                self.logger.info(f'★ New best model! Val Loss: {val_loss:.6f}')
            else:
                self.epochs_no_improve += 1
                
            # Early stopping
            if self.epochs_no_improve >= early_stopping_patience:
                self.logger.info(f'Early stopping at epoch {epoch+1}')
                break
                
        self.logger.info('=' * 60)
        self.logger.info('Training completed!')
        self.logger.info(f'Best validation loss: {self.best_val_loss:.6f}')
        self.logger.info('=' * 60)
        
        # Save history
        self.save_training_history()
        
    def save_training_history(self):
        """Save training history"""
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'total_epochs': self.current_epoch + 1
        }
        
        history_file = os.path.join(self.checkpoint_dir, 'training_history.json')
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2)
            
        self.logger.info(f'✓ Training history saved: {history_file}')
        
    def predict(self, test_loader):
        """
        Make predictions
        
        Returns:
            predictions, targets (numpy arrays)
        """
        self.model.eval()
        predictions = []
        targets = []
        
        with torch.no_grad():
            for batch_x, batch_y in tqdm(test_loader, desc='Predicting'):
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                output = self.model(batch_x, self.adj)
                
                predictions.append(output.cpu().numpy())
                targets.append(batch_y.cpu().numpy())
        
        predictions = np.concatenate(predictions, axis=0)
        targets = np.concatenate(targets, axis=0)
        
        return predictions, targets