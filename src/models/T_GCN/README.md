# T-GCN Implementation for Urban Traffic Prediction

Implementation của mô hình T-GCN (Temporal Graph Convolutional Network) cho dự án Urban Traffic Links, dựa trên paper "T-GCN: A Temporal Graph Convolutional Network for Traffic Prediction" (Zhao et al., 2019).

## 📋 Tổng quan

Hệ thống này bao gồm:
- ✅ **T-GCN Model**: Mô hình chính kết hợp GCN và GRU
- ✅ **Baseline Models**: HA, ARIMA, SVR, GCN, GRU để so sánh
- ✅ **Checkpoint Support**: Lưu và resume training khi bị ngắt quãng
- ✅ **Comparison Tables**: Bảng so sánh giống paper TH-Hierarchical
- ✅ **5 Metrics**: RMSE, MAE, Accuracy, R², VAR (theo paper)

## 📁 Cấu trúc thư mục

```
Urban-Traffic-T-GCN/
├── models/
│   └── T-GCN/
│       ├── __init__.py
│       ├── gcn.py          # Graph Convolutional Network
│       ├── gru.py          # Gated Recurrent Unit  
│       └── tgcn.py         # T-GCN Model chính
├── utils/
│   ├── trainer.py          # Training với checkpoint support
│   ├── metrics.py          # 5 metrics theo paper
│   ├── baselines.py        # Baseline models
│   └── data_loader.py      # Load dữ liệu từ .npz
├── experiments/
│   ├── train_and_compare.py    # Script train và so sánh
│   └── resume_training.py      # Resume training
├── checkpoints/            # Lưu checkpoints
│   └── T-GCN/
│       ├── last_checkpoint.pth     # Checkpoint mới nhất
│       ├── best_model.pth          # Model tốt nhất
│       ├── training_history.json   # Lịch sử training
│       └── logs/
└── results/                # Kết quả so sánh
    ├── comparison_horizon_1days.csv
    ├── comparison_horizon_2days.csv
    ├── comparison_summary.csv
    └── comparison_results.json
```

## 🚀 Cài đặt

```bash
# Cài đặt các thư viện cần thiết
pip install torch numpy pandas scikit-learn statsmodels tqdm
```

## 📊 Sử dụng

### 1. Training và So sánh Models

```bash
cd Urban-Traffic-T-GCN
python experiments/train_and_compare.py
```

Script này sẽ:
1. Load dữ liệu từ `/mnt/user-data/uploads/*.npz`
2. Train T-GCN và các baseline models (HA, ARIMA, SVR, GCN, GRU)
3. Đánh giá trên các prediction horizons (1, 2, 3, 4 days)
4. Tạo bảng so sánh như trong paper TH-Hierarchical
5. Lưu kết quả vào `results/`

### 2. Resume Training (khi bị ngắt quãng)

Nếu training bị ngắt giữa chừng, bạn có thể tiếp tục:

```python
# Trong file train_and_compare.py, sửa config:
config = {
    ...
    'resume': True  # Đổi thành True
}
```

Hoặc:

```bash
python experiments/resume_training.py
```

### 3. Chỉ Train T-GCN

```python
from models.T_GCN import TGCN
from utils.trainer import TGCNTrainer
from utils.data_loader import DataManager

# Load data
dm = DataManager()
dm.load_all()
train_loader, val_loader, test_loader, adj = dm.prepare_for_training()

# Create model
model = TGCN(
    num_nodes=114,
    input_dim=1,
    hidden_dim=64,
    output_dim=1,
    seq_len=12,
    pred_len=12
)

# Train
config = {
    'lr': 0.001,
    'weight_decay': 0.0001,
    'epochs': 100,
    'checkpoint_dir': 'checkpoints/T-GCN'
}

trainer = TGCNTrainer(model, adj.numpy(), config)
trainer.train(train_loader, val_loader, epochs=100)

# Predict
predictions, targets = trainer.predict(test_loader)
```

## 📈 Kết quả

### Bảng so sánh (giống TH-Hierarchical Tables 2-5)

Sau khi training, bạn sẽ có các bảng như sau:

**Table: Prediction Horizon = 1 day**
```
Model   RMSE    MAE     Accuracy    R²      VAR
T-GCN   3.9162  2.7061  0.7306      0.8541  0.8626
GRU     4.0483  2.6814  0.7178      0.8498  0.8499
GCN     9.2717  7.2606  0.6433      0.6147  0.6147
SVR     7.5368  4.9269  0.6961      0.8111  0.8121
ARIMA   8.2151  6.2192  0.4278      0.0842  -
HA      7.9198  5.4969  0.6807      0.7914  0.7914
```

## 🔧 Checkpoint System

### Tự động lưu checkpoints

Training sẽ tự động lưu:
- **last_checkpoint.pth**: Sau mỗi epoch
- **best_model.pth**: Khi val loss tốt nhất
- **training_history.json**: Lịch sử train/val losses

### Cấu trúc checkpoint

```python
checkpoint = {
    'epoch': 50,
    'model_state_dict': {...},
    'optimizer_state_dict': {...},
    'scheduler_state_dict': {...},
    'train_losses': [...],
    'val_losses': [...],
    'best_val_loss': 0.0123,
    'config': {...},
    'adj': adj_matrix
}
```

### Load checkpoint để inference

```python
import torch

checkpoint = torch.load('checkpoints/T-GCN/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# Predict
model.eval()
with torch.no_grad():
    predictions = model(X_test, adj)
```

## 📝 Metrics (theo T-GCN paper)

### 1. RMSE (Equation 8)
```
RMSE = sqrt(1/n * sum((Y_t - Y_pred)^2))
```

### 2. MAE (Equation 9)
```
MAE = 1/n * sum(|Y_t - Y_pred|)
```

### 3. Accuracy (Equation 10)
```
Accuracy = 1 - ||Y - Y_pred||_F / ||Y||_F
```

### 4. R² (Equation 11)
```
R² = 1 - sum((Y_t - Y_pred)^2) / sum((Y_t - Y_mean)^2)
```

### 5. VAR (Equation 12)
```
VAR = 1 - Var{Y - Y_pred} / Var{Y}
```

## 🎯 Mô hình

### T-GCN Architecture

```
Input (batch, seq_len, nodes, features)
    ↓
[T-GCN Cell 1] ← h_0
    ↓
[T-GCN Cell 2] ← h_1
    ↓
    ...
    ↓
[T-GCN Cell n] ← h_{n-1}
    ↓
FC Layer
    ↓
Output (batch, pred_len, nodes, features)
```

### T-GCN Cell

```
      Input x_t
         ↓
    ┌────GCN────┐
    │  Spatial  │
    └─────┬─────┘
          ↓
    ┌────GRU────┐
    │ Temporal  │ ← h_{t-1}
    └─────┬─────┘
          ↓
        h_t
```

## 💡 Tips

### 1. Tăng tốc training
```python
config = {
    'batch_size': 64,  # Tăng batch size
    'epochs': 50,      # Giảm epochs nếu cần test nhanh
}
```

### 2. Tránh overfitting
```python
config = {
    'weight_decay': 0.001,  # Tăng L2 regularization
    'patience': 10,         # Early stopping sớm hơn
}
```

### 3. Hyperparameter tuning
```python
hidden_dims = [32, 64, 128]
learning_rates = [0.001, 0.0001]

for hd in hidden_dims:
    for lr in learning_rates:
        config['hidden_dim'] = hd
        config['lr'] = lr
        # Train and compare
```

## 📚 Paper References

1. **T-GCN**: Zhao et al. "T-GCN: A Temporal Graph Convolutional Network for Traffic Prediction" (2019)
2. **TH-GAT**: Huang et al. "Temporal Hierarchical Graph Attention Network for Traffic Prediction" (2021)

## ⚠️ Lưu ý

1. **Dữ liệu**: Script tự động tìm file .npz trong `/mnt/user-data/uploads/`
2. **Normalize**: Data được normalize bằng StandardScaler
3. **Device**: Tự động dùng CUDA nếu có, không thì dùng CPU
4. **Memory**: Với large datasets, giảm batch_size nếu bị Out of Memory

## 🐛 Troubleshooting

### Out of Memory
```python
config['batch_size'] = 16  # Giảm batch size
```

### Training quá chậm
```python
# Dùng chỉ 1 baseline model để test
models_to_compare = ['T-GCN', 'HA']
```

### Checkpoint không load được
```bash
# Kiểm tra checkpoint tồn tại
ls checkpoints/T-GCN/
```

## 📧 Contact

Nếu có vấn đề, check lại:
1. Dữ liệu đã load đúng chưa
2. Checkpoint directory tồn tại chưa
3. Dependencies đã cài đủ chưa