# T-GCN: Temporal Graph Convolutional Network for Traffic Prediction

Implementation of T-GCN model based on the paper "T-GCN: A Temporal Graph Convolutional Network for Traffic Prediction" by Ling Zhao et al.

## 📚 Overview

This implementation combines **Graph Convolutional Network (GCN)** and **Gated Recurrent Unit (GRU)** to capture both **spatial** and **temporal** dependencies in traffic data.

### Key Features

- ✅ **Spatial Modeling**: GCN captures complex topological structures of road networks
- ✅ **Temporal Modeling**: GRU learns dynamic changes in traffic data over time
- ✅ **Correlation Analysis**: Comprehensive correlation matrices similar to TH-Hierarchical paper
- ✅ **Multi-step Prediction**: Support for forecasting multiple future timesteps

## 📁 Project Structure

```
DTC-STGCN/
├── t_gcn_model.py           # T-GCN model implementation
├── correlation_analysis.py   # Correlation analysis tools
├── main_tgcn_analysis.py    # Main training and evaluation script
└── README.md                 # This file
```

## 🏗️ Model Architecture

### T-GCN Cell

The T-GCN cell integrates graph convolution into GRU gates:

```
Update gate:    ut = σ(Wu[f(A,Xt), ht-1] + bu)
Reset gate:     rt = σ(Wr[f(A,Xt), ht-1] + br)
Candidate:      ct = tanh(Wc[f(A,Xt), (rt ⊙ ht-1)] + bc)
Hidden state:   ht = ut ⊙ ht-1 + (1-ut) ⊙ ct
```

Where:
- `f(A,Xt)` is the graph convolution operation
- `A` is the normalized adjacency matrix
- `Xt` is the input at timestep t
- `ht` is the hidden state

### Graph Convolution

2-layer GCN as described in the paper:

```
f(X,A) = σ(ÃReLU(ÃXW0)W1)
```

Where:
- `Ã = D^(-1/2) * (A + I) * D^(-1/2)` is the normalized adjacency matrix
- `W0, W1` are learnable weight matrices

## 🚀 Quick Start

### 1. Installation

```bash
pip install torch numpy pandas matplotlib seaborn scipy scikit-learn
```

### 2. Prepare Your Data

Your data should be in NPZ format with:
- **Graph structure**: `graph_structure.npz` containing node features, edge index, coordinates
- **Traffic features**: `traffic_features.npz` containing time series data

### 3. Run Training and Analysis

```bash
python main_tgcn_analysis.py
```

This will:
1. Load your traffic data
2. Train the T-GCN model
3. Generate predictions
4. Compute comprehensive correlation matrices
5. Save all results and visualizations

### 4. Use the Model Programmatically

```python
import torch
import numpy as np
from t_gcn_model import TGCN, normalize_adjacency_matrix

# Model parameters
num_nodes = 514
input_dim = 1
hidden_dim = 64
output_dim = 1
seq_len = 12

# Create model
model = TGCN(
    num_nodes=num_nodes,
    input_dim=input_dim,
    hidden_dim=hidden_dim,
    output_dim=output_dim
)

# Prepare data
x = torch.randn(32, seq_len, num_nodes, input_dim)  # (batch, seq_len, nodes, features)
adj = normalize_adjacency_matrix(your_adjacency_matrix)
adj = torch.FloatTensor(adj)

# Forward pass
output = model(x, adj)  # (batch, nodes, output_dim)
```

## 📊 Correlation Analysis

The implementation includes comprehensive correlation analysis tools similar to the TH-Hierarchical paper:

### 1. Node-to-Node Correlation

Computes Pearson correlation between all pairs of nodes:

```python
from correlation_analysis import CorrelationAnalyzer

analyzer = CorrelationAnalyzer()
corr_matrix, p_values = analyzer.compute_node_correlation_matrix(data)
```

### 2. Spatial Correlation

Correlation weighted by adjacency matrix:

```python
spatial_corr = analyzer.compute_spatial_correlation(data, adj_matrix)
```

### 3. Temporal Autocorrelation

Autocorrelation for each node at different lags:

```python
autocorr = analyzer.compute_temporal_autocorrelation(data, max_lag=10)
```

### 4. Prediction-Truth Correlation

Correlation between model predictions and ground truth:

```python
pred_corr, node_corr = analyzer.compute_prediction_correlation(
    predictions, ground_truth
)
```

### 5. Regional Correlation

If hierarchical structure is provided:

```python
regional_corr = analyzer.compute_regional_correlation(data, region_labels)
```

### Generate Full Report

```python
results = analyzer.generate_correlation_report(
    data=your_data,
    adj_matrix=adjacency_matrix,
    predictions=model_predictions,
    ground_truth=targets,
    region_labels=regions,  # optional
    save_dir='./correlation_analysis'
)
```

This generates:
- 📈 Correlation matrices (numpy arrays)
- 🖼️ Heatmap visualizations
- 📝 Summary report (text file)
- 📊 Autocorrelation plots

## 📈 Output Files

After running the analysis, you'll get:

```
tgcn_traffic_analysis/
├── best_model.pth                    # Trained model weights
├── training_curves.png               # Loss curves
├── predictions.npy                   # Model predictions
├── targets.npy                       # Ground truth
└── correlation_analysis/
    ├── node_correlation_matrix.npy
    ├── node_correlation_matrix.png
    ├── spatial_correlation_matrix.npy
    ├── spatial_correlation_matrix.png
    ├── temporal_autocorrelation.npy
    ├── temporal_autocorrelation.png
    ├── prediction_correlation_matrix.npy
    ├── prediction_correlation_matrix.png
    └── correlation_summary.txt       # Summary statistics
```

## 🔧 Model Configuration

Key hyperparameters in `main_tgcn_analysis.py`:

```python
train_tgcn_model(
    train_data=train_data,
    test_data=test_data,
    adj_matrix=adj_matrix,
    seq_len=12,           # Input sequence length
    horizon=1,            # Prediction horizon
    hidden_dim=64,        # Hidden layer dimension
    num_epochs=100,       # Training epochs
    batch_size=32,        # Batch size
    learning_rate=0.001,  # Learning rate
    device='cuda'         # 'cuda' or 'cpu'
)
```

## 📊 Evaluation Metrics

The model is evaluated using:

- **MSE** (Mean Squared Error)
- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Squared Error)
- **R²** (Coefficient of Determination)
- **Accuracy** (as defined in T-GCN paper)

## 🎯 Use Cases

This implementation can be used for:

1. **Traffic Speed Prediction** ✅
2. **Traffic Flow Forecasting** ✅
3. **Traffic Density Estimation** ✅
4. **Any spatio-temporal graph data** ✅

## 📖 References

### Main Paper
```
Zhao, L., Song, Y., Zhang, C., Liu, Y., Wang, P., Lin, T., Deng, M., & Li, H. (2020).
T-GCN: A temporal graph convolutional network for traffic prediction.
IEEE Transactions on Intelligent Transportation Systems, 21(9), 3848-3858.
```

### Correlation Analysis Inspired By
```
Huang, L., Liu, X., Huang, S., Wang, C., Tu, W., Xie, J., Tang, S., & Xie, W. (2021).
Temporal Hierarchical Graph Attention Network for Traffic Prediction.
ACM Transactions on Intelligent Systems and Technology, 12(6), Article 68.
```

## 💡 Tips for Best Results

1. **Normalize your data**: Always normalize traffic features to [0,1] or use z-score
2. **Tune sequence length**: Try different `seq_len` values (6, 12, 24)
3. **Adjust hidden dimension**: Larger networks for complex patterns
4. **Use early stopping**: Monitor validation loss to prevent overfitting
5. **Experiment with GCN depth**: 2-layer GCN works well in most cases

## 🐛 Troubleshooting

### CUDA Out of Memory
```python
# Reduce batch size or hidden dimension
batch_size = 16  # Instead of 32
hidden_dim = 32  # Instead of 64
```

### Poor Performance
1. Check data normalization
2. Ensure adjacency matrix is properly normalized
3. Try different hyperparameters
4. Increase training epochs
5. Add more GCN layers

### Data Loading Issues
```python
# Make sure data shapes are correct
print(f"Graph nodes: {graph_data['node_features'].shape[0]}")
print(f"Traffic samples: {len(traffic_data['average_speed'])}")
```

## 🤝 Contributing

Feel free to:
- Report bugs
- Suggest features
- Submit pull requests
- Share your results

## 📧 Contact

For questions or issues, please open an issue on the repository.

## 📜 License

This implementation is provided for research and educational purposes.

---

**Happy Predicting! 🚗📈**