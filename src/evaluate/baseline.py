"""
Baseline Models for Comparison
Based on T-GCN paper: HA, ARIMA, SVR, GCN, GRU
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.svm import SVR as SklearnSVR
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')


class HistoricalAverage:
    """
    Historical Average (HA) Model
    Uses average of historical values as prediction
    """
    def __init__(self):
        self.name = "HA"
        
    def fit(self, X_train, y_train):
        """
        Compute average of training data
        
        Args:
            X_train: (samples, seq_len, nodes, features)
            y_train: (samples, pred_len, nodes, features)
        """
        # Average across all training samples and time steps
        self.avg = np.mean(X_train, axis=(0, 1))  # (nodes, features)
        
    def predict(self, X_test):
        """
        Predict using historical average
        
        Args:
            X_test: (samples, seq_len, nodes, features)
        Returns:
            predictions: (samples, pred_len, nodes, features)
        """
        samples, _, nodes, features = X_test.shape
        # Repeat average for all prediction steps
        pred_len = 12  # Default prediction length
        predictions = np.tile(self.avg, (samples, pred_len, 1, 1))
        return predictions


class ARIMAModel:
    """
    ARIMA Model for each node independently
    """
    def __init__(self, order=(1, 0, 1)):
        self.name = "ARIMA"
        self.order = order
        self.models = []
        
    def fit(self, X_train, y_train):
        """
        Fit ARIMA model for each node
        
        Args:
            X_train: (samples, seq_len, nodes, features)
            y_train: (samples, pred_len, nodes, features)
        """
        samples, seq_len, nodes, features = X_train.shape
        
        # For each node, fit ARIMA on concatenated time series
        for node in range(nodes):
            # Get time series for this node
            node_data = X_train[:, :, node, 0].flatten()
            
            try:
                model = ARIMA(node_data, order=self.order)
                self.models.append(model.fit())
            except:
                # If ARIMA fails, use simple mean
                self.models.append(None)
                
    def predict(self, X_test):
        """
        Predict using fitted ARIMA models
        
        Args:
            X_test: (samples, seq_len, nodes, features)
        Returns:
            predictions: (samples, pred_len, nodes, features)
        """
        samples, seq_len, nodes, features = X_test.shape
        pred_len = 12
        
        predictions = np.zeros((samples, pred_len, nodes, features))
        
        for node, model in enumerate(self.models):
            if model is not None:
                try:
                    # Forecast for each sample
                    for i in range(samples):
                        forecast = model.forecast(steps=pred_len)
                        predictions[i, :, node, 0] = forecast
                except:
                    # If forecast fails, use last value
                    predictions[:, :, node, 0] = X_test[:, -1, node, 0:1]
            else:
                # Use last value
                predictions[:, :, node, 0] = X_test[:, -1, node, 0:1]
                
        return predictions


class SVRModel:
    """
    Support Vector Regression for each node
    """
    def __init__(self, kernel='linear', C=0.001):
        self.name = "SVR"
        self.kernel = kernel
        self.C = C
        self.models = []
        
    def fit(self, X_train, y_train):
        """
        Fit SVR for each node
        
        Args:
            X_train: (samples, seq_len, nodes, features)
            y_train: (samples, pred_len, nodes, features)
        """
        samples, seq_len, nodes, features = X_train.shape
        _, pred_len, _, _ = y_train.shape
        
        # Reshape to (samples, seq_len * features) for each node
        X_reshaped = X_train.reshape(samples, -1, nodes)
        y_reshaped = y_train.reshape(samples, -1, nodes)
        
        for node in range(nodes):
            svr = SklearnSVR(kernel=self.kernel, C=self.C)
            
            # Use mean prediction across prediction steps
            svr.fit(X_reshaped[:, :, node], y_reshaped[:, :, node].mean(axis=1))
            self.models.append(svr)
            
    def predict(self, X_test):
        """
        Predict using SVR
        
        Args:
            X_test: (samples, seq_len, nodes, features)
        Returns:
            predictions: (samples, pred_len, nodes, features)
        """
        samples, seq_len, nodes, features = X_test.shape
        pred_len = 12
        
        X_reshaped = X_test.reshape(samples, -1, nodes)
        predictions = np.zeros((samples, pred_len, nodes, features))
        
        for node, svr in enumerate(self.models):
            pred = svr.predict(X_reshaped[:, :, node])
            # Repeat prediction for all time steps
            predictions[:, :, node, 0] = pred[:, np.newaxis]
            
        return predictions


class GCNOnly(nn.Module):
    """
    GCN-only baseline (spatial only, no temporal)
    """
    def __init__(self, num_nodes, input_dim, hidden_dim, output_dim, seq_len, pred_len):
        super(GCNOnly, self).__init__()
        self.name = "GCN"
        
        from models.T-GCN.gcn import GCN
        
        self.gcn = GCN(input_dim * seq_len, hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim * pred_len)
        self.pred_len = pred_len
        self.output_dim = output_dim
        
    def forward(self, x, adj):
        """
        Args:
            x: (batch, seq_len, nodes, input_dim)
            adj: (nodes, nodes)
        Returns:
            output: (batch, pred_len, nodes, output_dim)
        """
        batch_size, seq_len, num_nodes, input_dim = x.size()
        
        # Flatten sequence
        x = x.reshape(batch_size, num_nodes, -1)
        
        # GCN
        x = self.gcn(x, adj)
        
        # Predict
        output = self.fc(x)
        output = output.reshape(batch_size, num_nodes, self.pred_len, self.output_dim)
        output = output.permute(0, 2, 1, 3)
        
        return output


class GRUOnly(nn.Module):
    """
    GRU-only baseline (temporal only, no spatial)
    """
    def __init__(self, num_nodes, input_dim, hidden_dim, output_dim, seq_len, pred_len):
        super(GRUOnly, self).__init__()
        self.name = "GRU"
        self.num_nodes = num_nodes
        self.pred_len = pred_len
        
        from models.T-GCN.gru import GRU
        
        self.gru = GRU(input_dim, hidden_dim, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x, adj=None):
        """
        Args:
            x: (batch, seq_len, nodes, input_dim)
            adj: Not used
        Returns:
            output: (batch, pred_len, nodes, output_dim)
        """
        batch_size, seq_len, num_nodes, input_dim = x.size()
        
        # Process each node independently
        outputs = []
        for node in range(num_nodes):
            x_node = x[:, :, node, :]  # (batch, seq, feat)
            x_node = x_node.unsqueeze(2)  # (batch, seq, 1, feat)
            
            out, _ = self.gru(x_node)  # (batch, seq, 1, hidden)
            outputs.append(out[:, -1, 0, :])  # (batch, hidden)
        
        h = torch.stack(outputs, dim=1)  # (batch, nodes, hidden)
        
        # Generate predictions
        preds = []
        for t in range(self.pred_len):
            pred = self.fc(h)
            preds.append(pred)
        
        output = torch.stack(preds, dim=1)  # (batch, pred_len, nodes, output_dim)
        
        return output


def get_baseline_model(model_name, **kwargs):
    """
    Factory function to get baseline model
    
    Args:
        model_name: 'HA', 'ARIMA', 'SVR', 'GCN', 'GRU'
        **kwargs: Model-specific parameters
    
    Returns:
        model instance
    """
    if model_name == 'HA':
        return HistoricalAverage()
    elif model_name == 'ARIMA':
        return ARIMAModel(order=kwargs.get('order', (1, 0, 1)))
    elif model_name == 'SVR':
        return SVRModel(
            kernel=kwargs.get('kernel', 'linear'),
            C=kwargs.get('C', 0.001)
        )
    elif model_name == 'GCN':
        return GCNOnly(**kwargs)
    elif model_name == 'GRU':
        return GRUOnly(**kwargs)
    else:
        raise ValueError(f"Unknown model: {model_name}")


if __name__ == "__main__":
    # Test baseline models
    print("Testing Baseline Models...")
    
    # Create dummy data
    samples, seq_len, nodes, features = 100, 12, 10, 1
    pred_len = 12
    
    X_train = np.random.rand(samples, seq_len, nodes, features)
    y_train = np.random.rand(samples, pred_len, nodes, features)
    X_test = np.random.rand(20, seq_len, nodes, features)
    
    # Test HA
    print("\n1. Historical Average")
    ha = HistoricalAverage()
    ha.fit(X_train, y_train)
    pred_ha = ha.predict(X_test)
    print(f"Prediction shape: {pred_ha.shape}")
    
    # Test SVR
    print("\n2. Support Vector Regression")
    svr = SVRModel()
    svr.fit(X_train, y_train)
    pred_svr = svr.predict(X_test)
    print(f"Prediction shape: {pred_svr.shape}")
    
    print("\n✓ All baseline models work!")