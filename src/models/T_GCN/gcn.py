"""
Graph Convolutional Network (GCN) Module
Based on T-GCN paper (Zhao et al., 2019)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math

class GraphConvolution(nn.Module):
    """
    Simple GCN layer: (batch_size, num_nodes, in_features) -> (batch_size, num_nodes, out_features)
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Tạo ma trận trọng số W (Learnable Weight)
        # Kích thước: [đầu_vào, đầu_ra]
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))

        # Tạo vector Bias (nếu cần)
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

        # Gọi hàm khởi tạo giá trị ngẫu nhiên (để model học tốt hơn từ đầu)
        self.reset_parameters()

    def reset_parameters(self):
        """Xavier uniform initialization"""
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain("tanh"))
        
        # Tính độ lệch chuẩn dựa trên kích thước weight
        stdv = 1. / math.sqrt(self.weight.size(1))

        # Khởi tạo ngẫu nhiên trong khoảng [-stdv, stdv]
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
    

    def forward(self, input, adj):    
        """
        Args:
            input: (batch_size, num_nodes, in_features)
            adj: (num_nodes, num_nodes) - normalized adjacency matrix
        Returns:
            output: (batch_size, num_nodes, out_features)
        """
        # Bước 1: Nhân đặc trưng với trọng số (Linear Transformation)
        # Công thức: support = X * W
        support = input @ self.weight

        # Bước 2: Nhân với ma trận kề để lan truyền thông tin (Graph Convolution)
        # Công thức: output = A * support
        # Lúc này, thông tin từ các nút hàng xóm sẽ được cộng gộp vào nút hiện tại
        output = adj @ support

        # Bước 3: Cộng thêm bias (nếu có)
        if self.bias is not None:
            output += self.bias

        return output
        
class GCN(nn.Module):
    """
    2-layer Graph Convolutional Network
    Equation: f(X, A) = σ(Â·ReLU(Â·X·W0)·W1)
    """
    def __init__(self, in_features, hidden_features, out_features, dropout=0.0):
        super(GCN, self).__init__()
        
        # Tầng 1: Từ Input -> Hidden
        self.gc1 = GraphConvolution(in_features, hidden_features)
        
        # Tầng 2: Từ Hidden -> Output
        self.gc2 = GraphConvolution(hidden_features, out_features)

        self.dropout = dropout
    
    def forward(self, x, adj):
        # Qua tầng 1 -> Dùng hàm kích hoạt ReLU -> Dropout (để chống Overfitting)
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)

        # Qua tầng 2 -> Ra kết quả
        x = self.gc2(x, adj)

        return x

def normalize_adj(adj):
    """
    Normalize adjacency matrix: D^(-1/2) * A * D^(-1/2)
    where A = A + I (with self-loops)
    
    Args:
        adj: (num_nodes, num_nodes) adjacency matrix (can be numpy or tensor)
    Returns:
        normalized adjacency matrix (same type as input)
    """
    # Chuyển sang Tensor nếu đầu vào là Numpy
    is_numpy = False
    if not torch.is_tensor(adj):
        is_numpy = True
        adj = torch.FloatTensor(adj)
    
    # 1. Thêm Self-loop (A = A + I)
    # Để mỗi nút giữ lại thông tin của chính nó khi cộng gộp hàng xóm
    adj += torch.eye(adj.size(0), device=adj.device)

    # 2. Tính ma trận bậc (Degree Matrix - D)
    rowsum = adj.sum(1)

    # 3. Tính D^(-1/2)
    d_inv_sqrt = torch.pow(rowsum, -1/2).flatten()
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.    # Xử lý chia cho 0
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)     # Tạo ma trận đường chéo

    # 4. Nhân ma trận: D^(-1/2) * A * D^(-1/2)
    adj_normalized = d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt

    # Convert back to numpy if needed
    if is_numpy:
        adj_normalized = adj_normalized.numpy()

    return adj_normalized


if __name__ == "__main__":
    # Test
    batch_size = 32
    num_nodes = 114
    in_features = 1
    hidden_features = 64
    out_features = 64
    
    model = GCN(in_features, hidden_features, out_features)
    
    x = torch.randn(batch_size, num_nodes, in_features)
    adj = torch.rand(num_nodes, num_nodes)
    adj = (adj + adj.T) / 2
    adj = normalize_adj(adj)
    
    output = model(x, adj)
    
    print(f"Input shape: {x.shape}")
    print(f"Adjacency matrix shape: {adj.shape}")
    print(f"Output shape: {output.shape}")