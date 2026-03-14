"""
T-GCN: Temporal Graph Convolutional Network
Based on: Zhao et al. "T-GCN: A Temporal Graph Convolutional Network for Traffic Prediction" (2019)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .gcn import GCN, GraphConvolution
from .gru import GRUCell

class TGCNCell(nn.Module):
    """
    T-GCN Cell: combines GCN and GRU
    
    Process:
        1. GCN extracts spatial features from current input
        2. GRU captures temporal dependencies with previous hidden state
    """
    def __init__(self, num_nodes, input_dim, hidden_dim, gcn_hidden_dim=64):
        super(TGCNCell, self).__init__()
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # 1. Bộ phận Không gian (GCN)
        # GCN for spatial features
        # Nhiệm vụ: Biến đổi input ban đầu thành đặc trưng có thông tin hàng xóm
        self.gcn = GCN(input_dim, gcn_hidden_dim, hidden_dim)
        
        # 2. Bộ phận Thời gian (GRU Cell)
        # Nhiệm vụ: Nhớ và cập nhật trạng thái
        # Input size của GRU bây giờ chính là output của GCN (hidden_dim)
        self.gru_cell = GRUCell(hidden_dim, hidden_dim)
    
    
    def forward(self, x, h, adj):
        """
        Args:
            x: (batch_size, num_nodes, input_dim)
            h: (batch_size, num_nodes, hidden_dim) - hidden state
            adj: (num_nodes, num_nodes) - adjacency matrix
        Returns:
            new_h: (batch_size, num_nodes, hidden_dim)
        """
        # Bước A: Nhìn sang hàng xóm (GCN)
        # x shape: (Batch, Nodes, Input_Dim)
        # adj shape: (Nodes, Nodes)
        x_gcn = self.gcn(x, adj)

        # Bước B: Nhớ lại quá khứ (GRU)
        # x_gcn shape: (Batch, Nodes, Hidden_Dim)
        # h (ký ức cũ): (Batch, Nodes, Hidden_Dim)
        new_h = self.gru_cell(x_gcn, h)
        
        return new_h

class TGCN(nn.Module):
    """
    T-GCN Model for Traffic Prediction
    
    Architecture:
        Input -> [T-GCN Cell x seq_len] -> FC Layer -> Output
    """
    def __init__(self, num_nodes, input_dim, hidden_dim, output_dim, 
                 seq_len, pred_len, gcn_hidden_dim=64):
        """
        Args:
            num_nodes: Number of nodes in graph Sẽ cập nhật khi chạy
            input_dim: Input feature dimension (usually 1)
            hidden_dim: Hidden dimension of GRU
            output_dim: Output dimension (usually 1)
            seq_len: Độ dài lịch sử (ví dụ: 12 bước)
            pred_len: Độ dài muốn dự báo (ví dụ: 3 bước)
            gcn_hidden_dim: Hidden dimension of GCN
        """
        super(TGCN, self).__init__()
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        # Khởi tạo Cell lai ghép
        self.tgcn_cell = TGCNCell(num_nodes, input_dim, hidden_dim, gcn_hidden_dim)
        
        # Lớp cuối cùng để đưa ra con số dự báo (ví dụ: vận tốc km/h)
        # Từ bộ nhớ 64 chiều -> nén xuống 1 con số (output_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, adj):
        """
        Args:
            x: (batch_size, seq_len, num_nodes, input_dim)
            adj: (num_nodes, num_nodes)
        Returns:
            output: (batch_size, pred_len, num_nodes, output_dim)
        """
        # x shape: (Batch, Seq_len, Nodes, Features)
        batch_size, seq_len, num_nodes, _ = x.size()
        self.num_nodes = num_nodes
        
        # 1. Khởi tạo ký ức rỗng (h0)
        h = torch.zeros(batch_size, num_nodes, self.hidden_dim, device=x.device)
        
        # 2. GIAI ĐOẠN ĐỌC HIỂU (Encoder)
        # Lặp qua từng thời điểm trong quá khứ
        for t in range(seq_len):
            # Lấy dữ liệu tại thời điểm t
            x_t = x[:, t, :, :] 
            # Cập nhật ký ức
            # Passing (x, adj, h) will swap adjacency and hidden state, causing matmul shape errors.
            h = self.tgcn_cell(x_t, h, adj)
        
        # Lúc này, 'h' chứa đựng toàn bộ tinh hoa của 12 bước quá khứ
        
        # 3. GIAI ĐOẠN DỰ BÁO (Decoder / Prediction)
        # Chúng ta sẽ dự báo từng bước một
        # Generate predictions
        predictions = []
        for t in range(self.pred_len):
            # Predict next step
            output = self.fc(h)
            predictions.append(output)
            
            # Use prediction as input for next step (auto-regressive)
            h = self.tgcn_cell(output, h, adj)
        
        # Stack: (batch, pred_len, nodes, output_dim)
        outputs = torch.stack(predictions, dim=1)
        
        return outputs
        
    def init_hidden(self, batch_size, device):
        """Initialize hidden state"""
        return torch.zeros(batch_size, self.num_nodes, self.hidden_dim, device=device)


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    

if __name__ == "__main__":
    # Test
    num_nodes = 114
    input_dim = 1
    hidden_dim = 64
    output_dim = 1
    seq_len = 12
    pred_len = 12
    batch_size = 32
    
    model = TGCN(num_nodes, input_dim, hidden_dim, output_dim, seq_len, pred_len)
    
    x = torch.randn(batch_size, seq_len, num_nodes, input_dim)
    adj = torch.rand(num_nodes, num_nodes)
    adj = (adj + adj.T) / 2
    adj = adj + torch.eye(num_nodes)
    
    output = model(x, adj)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Parameters: {count_parameters(model):,}")