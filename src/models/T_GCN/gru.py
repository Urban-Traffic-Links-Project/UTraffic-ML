"""
Gated Recurrent Unit (GRU) Module
Based on T-GCN paper (Zhao et al., 2019)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math

class GRUCell(nn.Module):
    """
    Custom GRU Cell
    
    Equations:
        r_t = σ(W_r·x_t + U_r·h_{t-1} + b_r)          # Reset gate
        z_t = σ(W_z·x_t + U_z·h_{t-1} + b_z)          # Update gate
        c_t = tanh(W_c·x_t + U_c·(r_t ⊙ h_{t-1}) + b_c)  # Candidate
        h_t = z_t ⊙ h_{t-1} + (1 - z_t) ⊙ c_t        # New hidden state
    """
    def __init__(self, input_size, hidden_size, bias=True):
        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        # --- 1. RESET GATE (Cổng Reset) ---
        self.weight_ir = Parameter(torch.FloatTensor(input_size, hidden_size))
        self.weight_hr = Parameter(torch.FloatTensor(hidden_size, hidden_size))

        # --- 2. UPDATE GATE (Cổng Update) ---
        self.weight_iz = Parameter(torch.FloatTensor(input_size, hidden_size))
        self.weight_hz = Parameter(torch.FloatTensor(hidden_size, hidden_size))
        
        # --- 3. CANDIDATE (Thông tin mới) ---
        self.weight_in = Parameter(torch.FloatTensor(input_size, hidden_size))
        self.weight_hn = Parameter(torch.FloatTensor(hidden_size, hidden_size))
        
        # --- Bias (Tùy chọn) ---
        if bias:
            self.bias_ir = Parameter(torch.FloatTensor(hidden_size))
            self.bias_hr = Parameter(torch.FloatTensor(hidden_size))
            self.bias_iz = Parameter(torch.FloatTensor(hidden_size))
            self.bias_hz = Parameter(torch.FloatTensor(hidden_size))
            self.bias_in = Parameter(torch.FloatTensor(hidden_size))
            self.bias_hn = Parameter(torch.FloatTensor(hidden_size))
        else:
            self.register_parameter('bias_ir', None)
            self.register_parameter('bias_hr', None)
            self.register_parameter('bias_iz', None)
            self.register_parameter('bias_hz', None)
            self.register_parameter('bias_in', None)
            self.register_parameter('bias_hn', None)
            
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters"""
        std = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-std, std)

        # ---- Bias (nếu có) ----
        if self.bias:
            nn.init.uniform_(self.bias_ir, -std, std)
            nn.init.uniform_(self.bias_hr, -std, std)
            nn.init.uniform_(self.bias_iz, -std, std)
            nn.init.uniform_(self.bias_hz, -std, std)
            nn.init.uniform_(self.bias_in, -std, std)
            nn.init.uniform_(self.bias_hn, -std, std)
        
    def forward(self, x, hidden):
        """
        Args:
            x: (batch_size, num_nodes, input_size)
            hidden: (batch_size, num_nodes, hidden_size)
        Returns:
            new_h: (batch_size, num_nodes, hidden_size)
        """
        # 1. Tính cổng Reset (r)
        # r = sigmoid(W_ir*x + b_ir + W_hr*h + b_hr)
        r = torch.sigmoid(
            x @ self.weight_ir + self.bias_ir + 
            hidden @ self.weight_hr + self.bias_hr
        )

        # 2. Tính cổng Update (z)
        # z_t = σ(W_z·x_t + U_z·h_{t-1} + b_z)
        z = torch.sigmoid(
            x @ self.weight_iz + self.bias_iz +
            hidden @ self.weight_hz + self.bias_hz
        )

        # 3. Tính thông tin mới tiềm năng (n - candidate)
        # Lưu ý: r * hidden -> Áp dụng cổng reset lên ký ức cũ
        # c_t = tanh(W_c·x_t + U_c·(r_t ⊙ h_{t-1}) + b_c)
        n = torch.tanh(
            x @ self.weight_in + self.bias_in + 
            r * (hidden @ self.weight_hn + self.bias_hn)
        )

        # 4. Tính Hidden State mới (h_t)
        # Công thức: (1 - z) * n + z * hidden
        # Ý nghĩa: Nếu z gần 1 -> Giữ lại ký ức cũ. Nếu z gần 0 -> Dùng thông tin mới.
        # h_t = z_t ⊙ h_{t-1} + (1 - z_t) ⊙ c_t
        new_h = (1 - z) * n + z * hidden
        
        return new_h

class GRU(nn.Module):
    """
    Multi-layer GRU
    """
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, 
                 batch_first=False, dropout=0.0):
        super(GRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        
        # Tạo danh sách các Cell (cho trường hợp nhiều lớp chồng lên nhau)
        self.gru_cells = nn.ModuleList()
        for layer in range(num_layers):
            # Input của layer sau là output (hidden) của layer trước
            input_dim = input_size if layer == 0 else hidden_size
            self.gru_cells.append(GRUCell(input_dim, hidden_size, bias))
        
    def forward(self, x, hidden=None):
        """
        Args:
            x: (seq_len, batch, nodes, input_size) if batch_first=False
            (batch, seq_len, nodes, input_size) if batch_first=True
            hidden: (num_layers, batch, nodes, hidden_size) or None
        Returns:
            output: same format as x but with hidden_size
            hidden: (num_layers, batch, nodes, hidden_size)
        """
        if self.batch_first:
            x = x.transpose(0, 1)  # -> (seq, batch, nodes, feat)

        seq_len, batch_size, num_nodes, _ = x.size()
        
        # Khởi tạo hidden state ban đầu bằng 0 nếu chưa có
        if hidden is None:
            hidden_list = [
                torch.zeros(batch_size, num_nodes, self.hidden_size, device=x.device)
                for _ in range(self.num_layers)
            ]
        else:
            # Avoid in-place writes to a stacked hidden tensor (breaks autograd).
            hidden_list = list(hidden.unbind(0))
        
        output_seq = []
        # --- VÒNG LẶP THỜI GIAN ---
        for t in range(seq_len):
            x_t = x[t]

            # Chạy qua từng lớp GRU (nếu num_layers > 1)
            for layer in range(self.num_layers):
                h_prev = hidden_list[layer]  # Ký ức cũ của lớp này

                # Gọi GRUCell
                h_new = self.gru_cells[layer](x_t, h_prev)

                # Cập nhật input cho lớp kế tiếp (hoặc dùng cho bước t+1)
                x_t = h_new
                hidden_list[layer] = h_new
                
                # Dropout between layers
                if layer < self.num_layers - 1 and self.dropout > 0:
                    x_t = F.dropout(x_t, p=self.dropout, training=self.training)
            
            # Lưu lại kết quả bước t
            output_seq.append(h_new)
            
        # Gom kết quả lại thành tensor
        output = torch.stack(output_seq)
        
        if self.batch_first:
            output = output.transpose(0, 1) # Trả về dạng (Batch, Seq, ...)
        
        hidden_out = torch.stack(hidden_list, dim=0)
        return output, hidden_out

if __name__ == "__main__":
    # Giả lập
    batch = 32
    seq_len = 12
    nodes = 100
    feat = 64
    hidden = 32
    
    model = GRU(feat, hidden, num_layers=1, batch_first=True)
    
    # Input giả: (32 mẫu, 12 bước thời gian, 100 nút, 64 đặc trưng)
    x = torch.randn(batch, seq_len, nodes, feat)
    
    output, h_last = model(x)
    
    print("Input:", x.shape)
    print("Output:", output.shape) 
    # Mong đợi: [32, 12, 100, 32] (Hidden size thay thế Feature size)