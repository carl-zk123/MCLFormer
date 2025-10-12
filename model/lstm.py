import torch
import torch.nn as nn
    
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, num_layers, dropout=0.1):
        super(LSTMModel, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # 1. Embedding层
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
        
        # 2. 多层双向LSTM（层间Dropout仅在num_layers>1时生效）
        self.LSTM = nn.LSTM(
            embedding_size, 
            hidden_size, 
            num_layers, 
            batch_first=True, 
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0  # 关键修改
        )
        
        # 3. 全连接层
        self.fc_1 = nn.Linear(hidden_size * 2, hidden_size)  # 双向LSTM输出维度是hidden_size*2
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.fc_2 = nn.Linear(hidden_size, 1)

    def init_hidden(self, batch_size):
        """初始化多层LSTM的隐藏状态（可选）"""
        device = next(self.parameters()).device
        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(device)  # 双向*层数
        c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(device)
        return (h0, c0)

    def forward(self, x):
        # 处理变长序列
        x_bool = (x == 0).int()
        x_lens = torch.argmax(x_bool, dim=1)
        x_lens[x_lens == 0] = x.shape[1]
        x_lens = x_lens.to(dtype=torch.int64).to('cpu')
        
        # 前向传播
        x = self.embedding(x)
        packed_output = nn.utils.rnn.pack_padded_sequence(x, x_lens, batch_first=True, enforce_sorted=False)
        
        # 可选：传入自定义初始状态
        hidden = self.init_hidden(x.size(0))
        _, (hidden, _) = self.LSTM(packed_output, hidden)  # 使用初始状态
        
        # 获取最终隐藏状态（双向LSTM需合并两个方向）
        out = hidden.view(self.num_layers, 2, -1, self.hidden_size)[-1]  # 取最后一层
        out = torch.cat([out[0], out[1]], dim=1)  # 合并前向和反向
        
        # 全连接层
        out = self.fc_1(out)
        out = self.dropout(out)
        out = self.relu(out)
        out = self.fc_2(out)
        return out