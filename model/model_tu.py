import torch
import math
import torch.nn as nn
from torch import Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class MultiScaleConv(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.conv3 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model)
        self.conv5 = nn.Conv1d(d_model, d_model, kernel_size=5, padding=2, groups=d_model)
        self.conv7 = nn.Conv1d(d_model, d_model, kernel_size=7, padding=3, groups=d_model)
        self.relu = nn.ReLU()
        # self.relu = nn.GELU()
        self.dropout = nn.Dropout(0.1)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        x = x.transpose(1, 2)
        out = (self.conv3(x) + self.conv5(x) + self.conv7(x)) / 3
        out = out.transpose(1, 2)
        out = self.dropout(self.relu(out))
        out = self.norm(out + identity)  # 残差+归一化
        return out
    
class LightweightConv(nn.Module):
    """轻量级卷积模块：局部建模"""
    def __init__(self, d_model):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        # x shape: [B, S, D]
        x = x.transpose(1, 2)  # -> [B, D, S]
        x = self.conv(x)
        return x.transpose(1, 2)  # -> [B, S, D]
    

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 2048):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[:x.size(1)].transpose(0, 1)
        return self.dropout(x)


class RegressionHead(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, 1)
            # nn.Linear(d_model, d_model // 2),
            # nn.GELU(),
            # nn.Linear(d_model // 2, d_model // 4),
            # nn.GELU(),
            # nn.Linear(d_model // 4, 1)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(x)

class Transformer(nn.Module):
    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.1):
        super().__init__()
        self.model_type = 'LWCResAttTransformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout, batch_first=True,activation="gelu")
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.token_encoder = nn.Embedding(ntoken, d_model)
        self.d_model = d_model

        self.light_weight_conv = LightweightConv(d_model)

        self.lstm = nn.LSTM(d_model, d_model, num_layers=1, batch_first=True, bidirectional=False)
        self.init_weights()


    def init_weights(self) -> None:
        nn.init.xavier_normal_(self.token_encoder.weight)

    def forward(self, src: Tensor) -> Tensor:
        # 输入 shape: [batch_size, seq_len]
        src = self.token_encoder(src) * math.sqrt(self.d_model)  # [B, S, D]
        src = self.pos_encoder(src)                              # [B, S, D]
      
        # src = self.multi_scale_conv(src)                        # [B, S, D]                              # [B, S, D]
        src = self.light_weight_conv(src) 
        lstm_out, _ = self.lstm(src)  # 现在可以安全解包
        src = src + lstm_out  # 残差连接

        output = self.transformer_encoder(src)                   # [B, S, D]
        return output

class TransformerRegressor(nn.Module):
    def __init__(self, transformer, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.transformer = transformer
        self.regressionHead = RegressionHead(d_model)

    def forward(self, src: Tensor) -> Tensor:
        output = self.transformer(src)              # [B, S, D]
        output = self.regressionHead(output[:, 0:1, :])  # [B, 1, D] → [B, 1]
        return output