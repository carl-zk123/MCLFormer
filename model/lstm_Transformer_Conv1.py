import torch
import torch.nn as nn
import math
from torch import Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer

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

class MultiScaleConv(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.conv3 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(d_model, d_model, kernel_size=5, padding=2)
        self.conv1x1 = nn.Conv1d(d_model * 2, d_model, kernel_size=1)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
        self.act = nn.GELU()

    def forward(self, x):
        identity = x
        x = x.transpose(1, 2)  # [B, D, S]
        c3 = self.conv3(x)
        c5 = self.conv5(x)
        out = torch.cat([c3, c5], dim=1)  # [B, 2D, S]
        out = self.conv1x1(out)  # [B, D, S]
        out = out.transpose(1, 2)  # [B, S, D]
        return self.norm(self.dropout(self.act(out)) + identity)

class AttentionPooling(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.att_fc = nn.Linear(d_model, 1)

    def forward(self, x):
        weights = torch.softmax(self.att_fc(x), dim=1)  # [B, S, 1]
        pooled = (weights * x).sum(dim=1)  # [B, D]
        return pooled

class RegressionHead(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1)
        )

    def forward(self, x):
        return self.mlp(x)

class Transformer(nn.Module):
    def __init__(self, ntoken, d_model, nhead, d_hid, nlayers, dropout=0.1):
        super().__init__()
        self.token_encoder = nn.Embedding(ntoken, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        self.conv_branch = MultiScaleConv(d_model)
        self.lstm_branch = nn.LSTM(d_model, d_model, num_layers=1, batch_first=True, bidirectional=False)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout, batch_first=True)
        self.transformer_branch = TransformerEncoder(encoder_layers, nlayers)

        self.norm = nn.LayerNorm(d_model)
        self.pooling = AttentionPooling(d_model)
        self.reg_head = RegressionHead(d_model)

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.token_encoder.weight)

    def forward(self, src):
        x = self.token_encoder(src) * math.sqrt(self.token_encoder.embedding_dim)
        x = self.pos_encoder(x)

        conv_out = self.conv_branch(x)
        lstm_out, _ = self.lstm_branch(x)
        trans_out = self.transformer_branch(x)

        fused = self.norm(conv_out + lstm_out + trans_out)
        return fused

class TransformerRegressor(nn.Module):
    def __init__(self, transformer, d_model: int):
        super().__init__()
        self.transformer = transformer
        self.pooling = AttentionPooling(d_model)
        self.regressionHead = RegressionHead(d_model)

    def forward(self, src: Tensor) -> Tensor:
        output = self.transformer(src)              # [B, S, D]
        pooled = self.pooling(output)               # [B, D]
        return self.regressionHead(pooled)          # [B, 1]
