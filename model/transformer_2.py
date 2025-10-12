import pandas as pd
import logging
import numpy as np
import torch
import math
from typing import Tuple
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class ChannelAttention(nn.Module):
    """通道注意力机制"""
    def __init__(self, d_model, reduction_ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(d_model // reduction_ratio, d_model)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        b, s, d = x.size()
        
        # 平均池化分支
        avg_out = self.avg_pool(x.transpose(1, 2)).view(b, d)
        avg_out = self.fc(avg_out)
        
        # 最大池化分支
        max_out = self.max_pool(x.transpose(1, 2)).view(b, d)
        max_out = self.fc(max_out)
        
        # 合并注意力权重
        out = avg_out + max_out
        out = self.sigmoid(out).view(b, 1, d)
        
        return x * out.expand_as(x)


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
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class regressoionHead(nn.Module):
    def __init__(self, d_embedding: int):
        super().__init__()
        self.layer1 = nn.Linear(d_embedding, d_embedding//2)
        self.layer2 = nn.Linear(d_embedding//2, d_embedding//4)
        self.layer3 = nn.Linear(d_embedding//4, d_embedding//8)
        self.layer4 = nn.Linear(d_embedding//8, 1)
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        return self.layer4(x)


class Transformer(nn.Module):
    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.1, use_channel_att: bool = True):
        super().__init__()
        self.model_type = 'EnhancedTransformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.token_encoder = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        
        # 仅保留通道注意力
        self.channel_attention = ChannelAttention(d_model) if use_channel_att else None
        
        self.init_weights()

    def init_weights(self) -> None:
        nn.init.xavier_normal_(self.token_encoder.weight)

    def forward(self, src: Tensor) -> Tensor:
        src = self.token_encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        
        # 应用通道注意力
        if self.channel_attention is not None:
            src = self.channel_attention(src)
        
        # Transformer编码
        output = self.transformer_encoder(src)
        
        return output


class TransformerRegressor(nn.Module):
    def __init__(self, transformer, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.transformer = transformer
        self.regressionHead = regressoionHead(d_model)

    def forward(self, src: Tensor) -> Tensor:
        output = self.transformer(src)
        output = self.regressionHead(output[:, 0:1, :])
        return output


class TransformerPretrain(nn.Module):
    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.1, use_attention: bool = True):
        super().__init__()
        self.model_type = 'EnhancedTransformerPretrain'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.token_encoder = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        
        # 仅保留通道注意力
        self.channel_attention = ChannelAttention(d_model) if use_attention else None
        
        self.proj_out = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Softplus(),
            nn.Linear(d_model, d_model) 
        )
        self.init_weights()

    def init_weights(self) -> None:
        nn.init.xavier_normal_(self.token_encoder.weight)

    def forward(self, src: Tensor) -> Tensor:
        src = self.token_encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        
        # 应用通道注意力
        if self.channel_attention is not None:
            src = self.channel_attention(src)
        
        # Transformer编码
        output = self.transformer_encoder(src)
        
        output_embed = output[:, 0:1, :]
        output_embed_proj = output_embed.squeeze(1)
        output_embed_proj = self.proj_out(output_embed_proj)
        return output_embed_proj