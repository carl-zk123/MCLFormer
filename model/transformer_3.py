import pandas as pd
import logging
import numpy as np
import torch
import math
from typing import Tuple
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class MultiScaleAttention(nn.Module):
    """多尺度融合注意力机制（移除了通道注意力）"""
    def __init__(self, d_model, num_scales=3):
        super().__init__()
        self.scales = num_scales
        self.conv_layers = nn.ModuleList()
        for i in range(num_scales):
            kernel_size = 3 + 2 * i  # 3, 5, 7等不同尺度的卷积核
            padding = (kernel_size - 1) // 2
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv1d(d_model, d_model, kernel_size=kernel_size, 
                              padding=padding, groups=d_model),
                    nn.BatchNorm1d(d_model),
                    nn.ReLU(inplace=True)
            ))
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        b, s, d = x.size()
        
        # 多尺度特征提取
        multi_scale_features = []
        x_t = x.transpose(1, 2)  # [b, d, s]
        for conv in self.conv_layers:
            conv_out = conv(x_t).transpose(1, 2)  # [b, s, d]
            multi_scale_features.append(conv_out)
        
        # 合并多尺度特征
        multi_scale_features = torch.stack(multi_scale_features, dim=1)  # [b, num_scales, s, d]
        
        # 空间注意力
        spatial_att = torch.mean(multi_scale_features, dim=3, keepdim=True)  # [b, num_scales, s, 1]
        spatial_att = self.softmax(spatial_att)
        
        # 特征融合
        out = (multi_scale_features * spatial_att).sum(dim=1)  # [b, s, d]
        return out


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
                 nlayers: int, dropout: float = 0.1, use_multi_scale: bool = True):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.token_encoder = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        
        # 仅保留多尺度注意力
        self.multi_scale_attention = MultiScaleAttention(d_model) if use_multi_scale else None
        
        self.init_weights()

    def init_weights(self) -> None:
        nn.init.xavier_normal_(self.token_encoder.weight)

    def forward(self, src: Tensor) -> Tensor:
        src = self.token_encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        
        # Transformer编码
        output = self.transformer_encoder(src)
        
        # 应用多尺度注意力
        if self.multi_scale_attention is not None:
            output = self.multi_scale_attention(output)
        
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
                 nlayers: int, dropout: float = 0.1, use_multi_scale: bool = True):
        super().__init__()
        self.model_type = 'TransformerPretrain'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.token_encoder = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        
        # 仅保留多尺度注意力
        self.multi_scale_attention = MultiScaleAttention(d_model) if use_multi_scale else None
        
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
        
        # Transformer编码
        output = self.transformer_encoder(src)
        
        # 应用多尺度注意力
        if self.multi_scale_attention is not None:
            output = self.multi_scale_attention(output)
        
        output_embed = output[:, 0:1, :]
        output_embed_proj = output_embed.squeeze(1)
        output_embed_proj = self.proj_out(output_embed_proj)
        return output_embed_proj