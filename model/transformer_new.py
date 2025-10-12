import pandas as pd
import logging
import numpy as np
import torch
import math
from typing import Tuple
from torch import nn, Tensor
import torch.nn.functional as F
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
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class regressionHead(nn.Module):

    def __init__(self, d_embedding: int):
        super().__init__()
        self.layer1 = nn.Linear(d_embedding, d_embedding//2)
        self.layer2 = nn.Linear(d_embedding//2, d_embedding//4)
        self.layer3 = nn.Linear(d_embedding//4, d_embedding//8)
        self.layer4 = nn.Linear(d_embedding//8, 1)
        self.relu=nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        
        return self.layer4(x)

class SEAttention(nn.Module):
    """Squeeze-and-Excitation Attention for channel-wise weighting"""
    def __init__(self, d_model: int, reduction: int = 4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model // reduction),
            nn.ReLU(),
            nn.Linear(d_model // reduction, d_model),
            nn.Sigmoid()
        )
    
    def forward(self, x: Tensor) -> Tensor:
        # x shape: [batch_size, seq_len, d_model]
        weights = self.fc(x.mean(dim=1))  # Global average pooling
        return x * weights.unsqueeze(1)  # Broadcast to sequence length

class PyramidSplitAttention(nn.Module):
    """Multi-scale feature fusion inspired by Pyramid Split Attention"""
    def __init__(self, d_model: int, scales: list = [1, 2, 4]):
        super().__init__()
        self.scales = scales
        self.convs = nn.ModuleList([
            nn.Conv1d(d_model, d_model // len(scales), kernel_size=s, padding=s//2)
            for s in scales
        ])
        self.fusion = nn.Linear(d_model, d_model)

    def forward(self, x: Tensor) -> Tensor:
        # x shape: [batch_size, seq_len, d_model]
        batch_size, seq_len, _ = x.shape
        x_t = x.transpose(1, 2)  # [batch_size, d_model, seq_len]
        
        # Multi-scale processing
        split_features = []
        for conv, scale in zip(self.convs, self.scales):
            if scale == 1:
                split_features.append(x)  # Original scale
            else:
                pooled = F.avg_pool1d(x_t, kernel_size=scale)
                convolved = conv(pooled)
                convolved = F.interpolate(convolved, size=seq_len, mode='linear')
                split_features.append(convolved.transpose(1, 2))
        
        # Concatenate and fuse
        fused = torch.cat(split_features, dim=-1)
        return self.fusion(fused)

class EnhancedTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1):
        super().__init__()
        # 原始Transformer层结构
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        # 新增模块 (完全兼容原始参数)
        self.se_attention = SEAttention(d_model)  # 保持您的SE实现
        self.pyramid_attention = PyramidSplitAttention(d_model)  # 保持您的金字塔实现

    def forward(self, src: Tensor, src_mask: Tensor = None, src_key_padding_mask: Tensor = None, is_causal : bool = False) -> Tensor:
        # 原始自注意力
        src2, _ = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        # 新增模块调用 (保持原始残差连接风格)
        src = src + self.se_attention(src)  # SE注意力
        src = src + self.pyramid_attention(src)  # 金字塔注意力
        
        # 原始前馈网络
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class Transformer(nn.Module):
    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                nlayers: int, dropout: float = 0.1):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # 使用增强的EncoderLayer（保持参数完全一致）
        encoder_layers = EnhancedTransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        
        self.token_encoder = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.init_weights()

    def init_weights(self) -> None:
        nn.init.xavier_normal_(self.token_encoder.weight)

    def forward(self, src: Tensor) -> Tensor:
        src = self.token_encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        return output

class TransformerRegressor(nn.Module):
    def __init__(self, transformer, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.transformer = transformer  # 直接复用您的Transformer类
        
        # 完全保持您的回归头结构
        self.regressionHead = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.ReLU(),
            nn.Linear(d_model//2, d_model//4),
            nn.ReLU(),
            nn.Linear(d_model//4, 1))
        
    def forward(self, src: Tensor) -> Tensor:
        # 完全保持您的调用方式
        output = self.transformer(src)
        return self.regressionHead(output[:, 0:1, :])  # 保持[:, 0:1, :]切片风格

def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

class TransformerPretrain(nn.Module):

    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.1):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.token_encoder = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.proj_out = nn.Sequential(
            nn.Linear(d_model,d_model),
            nn.Softplus(),
            nn.Linear(d_model, d_model) 
        )
        self.init_weights()

    def init_weights(self) -> None:
        nn.init.xavier_normal_(self.token_encoder.weight)

    def forward(self, src: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        src = self.token_encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output_embed = output[:, 0:1, :]
        output_embed_proj = output_embed.squeeze(1)
        output_embed_proj = self.proj_out(output_embed_proj)
        return output_embed_proj