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

class FusionConv(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.multi = MultiScaleConv(d_model)
        self.light = LightweightConv(d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        out = self.multi(x) + self.light(x)  # 并行融合
        return self.norm(self.dropout(out + x))  # 残差连接
    

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

class CustomTransformerEncoderLayer(TransformerEncoderLayer):
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # 保存注意力权重
        attn_weights = None
        
        def attn_hook(module, input, output):
            nonlocal attn_weights
            attn_weights = output[1]  # (attn_output, attn_weights)
        
        # 注册钩子
        hook = self.self_attn.register_forward_hook(attn_hook)
        
        # 原始前向传播
        src2 = self.self_attn(
            src, src, src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask
        )[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        # 移除钩子
        hook.remove()
        
        return src, attn_weights

class CustomTransformerEncoder(TransformerEncoder):
    def forward(self, src, mask=None, src_key_padding_mask=None):
        all_attn_weights = []
        output = src
        
        for mod in self.layers:
            output, attn_weights = mod(
                output, 
                src_mask=mask, 
                src_key_padding_mask=src_key_padding_mask
            )
            all_attn_weights.append(attn_weights)
        
        if self.norm is not None:
            output = self.norm(output)
            
        return output, all_attn_weights

class Transformer(nn.Module):
    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.1):
        super().__init__()
        self.model_type = 'LWCResAttTransformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        # 使用自定义的Encoder
        encoder_layers = CustomTransformerEncoderLayer(
            d_model, nhead, d_hid, dropout, 
            batch_first=True, activation="gelu"
        )
        self.transformer_encoder = CustomTransformerEncoder(encoder_layers, nlayers)
        # self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.token_encoder = nn.Embedding(ntoken, d_model)
        self.d_model = d_model

        self.light_weight_conv = LightweightConv(d_model)

        self.lstm = nn.LSTM(d_model, d_model, num_layers=1, batch_first=True, bidirectional=False)
        self.init_weights()

        # self.attention_weights = None  # 用于存储attention权重
        # # 添加对norm层的引用
        # self.final_norm = self.transformer_encoder.norm  # 保存最后的norm层

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

        # output = self.transformer_encoder(src)                   # [B, S, D]
        output, all_attn_weights = self.transformer_encoder(src)
            
        return output, all_attn_weights

        # attn_weights_list = []

        # def save_attn(module, input, output):
        #     if hasattr(module.self_attn, 'attn_output_weights'):
        #         attn_weights_list.append(module.self_attn.attn_output_weights.detach().cpu())

        # for layer in self.transformer_encoder.layers:
        #     layer.register_forward_hook(save_attn)

        # output = self.transformer_encoder(src)
        # return output, attn_weights_list

    # def _custom_encoder_layer_forward(self, layer, src):
    #     """自定义的encoder层前向传播以捕获attention权重"""
    #     # 这是对TransformerEncoderLayer.forward的修改
    #     src2, attn_weights = layer.self_attn(
    #         src, src, src, 
    #         attn_mask=None,
    #         key_padding_mask=None,
    #         need_weights=True
    #     )
    #     src = src + layer.dropout1(src2)
    #     src = layer.norm1(src)
    #     src2 = layer.linear2(layer.dropout(layer.activation(layer.linear1(src))))
    #     src = src + layer.dropout2(src2)
    #     src = layer.norm2(src)
    #     return src, attn_weights

    # def get_attention_maps(self):
    #     """返回所有层的attention权重"""
    #     return self.attention_weights

class TransformerRegressor(nn.Module):
    def __init__(self, transformer, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.transformer = transformer
        self.regressionHead = RegressionHead(d_model)

    def forward(self, src: Tensor) -> Tensor:
        # output = self.transformer(src)              # [B, S, D]
        # output = self.regressionHead(output[:, 0:1, :])  # [B, 1, D] → [B, 1]
        # return output
        features, all_attn_weights = self.transformer(src)
        output = self.regressionHead(features[:, 0:1, :])
        return output, all_attn_weights