import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, List, Optional
from torch import Tensor

# 导入两个模型
from transformer_1 import Transformer, ChannelAttention, MultiScaleAttention, PositionalEncoding
from cgcnn_finetune import CrystalGraphConvNet


class FeatureFusionModule(nn.Module):
    """特征融合模块，用于融合Transformer和CGCNN的特征"""
    def __init__(self, transformer_dim: int, cgcnn_dim: int, fusion_dim: int, dropout: float = 0.1):
        super().__init__()
        self.transformer_dim = transformer_dim
        self.cgcnn_dim = cgcnn_dim
        self.fusion_dim = fusion_dim
        
        # 特征映射层，将两个模型的特征映射到相同的维度
        self.transformer_proj = nn.Linear(transformer_dim, fusion_dim)
        self.cgcnn_proj = nn.Linear(cgcnn_dim, fusion_dim)
        
        # 注意力融合
        self.attention = nn.MultiheadAttention(fusion_dim, num_heads=4, dropout=dropout, batch_first=True)
        
        # 特征融合后的处理
        self.norm1 = nn.LayerNorm(fusion_dim)
        self.norm2 = nn.LayerNorm(fusion_dim)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim * 4, fusion_dim)
        )
        
        # 交叉注意力权重
        self.cross_weight = nn.Parameter(torch.ones(1))
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, transformer_feat: Tensor, cgcnn_feat: Tensor) -> Tensor:
        """融合Transformer和CGCNN的特征
        
        Args:
            transformer_feat: Transformer特征 [batch_size, seq_len, transformer_dim]
            cgcnn_feat: CGCNN特征 [batch_size, cgcnn_dim]
            
        Returns:
            融合后的特征 [batch_size, fusion_dim]
        """
        batch_size = transformer_feat.size(0)
        
        # 提取Transformer的CLS token特征
        if transformer_feat.dim() == 3:
            transformer_feat = transformer_feat[:, 0]  # 使用第一个token作为序列表示 [batch_size, transformer_dim]
        
        # 特征映射
        trans_proj = self.transformer_proj(transformer_feat)  # [batch_size, fusion_dim]
        cgcnn_proj = self.cgcnn_proj(cgcnn_feat)  # [batch_size, fusion_dim]
        
        # 特征拼接并进行自注意力
        fused_feat = torch.stack([trans_proj, cgcnn_proj], dim=1)  # [batch_size, 2, fusion_dim]
        attn_output, _ = self.attention(fused_feat, fused_feat, fused_feat)
        
        # 残差连接和层归一化
        fused_feat = self.norm1(fused_feat + self.dropout(attn_output))
        
        # 前馈网络
        ffn_output = self.ffn(fused_feat)
        fused_feat = self.norm2(fused_feat + self.dropout(ffn_output))
        
        # 计算特征重要性权重
        weights = self.softmax(torch.tensor([1.0, self.cross_weight], device=fused_feat.device))
        
        # 加权融合
        weighted_feat = weights[0] * fused_feat[:, 0] + weights[1] * fused_feat[:, 1]
        
        return weighted_feat


class FusionModel(nn.Module):
    """融合Transformer和CGCNN的模型"""
    def __init__(self, 
                 # Transformer参数
                 ntoken, 
                 d_model, 
                 nhead, 
                 d_hid,
                 nlayers, 
                 # CGCNN参数
                 orig_atom_fea_len,
                 nbr_fea_len,

                 atom_fea_len: int = 64, 
                 n_conv: int = 3, 
                 h_fea_len: int = 128, 
                 n_h: int = 1,
                 cgcnn_drop_ratio: float = 0.4,
                 # Transformer参数
                 dropout: float = 0.1,
                 use_channel_att: bool = True,
                 use_multi_scale: bool = True,
                 # 融合参数
                 fusion_dim: int = 256,
                 n_classes: int = 1,
                 classification: bool = False):
        super().__init__()
        
        # 初始化Transformer模型
        self.transformer = Transformer(
            ntoken=ntoken,
            d_model=d_model,
            nhead=nhead,
            d_hid=d_hid,
            nlayers=nlayers,
            dropout=dropout,
            use_channel_att=use_channel_att,
            use_multi_scale=use_multi_scale
        )
        
        # 初始化CGCNN模型
        self.cgcnn = CrystalGraphConvNet(
            orig_atom_fea_len=orig_atom_fea_len,
            nbr_fea_len=nbr_fea_len,
            atom_fea_len=atom_fea_len,
            n_conv=n_conv,
            h_fea_len=h_fea_len,
            n_h=n_h,
            drop_ratio=cgcnn_drop_ratio,
            classification=classification
        )
        
        # 特征融合模块
        self.fusion_module = FeatureFusionModule(
            transformer_dim=d_model,
            cgcnn_dim=h_fea_len,
            fusion_dim=fusion_dim,
            dropout=dropout
        )
        
        # 输出层
        if classification:
            self.output_layer = nn.Sequential(
                nn.Linear(fusion_dim, fusion_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fusion_dim // 2, n_classes)
            )
            self.logsoftmax = nn.LogSoftmax(dim=1) if n_classes > 1 else None
        else:
            self.output_layer = nn.Sequential(
                nn.Linear(fusion_dim, fusion_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fusion_dim // 2, 1)
            )
        
        self.classification = classification
        self.n_classes = n_classes
        
    def forward(self, 
                transformer_input: Tensor,
                atom_fea: Tensor, 
                nbr_fea: Tensor, 
                nbr_fea_idx: Tensor, 
                crystal_atom_idx: List[Tensor]) -> Tensor:
        """前向传播
        
        Args:
            transformer_input: Transformer的输入 [batch_size, seq_len]
            atom_fea: 原子特征 [N, orig_atom_fea_len]
            nbr_fea: 邻居特征 [N, M, nbr_fea_len]
            nbr_fea_idx: 邻居索引 [N, M]
            crystal_atom_idx: 晶体原子索引 list of [batch_size]
            
        Returns:
            预测结果
        """
        # Transformer前向传播
        transformer_output = self.transformer(transformer_input)  # [batch_size, seq_len, d_model]
        
        # CGCNN前向传播
        cgcnn_output, cgcnn_features = self.cgcnn(atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx)
        
        # 特征融合
        fused_features = self.fusion_module(transformer_output, cgcnn_features)
        
        # 输出层
        output = self.output_layer(fused_features)
        
        # 分类任务应用LogSoftmax
        if self.classification and self.n_classes > 1:
            output = self.logsoftmax(output)
            
        return output


class AdaptiveFusionModel(nn.Module):
    """自适应融合模型，可以处理单一输入或双输入情况"""
    def __init__(self, fusion_model: FusionModel):
        super().__init__()
        self.fusion_model = fusion_model
        
        # 添加门控机制，用于决定使用哪种模型的特征
        self.gate = nn.Sequential(
            nn.Linear(self.fusion_model.fusion_module.fusion_dim * 2, 2),
            nn.Softmax(dim=1)
        )
    
    def forward(self, 
                transformer_input: Optional[Tensor] = None,
                atom_fea: Optional[Tensor] = None, 
                nbr_fea: Optional[Tensor] = None, 
                nbr_fea_idx: Optional[Tensor] = None, 
                crystal_atom_idx: Optional[List[Tensor]] = None) -> Tensor:
        """自适应前向传播，可以处理单一输入或双输入情况
        
        Args:
            transformer_input: Transformer的输入，可选
            atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx: CGCNN的输入，可选
            
        Returns:
            预测结果
        """
        # 检查输入情况
        has_transformer = transformer_input is not None
        has_cgcnn = all([x is not None for x in [atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx]])
        
        # 如果两种输入都有，使用完整的融合模型
        if has_transformer and has_cgcnn:
            return self.fusion_model(transformer_input, atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx)
        
        # 如果只有Transformer输入
        elif has_transformer:
            # 创建一个虚拟的CGCNN特征
            batch_size = transformer_input.size(0)
            dummy_cgcnn_feat = torch.zeros(batch_size, self.fusion_model.cgcnn.conv_to_fc.out_features, 
                                          device=transformer_input.device)
            
            # Transformer前向传播
            transformer_output = self.fusion_model.transformer(transformer_input)
            
            # 特征融合，但给CGCNN特征很低的权重
            fused_features = self.fusion_model.fusion_module(transformer_output, dummy_cgcnn_feat)
            
            # 输出层
            output = self.fusion_model.output_layer(fused_features)
            
            # 分类任务应用LogSoftmax
            if self.fusion_model.classification and self.fusion_model.n_classes > 1:
                output = self.fusion_model.logsoftmax(output)
                
            return output
        
        # 如果只有CGCNN输入
        elif has_cgcnn:
            # CGCNN前向传播
            cgcnn_output, cgcnn_features = self.fusion_model.cgcnn(atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx)
            
            # 创建一个虚拟的Transformer特征
            batch_size = cgcnn_features.size(0)
            dummy_transformer_feat = torch.zeros(batch_size, self.fusion_model.transformer.d_model, 
                                               device=cgcnn_features.device)
            
            # 特征融合，但给Transformer特征很低的权重
            fused_features = self.fusion_model.fusion_module(dummy_transformer_feat, cgcnn_features)
            
            # 输出层
            output = self.fusion_model.output_layer(fused_features)
            
            # 分类任务应用LogSoftmax
            if self.fusion_model.classification and self.fusion_model.n_classes > 1:
                output = self.fusion_model.logsoftmax(output)
                
            return output
        
        else:
            raise ValueError("至少需要提供一种模型的输入")


# # 示例：如何创建和使用融合模型
# def create_fusion_model(classification=False, n_classes=1):
#     """创建融合模型的示例函数"""
#     # Transformer参数
#     ntoken = 10000  # 词汇表大小
#     d_model = 512   # 模型维度
#     nhead = 8       # 注意力头数
#     d_hid = 2048    # 前馈网络隐藏层维度
#     nlayers = 6     # Transformer层数
#     dropout = 0.1   # Dropout比例
    
#     # CGCNN参数
#     orig_atom_fea_len = 92  # 原子特征长度
#     nbr_fea_len = 41        # 邻居特征长度
#     atom_fea_len = 64       # 原子隐藏特征长度
#     n_conv = 3              # 卷积层数
#     h_fea_len = 128         # 隐藏特征长度
    
#     # 融合参数
#     fusion_dim = 256        # 融合特征维度
    
#     # 创建融合模型
#     model = FusionModel(
#         # Transformer参数
#         ntoken=ntoken,
#         d_model=d_model,
#         nhead=nhead,
#         d_hid=d_hid,
#         nlayers=nlayers,
#         dropout=dropout,
#         use_channel_att=True,
#         use_multi_scale=True,
        
#         # CGCNN参数
#         orig_atom_fea_len=orig_atom_fea_len,
#         nbr_fea_len=nbr_fea_len,
#         atom_fea_len=atom_fea_len,
#         n_conv=n_conv,
#         h_fea_len=h_fea_len,
#         cgcnn_drop_ratio=0.4,
        
#         # 融合参数
#         fusion_dim=fusion_dim,
#         n_classes=n_classes,
#         classification=classification
#     )
    
#     # 创建自适应融合模型
#     adaptive_model = AdaptiveFusionModel(model)
    
#     return adaptive_model


# # 示例：如何训练融合模型
# def train_fusion_model(model, transformer_data, cgcnn_data, target, optimizer, criterion):
#     """训练融合模型的示例函数"""
#     model.train()
#     optimizer.zero_grad()
    
#     # 解包数据
#     transformer_input = transformer_data
#     atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx = cgcnn_data
    
#     # 前向传播
#     output = model(transformer_input, atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx)
    
#     # 计算损失
#     loss = criterion(output, target)
    
#     # 反向传播
#     loss.backward()
#     optimizer.step()
    
#     return loss.item()