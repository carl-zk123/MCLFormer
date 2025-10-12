import torch
import torch.nn as nn

class GRUModel(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, num_layers, dropout=0.5):
        super(GRUModel, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=0)

        # GRU 层
        self.gru = nn.GRU(embedding_size, hidden_size, num_layers, batch_first=True, bidirectional=True, dropout=dropout)

        # 全连接层
        self.fc_1 = nn.Linear(hidden_size * 2 * num_layers, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc_2 = nn.Linear(hidden_size, 1)

        # 用于存储中间输出
        self.hidden_output = None
        self.out1 = None
        self.out2 = None

    def forward(self, x):
        # 计算序列长度
        x_lens = (x != 0).sum(dim=1).to(dtype=torch.int64).to('cpu')

        # 嵌入层
        x = self.embedding(x)

        # 打包变长序列
        packed_output = nn.utils.rnn.pack_padded_sequence(x, x_lens, batch_first=True, enforce_sorted=False)

        # GRU 前向传播
        _, out = self.gru(packed_output)

        # 处理 GRU 的输出
        out = out.transpose(0, 1).contiguous().view(out.size(1), -1)  # 形状: (batch_size, hidden_size * 2)
        #print("GRU output shape:", out.shape)  # 打印 GRU 输出形状
        self.hidden_output = out

        # 全连接层 1
        out = self.fc_1(out)
        #print("FC1 output shape:", out.shape)  # 打印全连接层 1 输出形状
        self.out1 = out

        # ReLU 和 Dropout
        out = self.relu(out)
        out = self.dropout(out)
        self.out2 = out

        # 全连接层 2
        out = self.fc_2(out)
        #print("FC2 output shape:", out.shape)  # 打印全连接层 2 输出形状
        return out

    def get_hidden_layer_output(self):
        return self.hidden_output

    def get_fc_layer_output1(self):
        return self.out1

    def get_fc_layer_output2(self):
        return self.out2