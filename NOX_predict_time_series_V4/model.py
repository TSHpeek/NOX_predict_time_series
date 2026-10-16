import math

import torch
import torch.nn as nn

# 创建模型
# ============================================================
# GRU 模型 + 注意力机制 
# ============================================================
class GRUAttentionForecaster(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_len):
        super().__init__()
        self.output_len = output_len
        self.hidden_size = hidden_size

        # 1. GRU 特征提取
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0
        )

        # 2. 标准 QKV 注意力投影层
        self.query_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.key_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.value_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.scale = math.sqrt(hidden_size)  #缩放因子

        # 3. 输出头
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),  #context 是 (batch_size, 128)，last_hidden 是 (batch_size, 128)，拼接后是 (batch_size, 256)
            nn.ReLU(),
            nn.Linear(hidden_size, output_len)
        )

    def forward(self, x):
        gru_out, _ = self.gru(x)  # gru_out: (batch_size, seq_len, hidden_size)  所有时间步的隐藏状态

        # 最后一个时间步的隐藏状态作为 Query
        # 它表示每个样本在最新时刻的状态特征。因为时间序列预测通常基于“当前/最近状态”来判断要从历史中找什么信息，所以它很适合作为 Query
        last_hidden = gru_out[:, -1, :]  # last_hidden: (batch_size, hidden_size)
        query = self.query_proj(last_hidden).unsqueeze(1)  # Q: (batch_size, 1, hidden_size)

        keys = self.key_proj(gru_out)  # K : (batch_size, seq_len, hidden_size)
        values = self.value_proj(gru_out)  # V: (batch_size, seq_len, hidden_size)

        # 计算注意力分数
        attention_scores = torch.bmm(query,keys.transpose(1, 2)) / self.scale  # attention_scores: (batch_size, 1, seq_len)
        # 相关性分数-->概率--权重矩阵
        attention_weights = torch.softmax(attention_scores, dim=-1)  # attention_weights: (batch_size, 1, seq_len)

        # 全局信息向量 = V * 注意力权重矩阵
        context = torch.bmm(attention_weights, values).squeeze(1)  # context: (batch_size, hidden_size)

        # 同时使用注意力上下文和最后时刻状态
        out = self.fc(torch.cat([context, last_hidden], dim=1))

        return out.view(-1, self.output_len, 1)




