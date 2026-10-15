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
            dropout=0.2 if num_layers > 1 else 0.0
        )

        # 2. 注意力得分计算层
        self.attn_score = nn.Linear(hidden_size, 1, bias=False)  # 把 128维向量，压缩成 1 个分数
        self.scale = math.sqrt(hidden_size)  # 缩放因子，到时候分数/缩放因子，把分数缩小一点

        # 3. 全连接输出头（输出预测结果）
        fc_input_size = hidden_size * 2
        self.fc = nn.Sequential(
            nn.Linear(fc_input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_len),
        )

    def forward(self, x):
        # GRU 特征提取
        gru_out, _ = self.gru(x)  # gru_out：(batch_size, seq_len, hidden_size)

        # 计算注意力得分并缩放
        attn_scores = self.attn_score(gru_out) / self.scale  #计算分数
        attn_weights = torch.softmax(attn_scores, dim=1)  # 把分数变成权重

        # 加权求和获取上下文向量
        context = torch.bmm(attn_weights.transpose(1, 2), gru_out).squeeze(1)  # 加入注意力机制后的全局信息摘要
        last_hidden = gru_out[:, -1, :]  # last_hidden：(batch_size, hidden_size)  最后一个时间步的隐藏状态

        # 拼起来并预测
        out = self.fc(torch.cat([context, last_hidden], dim=1))  #

        return out.view(-1, self.output_len, 1)  # 调整输出形状


