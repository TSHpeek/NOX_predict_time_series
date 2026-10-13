import math

import torch
import torch.nn as nn

# 创建模型
# ============================================================
# GRU 模型 + 注意力机制 
# ============================================================
class GRUAttentionForecaster(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_targets, output_len,
                 target_feature_indices):
        super().__init__()
        self.output_len = output_len
        self.num_targets = num_targets
        self.hidden_size = hidden_size
        if len(target_feature_indices) != num_targets:
            raise ValueError("target_feature_indices 必须与 num_targets 数量一致")
        if any(index < 0 or index >= input_size for index in target_feature_indices):
            raise ValueError("target_feature_indices 超出输入特征范围")
        self.register_buffer(
            "target_feature_indices",
            torch.as_tensor(target_feature_indices, dtype=torch.long),
            persistent=True,
        )

        # 1. GRU 特征提取
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0
        )

        # 2. 注意力得分计算层
        self.attn_score = nn.Linear(hidden_size, 1, bias=False)
        self.scale = math.sqrt(hidden_size)  # 缩放因子，防止数值过大

        # 3. 全连接输出头
        # 同时使用注意力上下文、最后时刻状态和最后时刻目标值。
        # 最后目标值作为持久性基线，网络只需学习未来变化量。
        fc_input_size = hidden_size * 2 + num_targets
        self.fc = nn.Sequential(
            nn.Linear(fc_input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_len * num_targets),
        )

    def forward(self, x):
        # GRU 特征提取
        gru_out, _ = self.gru(x)  # (batch_size, seq_len, hidden_size)

        # 计算注意力得分并缩放
        attn_scores = self.attn_score(gru_out) / self.scale
        attn_weights = torch.softmax(attn_scores, dim=1)

        # 加权求和获取上下文向量
        context = torch.bmm(attn_weights.transpose(1, 2), gru_out).squeeze(1)
        last_hidden = gru_out[:, -1, :]
        last_targets = x[:, -1, :].index_select(1, self.target_feature_indices)

        # 输出相对最后观测值的变化量，避免所有步长都退化到总体均值。
        delta = self.fc(torch.cat([context, last_hidden, last_targets], dim=1))
        delta = delta.view(-1, self.output_len, self.num_targets)
        persistence = last_targets.unsqueeze(1).expand(-1, self.output_len, -1)
        return persistence + delta
