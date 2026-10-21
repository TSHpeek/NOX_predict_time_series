import math

import torch
import torch.nn as nn

# 创建模型
# ============================================================
# LSTM 模型 + 多头头QKV注意力机制
# ============================================================
class LSTMMultiHeadAttentionForecaster(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_len,num_heads=4, dropout=0.1):
        # 定义时多传了一个num_heads参数（相比于单头注意力）
        super().__init__()
        # 多头注意力 要求 hidden_size 能被 num_heads 整除
        assert hidden_size % num_heads == 0, \
            "hidden_size 必须能被 num_heads 整除"

        self.output_len = output_len
        self.hidden_size = hidden_size
        # LSTM 层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # 内部自动完成 Q、K、V 投影、分头、缩放点积注意力、拼接
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # 输出头
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),  # context + last_hidden，所以输入仍然是 hidden_size * 2
            nn.ReLU(),
            nn.Dropout(dropout),  # 多了一个Dropout层
            nn.Linear(hidden_size, output_len)
        )

    def forward(self, x, return_attention=False):
        # lstm_out: (batch_size, seq_len, hidden_size)
        lstm_out, _ = self.lstm(x)

        # 取最后一个时间步的隐藏状态作为 Query
        last_hidden = lstm_out[:, -1:, :]  # (batch_size, 1, hidden_size)

        # Query: 最后一个时间步、Key / Value: 全部历史时间步
        context, attention_weights = self.attention(
            query=last_hidden,
            key=lstm_out,
            value=lstm_out,
            need_weights=True,
            average_attn_weights=False
        )

        # 改变“全局信息矩阵”形状，用于计算
        context = context.squeeze(1)  # context: (batch_size, 1, hidden_size)

        # 最后一个时间步的隐藏状态
        last_hidden = last_hidden.squeeze(1)  # last_hidden: (batch_size, hidden_size)

        # 拼接后输出
        out = self.fc(torch.cat([context, last_hidden], dim=1))  # 拼接后：(batch_size, hidden_size * 2)

        # 预测结果: (batch_size, output_len, 1)
        out = out.view(-1, self.output_len, 1)

        if return_attention:
            # (batch_size, num_heads, 1, seq_len)
            return out, attention_weights

        return out


# ============================================================
# LSTM 模型 + 单头QKV注意力机制
# ============================================================
class LSTMAttentionForecaster(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_len):
        super().__init__()
        self.output_len = output_len
        self.hidden_size = hidden_size

        # 1. LSTM 特征提取
        self.lstm = nn.LSTM(
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
        lstm_out, _ = self.lstm(x)  # lstm_out: (batch_size, seq_len, hidden_size)  所有时间步的隐藏状态

        # 最后一个时间步的隐藏状态作为 Query
        # 它表示每个样本在最新时刻的状态特征。因为时间序列预测通常基于“当前/最近状态”来判断要从历史中找什么信息，所以它很适合作为 Query
        last_hidden = lstm_out[:, -1, :]  # last_hidden: (batch_size, hidden_size)
        query = self.query_proj(last_hidden).unsqueeze(1)  # Q: (batch_size, 1, hidden_size)

        keys = self.key_proj(lstm_out)  # K : (batch_size, seq_len, hidden_size)
        values = self.value_proj(lstm_out)  # V: (batch_size, seq_len, hidden_size)

        # 计算注意力分数
        attention_scores = torch.bmm(query,keys.transpose(1, 2)) / self.scale  # attention_scores: (batch_size, 1, seq_len)
        # 相关性分数-->概率--权重矩阵
        attention_weights = torch.softmax(attention_scores, dim=-1)  # attention_weights: (batch_size, 1, seq_len)

        # 全局信息向量 = V * 注意力权重矩阵
        context = torch.bmm(attention_weights, values).squeeze(1)  # context: (batch_size, hidden_size)

        # 同时使用注意力上下文和最后时刻状态
        out = self.fc(torch.cat([context, last_hidden], dim=1))

        return out.view(-1, self.output_len, 1)




