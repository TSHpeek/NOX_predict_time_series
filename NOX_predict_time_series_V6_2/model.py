import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """为 Transformer 注入时间步顺序信息。"""
    # 输入和输出形状完全相同，只是在每个时间步的特征上加了对应的位置向量
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        # 正弦/余弦位置编码
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # buffer 会随模型迁移到 CPU/GPU，但不会参与梯度训练。
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        return x + self.pe[:, :x.size(1)]

class LearnedPositionalEncoding(nn.Module):
    # 可学习的位置编码
    def __init__(self, d_model, max_len):
        super().__init__()
        self.position_embedding = nn.Embedding(max_len, d_model)

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device)
        return x + self.position_embedding(positions).unsqueeze(0)


class TransformerForecaster(nn.Module):
    """Encoder-only Transformer：用历史窗口直接预测未来多个时间步。"""
    # 初始化
    def __init__(self,input_size,d_model,num_heads,num_layers,output_len,dim_feedforward=256,dropout=0.1,max_len=5000,):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model 必须能被 num_heads 整除。")
        self.output_len = output_len

        # 线性投影--将原始传感器特征映射到 Transformer 的统一隐藏维度。
        self.input_projection = nn.Linear(input_size, d_model)  # 原始特征映射到 Transformer 隐藏维度
        # 输入：(batch_size, seq_len, input_size)；  输出：(batch_size, seq_len, d_model)

        # 位置编码--注入时间顺序信息
        # self.positional_encoding = PositionalEncoding(d_model, max_len=max_len)
        self.positional_encoding = LearnedPositionalEncoding(d_model, max_len=max_len)

        # Transformer编码层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        # Transformer编码器--将上面的编码层堆叠 num_layers 次
        self.encoder = nn.TransformerEncoder(
            encoder_layer,  # 标准编码层
            num_layers=num_layers,  # 堆叠层数
            norm=nn.LayerNorm(d_model),  # 对最终输出做一次 LayerNorm 归一化
        )

        # 最后一个编码向量对应输入窗口中最近的时刻，直接输出未来 output_len 步。
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),  # 比 ReLU 更平滑，常见于 Transformer 类模型
            nn.Dropout(dropout),
            nn.Linear(d_model, output_len),
        )

    def forward(self, x):  # x: (batch_size, seq_len, input_size)
        # 线性投影
        x = self.input_projection(x)
        # 位置编码
        x = self.positional_encoding(x)
        # 编码器
        encoded = self.encoder(x)  # (batch_size, seq_len, d_model)
        # 输出头
        forecast = self.head(encoded[:, -1, :])
        return forecast.unsqueeze(-1)  # (batch_size, output_len, 1)

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




