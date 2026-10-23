import torch
import torch.nn as nn
from ts_benchmark.baselines.seer.layers.Embed import (AugmentedEmbedding)
from ts_benchmark.baselines.seer.layers.RevIN import RevIN
from ts_benchmark.baselines.seer.utils.masked_attention import (Encoder,EncoderLayer,FullAttention,AttentionLayer,)
from ..layers.Transformer_EncDec import Encoder, EncoderLayer


##############################################
# 预测头（Prediction Head）
##############################################
class FlattenHead(nn.Module):
    """
    如果把整个神经网络比作一个工厂，前面的 Encoder（编码器）是“车间”，负责把原材料加工成高维度的抽象特征；
    那么这个 FlattenHead 就是“出货口”，负责把车间加工好的特征，转换成你真正想要的最终预测结果
    """
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars  # 变量数量（比如你的4个目标：功率、A流量、B流量、NOX）
        self.flatten = nn.Flatten(start_dim=-2)  # 【核心工具】：展平层，把多维数据压扁成一维
        self.linear = nn.Linear(nf, target_window)  # 【核心工具】：全连接层，负责把特征映射成预测长度；  nf：输入数据的特征维度; target_window:比如未来 5 分钟
        self.dropout = nn.Dropout(head_dropout)  # 【防过拟合】：随机丢弃一些神经元，防止模型死记硬背

    def forward(self, x):  # x: [bs x n_vars x d_model x patch_num]
        """
        bs (Batch Size)：批次大小（比如 64 个样本）
        n_vars (Variables)：变量数（比如 4 个预测目标）
        d_model：特征维度（比如 256，模型对每个 Patch 的理解深度）
        patch_num：Patch 的数量（比如 10 个时间块）
        """
        x = self.flatten(x)  #第一步：展平 [64, 4, 256, 10] -> [64, 4, 2560]
        x = self.linear(x) #第二步：线性映射  把包含历史信息的长向量，压缩/映射成未来 5 分钟的数值  [64, 4, 2560] -> [64, 4, 5]
        x = self.dropout(x) #第三步：随机丢弃  提高泛化能力（预测时会自动关闭）
        return x


##############################################
# SEER模型
##############################################
class SEERModel(nn.Module):
    # 初始化
    def __init__(self, config):
        # 第一部分：__init__（建厂：准备各个车间和工具）  模型根据你传入的 config（配置说明书），把需要的零件都组装好
        super(SEERModel, self).__init__()
        self.n_vars = config.enc_in
        self.revin = RevIN(self.n_vars)  # 负责在数据进来时洗个澡（归一化），出去时换回原来的衣服（反归一化）
        self.patch_len = config.patch_len
        self.stride = config.stride
        self.n_heads = config.n_heads
        self.config = config
        padding = self.stride
        self.patch_num = int((config.seq_len - self.patch_len) / self.stride + 2)
        self.augmented_embedding = AugmentedEmbedding(  #把长长的时间序列切成一个个小块（Patch），并提取特征
            self.config,
            config.d_model,
            self.patch_len,
            self.stride,
            padding,
            config.dropout,
        )
        # 双路编码器设计
        self.score_threshold = config.score_threshold
        if self.score_threshold != 0:
            self.casual_encoder = Encoder(  #因果编码器（只有1层），用于初步处理
                [
                    EncoderLayer(
                        AttentionLayer(
                            FullAttention(
                                True,
                                config.factor,
                                attention_dropout=config.dropout,
                                output_attention=config.output_attention,
                            ),
                            config.d_model,
                            config.n_heads,
                        ),
                        config.d_model,
                        config.d_ff,
                        dropout=config.dropout,
                        activation=config.activation,
                    )
                    for l in range(1)
                ],
                norm_layer=torch.nn.LayerNorm(config.d_model),
            )

        self.full_encoder = Encoder(  #全量编码器（多层），用于深度挖掘数据规律
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            config.factor,
                            attention_dropout=config.dropout,
                            output_attention=config.output_attention,
                        ),
                        config.d_model,
                        config.n_heads,
                    ),
                    config.d_model,
                    config.d_ff,
                    dropout=config.dropout,
                    activation=config.activation,
                )
                for l in range(config.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(config.d_model),
        )


        self.head_nf = int(
            config.d_model
            * config.scaling
            * int((config.seq_len - self.patch_len) / self.stride + 2 + 1)
        )
        if self.config.scaling != 1:
            self.linear_projection = nn.Linear(
                config.d_model, int(config.d_model * config.scaling), bias=False
            )

        # 预测头  这是一个简单的线性映射层。它把 Encoder 输出的高维特征 (Batch, Vars, Patches, Dim) 展平并映射到最终的预测长度 pred_len
        self.head = FlattenHead(
            config.enc_in,   # 输入特征总数
            self.head_nf,
            config.pred_len,  # 预测未来长度（比如5分钟）
            head_dropout=config.head_dropout,
        )

        if self.score_threshold != 0:  # 一个打分器，用来判断切出来的 Patch 是好数据还是坏数据（噪声）
            # 质量过滤器
            self.linear_filter = nn.Sequential(
                nn.Linear(config.d_model, 1), nn.Sigmoid()
            )

    def replaced_embedding(self, enc_out, score):
        mask_zero = score != 0
        inv = torch.zeros_like(score)
        inv[mask_zero] = (1 / score[mask_zero]).detach()
        identity = score * inv
        # 计算掩码  分数高于阈值的 Patch 被标记为 1（保留），低于阈值的被标记为 0（视为噪声）
        cond = (score > self.score_threshold).float()
        raw_mask = identity * cond
        raw_mask[:, 0] = 1
        # 生成替代物
        series_token = enc_out[:, 0:1, :].expand(-1, self.patch_num + 1, -1)
        # 执行替换
        series_token = series_token * (1 - raw_mask.unsqueeze(-1))
        enc_out = enc_out * raw_mask.unsqueeze(-1)
        enc_out = enc_out + series_token
        return enc_out

    # 第二部分：模型真正开始处理数据
    ######输入######
    def forward(self, input, input_mark):
        # 第 1 步：数据清洗与切块  原始数据进来后，先归一化，然后被切成一个个小 Patch，变成了高维度的特征向量
        ######归一化######
        input = self.revin(input, "norm")  # 1. 先做归一化  [batch_size, seq_len, n_vars]
        x_enc = input.permute(0, 2, 1)  # 2. 调整数据维度   [batch_size, n_vars, seq_len]
        ######增强嵌入######
        enc_out, l_importance = self.augmented_embedding(x_enc, input_mark)  # 3. 切块并提取特征   [batch_size * n_vars, n_patch+1, dim]

        # 第 2 步：自动质检与替换  先给数据“体检”（打分），然后把“生病”的部分用“健康”的特征替换掉，最后再进行分析。
        if self.score_threshold != 0:
            ######自动化令牌过滤器######
            score = self.linear_filter(enc_out).squeeze(-1)  # 质检员给每个 Patch 打分  [batch_size * n_vars, n_patch+1]
            ######替换注意力######
            enc_out = self.replaced_embedding(enc_out, score)  # 把坏 Patch 替换掉  [batch_size * n_vars, n_patch+1, dim]

            # 第 3 步：深度思考（Transformer 编码）  处理干净的数据被送进 Transformer 车间，通过注意力机制，模型开始寻找历史数据与未来预测之间的复杂关联
            enc_out, attns = self.casual_encoder(enc_out)  # 因果编码器  初步处理  [batch_size * n_vars, n_patch+1, dim]
            ######多头自注意力######
            enc_out, attns = self.full_encoder(enc_out)  # 全量编码器  深度挖掘规律  [batch_size * n_vars, n_patch+1, dim]
        else:
            enc_out, attns = self.full_encoder(enc_out)  # [batch_size * n_vars, n_patch+1, dim]

        enc_out = enc_out.reshape(-1, self.n_vars, enc_out.shape[-2], enc_out.shape[-1])  # [batch_size, n_vars, n_patch+1, dim]

        # 第 4 步：特征缩放（可选）
        if self.config.scaling != 1:
            enc_out = self.linear_projection(enc_out)

        # 第 5 步：预测与反归一化  高维特征被“拍扁”并映射成未来 5 分钟的数值
        ######预测器######
        dec_out = self.head(enc_out)  # 1. 通过预测头，输出未来的预测值  [batch_size, n_vars, horizon]
        dec_out = dec_out.permute(0, 2, 1)  # 2. 调整维度顺序  [batch_size, horizon, n_vars]
        output = self.revin(dec_out, "denorm")  # 3. 反归一化，还原成真实的物理量（如真实的功率值）
        return output, l_importance
