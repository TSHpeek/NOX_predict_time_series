import torch
import torch.nn as nn
import numpy as np

# 1. 假设的输入数据：Batch=2, 历史序列长10分钟, 5个变量
input_tensor = torch.randn(64, 60, 8)


# input_tensor 形状: [2, 10, 5] (Batch, Seq_len, n_vars)

class SimpleSEERSkeleton(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_len = 3  # 3分钟切一个块
        self.stride = 2  # 步长为2
        self.d_model = 128  # Transformer隐层维度
        self.horizon = 5  # 预测未来5分钟

        # 模拟 RevIN 归一化 (简化版)
        self.revin_norm = lambda x: x
        self.revin_denorm = lambda x: x

        # 模拟 Patch 切分后的线性映射 (Embedding)
        self.patch_embedding = nn.Linear(self.patch_len * 8, self.d_model)

        # 模拟 SEER 的核心创新：AEM (增强嵌入)
        self.aem_layer = nn.Linear(self.d_model, self.d_model)

        # 模拟 SEER 的核心创新：LPRM (自动替换)
        self.lprm_filter = nn.Linear(self.d_model, 1)
        self.global_prototype = nn.Parameter(torch.randn(1, 1, self.d_model))

        # 模拟 Transformer 编码器 (这里用简单的全连接代替)
        self.transformer_encoder = nn.Linear(self.d_model, self.d_model)

        # 模拟预测头 (Projection Head)
        self.head = nn.Linear(self.d_model, self.horizon)

    def forward(self, x):
        print(f"1. 输入原始数据形状: {x.shape} -> [Batch, Seq_len, n_vars]")

        # --- 第1步：归一化 ---
        x = self.revin_norm(x)

        # --- 第2步：Patching (切块) 并展平 ---
        # 核心：把时间序列切成一块块的
        n_patches = int((x.shape[1] - self.patch_len) / self.stride) + 1
        x_patches = x.unfold(1, self.patch_len, self.stride)
        # 展平每个 patch 的特征：[Batch, n_patches, patch_len * n_vars]
        x_patches = x_patches.reshape(x.shape[0], n_patches, -1)

        # 线性映射到高维空间
        x = self.patch_embedding(x_patches)
        print(f"2. Patching后(Embedding)形状: {x.shape} -> [Batch, n_patches, d_model]")

        # --- 第3步：AEM 增强嵌入 ---
        x = self.aem_layer(x)
        print(f"3. AEM增强后形状: {x.shape} -> [Batch, n_patches, d_model]")

        # --- 第4步：LPRM 自动替换 (鲁棒性核心) ---
        # 给每个 Patch 打分
        scores = self.lprm_filter(x).squeeze(-1)
        # 找出分数最低的 Patch (这里假设最后一个是最差的)
        # 用全局原型替换掉那个最差的 Patch
        x[:, -1, :] = self.global_prototype.squeeze(0)
        print(f"4. LPRM替换后形状: {x.shape} -> [Batch, n_patches, d_model]")

        # --- 第5步：Transformer 编码 ---
        x = self.transformer_encoder(x)
        print(f"5. Transformer编码后形状: {x.shape} -> [Batch, n_patches, d_model]")

        # --- 第6步：预测与反归一化 ---
        # 预测头：把高维特征映射成未来的预测值 (这里简单地对 Patch 维度求平均后预测)
        x = x.mean(dim=1)
        output = self.head(x)
        output = self.revin_denorm(output)
        print(f"6. 最终输出预测形状: {output.shape} -> [Batch, horizon]")

        return output


# 运行一下，观察控制台打印的形状变化
model = SimpleSEERSkeleton()
output = model(input_tensor)