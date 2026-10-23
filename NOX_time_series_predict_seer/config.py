"""分钟级 NOX 预测实验的公共参数和通用工具。"""
from __future__ import annotations

import random

import numpy as np


# ========================== 可在 PyCharm 中直接修改的实验参数 ==================
EPOCHS = 60  # 最大训练轮数；通常会由 PATIENCE 提前停止。
BATCH_SIZE = 64  # 每次送入模型的滑动窗口样本数；数值越大通常显存占用越高。
LEARNING_RATE = 1e-4  # Adam 的学习率；适合此规模 Transformer 的起始值。
PATIENCE = 8  # 验证集 MAE 连续 8 轮不改善时提前停止。
USE_GPU = True  # 是否优先使用 CUDA 显卡；若本机没有可用 CUDA，会自动使用 CPU。
AUTO_RESUME = True  # True：允许自动续训。程序发现 last_checkpoint.pth 时，会恢复上次保存的模型权重、优化器状态和训练轮数。
NO_RESUME = True  # 强制不续训。即使存在 last_checkpoint.pth，也从第 1 轮重新训练。
TEST_ONLY = True  # True：只测试，不训练；

RANDOM_SEED = 2026  # 随机数种子
HISTORY_MINUTES = 60  # 使用过去 60 分钟作为输入。
FORECAST_MINUTES = 5  # 连续预测的未来分钟数
NOX_TARGET_INDEX = 4  # 预测目标在原始 8 个输入特征中的下标；4 表示第 5 列脱硫出口 NOX。
OUTPUT_DECIMALS = 4  # 导出预测结果和指标 CSV 时保留的小数位数。
TRAIN_RATIO = 0.80  # 训练集  0.0--0.8
VALID_RATIO = 0.10  # 验证集  0.8--0.9

# 原始秒级数据、分钟均值缓存及实验结果目录。
DATA_FILE = "常熟发电#5机组脱硝10天.csv"  # 原始秒级工况数据的文件名。
MINUTE_DATA_FILE = "changshu_minute_mean.csv"  # 从原始数据计算得到的每分钟均值缓存文件名，可减少重复预处理耗时。
# 所有模型权重、指标、预测 CSV 和图片的保存根目录。
OUTPUT_ROOT_DIR = "result"
# 输出根目录下本实验专用的子目录，用于避免不同实验的结果互相覆盖。
RESULT_SUBDIR = "changshu/minute_nox_multivariate"

# SEER 网络的结构与优化器相关参数。
MODEL_PARAMS = {
    "seq_len": HISTORY_MINUTES,
    "horizon": FORECAST_MINUTES,
    "batch_size": BATCH_SIZE,

    # 60 分钟、约 11 天数据的轻量模型；避免过多参数记住高度重叠的窗口。
    "d_model": 64,
    "d_ff": 128,
    "d_score": 16,
    "e_layers": 1,
    "n_heads": 4,
    "patch_len": 3,  # 一个时间块（patch）包含的连续分钟数，此处每 3 分钟组成一个局部片段。
    "stride": 2,  # 相邻 patch 的滑动步长；小于 patch_len 时，patch 之间会有重叠。
    "patience": PATIENCE,  # 传给 SEER 配置的早停容忍轮数，与训练逻辑中的 PATIENCE 保持一致。
    "lr": LEARNING_RATE,
    "loss": "MAE",
    "norm": True,  # 是否启用模型内部归一化，使不同量纲的输入特征更容易共同训练。

    # Transformer 的注意力、正则化与激活函数设置。
    "factor": 3,
    "output_attention": 0,
    "dropout": 0.1,
    "head_dropout": 0.1,
    "activation": "gelu",

    # SEER 的稀疏专家路由：每个 patch 由 4 个专家中的 1 个负责处理。
    "num_experts": 4,
    "noisy_gating": True,
    "k": 1,
    "CI": True,

    # 将编码特征压缩为一半后送入预测头，并启用标准的低质量 Patch 替换。
    "scaling": 0.5,  # 表示预测头之前，先把每个 Patch 的特征维度压缩一半,降低过拟合风险
    "score_threshold": 0.2,  # 分数大于 0.2：保留该 Patch;  设为 0.0，这会关闭 Patch 替换机制
}


def configure_plot_font() -> None:
    """配置 Windows 中文字体，保证输出图中的中文测点名称正常显示。"""
    import matplotlib

    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun", "Arial Unicode MS", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False


def set_seed(seed: int) -> None:
    """固定 Python、NumPy 和 PyTorch 的随机数种子。"""
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
