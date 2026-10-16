# -*- coding: utf-8 -*-
import torch
from pathlib import Path


# 文件路径
BASE_DIR = Path(__file__).resolve().parent  # 取这个文件的上一级路径，D:\python\school_Project\NOX_predict
CSV_PATH = BASE_DIR / "data" / "常熟发电#5机组脱硝10天.csv"

# 开关
TRAIN = True
TEST = True

# 超参数
feature_num = 0 # 全局变量--x的特征数(第一层神经元个数)
lr = 0.0001  # 学习率
epoch_num = 20  # 训练轮数
batch_size = 64  # 数据集加载器的每批数据数
device = 'cuda' if torch.cuda.is_available() else 'cpu'
WINDOW_LEN = 120 # 特征窗口序列长度，例如根据过去60分钟预测...
OUTPUT_LEN = 10 # 目标窗口序列长度，例如...预测未来5分钟
# 数据集占比--训练集0.9；验证集+测试集0.1（）
TRAIN_RATIO = 0.81         # 训练序列占比
VAL_RATIO = 0.5          # 验证+测试序列中验证集占比
# TEST_RATIO = 0.1         # 测试序列占比
# VAL_RATIO = 0.1          # 训练验证序列中验证集占比


# GRU-Attention 模型参数。NOX 是 SENSOR_COLS 中的第 5 列（下标 4），
# 它需保留在输入 X 中，以便模型将最后观测值作为预测基线。
HIDDEN_SIZE = 128  # 注意力隐藏层维度
NUM_HEADS = 2  # 注意头数
NUM_LAYERS = 2  # gru层数
TARGET_FEATURE_INDICES = [4]  # 预测目标的索引列表

SENSOR_COLS = [
    "5号机机组功率",
    "5号机@OvationControlBuilde",
    "5号机A侧NO测量混合值(6%氧量)",
    "5号机B侧NO测量混合值(6%氧量)",
    "#5炉FGD出口NOX浓度(6%氧量)",
    "5号机烟囱入口NOX浓度设定值",
    "A尿素喷氨流量",
    "B尿素喷氨流量",
]  # 用于指定分钟级数据取出哪些列

