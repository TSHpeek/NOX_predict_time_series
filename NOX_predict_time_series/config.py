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
lr = 0.001
epoch_num = 20
batch_size = 64
device = 'cuda' if torch.cuda.is_available() else 'cpu'
WINDOW_LEN = 60 # 特征窗口序列长度，例如根据过去60分钟预测...
OUTPUT_LEN = 5 # 目标窗口序列长度，例如...预测未来5分钟

# GRU-Attention 模型参数。NOX 是 SENSOR_COLS 中的第 5 列（下标 4），
# 它需保留在输入 X 中，以便模型将最后观测值作为预测基线。
HIDDEN_SIZE = 64
NUM_LAYERS = 2
NUM_TARGETS = 1
TARGET_FEATURE_INDICES = [4]

SENSOR_COLS = [
    "5号机机组功率",
    "5号机@OvationControlBuilde",
    "5号机A侧NO测量混合值(6%氧量)",
    "5号机B侧NO测量混合值(6%氧量)",
    "#5炉FGD出口NOX浓度(6%氧量)",
    "5号机烟囱入口NOX浓度设定值",   # 控制系统目标值，影响喷氨量
    "A尿素喷氨流量",
    "B尿素喷氨流量",
]

