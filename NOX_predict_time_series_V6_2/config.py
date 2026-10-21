# -*- coding: utf-8 -*-
import torch
from pathlib import Path


# 文件路径
BASE_DIR = Path(__file__).resolve().parent  # 取这个文件的上一级路径，D:\python\school_Project\NOX_predict
CSV_PATH = BASE_DIR / "data" / "常熟发电#5机组脱硝10天.csv"
MODEL_PATH = BASE_DIR / "model" / "transformer_model.pth"  # 保存/调用模型的位置

# 开关
TRAIN = True
TEST = True

# 超参数
feature_num = 0 # 全局变量--x的特征数(第一层神经元个数)
lr = 0.00005  # 学习率
epoch_num = 30  # 训练轮数
batch_size = 32  # 数据集加载器的每批数据数
device = 'cuda' if torch.cuda.is_available() else 'cpu'
WINDOW_LEN = 60 # 特征窗口序列长度，例如根据过去60分钟预测...
OUTPUT_LEN = 5 # 目标窗口序列长度，例如...预测未来5分钟
# 数据集占比--训练集0.9；验证集+测试集0.1（）
TRAIN_RATIO = 0.82         # 训练序列占比
VAL_RATIO = 0.5          # 验证+测试序列中验证集占比


# Transformer 模型参数。D_MODEL 必须能被 NUM_HEADS 整除。
# NOX 需保留在输入 X 中，使模型可利用最近一次 NOX 观测。
D_MODEL = 128  # 隐藏层维度
NUM_HEADS = 2  # 多头自注意力的“注意力头”数量
NUM_LAYERS = 2  # Transformer Encoder 层数
FFN_DIM = 256  # 每层前馈网络的隐藏维度
DROPOUT = 0.1

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

