"""模型构建：初始化 SEER 并替换为本实验使用的多变量预测网络。"""
from __future__ import annotations

import pandas as pd

from config import MODEL_PARAMS, NOX_TARGET_INDEX


def make_model(data: pd.DataFrame, device):
    """根据输入特征配置并创建部署到指定设备的多变量 SEER 模型。"""
    from ts_benchmark.baselines.seer.models.multivariate_seer_model import (MultivariateSEERModel,)
    from ts_benchmark.baselines.seer.seer import SEER

    model = SEER(**MODEL_PARAMS)
    model.multi_forecasting_hyper_param_tune(data)
    model.config.freq = "t"  # 基准库中表示分钟频率的代码。
    model.config.target_index = NOX_TARGET_INDEX
    model.model = MultivariateSEERModel(model.config).to(device)
    return model
