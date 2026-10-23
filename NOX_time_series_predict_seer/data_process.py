"""数据处理：秒级 CSV 转分钟均值，并构造按时间顺序滚动的样本窗口。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    BATCH_SIZE,
    FORECAST_MINUTES,
    HISTORY_MINUTES,
    TRAIN_RATIO,
    VALID_RATIO,
)


def minute_data(source: Path, cache: Path) -> pd.DataFrame:
    """从原始秒级数据计算每分钟均值，优先使用未过期的缓存文件。"""
    if cache.exists() and cache.stat().st_mtime >= source.stat().st_mtime:
        data = pd.read_csv(cache, index_col=0, parse_dates=True, encoding="utf-8-sig")
        data.index.name = "date"
        print(f"Using cached minute data: {cache} ({len(data):,} rows)")
        return data

    from ts_benchmark.data.utils import read_data

    print("Reading original CSV and calculating a mean for every 60 seconds ...")
    # 将秒级观测按 1 分钟重采样；任一输入变量缺失的分钟会被丢弃。
    data = read_data(str(source)).resample("1min").mean().dropna(how="any")
    data.index.name = "date"
    data.to_csv(cache, encoding="utf-8-sig")
    print(f"Created minute data: {cache} ({len(data):,} rows)")
    return data


def time_marks(index: pd.DatetimeIndex) -> np.ndarray:
    """把时间戳转换为 SEER 所需的分钟级时间特征。"""
    from ts_benchmark.baselines.utils import get_time_mark

    return get_time_mark(index.to_numpy().reshape(1, -1), timeenc=1, freq="t")[0]


class SlidingMinuteDataset:
    """每个样本包含 60 分钟历史输入和紧随其后的 5 分钟真实标签。"""

    def __init__(self, values: np.ndarray, marks: np.ndarray, starts: np.ndarray):
        self.values = values
        self.marks = marks
        self.starts = starts

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, index: int):
        import torch

        start = int(self.starts[index])
        split = start + HISTORY_MINUTES
        end = split + FORECAST_MINUTES
        return (
            torch.tensor(self.values[start:split], dtype=torch.float32),
            torch.tensor(self.values[split:end], dtype=torch.float32),
            torch.tensor(self.marks[start:split], dtype=torch.float32),
        )


def starts_for_labels(first_label: int, last_label_exclusive: int) -> np.ndarray:
    """生成窗口起点，确保每个预测标签都在同一个训练、验证或测试分区内。"""
    return np.arange(
        first_label - HISTORY_MINUTES,
        last_label_exclusive - HISTORY_MINUTES - FORECAST_MINUTES + 1,
        dtype=np.int64,
    )


def create_loaders(data: pd.DataFrame):
    """按时间顺序划分数据，并创建训练、验证和测试用的 DataLoader。"""
    from torch.utils.data import DataLoader

    count = len(data)
    train_end = int(count * TRAIN_RATIO)
    valid_end = int(count * (TRAIN_RATIO + VALID_RATIO))
    if train_end < HISTORY_MINUTES + FORECAST_MINUTES or valid_end >= count:
        raise ValueError("Insufficient data for the requested time split.")

    values = data.to_numpy(dtype=np.float32)
    marks = time_marks(data.index)
    # 验证/测试窗口可使用分区前的历史值作为输入，但其未来标签不跨越分区边界。
    train_dataset = SlidingMinuteDataset(
        values, marks, starts_for_labels(HISTORY_MINUTES, train_end)
    )
    valid_dataset = SlidingMinuteDataset(
        values, marks, starts_for_labels(train_end, valid_end)
    )
    test_dataset = SlidingMinuteDataset(values, marks, starts_for_labels(valid_end, count))
    print(
        f"Chronological split: train 0:{train_end} ({len(train_dataset):,} windows), "
        f"val {train_end}:{valid_end} ({len(valid_dataset):,} windows), "
        f"test {valid_end}:{count} ({len(test_dataset):,} rolling windows)."
    )
    return (
        DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0),
        DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0),
        DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0),
        test_dataset,
    )
