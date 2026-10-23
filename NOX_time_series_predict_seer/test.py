"""测试与评估：输出 NOX 滚动预测、回归指标和可视化图表。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    FORECAST_MINUTES,
    HISTORY_MINUTES,
    NOX_TARGET_INDEX,
    OUTPUT_DECIMALS,
    configure_plot_font,
)


def metric_values(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """计算 MAE、MSE、RMSE、MAPE 和 R2 五项回归指标。"""
    error = predicted - actual
    nonzero = np.abs(actual) > 1e-8  # MAPE 不使用真实值为 0 的样本。
    denominator = np.sum((actual - actual.mean()) ** 2)
    return {
        "mae": float(np.mean(np.abs(error))),
        "mse": float(np.mean(error**2)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mape_percent": (
            float(np.mean(np.abs(error[nonzero] / actual[nonzero])) * 100)
            if nonzero.any()
            else np.nan
        ),
        "r2": float(1 - np.sum(error**2) / denominator) if denominator else np.nan,
    }


def evaluate(model, loader, dataset, data: pd.DataFrame, output: Path, device) -> None:
    """在测试集滚动预测，并按提前量导出预测 CSV、图表和汇总指标。"""
    import matplotlib.pyplot as plt
    import torch

    configure_plot_font()
    plot_dir = output / "plots"
    plot_dir.mkdir(exist_ok=True)
    model.model.eval()
    prediction_parts, actual_parts, origin_parts = [], [], []
    offset = 0
    with torch.no_grad():
        for inputs, targets, input_marks in loader:
            batch_size = len(inputs)
            predicted, _ = model.model(inputs.to(device), input_marks.to(device))
            prediction_parts.append(predicted[:, :, 0].cpu().numpy())
            actual_parts.append(targets[:, :, NOX_TARGET_INDEX].numpy())
            origin_parts.append(dataset.starts[offset:offset + batch_size] + HISTORY_MINUTES)
            offset += batch_size

    predicted = np.concatenate(prediction_parts)
    actual = np.concatenate(actual_parts)
    origins = np.concatenate(origin_parts)
    metrics = [{"scope": "overall_all_5_leads", **metric_values(actual.ravel(), predicted.ravel())}]
    target_label = str(data.columns[NOX_TARGET_INDEX])

    for lead in range(FORECAST_MINUTES):
        timestamps = data.index[origins + lead]
        lead_actual = actual[:, lead]
        lead_predicted = predicted[:, lead]
        metrics.append(
            {"scope": f"lead_{lead + 1}min", **metric_values(lead_actual, lead_predicted)}
        )
        pd.DataFrame(
            {
                "timestamp": timestamps,
                "lead_minutes": lead + 1,
                "actual_nox": lead_actual,
                "predicted_nox": lead_predicted,
            }
        ).to_csv(
            output / f"lead_{lead + 1}min_predictions.csv",
            index=False,
            encoding="utf-8-sig",
            float_format=f"%.{OUTPUT_DECIMALS}f",
        )
        fig, ax = plt.subplots(figsize=(13, 4.8))
        ax.plot(timestamps, lead_actual, label="Actual NOX", linewidth=0.5)
        ax.plot(timestamps, lead_predicted, label="Predicted NOX", linewidth=0.5, alpha=0.85)
        ax.set(
            title=f"Rolling test: NOX {lead + 1} minute(s) ahead",
            xlabel="Time",
            ylabel=target_label,
        )
        ax.grid(alpha=0.25)
        ax.legend()
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(plot_dir / f"lead_{lead + 1}min.png", dpi=160)
        plt.close(fig)

    pd.DataFrame(metrics).to_csv(
        output / "metrics.csv",
        index=False,
        encoding="utf-8-sig",
        float_format=f"%.{OUTPUT_DECIMALS}f",
    )
    # 绘制测试集第一条样本的完整 5 分钟预测，便于检查单次预报。
    example_time = data.index[origins[0]: origins[0] + FORECAST_MINUTES]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(example_time, actual[0], marker="o", label="Actual NOX")
    ax.plot(example_time, predicted[0], marker="o", label="Predicted NOX")
    ax.set(title="One five-minute NOX forecast", xlabel="Forecast time", ylabel=target_label)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(plot_dir / "all_5min_forecast_example.png", dpi=160)
    plt.close(fig)
