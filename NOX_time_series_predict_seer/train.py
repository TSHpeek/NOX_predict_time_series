"""训练：执行训练/验证轮次，管理检查点、早停和损失曲线。"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from config import (
    AUTO_RESUME,
    FORECAST_MINUTES,
    HISTORY_MINUTES,
    NOX_TARGET_INDEX,
    PATIENCE,
    configure_plot_font,
)


def run_epoch(model, loader, optimizer, device, training: bool) -> float:
    """执行一个训练或验证轮次，并返回所有批次平均的 NOX MAE。"""
    import torch

    criterion = torch.nn.L1Loss()
    model.model.train(training)
    losses = []
    for inputs, targets, input_marks in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        input_marks = input_marks.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            output, importance = model.model(inputs, input_marks)
            # 8 个变量均参与特征学习，但只以目标 NOX 列计算监督损失。
            loss = criterion(output[:, :, 0], targets[:, :, NOX_TARGET_INDEX])
            if training:
                (loss + importance.mean()).backward()
                optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


def checkpoint_payload(
    model,
    optimizer,
    completed_epoch: int,
    best_loss: float,
    stale_epochs: int,
    train_history: list[float],
    valid_history: list[float],
    columns,
) -> dict:
    """将模型权重、优化器状态及训练进度组装为可续训的检查点。"""
    return {
        "architecture": "multivariate_cross_attention_single_nox_v1",
        "epoch": completed_epoch,
        "model_state_dict": model.model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_valid_loss": best_loss,
        "stale_epochs": stale_epochs,
        "train_history": train_history,
        "valid_history": valid_history,
        "feature_columns": list(columns),
        "target_column_index": NOX_TARGET_INDEX,
        "history_minutes": HISTORY_MINUTES,
        "forecast_minutes": FORECAST_MINUTES,
    }


def restore_checkpoint(path: Path, model, optimizer, device):
    """恢复检查点，并校验网络结构、预测目标与窗口长度是否兼容。"""
    import torch

    state = torch.load(path, map_location=device, weights_only=False)
    if state.get("architecture") != "multivariate_cross_attention_single_nox_v1":
        raise ValueError(
            "The checkpoint uses the old channel-independent architecture. "
            "Use the new multivariate result directory."
        )
    if (
        state.get("target_column_index") != NOX_TARGET_INDEX
        or state.get("history_minutes") != HISTORY_MINUTES
        or state.get("forecast_minutes") != FORECAST_MINUTES
    ):
        raise ValueError("The existing checkpoint has different target/window settings. Use --no-resume.")
    model.model.load_state_dict(state["model_state_dict"])
    optimizer.load_state_dict(state["optimizer_state_dict"])
    return (
        state["epoch"] + 1,
        state["best_valid_loss"],
        state["stale_epochs"],
        list(state["train_history"]),
        list(state["valid_history"]),
    )


def save_loss_curve(train_history: list[float], valid_history: list[float], output: Path) -> None:
    """保存训练集和验证集 MAE 随训练轮数变化的曲线图。"""
    import matplotlib.pyplot as plt

    configure_plot_font()
    epochs = np.arange(1, len(train_history) + 1)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs, train_history, marker="o", label="Training MAE")
    ax.plot(epochs, valid_history, marker="o", label="Validation MAE")
    ax.set(xlabel="Epoch", ylabel="NOX MAE", title="SEER minute-level NOX loss")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "loss_curve.png", dpi=160)
    plt.close(fig)


def train_model(
    model,
    train_loader,
    valid_loader,
    optimizer,
    output: Path,
    columns,
    device,
    total_epochs: int,
    no_resume: bool,
):
    """训练模型、按验证 MAE 保存最优权重，并在需要时从最后检查点续训。"""
    import torch

    last_path = output / "last_checkpoint.pth"
    best_path = output / "best_seer_model.pth"
    start, best_loss, stale = 0, float("inf"), 0
    train_history: list[float] = []
    valid_history: list[float] = []

    if AUTO_RESUME and not no_resume and last_path.exists():
        start, best_loss, stale, train_history, valid_history = restore_checkpoint(
            last_path, model, optimizer, device
        )
        print(f"Resuming from epoch {start + 1}; best validation MAE: {best_loss:.6f}")

    for current in range(start, total_epochs):
        train_loss = run_epoch(model, train_loader, optimizer, device, training=True)
        valid_loss = run_epoch(model, valid_loader, optimizer, device, training=False)
        train_history.append(train_loss)
        valid_history.append(valid_loss)
        improved = valid_loss < best_loss
        if improved:
            best_loss, stale = valid_loss, 0
            torch.save(
                checkpoint_payload(
                    model, optimizer, current, best_loss, stale, train_history, valid_history, columns
                ),
                best_path,
            )
        else:
            stale += 1

        # 每轮保存最新状态，保证训练被中断后仍可从下一轮继续。
        torch.save(
            checkpoint_payload(
                model, optimizer, current, best_loss, stale, train_history, valid_history, columns
            ),
            last_path,
        )
        save_loss_curve(train_history, valid_history, output)
        suffix = " | best model saved" if improved else f" | no improvement {stale}/{PATIENCE}"
        print(
            f"Epoch {current + 1}/{total_epochs} | train NOX MAE {train_loss:.6f} | "
            f"validation NOX MAE {valid_loss:.6f}{suffix}"
        )
        if stale >= PATIENCE:
            print("Early stopping reached.")
            break

    if not best_path.exists():
        raise RuntimeError("No best model was saved; inspect the data and training losses.")
    model.model.load_state_dict(
        torch.load(best_path, map_location=device, weights_only=False)["model_state_dict"]
    )
    return best_path
