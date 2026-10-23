"""程序入口：连接数据处理、模型、训练与测试模块。"""
from __future__ import annotations

import argparse
from pathlib import Path

from data_process import create_loaders, minute_data
from config import (
    DATA_FILE,
    EPOCHS,
    LEARNING_RATE,
    MINUTE_DATA_FILE,
    NO_RESUME,
    NOX_TARGET_INDEX,
    OUTPUT_ROOT_DIR,
    RANDOM_SEED,
    RESULT_SUBDIR,
    TEST_ONLY,
    USE_GPU,
    set_seed,
)
from model import make_model
from test import evaluate
from train import train_model


def parse_args() -> argparse.Namespace:
    """读取是否续训、是否只测试和总训练轮数等命令行选项。"""
    parser = argparse.ArgumentParser(
        description="分钟级 SEER 实验：过去 60 分钟预测未来 5 分钟脱硫出口 NOX。"
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        default=NO_RESUME,
        help="Ignore last_checkpoint.pth and begin at epoch 1.",
    )
    parser.add_argument(
        "--test-only",
        action="store_true",
        default=TEST_ONLY,
        help="Load best_seer_model.pth and run only rolling test evaluation; do not train.",
    )
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Total number of training epochs.")
    return parser.parse_args()


def main() -> None:
    """准备数据和模型，执行训练或仅测试模式，最后生成测试集输出。"""
    args = parse_args()
    import torch

    # 全部文件路径相对于项目根目录，而不是终端当前目录。
    root = Path(__file__).resolve().parent
    source = root / "dataset" / "forecasting" / DATA_FILE
    cache = root / "dataset" / "forecasting" / MINUTE_DATA_FILE
    # 所有训练产物保存至 model/changshu/minute_nox_multivariate。
    # output = root / OUTPUT_ROOT_DIR / RESULT_SUBDIR
    output = root / OUTPUT_ROOT_DIR
    output.mkdir(parents=True, exist_ok=True)
    if not source.is_file():
        raise FileNotFoundError(f"Cannot find data file: {source}")

    set_seed(RANDOM_SEED)
    data = minute_data(source, cache)
    if data.shape[1] != 8:
        raise ValueError(f"Expected 8 input features but found {data.shape[1]}.")
    print("Input features (8):", list(data.columns))
    print(f"Prediction target: feature {NOX_TARGET_INDEX + 1}: {data.columns[NOX_TARGET_INDEX]}")
    train_loader, valid_loader, test_loader, test_dataset = create_loaders(data)

    device = torch.device("cuda" if USE_GPU and torch.cuda.is_available() else "cpu")
    print("Training device:", torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU")
    model = make_model(data, device)
    optimizer = torch.optim.Adam(model.model.parameters(), lr=LEARNING_RATE)
    best_path = output / "best_seer_model.pth"

    if args.test_only:
        if not best_path.exists():
            raise FileNotFoundError(
                f"Cannot test because the best model does not exist: {best_path}\n"
                "Train the model first, or copy a compatible best_seer_model.pth into this folder."
            )
        model.model.load_state_dict(
            torch.load(best_path, map_location=device, weights_only=False)["model_state_dict"]
        )
        print(f"Test-only mode: loaded best model from {best_path}")
        evaluate(model, test_loader, test_dataset, data, output, device)
        print(f"Test complete. Updated metrics and plots: {output}")
        return

    train_model(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        optimizer=optimizer,
        output=output,
        columns=data.columns,
        device=device,
        total_epochs=args.epochs,
        no_resume=args.no_resume,
    )
    evaluate(model, test_loader, test_dataset, data, output, device)
    print(f"Finished. Results: {output}")
    print(
        "Files: last_checkpoint.pth, best_seer_model.pth, loss_curve.png, metrics.csv, "
        "plots/lead_1min.png ... lead_5min.png"
    )


if __name__ == "__main__":
    main()
