"""
不用Sequential构建模型,训练集训练完所有轮，测试集才开始测试
1.一个文件写所有逻辑
2.特征工程方法
3.MSE损失函数
"""
import os
import matplotlib.pyplot as plt
import numpy as np

from config import *
from data_process import create_dataset
from model import GRUAttentionForecaster
from test import test
from train import train

if __name__ == "__main__":
    # 1. 加载数据
    train_dataset, val_dataset, test_dataset, feature_num, target_mean, target_scale = create_dataset()
    print(f"feature_num:{feature_num}")

    # 2. 定义模型
    model = GRUAttentionForecaster(
        input_size=feature_num,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        output_len=OUTPUT_LEN,
    ).to(device)

    # 3. 训练、验证 + 画"损失函数"曲线
    train_loss_list = []
    val_loss_list = []
    if TRAIN:
        # 训练并保存模型
        train_loss_list,val_loss_list = train(model, train_dataset, val_dataset, lr, epoch_num, batch_size, device)
        os.makedirs("model", exist_ok=True)
        torch.save(model.state_dict(), "model/model.pth")
        print("模型已保存到 model/model.pth")
        # 画"损失函数"曲线
        plt.figure(figsize=(10, 5))
        plt.plot(train_loss_list, 'r-', label='train loss', linewidth=1)
        plt.plot(val_loss_list, 'b-', label='val loss', linewidth=1)
        plt.legend()
        plt.title("train loss")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.tight_layout()
        plt.savefig("outputs/train_loss.png", dpi=150)
        plt.show()

    # 4. 测试：将每个预测分钟分开评估和作图
    if TEST:
        if not TRAIN:  # 没有训练就导入模型，训练了就用训练的模型
            model_path = "model/model.pth"
            state_dict = torch.load(model_path, map_location=device, weights_only=True)
            model.load_state_dict(state_dict)
            print(f"已加载模型：{model_path}")
        # 测试
        all_preds, all_targets = test(test_dataset, model, device)

        # 模型在标准化空间训练，评估和作图前先还原到原始 NOX 单位。
        target_mean = np.asarray(target_mean).reshape(1, 1, -1)
        target_scale = np.asarray(target_scale).reshape(1, 1, -1)
        all_preds = all_preds * target_scale + target_mean
        all_targets = all_targets * target_scale + target_mean

        print("\n" + "=" * 60)
        print("按预测步长评估...")
        print("=" * 60)
        os.makedirs("outputs", exist_ok=True)
        metrics = []
        for horizon in range(OUTPUT_LEN):  # horizon:预测窗口中的第几分钟
            preds = all_preds[:, horizon, 0]  # 取所有预测窗口中第horizon分钟的预测值
            targets = all_targets[:, horizon, 0]  # # 取所有预测窗口中第horizon分钟的真实值
            # 评估
            mae = np.mean(np.abs(preds - targets))
            mse = np.mean((preds - targets) ** 2)
            rmse = np.sqrt(mse)
            mape = np.mean(np.abs((targets - preds) / (targets + 1e-8))) * 100
            metrics.append([horizon + 1, mae, mse, rmse, mape])
            print(f"第 {horizon + 1} 分钟 - MAE: {mae:.4f}, MSE: {mse:.4f},RMSE: {rmse:.4f}, MAPE: {mape:.2f}% ")

            # 画图
            plt.figure(figsize=(12, 6))
            plt.plot(targets, 'g-', label='True NOX', linewidth=0.5)
            plt.plot(preds, 'r-', label='Predicted NOX', linewidth=0.5)
            plt.title(f'NOX prediction: horizon {horizon + 1} minute')
            plt.xlabel('Test Window Index')
            plt.ylabel('NOX')
            plt.legend()
            plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域，并防止标签、标题等元素重叠或被裁剪
            plt.savefig(f"outputs/test_pred_vs_true_horizon_{horizon + 1}.png", dpi=150)
            plt.close()
        # 把评估参数保存为.csv文件
        np.savetxt(
            "outputs/horizon_metrics.csv",  # 路径/名称
            np.asarray(metrics),
            delimiter=",", # 列之间用,间隔，标准的csv格式
            header="horizon_minute,mae,mse,rmse,mape_percent",  # .csv文件表头
            comments="",
            fmt="%.4f"  # 保留4位小数，不用科学计数法
        )

        print("\n" + "=" * 60)
        print("打印模型的结构和参数量...")
        print("=" * 60)
        print(model)
        # 计算总参数量
        total_params = sum(p.numel() for p in model.parameters())
        print(f"模型总参数量: {total_params:,}")



