from torch.utils.data import DataLoader  # 数据集和数据加载器
from config import *
import numpy as np

def test(test_dataset, model, device, target_mean, target_scale):
    print("\n" + "=" * 60)
    print("开始测试...")
    print("=" * 60)
    # 创建空列表，用于收集预测值和真实值
    all_preds = []
    all_targets = []
    # 创建DataLoader
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    # 收集预测和标签，反标准化后统一计算指标。
    with torch.no_grad():  # 测试时关闭梯度计算
        for X, y in test_loader:  # X.shape:(batch_size,seq_len,feature_num);y.shape:(batch_size,OUTPUT_LEN,1)
            # 将数据加载到设备
            X, y = X.to(device), y.to(device)
            # 前向传播
            y_pred = model(X)
            all_preds.append(y_pred.cpu().numpy())  # 记录测试集的预测结果
            all_targets.append(y.cpu().numpy())  # 记录测试集的真实值

        # 把测试过程中各个 batch 的预测结果拼接成一个完整数组再返回
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        # 反归一化后再计算损失
        target_mean = np.asarray(target_mean).reshape(1, 1, -1)
        target_scale = np.asarray(target_scale).reshape(1, 1, -1)
        preds_original = all_preds * target_scale + target_mean
        targets_original = all_targets * target_scale + target_mean
        test_mae = np.mean(np.abs(preds_original - targets_original))
        test_mse = np.mean((preds_original - targets_original) ** 2)

        print(f"test MAE : {test_mae:.4f}")
        print(f"test MSE : {test_mse:.4f}")
        return all_preds, all_targets
