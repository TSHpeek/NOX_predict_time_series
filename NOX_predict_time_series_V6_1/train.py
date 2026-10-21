from torch.utils.data import TensorDataset, DataLoader  # 数据集和数据加载器
from config import *
import torch.nn as nn

# 模型训练
def train(model, train_dataset, val_dataset, lr, epoch_num, batch_size, device):
    print("\n" + "=" * 60)
    print("开始训练...")
    print("=" * 60)
    # 将模型加载到设备
    model = model.to(device)
    # 定义优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # 定义训练和验证误差每轮变化列表
    train_loss_list = []
    val_loss_list = []

    # 创建DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    # 创建损失函数对象
    mse_loss = nn.MSELoss()
    # epoch循环
    for epoch in range(epoch_num):
        # --- 训练阶段 ---
        model.train()
        train_loss_total = 0
        # batch循环
        for x, y in train_loader:
            # 将数据加载到设备
            x, y = x.to(device), y.to(device)
            # 前向传播
            y_pred = model(x)
            if y_pred.shape != y.shape:
                raise ValueError(
                    f"模型输出与标签形状不一致: y_pred={tuple(y_pred.shape)}, "
                    f"y={tuple(y.shape)}。请确保模型 output_len 与 OUTPUT_LEN 一致。"
                )
            # 计算损失
            loss_value = mse_loss(y_pred, y)  # 两者都是 (batch, output_len, num_targets)
            train_loss_total += loss_value.item() * x.shape[0]  # 累加损失
            # 梯度清零
            optimizer.zero_grad()
            # 反向传播
            loss_value.backward()
            # 更新参数
            optimizer.step()
        # 计算每轮损失+添加到列表
        this_train_loss = train_loss_total / len(train_dataset)

        # --- 验证阶段 (不加梯度计算，纯评估) ---
        model.eval()  # 切换到评估模式
        val_loss_total = 0
        with torch.no_grad():  # 关闭梯度追踪，节省显存
            for x_val, y_val in val_loader:
                # 将数据加载到设备
                x_val, y_val = x_val.to(device), y_val.to(device)
                # 前向传播
                y_pred_val = model(x_val)
                # 计算损失
                val_loss_val = mse_loss(y_pred_val, y_val)
                val_loss_total += val_loss_val.item() * x_val.shape[0]
        # 计算每轮损失+添加到列表
        this_val_loss = val_loss_total / len(val_dataset)

        # --- 打印和记录 ---
        train_loss_list.append(this_train_loss)
        val_loss_list.append(this_val_loss)
        print(f"epoch: {epoch + 1}/{epoch_num} | train loss: {this_train_loss:.5f} | val loss: {this_val_loss:.5f}")

    # 返回记录每轮误差的列表，用于画损失曲线
    return train_loss_list, val_loss_list
