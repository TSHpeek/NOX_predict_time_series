from torch.utils.data import TensorDataset, DataLoader  # 数据集和数据加载器
from config import *
import torch.nn as nn

# 模型训练
def train(model, train_dataset, test_dataset, lr, epoch_num, batch_size, device):
    print("\n" + "=" * 60)
    print("开始训练...")
    print("=" * 60)
    # 将模型加载到设备
    model = model.to(device)
    # 定义优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # 定义训练误差每轮变化列表
    train_loss_list = []
    # 创建DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # 创建损失函数对象
    mse_loss = nn.MSELoss()
    # epoch循环
    for epoch in range(epoch_num):
        model.train()
        train_loss_total = 0
        # batch循环
        for x, y in train_loader:
            # 将数据加载到设备
            x, y = x.to(device), y.to(device)
            # 前向传播
            y_pred = model(x)
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
        train_loss_list.append(this_train_loss)
        print(f"epoch: {epoch+1}/{epoch_num}, train loss: {this_train_loss}")
    # 返回记录每轮误差的列表，用于画损失曲线
    return train_loss_list
