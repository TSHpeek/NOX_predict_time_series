import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer    # 缺省值处理
from sklearn.preprocessing import StandardScaler, OneHotEncoder # 标准化和独热编码
from torch.utils.data import TensorDataset, DataLoader  # 数据集和数据加载器
from config import *

# 创建数据集dataset
def create_dataset():
    print("\n" + "=" * 60)
    print("加载数据并重采样为分钟级...")
    print("=" * 60 )
    # 1、获取数据
    df_raw = pd.read_csv(CSV_PATH, encoding='gbk')
    print(f"df_raw.shape:{df_raw.shape}")
    # 把第一列重命名为 datetime 并转换为 Pandas 的时间格式，然后将其设为索引
    df_raw.rename(columns={df_raw.columns[0]: "datetime"}, inplace=True)
    df_raw["datetime"] = pd.to_datetime(df_raw["datetime"])
    df_raw.set_index("datetime", inplace=True)
    df_raw = df_raw[SENSOR_COLS].sort_index()  # 取出指定列，并按时间索引排序

    data = df_raw.resample("1min").mean().dropna()  # 按分钟聚合，计算每分钟内所有秒级数据的平均值;  dropna()直接删除所有包含空值（NaN）的行
    print(f'df_min.shape:{data.shape}(去除时间列)')

    # 2、划分特征和目标
    print("\n" + "=" * 60)
    print("划分特征和目标...")
    print("=" * 60)
    # 保留 NOX 在输入中，模型需要读取它的最后观测值作为预测基线。
    feature = data
    print(f"x.shape:{feature.shape}")
    target = data["#5炉FGD出口NOX浓度(6%氧量)"]
    print(f"y.shape:{target.shape}")
    print(f"feature列名：{feature.columns.tolist()}")  # Dataframe用.columns.tolist()查看列名
    print(f"target列名：{target.name}")  # series要用.name查看列名

    # 3、划分训练集和测试集
    print("\n" + "=" * 60)
    print("划分训练集和测试集...")
    print("=" * 60 )
    # 时间序列不能随机打乱，否则每个滑动窗口内的点不再是连续时间。
    split_index = int(len(feature) * TRAINDATA_PERSCENT)
    train_feature, test_feature = feature.iloc[:split_index], feature.iloc[split_index:]
    train_target, test_target = target.iloc[:split_index], target.iloc[split_index:]
    print(f"训练集特征train_feature.shape:{train_feature.shape}")
    print(f"训练集目标train_target.shape:{train_target.shape}")
    print(f"测试集特征test_feature.shape:{test_feature.shape}")
    print(f"测试集目标test_target.shape:{test_target.shape}")

    # 4、特征处理：填缺失 + 标准化
    imputer = SimpleImputer(strategy="mean")  # 均值填充
    scaler = StandardScaler()  # 标准化归一化器--适用波动大的数据
    # 用"训练集特征"训练归一化器
    train_feature = imputer.fit_transform(train_feature)
    train_feature_scaled = scaler.fit_transform(train_feature)
    # 对"测试集特征"使用归一化器
    test_feature = imputer.transform(test_feature)
    test_feature_scaled = scaler.transform(test_feature)

    # 5、滑动窗口构建
    print("\n" + "=" * 60)
    print("滑动窗口构建...")
    print("=" * 60 )
    X_train, Y_train= [], []
    for i in range(len(train_feature_scaled)-WINDOW_LEN-OUTPUT_LEN):
        # 切取输入数据（X）
        X_train.append(train_feature_scaled[i:i + WINDOW_LEN])
        # 切取对应的标签（Y）
        Y_train.append(
            train_feature_scaled[
                i + WINDOW_LEN:i + WINDOW_LEN + OUTPUT_LEN,
                TARGET_FEATURE_INDICES,  # 【关键点】只提取我们关心的列（比如第5列NOX）
            ]
        )
    X_train, Y_train = np.array(X_train), np.array(Y_train)
    print(f"训练集序列总数: {len(X_train)}")  # 序列总数:
    print(f"输入形状: {X_train.shape}, 输出形状: {Y_train.shape}  (直接预测未来{OUTPUT_LEN}分钟)")

    X_test, Y_test = [], []
    for i in range(len(test_feature_scaled)-WINDOW_LEN-OUTPUT_LEN):
        # 切取输入数据（X）
        X_test.append(test_feature_scaled[i:i + WINDOW_LEN])
        # 切取对应的标签（Y）  只取标准化的NOX列
        Y_test.append(
            test_feature_scaled[
                i + WINDOW_LEN:i + OUTPUT_LEN + WINDOW_LEN,
                TARGET_FEATURE_INDICES,
            ]
        )
    X_test, Y_test = np.array(X_test), np.array(Y_test)
    print(f"测试集序列总数: {len(X_test)}")  # 序列总数:
    print(f"输入形状: {X_test.shape}, 输出形状: {Y_test.shape}  (直接预测未来{OUTPUT_LEN}分钟)")

    # 6、构建Tensor数据集
    train_dataset = TensorDataset(torch.tensor(X_train).float(), torch.tensor(Y_train).float())
    test_dataset = TensorDataset(torch.tensor(X_test).float(), torch.tensor(Y_test).float())
    
    # 返回目标的反标准化参数，供测试阶段还原为原始 NOX 单位。
    target_mean = scaler.mean_[TARGET_FEATURE_INDICES]
    target_scale = scaler.scale_[TARGET_FEATURE_INDICES]
    return train_dataset, test_dataset, X_train.shape[2], target_mean, target_scale


