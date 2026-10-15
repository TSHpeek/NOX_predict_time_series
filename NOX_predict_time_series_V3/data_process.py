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
    total_len = len(feature)
    # 第一刀：切分出测试集 (后 10%)
    split_test = int(total_len * (1 - TEST_RATIO))
    # 第二刀：在前 90% 的数据中切分出验证集 (最后 10%)
    split_val = int(split_test * (1 - VAL_RATIO))

    train_feature = feature.iloc[:split_val]
    val_feature = feature.iloc[split_val:split_test]
    test_feature = feature.iloc[split_test:]

    # 对应切分目标列
    train_target = target.iloc[:split_val]
    val_target = target.iloc[split_val:split_test]
    test_target = target.iloc[split_test:]

    print(f"训练集特征 train_feature.shape: {train_feature.shape}")
    print(f"验证集特征 val_feature.shape:   {val_feature.shape}")
    print(f"测试集特征 test_feature.shape:  {test_feature.shape}")

    # 4、特征处理：填缺失 + 标准化
    imputer = SimpleImputer(strategy="mean")  # 均值填充
    scaler = StandardScaler()  # 标准化归一化器--适用波动大的数据
    # 用"训练集特征"训练归一化器
    train_feature = imputer.fit_transform(train_feature)
    train_feature_scaled = scaler.fit_transform(train_feature)
    # 对"测试集特征"使用归一化器
    test_feature = imputer.transform(test_feature)
    test_feature_scaled = scaler.transform(test_feature)
    # 对"验证集特征"使用归一化器
    val_feature = imputer.transform(val_feature)
    val_feature_scaled = scaler.transform(val_feature)

    # 5、滑动窗口构建
    print("\n" + "=" * 60)
    print("滑动窗口构建...")
    print("=" * 60 )

    # 统一构建滑动窗口的函数
    def build_windows(feature_scaled, target_series):
        X, Y = [], []
        for i in range(len(feature_scaled) - WINDOW_LEN - OUTPUT_LEN):
            X.append(feature_scaled[i:i + WINDOW_LEN])
            Y.append(
                feature_scaled[i + WINDOW_LEN:i + WINDOW_LEN + OUTPUT_LEN, TARGET_FEATURE_INDICES]
            )
        return np.array(X), np.array(Y)

    X_train, Y_train = build_windows(train_feature_scaled, train_target)
    print(f"训练集序列总数: {len(X_train)}")
    print(f"训练集形状: X={X_train.shape}, Y={Y_train.shape}")

    X_val, Y_val = build_windows(val_feature_scaled, val_target)
    print(f"验证集序列总数: {len(X_val)}")
    print(f"验证集形状: X={X_val.shape}, Y={Y_val.shape}")

    X_test, Y_test = build_windows(test_feature_scaled, test_target)
    print(f"测试集序列总数: {len(X_test)}")
    print(f"测试集形状: X={X_test.shape}, Y={Y_test.shape}")

    print(f"输入形状: {X_test.shape}, 输出形状: {Y_test.shape}  (直接预测未来{OUTPUT_LEN}分钟)")

    # 6、构建 Tensor 数据集
    train_dataset = TensorDataset(torch.tensor(X_train).float(), torch.tensor(Y_train).float())
    val_dataset = TensorDataset(torch.tensor(X_val).float(), torch.tensor(Y_val).float())
    test_dataset = TensorDataset(torch.tensor(X_test).float(), torch.tensor(Y_test).float())
    
    # 返回目标的反标准化参数，供测试阶段还原为原始 NOX 单位。
    target_mean = scaler.mean_[TARGET_FEATURE_INDICES]
    target_scale = scaler.scale_[TARGET_FEATURE_INDICES]

    return train_dataset, val_dataset, test_dataset, X_train.shape[2], target_mean, target_scale

