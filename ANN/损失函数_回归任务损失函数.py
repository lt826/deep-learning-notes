"""

回归任务常用损失函数：
    MAE：Mean Absolute Error，平均绝对误差
        公式：
            误差绝对值之和 / 样本总数
        类似于L1正则化，权重可以降为0，数据会变得稀疏
        弊端：
            在0点不平滑，可能错过最小值

    MSE：Mean Squared Error，均方误差
        公式：
            误差平方和 / 样本总数
        弊端：
            如果差值过大，可能存在梯度爆炸的情况

    Smooth L1：
    就是基于MAE和MSE做的综合，在[-1,1]是L2(MSE),其它段是L1
    这样既解决了L1不平滑的问题(0点不可导，可能错过最小值)
    也解决了L2梯度爆炸的问题
"""

import torch
import torch.nn as nn


#1. 定义函数，演示：MAE损失函数
def dm01():
    #1. 定义变量，记录真实值
    y_true = torch.tensor([1.0, 2.0, 3.0],dtype=torch.float)

    #2. 定义变量，记录预测值
    y_pred = torch.tensor([1.2, 2.4, 3.1],requires_grad=True)

    #3. 创建MAE损失函数对象
    criterion = nn.L1Loss()

    #4. 计算损失值
    loss = criterion(y_pred, y_true)

    #5. 输出结果
    print(f"MAE:{loss}")

#2. 定义函数，演示：MSE损失函数
def dm02():
    # 1. 定义变量，记录真实值
    y_true = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float)

    # 2. 定义变量，记录预测值
    y_pred = torch.tensor([1.2, 2.4, 3.1], requires_grad=True)

    # 3. 创建MSE损失函数对象
    criterion = nn.MSELoss()

    # 4. 计算损失值
    loss = criterion(y_pred, y_true)

    # 5. 输出结果
    print(f"MSE:{loss}")

#2. 定义函数，演示：SmoothL1损失函数
def dm03():
    # 1. 定义变量，记录真实值
    y_true = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float)

    # 2. 定义变量，记录预测值
    y_pred = torch.tensor([1.2, 2.4, 3.1], requires_grad=True)

    # 3. 创建SmoothL1损失函数对象
    criterion = nn.SmoothL1Loss()

    # 4. 计算损失值
    loss = criterion(y_pred, y_true)

    # 5. 输出结果
    print(f"SmoothL1:{loss}")



if __name__ == '__main__':
    dm01()
    dm02()
    dm03()




