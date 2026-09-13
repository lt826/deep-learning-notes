"""

二分类任务的损失函数(BCELoss):
    公式：
        Loss = -ylog(预测值) - (1-y)log(1-预测值)
    细节：
        因为公式中没有包含Sigmoid激活函数，所以使用BCELoss的时候，还需要手动指定Sigmoid激活函数

"""

import torch
import torch.nn as nn

def dm01():
    # 1. 手动创建样本的真实值
    y_true = torch.tensor([1, 0, 0], dtype=torch.float)

    # 2. 手动创建样本的预测值
    y_pred = torch.tensor([0.856, 0.556, 0.415])

    # 3. 创建二分类交叉熵损失函数
    criterion = nn.BCELoss()

    # 4. 计算损失值
    loss = criterion(y_pred, y_true)
    print(f'loss:{loss}')


if __name__ == '__main__':
    dm01()


