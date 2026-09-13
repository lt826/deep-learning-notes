"""

损失函数介绍：
    概述：
        损失函数也叫成本函数、目标函数、代价函数、误差函数，就是用来衡量模型好坏的，即：模型预测值与真实值之间的差异。

    分类：
        分类问题：
            多分类交叉熵损失：CrossEntropyloss
            二分类交叉熵损失：BCELoss
        回归问题：
            平均平方误差：MSE
            均方误差：MSE
            Smooth L1：结合上述两个的特点做的升级，优化


多分类交叉熵损失：CrossEntropyLoss
    设计思路：
        Loss = -∑ylog(S(f(x)))
        x:          样本
        f(x):       加权求和
        S(f(x)):    处理后的概率
        y:          样本x属于某一个类别的真实概率

    细节：
        CrossEntropyLoss = Softmax() + 损失计算，后续如果使用这个损失函数，则：输出层不需要使用Softmax激活函数
"""

import torch
import torch.nn as nn



def dm01():
    #1. 手动创建样本的真实值 -> 就是上述公式中的y
    y_true = torch.tensor([[1,0,0],[0,1,1]],dtype=torch.float)

    #2. 手动创建样本的预测值 -> 就是上述公式中的f(x)
    y_pred = torch.tensor([[0.8,0.1,0.1],[0.2,0.4,0.4]],requires_grad=True,dtype=torch.float)

    #3. 创建多分类交叉熵损失函数
    criterion = nn.CrossEntropyLoss()

    #4. 计算损失值
    loss = criterion(y_pred, y_true)
    print(f'loss:{loss}')


if __name__ == '__main__':
    dm01()