"""

结论：
    1. 先前向传播（正向传播），计算出预测值
    2. 基于损失函数，结合预测值和真实值，来计算梯度
    3. 结合权重更新公式：W新 = W旧 - 学习率 * 梯度，实现权重更新


"""

import torch

#定义x,表示输入数据(特征)，假设：2行5列，全1矩阵
x = torch.ones(2,5)
print(f'x:{x}')

#定义y，表示真实值(标签)，假设：2行3列，全0矩阵
y = torch.zeros(2,3)
print(f'y:{y}')

#初始化(可自动微分的)权重 和 偏置
w = torch.randn(5,3, requires_grad=True)        #因为x@w后要为y,所以w是5行3列
print(f'w:{w.data}')
b = torch.randn(3, requires_grad=True)          #因为x@w+b后要为y,所以b是3列
print(f'b:{b.data}')
print('-'*50)

#前向传播，计算出预测值
z = torch.matmul(x,w) + b         #z = x@w + b
print(f'z:{z.data}')
print('-'*50)

#定义损失函数
criterion = torch.nn.MSELoss()      #neural network:神经网络
loss = criterion(z, y)
print(f'loss:{loss}')
print('-'*50)

#进行自动微分，求导
loss.backward()

print(f'w的梯度:{w.grad}')
print(f'b的梯度:{b.grad}')

#后续就是：W新 = W旧 - 学习率 * 梯度，实现权重更新
