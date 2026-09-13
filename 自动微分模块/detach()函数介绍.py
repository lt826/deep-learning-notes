"""

自动微分 = 求导，即： 基于损失函数，计算梯度
结合权重更新公式：W新 = W旧 - 学习率 * 梯度，实现参数更新

一个张量一旦设置了自动微分，这个张量就不能直接转成numpy的ndarray对象了，需要使用detach()函数解决

"""

import torch
import numpy as np


#定义张量
t1 = torch.tensor([10,20],requires_grad=True,dtype=torch.float)
print(f't1:{t1},type:{t1.type()}')

# n1 = t1.numpy()         #报错

#通过detach()函数解决
t2 = t1.detach()
print(f't2:{t2},type:{t2.type()}')

#t1和t2是否共享同一块空间     -> 共享
t1.data[0] = 100
print(f't1:{t1},type:{t1.type()}')
print(f't2:{t2},type:{t2.type()}')

#查看t1和t2是否可以自动微分
print(f't1:{t1.requires_grad}')         #True
print(f't2:{t2.requires_grad}')         #False

n1 = t2.numpy()
print(f'n1:{n1},type:{type(n1)}')

#最终版
n1 = t1.detach().numpy()
