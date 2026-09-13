"""

涉及到的函数：
    torch.ones 和 torch.ones_like        创建全为1的张量
    torch.zeros 和 torch.zeros_like      创建全为0的张量
    torch.full 和 torch.full_like        创建全为指定值的张量

需要掌握的函数：
    zeros(),full()
"""

import torch


#场景1：torch.ones 和 torch.ones_like        创建全为1的张量
t1 = torch.ones(2,3)
print(f"t1:{t1}, type:{t1.type()}")
print('-' * 30)

t2 = torch.tensor([[1,2,3],[4,5,6]])

#基于t2的形状创建全为0的张量
t3 = torch.ones_like(t2)
print(f"t3:{t3}, type:{t3.type()}")
print('-' * 30)


#场景2：torch.zeros 和 torch.zeros_like      创建全为0的张量
t4 = torch.zeros(2,3)
print(f"t4:{t4}, type:{t4.type()}")
print('-' * 30)

t5 = torch.tensor([[1,2,3],[4,5,6]])

#基于t6的形状创建全为0的张量
t6 = torch.zeros_like(t2)
print(f"t6:{t6}, type:{t6.type()}")
print('-' * 30)

#场景3：torch.full 和 torch.full_like        创建全为指定值的张量
t7 = torch.full(size=(2,3), fill_value=555)
print(f"t7:{t7}, type:{t7.type()}")
print('-' * 30)

t8 = torch.tensor([[1,2,3],[4,5,6]])

#基于t8的形状创建全为255的张量
t9 = torch.full_like(t2, fill_value=255)
print(f"t9:{t9}, type:{t9.type()}")
print('-' * 30)

