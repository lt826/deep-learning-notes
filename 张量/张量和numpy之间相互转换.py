"""

涉及到的API:
    场景1：张量 -> numpy nd数组对象
        张量对象.numpy()            共享内存
        张量对象.numpy().copy()    不共享内存
    场景2：numpy nd数组 -> 张量
        from_numpy()              共享内存
        torch.tensor(nd数组)       不共享内存
    场景3：从标量张量中提取其内容
        标量张量.item()

掌握：
    张量对象.numpy()
    torch.tensor(nd数组)
    标量张量.item()
"""
import torch
import numpy as np

#1.定义函数，演示：张量 -> numpy
def dm01():
    t1 = torch.tensor([1,2,3,4,5])
    #n1 = t1.numpy()                 #共享内存
    n1 = t1.numpy().copy()          #不共享内存

    n1[0] = 100
    print(t1)
    print(n1)

#2.定义函数，演示：numpy -> 张量
def dm02():
    n1 = np.array([11,22,33])
    t1 = torch.from_numpy(n1)       #共享内存
    t2 = torch.tensor(n1)           #不共享内存

    n1[0] = 100
    print(t1)
    print(t2)

#3.定义函数，演示：从标量张量中提取其内容
def dm03():
    t1 = torch.tensor(100)
    t2 = torch.tensor([10.10,])

    a = t1.item()
    b = t2.item()
    print(f"value:{a},type:{type(a)}")
    print(f"value:{b},type:{type(b)}")


if __name__ == '__main__':
    # dm01()
    # dm02()
    dm03()