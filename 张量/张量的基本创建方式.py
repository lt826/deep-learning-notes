"""

张量（Tensor）：
    存储同一个类型元素的容器，且元素值必须是数值类型

PyTorch框架属于最常用的深度学习框架，无论是后续要学的ANN（人工神经网络），还是CNN（卷积神经网络），RNN（循环神经网络）
底层在处理数据时，都是使用张量来处理的

张量的基本创建方式：
    torch.tensor    根据指定数据创建张量
    torch.Tensor    根据形状创建张量，也可以用来创建指定数据的张量
    torch.IntTensor、torch.FloatTensor、torch.DoubleTensor    创建指定类型的张量

要掌握的函数：
    tensor()
"""
import torch
import numpy as np

#1.定义函数，演示： torch.tensor    根据指定数据创建张量
def dm01():
    #场景1：标量 -> 张量
    t1 = torch.tensor(10)
    print(f"t1:{t1}, type:{t1.type()}")
    print('-' * 30)

    #场景2：二维列表 -> 张量
    data = [[1,2,3],[4,5,6]]
    t2 = torch.tensor(data)
    print(f"t2:{t2}, type:{t2.type()}")
    print('-' * 30)

    #场景3：numpy nd数组 -> 张量
    data = np.random.randint(1,10, size=(2,3))
    t3 = torch.tensor(data)
    print(f"t3:{t3}, type:{t3.type()}")
    print('-' * 30)

#2.定义函数，演示： torch.Tensor    根据形状创建张量，也可以用来创建指定数据的张量
def dm02():
    #场景1：标量 -> 张量
    t1 = torch.Tensor(10)
    print(f"t1:{t1}, type:{t1.type()}")
    print('-' * 30)

    #场景2：二维列表 -> 张量
    data = [[1,2,3],[4,5,6]]
    t2 = torch.Tensor(data)
    print(f"t2:{t2}, type:{t2.type()}")
    print('-' * 30)

    #场景3：numpy nd数组 -> 张量
    data = np.random.randint(1,10, size=(2,3))
    t3 = torch.Tensor(data)
    print(f"t3:{t3}, type:{t3.type()}")
    print('-' * 30)

    #场景4：直接创建指定形状的张量
    t4 = torch.Tensor(2,3)
    print(f"t4:{t4}, type:{t4.type()}")
    print('-' * 30)

#3.定义函数，演示： torch.IntTensor、torch.FloatTensor、torch.DoubleTensor    创建指定类型的张量
def dm03():
    # 场景1：标量 -> 张量
    t1 = torch.IntTensor(10)
    print(f"t1:{t1}, type:{t1.type()}")
    print('-' * 30)

    # 场景2：二维列表 -> 张量
    data = [[1, 2, 3], [4, 5, 6]]
    t2 = torch.IntTensor(data)
    print(f"t2:{t2}, type:{t2.type()}")
    print('-' * 30)

    # 场景3：numpy nd数组 -> 张量
    data = np.random.randint(1, 10, size=(2, 3))
    t3 = torch.IntTensor(data)
    print(f"t3:{t3}, type:{t3.type()}")
    print('-' * 30)

    #场景4：如果类型不匹配，会尝试自动转换类型
    data = np.random.randint(1, 10, size=(2, 3))
    t4 = torch.FloatTensor(data)
    print(f"t4:{t4}, type:{t4.type()}")
    print('-' * 30)

if __name__ == '__main__':
    #dm01()                 #掌握
    #dm02()
    dm03()