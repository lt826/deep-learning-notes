"""

涉及到的函数：
    torch.arange() 和 torch.linspace() 创建线性张量
    torch.rand() 和 torch.randn() 创建随机浮点类型张量
    torch.random.initial_seed() 和 torch.random.manual_seed() 设置随机种子
    torch.randint(low,high,size=()) 创建随机整数张量

要掌握的函数：
    arange(), linspace(), manual_seed(), randint()
"""

import torch

#1.定义函数，演示：创建线性张量
def dm01():
    #场景1：创建指定范围的线性张量
    #参1：起始值，参2：结束值，参3：步长
    t1 = torch.arange(0, 10, 2)
    print(f"t1:{t1}, type:{t1.type()}")
    print('-' * 30)

    #场景2：创建指定范围的线性张量  -> 等差数列
    #参1：起始值，参2：结束值，参3：元素个数
    t2 = torch.linspace(1, 10, 5)
    print(f"t2:{t2}, type:{t2.type()}")
    print('-' * 30)

#2.定义函数，演示：创建随机张量
def dm02():
    #设置随机种子
    #torch.initial_seed()       #默认采用当前系统的时间戳作为随机种子
    torch.manual_seed(10)       #设置随机种子

    #创建随机张量
    #场景1：均匀分布的（0，1）随机张量
    t1 = torch.rand(size= (2,3))
    print(f"t1:{t1}, type:{t1.type()}")
    print('-' * 30)

    #场景2：符合正态分布的随机张量
    t2 = torch.randn(size= (2,3))
    print(f"t2:{t2}, type:{t2.type()}")
    print('-' * 30)

    #场景3：创建随机整数张量
    t3 = torch.randint(low=1, high=10, size=(4,5))
    print(f"t3:{t3}, type:{t3.type()}")
    print('-' * 30)

if __name__ == '__main__':
    #dm01()
    dm02()