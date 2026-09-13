"""

结论：
    针对于β(调节权重参数)来讲，其值越大说明：越依赖指数加权平均，越不依赖本次的梯度，数据就越平缓

"""

import torch
import matplotlib.pyplot as plt

ELEMENT_NUMBER = 30

#1. 实际平均温度
def dm01():
    #固定随机数种子
    torch.manual_seed(10)
    #产生30天的随机温度
    temperature = torch.randn(size=[ELEMENT_NUMBER,]) * 10
    print(temperature)
    #绘制平均温度
    days = torch.arange(1,ELEMENT_NUMBER + 1 ,1)
    plt.plot(days,temperature)
    plt.scatter(days,temperature)
    plt.show()


#2. 指数加权平均温度
def dm02(beta):
    #固定随机数种子
    torch.manual_seed(10)
    #产生30天的随机温度
    temperature = torch.randn(size=[ELEMENT_NUMBER,]) * 10
    print(temperature)

    exp_weight_avg = []
    #idx从1开始
    for idx,temp in enumerate(temperature,1):
        #第一个元素的EWA值等于自身
        if idx == 1:
            exp_weight_avg.append(temp)
            continue

        #第二个元素的EWA值等于上一个EWA乘以β + 本次温度乘以(1-β)
        new_temp = exp_weight_avg[idx - 2] * beta + (1 - beta) * temp
        exp_weight_avg.append(new_temp)

    #绘制平均温度
    days = torch.arange(1,ELEMENT_NUMBER + 1 ,1)
    plt.plot(days,exp_weight_avg,c='r')
    plt.scatter(days,temperature)
    plt.show()


if __name__ == '__main__':
    dm01()
    dm02(0.9)
    dm02(0.5)
