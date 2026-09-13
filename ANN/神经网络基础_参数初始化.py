"""

参数初始化的目的：
    1. 防止梯度消失或梯度爆炸
    2. 提高收敛速度
    3. 打破对称性，即每层的每个神经元的参数都不同

参数初始化的方式：
    无法打破对称性的：
        全0，全1，固定值
    可以打破对称性的：
        随机初始化，正态分布初始化，kaiming初始化，xavier初始化

总结：
    1. 记忆kaiming初始化，xavier初始化，全0初始化
    2. 关于初始化选择上：
        1. 激活函数ReLU及其变种：优先用kaiming
        2. 激活函数非ReLU：优先用xavier
        3. 如果是浅层网络：可以考虑使用随机初始化
"""

import torch.nn as nn


#1. 均匀分布随机初始化
def dm01():
    #1.创建一个线性层，输入维度5，输出维度3
    linear = nn.Linear(5, 3)
    #2.对权重进行随机初始化，从0-1均匀分布产生参数
    nn.init.uniform_(linear.weight)
    #3.对偏置进行随机初始化，从0-1均匀分布产生参数
    nn.init.uniform_(linear.bias)

    print(linear.weight.data)
    print(linear.bias.data)


#2. 固定初始化
def dm02():
    # 1.创建一个线性层，输入维度5，输出维度3
    linear = nn.Linear(5, 3)
    # 2.对权重进行固定初始化，设置固定值为：3
    nn.init.constant_(linear.weight, 3)
    # 3.对偏置进行固定初始化，设置固定值为：3
    nn.init.constant_(linear.bias,3)

    print(linear.weight.data)
    print(linear.bias.data)

#3. 全0初始化
def dm03():
    # 1.创建一个线性层，输入维度5，输出维度3
    linear = nn.Linear(5, 3)
    # 2.对权重进行全0初始化
    nn.init.zeros_(linear.weight)
    # 3.对偏置进行全0初始化
    nn.init.zeros_(linear.bias)

    print(linear.weight.data)
    print(linear.bias.data)



#4. 全1初始化
def dm04():
    # 1.创建一个线性层，输入维度5，输出维度3
    linear = nn.Linear(5, 3)
    # 2.对权重进行全1初始化
    nn.init.ones_(linear.weight)
    # 3.对偏置进行全1初始化
    nn.init.ones_(linear.bias)

    print(linear.weight.data)
    print(linear.bias.data)




#5. 正态分布随机初始化
def dm05():
    # 1.创建一个线性层，输入维度5，输出维度3
    linear = nn.Linear(5, 3)
    # 2.对权重进行正态分布随机初始化，均值为0，标准差为1
    nn.init.normal_(linear.weight, mean=0, std=1)
    # 3.对偏置进行正态分布随机初始化，均值为0，标准差为1
    nn.init.normal_(linear.bias, mean=0, std=1)

    print(linear.weight.data)
    print(linear.bias.data)


#6. kaiming初始化
def dm06():
    # 1.创建一个线性层，输入维度5，输出维度3
    linear = nn.Linear(5, 3)
    # 2.对权重进行kaiming正态初始化
    nn.init.kaiming_normal_(linear.weight)
    print(linear.weight.data)


    #4.对权重进行kaiming均匀初始化
    nn.init.kaiming_uniform_(linear.weight)
    print(linear.weight.data)


#7. xavier初始化
def dm07():
    # 1.创建一个线性层，输入维度5，输出维度3
    linear = nn.Linear(5, 3)
    # 2.对权重进行xavier正态初始化
    nn.init.xavier_normal_(linear.weight)
    print(linear.weight.data)

    # 4.对权重进行xavier均匀初始化
    nn.init.xavier_uniform_(linear.weight)
    print(linear.weight.data)


if __name__ == '__main__':
    # dm01()
    # dm02()
    # dm03()
    # dm04()
    # dm05()
    # dm06()
    dm07()