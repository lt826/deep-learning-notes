"""

学习率衰减策略介绍：
    目的：
        较之于AdaGrad，RMSProp，Adam方式，我们可以通过等间隔、指定间隔、指数等方式，来手动控制学习率的调整

    分类：
        1. 等间隔学习率衰减
        2. 指定间隔学习率衰减
        3. 指数学习率衰减

    等间隔学习率衰减：
        step_size: 间隔的轮数，即：每多少轮进行一次学习率衰减
        gamma: 学习率衰减系数，即：lr新 = lr旧 * gamma
    指定间隔学习率衰减：
        milestones: 指定的轮数，即：在哪些轮数进行学习率衰减
        gamma: 学习率衰减系数，即：lr新 = lr旧 * gamma
    指数学习率衰减：
        前期学习率衰减快，后期学习率衰减慢
        公式：
        lr新 = lr旧 * gamma**epoch

总结：
    等间隔学习率衰减：
        优点：
            直观、易于调试、适用于大批量数据
        缺点：
            学习率变化较大，可能跳过最优解
        应用场景：
            大型数据集，较为简单的任务

    指定间隔学习率衰减：
        优点：
            易于调试，稳定训练过程
        缺点：
            在某些情况下可能衰减过快，导致优化提前停滞
        应用场景：
            对训练平稳性要求较高的任务

    指数学习率衰减：
        优点：
            平滑，且考虑历史更新，收敛稳定性较强
        缺点：
            超参调节较为复杂，可能需要更多的资源
        应用场景：
            高精度训练，避免过快收敛

"""
import sympy.physics.units
import torch
from torch import optim
import matplotlib.pyplot as plt


#1. 定义函数，演示：等间隔学习率衰减
def dm01():
    #1. 定义变量，记录初始的学习率，训练的轮数，每轮训练的批次数
    lr,epochs,iteration = 0.1,200,10

    #2. 创建数据集
    y_true = torch.tensor([0])
    x = torch.tensor([1.0],dtype=torch.float)
    w = torch.tensor([1.0],dtype=torch.float,requires_grad=True)

    #3.创建优化器对象，动量法
    optimizer = optim.SGD([w],lr=lr,momentum=0.9)

    #4. 创建等间隔学习率衰减对象
    scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=50,gamma=0.5)

    #5. 创建两个列表，分别表示：训练轮数和对应的学习率
    epochs_list,lr_list = [],[]

    #6. 循环遍历训练轮数，进行具体的训练
    for epoch in range(epochs):
        #7. 获取当前轮数和学习率，并保存到列表中
        epochs_list.append(epoch)
        lr_list.append(scheduler.get_last_lr())             #获取当前学习率

        #8. 循环遍历，每轮每批次进行训练
        for batch in range(iteration):
            #9. 先计算预测值，然后基损失函数计算损失
            y_pred = w * x
            loss = (y_pred - y_true) ** 2                   #最小二乘
            #10. 梯度清零 + 反向传播 + 优化器更新参数
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        #11. 更新学习率
        scheduler.step()

    #12. 打印结果
    #print(f'lr_list:{lr_list}')

    #13. 可视化
    plt.plot(epochs_list,lr_list)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.legend()
    plt.show()

#2. 定义函数，演示：指定间隔学习率衰减
def dm02():
    # 1. 定义变量，记录初始的学习率，训练的轮数，每轮训练的批次数
    lr, epochs, iteration = 0.1, 200, 10

    # 2. 创建数据集
    y_true = torch.tensor([0])
    x = torch.tensor([1.0], dtype=torch.float)
    w = torch.tensor([1.0], dtype=torch.float, requires_grad=True)

    # 3.创建优化器对象，动量法
    optimizer = optim.SGD([w], lr=lr, momentum=0.9)

    # 4. 创建指定间隔学习率衰减对象
    #参1：优化器对象，参2：指定间隔的轮数，参3：学习率衰减系数
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50,125,160], gamma=0.5)

    # 5. 创建两个列表，分别表示：训练轮数和对应的学习率
    epochs_list, lr_list = [], []

    # 6. 循环遍历训练轮数，进行具体的训练
    for epoch in range(epochs):
        # 7. 获取当前轮数和学习率，并保存到列表中
        epochs_list.append(epoch)
        lr_list.append(scheduler.get_last_lr())  # 获取当前学习率

        # 8. 循环遍历，每轮每批次进行训练
        for batch in range(iteration):
            # 9. 先计算预测值，然后基损失函数计算损失
            y_pred = w * x
            loss = (y_pred - y_true) ** 2  # 最小二乘
            # 10. 梯度清零 + 反向传播 + 优化器更新参数
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # 11. 更新学习率
        scheduler.step()

    # 12. 打印结果
    #print(f'lr_list:{lr_list}')

    # 13. 可视化
    plt.plot(epochs_list, lr_list)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.legend()
    plt.show()

#3. 定义函数，演示：指数学习率衰减
def dm03():
    # 1. 定义变量，记录初始的学习率，训练的轮数，每轮训练的批次数
    lr, epochs, iteration = 0.1, 200, 10

    # 2. 创建数据集
    y_true = torch.tensor([0])
    x = torch.tensor([1.0], dtype=torch.float)
    w = torch.tensor([1.0], dtype=torch.float, requires_grad=True)

    # 3.创建优化器对象，动量法
    optimizer = optim.SGD([w], lr=lr, momentum=0.9)

    # 4. 创建指数学习率衰减对象
    # 参1：优化器对象，参2：学习率衰减系数
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    # 5. 创建两个列表，分别表示：训练轮数和对应的学习率
    epochs_list, lr_list = [], []

    # 6. 循环遍历训练轮数，进行具体的训练
    for epoch in range(epochs):
        # 7. 获取当前轮数和学习率，并保存到列表中
        epochs_list.append(epoch)
        lr_list.append(scheduler.get_last_lr())  # 获取当前学习率

        # 8. 循环遍历，每轮每批次进行训练
        for batch in range(iteration):
            # 9. 先计算预测值，然后基损失函数计算损失
            y_pred = w * x
            loss = (y_pred - y_true) ** 2  # 最小二乘
            # 10. 梯度清零 + 反向传播 + 优化器更新参数
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # 11. 更新学习率
        scheduler.step()

    # 12. 打印结果
    # print(f'lr_list:{lr_list}')

    # 13. 可视化
    plt.plot(epochs_list, lr_list)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    # dm01()
    # dm02()
    dm03()