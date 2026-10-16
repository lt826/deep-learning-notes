"""

需求：
    求y = x**2 + 20 的极小值点并打印y是最小值时w的值（梯度）

解题步骤：
    1.定义点 x = 10
    2.定义函数 y = x**2 + 20
    3.利用梯度下降法，循环迭代100次，求最优解
    3.1 正向计算(前向传播)
    3.2 梯度清零
    3.3 反向传播
    3.4 梯度更新

"""

import torch


# 1.定义点 x = 10
x = torch.tensor(10.0, requires_grad=True,dtype=torch.float)

# 2.定义函数 y = x**2 + 20
loss = x**2 + 20

# 3.利用梯度下降法，循环迭代100次，求最优解
print(f"初始值:{x}, loss:{loss}")


for i in range(300):
    # 3.1 正向计算(前向传播)
    loss = x**2 + 20

    # 3.2 梯度清零,默认梯度会累加,第一次还没有梯度，所以需要做非空判断
    if x.grad is not None:
        x.grad.zero_()

    # 3.3 反向传播
    loss.sum().backward()

    # 3.4 梯度更新
    x.data = x.data - 0.01 * x.grad

    print(f'第{i+1}次，权重初始值：{x:.4f},loss:{loss:.4f}')

#打印最终结果
print(f'最终结果 x:{x},梯度：{x.grad:.4f},loss:{loss:.4f} ')