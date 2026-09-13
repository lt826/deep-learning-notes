"""

梯度下降相关介绍：
    概述：
        梯度下降是结合本次损失函数的导数(作为梯度)基于学习率来更新权重的
    公式：
        W新 = W旧 - 学习率 * (本次的)梯度
    存在的问题：
        1. 遇到平缓区域，梯度下降(权重更新)可能会变慢
        2. 可能会遇到鞍点(梯度为0)
        3. 可能会遇到局部最小值
    解决思路：
        从上述的学习率或者梯度入手，进行优化，于是有了：动量法Momentum，自适应学习率AdaGrad，RMSProp，综合衡量：Adam

    动量法Momentum：
        公式：
            St = β * St-1 - (1-β) * Gt
            解释：
                St: 本次的指数移动加权平均结果
                β: 调节权重系数，越大，数据越平缓，历史指数移动加权平均比重越大，本次梯度权重越小
                St-1: 历史的指数移动加权平均结果
                Gt: 本次计算出的梯度
        加入动量法后的梯度更新公式：
            W新 = W旧 - 学习率 * St

    自适应学习率AdaGrad：
        公式：
            St = St-1 + Gt * Gt
            解释：
                Gt: 本次计算出的梯度
                St: 累计平方梯度
                St-1: 历史的累计平方梯度
        学习率：
            学习率 = 学习率 / (sqrt(St) + ε)
            解释：
                ε：1e-10，目的：防止分母为0
        梯度下降公式：
            W新 = W旧 - 调整后的学习率 * Gt
        缺点：
            可能会导致学习率过早，过量的降低，导致模型后期学习率太小，较难找到最优解

    自适应学习率RMSProp -> 可以看做是AdaGrad的改进版，加入了调和权重系数
        公式：
            St = β * St-1 + (1-β) * Gt * Gt
            解释：
                Gt: 本次计算出的梯度
                St: 累计平方梯度
                St-1: 历史的累计平方梯度
                β: 调节权重系数
        学习率：
            学习率 = 学习率 / (sqrt(St) + ε)
            解释：
                ε：1e-10，目的：防止分母为0
        梯度下降公式：
            W新 = W旧 - 调整后的学习率 * Gt
        优点：
            RMSProp通过引入了权重系数β，控制历史梯度对历史梯度信息获取的多少，从而避免了AdaGrad过早的降低学习率

    自适应矩估计Adam：
        思路：
            既优化学习率，也优化梯度
        公式：
            一阶矩：算均值
                Mt = β1 * Mt-1 + (1-β1) * Gt            充当：梯度
                St = β2 * St-1 + (1-β2) * Gt * Gt       充当：学习率
            二阶矩：梯度的方差
                Mt^ = Mt / (1 - β1^t)
                St^ = St / (1 - β2^t)
            权重更新公式：
                W新 = W旧 - 学习率 / (sqrt(St^) + ε) * Mt^
        Adam = RMSProp + Momentum

    总结：如何选择梯度下降优化方法
        简单任务和较小的模型：
            SGD,动量法
        复杂任务或者有大量数据：
            Adam
        需要处理稀疏数据或者文本数据：
            AdaGrad,RMSProp
"""
import torch
import torch.optim as optim

#1. 定义函数，演示：梯度下降优化方法 -> 动量法(Momentum)
def dm01_momentum():
    #初始化权重系数
    w = torch.tensor(1.0,requires_grad=True,dtype=torch.float)
    #定义损失函数
    criterion = ((w**2)/2.0)
    #创建优化器 -> 基于SGD(随机梯度下降)，加入参数momentum，就是动量法
    optimizer = optim.SGD([w],lr=0.01,momentum=0.9)         #momentum默认为0.0，只考虑本次的梯度
    #计算梯度值：梯度清零 + 反向传播 + 参数更新
    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    print(f"w: {w},w.grad: {w.grad}")

    #重复上述步骤
    criterion = ((w**2)/2.0)
    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    print(f"w: {w},w.grad: {w.grad}")

#2. 定义函数，演示：梯度下降优化方法 -> 自适应学习率(AdaGrad)
def dm02_adagrad():
    # 初始化权重系数
    w = torch.tensor(1.0, requires_grad=True, dtype=torch.float)
    # 定义损失函数
    criterion = ((w ** 2) / 2.0)
    # 创建优化器 -> 基于AdaGrad(自适应学习率)
    optimizer = optim.Adagrad([w], lr=0.01)
    # 计算梯度值：梯度清零 + 反向传播 + 参数更新
    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    print(f"w: {w},w.grad: {w.grad}")

    # 重复上述步骤
    criterion = ((w ** 2) / 2.0)
    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    print(f"w: {w},w.grad: {w.grad}")


#3. 定义函数，演示：梯度下降优化方法 -> 自适应学习率(RMSProp)
def dm03_rmsprop():
    # 初始化权重系数
    w = torch.tensor(1.0, requires_grad=True, dtype=torch.float)
    # 定义损失函数
    criterion = ((w ** 2) / 2.0)
    # 创建优化器 -> 基于RMSProp(自适应学习率)
    optimizer = optim.RMSprop([w], lr=0.01,alpha=0.9)   # alpha: 调节权重系数
    # 计算梯度值：梯度清零 + 反向传播 + 参数更新
    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    print(f"w: {w},w.grad: {w.grad}")

    # 重复上述步骤
    criterion = ((w ** 2) / 2.0)
    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    print(f"w: {w},w.grad: {w.grad}")


#4. 定义函数，演示：梯度下降优化方法 -> 自适应矩估计(Adam)
def dm04_adam():
    # 初始化权重系数
    w = torch.tensor(1.0, requires_grad=True, dtype=torch.float)
    # 定义损失函数
    criterion = ((w ** 2) / 2.0)
    # 创建优化器 -> 基于Adam(自适应矩估计)
    optimizer = optim.Adam([w], lr=0.01, betas=[0.9, 0.999])  #betas = (梯度用的衰减系数，学习率用的衰减系数)
    # 计算梯度值：梯度清零 + 反向传播 + 参数更新
    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    print(f"w: {w},w.grad: {w.grad}")

    # 重复上述步骤
    criterion = ((w ** 2) / 2.0)
    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    print(f"w: {w},w.grad: {w.grad}")


if __name__ == '__main__':
    dm01_momentum()
    print('-'*40)
    dm02_adagrad()
    print('-'*40)
    dm03_rmsprop()
    print('-'*40)
    dm04_adam()