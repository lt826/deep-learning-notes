"""

正则化的作用：
    缓解模型的过拟合情况

正则化的方式：
    L1正则化：权重可以变为0，相当于降维
    L2正则化：权重可以无限接近0
    Dropout：随机失活，每批次样本训练时，随机让一部分神经元死亡，防止一些特征对结果的影响较大(防止过拟合)，只有在训练时使用，测试时不需要
    BN(Batch Normalization):批量归一化
        思路：
            先对数据做标准化(会丢失一些信息)，然后再对数据做缩放(λ，理解为：w权重) 和 平移(β，理解为：b偏置)，再找补回一些信息
        应用场景：
            批量归一化在计算机视觉领域使用较多
"""

import torch
import torch.nn as nn

#1. 定义函数，演示：随机失活(Dropout)
def dm01():
    #1. 创建隐藏层输出结果
    t1 = torch.randint(1,10,(1,4)).float()
    print("t1:",t1)

    #2. 创建下一层的加权求和
    linear1 = nn.Linear(4,5)

    l1 = linear1(t1)
    print("l1:",l1)

    #3. 创建下一层的激活函数
    r1 = torch.relu(l1)
    print("r1:",r1)

    #4. 对激活值进行随机失活处理
    dropout = nn.Dropout(p=0.5)       #每个神经元有p(50%)的概率被失活
    d1 = dropout(r1)
    print("d1:",d1)                 #未被失活的进行缩放，缩放比例为：1 / (1-p)


#2. 定义函数，演示：批量归一化,处理一维数据
def dm02():
    #1. 创建样本数据
    t1 = torch.randn(size=(2,2))
    print("t1:",t1)

    #2. 创建线性层
    linear1 = nn.Linear(2,4)

    #3. 对数据进行线性变换
    l1 = linear1(t1)
    print("l1:",l1)

    #4. 创建批量归一化层
    bn1 = nn.BatchNorm1d(num_features=4)
    #5. 对线性处理结果l1进行批量归一化处理
    b1 = bn1(l1)
    print("b1:",b1)

#3. 定义函数，演示：批量归一化,处理二维数据
def dm03():
    #1. 创建图像样本数据
    #1张图片，2个通道，3行4列
    t1 = torch.randn(size=(1,2,3,4))
    print("t1:",t1)

    #2. 创建批量归一化层
    #参1：输入特征数=图片的通道数，参2：噪声值，参3：动量，参4：是否使用仿射参数
    bn1 = nn.BatchNorm2d(num_features=2,eps=1e-05, momentum=0.1, affine=True)

    #3. 对图像数据进行批量归一化处理
    b1 = bn1(t1)
    print("b1:",b1)


if __name__ == '__main__':
    # dm01()
    dm02()
    print('-'*30)
    dm03()
