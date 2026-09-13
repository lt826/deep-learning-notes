"""

神经网络搭建流程：
    1. 定义一个类，继承：nn.Module
    2. 在__init__()方法中，搭建神经网络
    3. 在forward()方法中，完成：前向传播

"""

import torch
import torch.nn as nn
from torchsummary import summary    #计算模型参数，查看模型结构


#todo：搭建神经网络，即：自定义一个类，继承nn.Module
class ModelDemo(nn.Module):
    #todo：1. 在init魔法方法中，完成初始化：父类成员，及神经网络搭建
    def __init__(self):
        super().__init__()
        #搭建神经网络 -> 隐藏层 + 输出层
        #隐藏层1：输入特征数：3，输出特征数：3
        self.linear1 = nn.Linear(3,3)
        #隐藏层2：输入特征数：3，输出特征数：2
        self.linear2 = nn.Linear(3,2)
        #输出层：输入特征数：2，输出特征数：2
        self.output = nn.Linear(2,2)

        #对隐藏层参数进行初始化
        #隐藏层1
        nn.init.xavier_normal_(self.linear1.weight)
        nn.init.zeros_(self.linear1.bias)

        #隐藏层2
        nn.init.kaiming_normal(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)

    #todo:2. 前向传播：输入层 -> 隐藏层 -> 输出层
    def forward(self,x):                     #forward方法会因为类实例化后自动调用，所以方法名不能修改
        #隐藏层1 计算： 加权求和 + Sigmoid激活函数
        #分解版
        # x = self.linear1(x)     #加权求和
        # x = torch.sigmoid(x)    #激活函数
        #合并版
        x = torch.sigmoid(self.linear1(x))

        #隐藏层2 计算： 加权求和 + ReLU激活函数
        x = torch.relu(self.linear2(x))

        #输出层  计算： 加权求和 + Softmax激活函数
        #dim = -1,表示按行求和,即一个样本一个样本的处理
        x = torch.softmax(self.output(x),dim=-1)

        #返回预测值
        return x


#todo：模型训练
def train():
    my_model = ModelDemo()
    # print(my_model)

    #创建数据集样本
    data = torch.randn(size=(5,3))

    #调用神经网络大模型
    output = my_model(data)         #底层调用forward方法，进行前向传播
    print(f'output:{output}')
    print(f'output.shape:{output.shape}')
    print(f'output.requires_grad:{output.requires_grad}')       #是否自动微分

    #计算和查看模型参数
    print('======================计算模型参数======================')
    summary(my_model, (5,3))          #查看模型结构，计算模型参数

    print('======================查看模型参数======================')
    for name, param in my_model.named_parameters():
        print(f'name:{name}')
        print(f'param:{param}\n')



if __name__ == '__main__':
    train()