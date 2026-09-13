"""

利用pytorch模拟线性回归


numpy对象  -> torch张量对象 -> 数据集对象TensorDataset -> 数据加载器DataLoader
"""

import torch
from torch.utils.data import TensorDataset          #构造数据集对象
from torch.utils.data import DataLoader             #数据加载器
from torch import nn                                #nn模块中有平方损失函数和假设函数
from torch import optim                             #optim模块中有优化器函数
from sklearn.datasets import make_regression        #创建线性回归模型数据集
import matplotlib.pyplot as plt                     #可视化

import 自动微分模块.自动微分真实应用场景

plt.rcParams['font.sans-serif'] = ['SimHei']        #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False          #用来正常显示负号


def create_dataset():
    #创建数据集对象
    x, y ,coef = make_regression(
        n_samples=100,              #样本数
        n_features=1,               #特征数
        noise=20,                   #噪声,噪声越大，样本点越散
        coef=True,                   #是否返回系数，默认为False
        bias=12,                    #偏置
        random_state=23             #随机种子
    )

    #把上述数据封装成张量对象
    x = torch.tensor(x,dtype=torch.float)
    y = torch.tensor(y,dtype=torch.float)

    return x,y,coef

def train(x,y,coef):
    #创建数据集对象，把tensor -> 数据集对象 -> 数据加载器
    dataset = TensorDataset(x,y)

    #创建数据加载器对象
    #参1：数据集对象，参2：批次大小，参3：是否打乱数据(训练集打乱，测试集不打乱)
    dataloader = DataLoader(dataset,batch_size=10,shuffle=True)

    #创建线性模型
    #in_features:输入特征数，out_features:输出特征数
    model = nn.Linear(in_features=1,out_features=1)

    #创建损失函数对象
    criterion = nn.MSELoss()

    #创建优化器对象
    #参1：模型参数，参2：学习率
    optimizer = optim.SGD(model.parameters(),lr=0.01)

    #定义变量，分别表示：训练轮数，每轮的(平均)损失值
    epochs,loss_list = 100,[]

    #开始训练，按轮训练
    for epoch in range(epochs):
        #每轮训练总损失值，训练的样本数
        total_loss, total_samples = 0, 0
        #每轮按批次训练，所以从数据加载器中获取批次数据
        for train_x,train_y in dataloader:      #10批(10，10，10，10，10，10，10，10，10，10)
            y_pre = model(train_x)
            #计算(每轮的平均)损失值
            loss = criterion(y_pre,train_y.reshape(-1,1))
            #计算总损失和样本(批次)数
            total_loss += loss.item()
            total_samples += 1
            #梯度清零 + 反向传播 + 梯度更新
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        #把本轮的平均损失值添加到损失值列表中
        loss_list.append(total_loss/total_samples)
        print(f'第{epoch+1}轮平均损失值为：{total_loss/total_samples}')

    print(f'训练完成，最终的平均损失值为：{loss_list}')
    print(f"模型参数，权重为：{model.weight}，偏置为：{model.bias}")


    #绘制损失曲线
    plt.plot(range(epochs), loss_list)
    plt.title('损失曲线')
    plt.grid()
    plt.show()

    #绘制模型的预测值和真实值的折线图
    y_pred = torch.tensor(data = [v * model.weight + model.bias for v in x])
    y_true = torch.tensor(data = [v * coef + 12 for v in x])

    plt.plot(x, y_pred,color = 'red' ,label='预测值')
    plt.plot(x, y_true,color = 'green' ,label='真实值')
    plt.title('线性回归模型的预测值和真实值')
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == '__main__':
    x,y,coef = create_dataset()
    #print(f'x:{x}\ny:{y}\ncoef:{coef}')
    train(x,y,coef)