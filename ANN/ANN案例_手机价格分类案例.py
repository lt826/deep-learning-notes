"""
背景：
    基于手机的20列特征 -> 预测手机的价格区间(4个区间)     分类问题

ANN案例的实现步骤：
    1. 构建数据集
    2. 搭建神经网络
    3. 模型训练
    4. 模型测试

优化思路：
    1. 优化方法由SGD调整为Adam
    2. 学习率由0.01调整为0.001
    3. 对数据进行标准化
    4. 增加网络深度
    5. 调整训练轮数
"""

import torch                                    #PyTorch框架，封装了张量的各种操作
from torchsummary import summary                #模型结构可视化
from torch.utils.data import TensorDataset      #数据集对象          数据 -> Tensor -> 数据集 -> 数据加载器
from torch.utils.data import DataLoader         #数据加载器
import torch.nn as nn                           #神经网络模块
import torch.optim as optim                     #优化器模块
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split        #数据集分割模块
import pandas as pd                             #数据处理模块
import numpy as np                              #数组操作模块
import time                                     #时间模块


#todo 1. 构建数据集
def create_dataset():
    #1. 加载csv文件数据集
    data = pd.read_csv('./data/手机价格预测.csv')

    #2. 获取x特征列，y标签列
    x,y = data.iloc[:,:-1],data.iloc[:,-1]
    # print(f"x:{x.head()},{x.shape}")
    # print(f"y:{y.head()},{y.shape}")

    #3. 把特征列转成浮点型
    x = x.astype('float32')

    #4. 切分数据集
    #stratify=y 表示按y的类别比例切分数据
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42,stratify=y)

    #5. 标准化处理
    scaler = StandardScaler()
    # 必须在训练集上 fit_transform，测试集上仅 transform
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    #6. 把数据集封装成张量数据集  思路：数据 -> Tensor -> 数据集TensorDateSet -> 数据加载器DataLoader
    train_dataset = TensorDataset(torch.tensor(x_train_scaled),torch.tensor(y_train.values))
    test_dataset = TensorDataset(torch.tensor(x_test_scaled),torch.tensor(y_test.values))

    #7.返回结果
    #x_train.shape[1]:20,充当输入特征数，len(np.unique(y)):4,充当输出标签数
    return train_dataset,test_dataset,x_train.shape[1],len(np.unique(y))


#todo 2. 搭建神经网络
class PhonePriceModel(nn.Module):
    #1.定义__init__魔法方法，初始化父类，构建神经网络
    def __init__(self,input_dim,output_dim):
        super().__init__()
        #隐藏层1
        self.linear1 = nn.Linear(input_dim,128)
        #隐藏层2
        self.linear2 = nn.Linear(128,64)
        #输出层
        self.output = nn.Linear(64,output_dim)

    #2.前向传播
    def forward(self,x):
        #隐藏层1：加权求和 + ReLU激活函数
        x = torch.relu(self.linear1(x))
        #隐藏层2：加权求和 + ReLU激活函数
        x = torch.relu(self.linear2(x))
        #输出层：加权求和 + Sigmoid激活函数
        #因为后续要用多分类交叉熵损失函数CrossEntropyLoss()
        #CrossEntropyLoss() = softmax() +损失计算，所以此处只需要加权求和
        x = self.output(x)
        return x


#todo 3. 模型训练
def train(train_dataset,input_dim,output_dim):
    #1.构建数据加载器
    #参1：数据集对象，参2：每批的数据条数，参3：是否打乱数据(训练集：打乱，测试集：不打乱)
    train_loader = DataLoader(train_dataset,batch_size=16,shuffle=True)
    #2.构建模型
    model = PhonePriceModel(input_dim,output_dim)
    #3.定义损失函数，因为是多分类，所以用多分类交叉熵损失函数
    criterion = nn.CrossEntropyLoss()
    #4.创建优化器
    optimizer = optim.Adam(model.parameters(),lr=0.001)
    #5.模型训练
    epochs = 100
    for epoch in range(epochs):
        #定义变量，表示训练的损失值，训练批次数
        total_loss,batch_num = 0.0,0
        #定义变量，表示训练开始时间
        start = time.time()
        #开始本轮的各个批次训练
        for x,y in train_loader:
            #切换模型(状态)
            model.train()       #训练模式           model.eval()为测试模式
            y_pred = model(x)
            #计算损失
            loss = criterion(y_pred,y)
            #梯度清零，反向传播，优化参数
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #累加损失值
            total_loss += loss.item()
            batch_num += 1
        #本轮训练结束，打印训练信息
        print(f'epoch:{epoch+1},loss:{total_loss/batch_num:.4f},time:{time.time()-start:.2f}s')

    #6.保存模型(参数)
    #参1：模型对象参数(权重矩阵，偏置矩阵)
    torch.save(model.state_dict(),'./model/phone_price_model.pth')


#todo 4. 模型测试
def evaluate(test_dataset,input_dim,output_dim):
    #1.创建神经网络分类对象
    model = PhonePriceModel(input_dim,output_dim)
    #2.加载模型参数
    model.load_state_dict(torch.load('./model/phone_price_model.pth'))
    #3.创建测试集的数据加载器对象
    # 参1：数据集对象，参2：每批的数据条数，参3：是否打乱数据(训练集：打乱，测试集：不打乱)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    correct = 0
    #4.获取每批次的数据
    for x,y in test_loader:
        #切换模型(状态)
        model.eval()        #测试模式
        y_pred = model(x)
        #根据加权求和，得到类别，用argmax()获取最大值对应的下标，就是类别
        y_pred = torch.argmax(y_pred,dim=1)
        correct += (y_pred == y).sum()

    print(f'准确率(accuracy):{correct/len(test_dataset):.4f}')


if __name__ == '__main__':
    train_dataset, test_dataset, input_dim, output_dim = create_dataset()
    # print(f"train_dataset:{train_dataset}")
    # print(f'test_dataset:{test_dataset}')
    # print(f'输入特征数:{input_dim}')
    # print(f'输出标签数:{output_dim}')

    # model = PhonePriceModel(input_dim,output_dim)
    # #计算模型参数
    # summary(model, input_size=(16,input_dim,))      #每批16条，每条20列特征

    train(train_dataset,input_dim,output_dim)
    evaluate(test_dataset,input_dim,output_dim)