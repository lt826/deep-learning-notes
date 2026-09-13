"""

深度学习项目步骤
    1. 准备数据集
    2. 搭建神经网络
    3. 模型训练
    4. 模型测试

卷积层：
    提取图像的局部特征 -> 特征图
    卷积核(每个卷积核 = 1个神经元) -> 卷积 -> 特征图

池化层：
    降维，有最大池化、平均池化
    池化只在HW上做调整，通道上不改变

"""

#每批次样本数
BATCH_SIZE = 8

import torch
import torch.nn as nn
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torch.optim as optim
import time
import matplotlib.pyplot as plt
from torchsummary import summary
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 1. 准备数据集
def create_dataset():
    #获取训练集
    #参1：数据集路径，参2：是否为训练集，参3：数据预处理，参4：是否下载数据集
    train_dataset = CIFAR10(root='./data', train=True, transform=ToTensor(), download=True)

    #获取测试集
    test_dataset = CIFAR10(root='./data', train=False, transform=ToTensor(), download=True)

    return train_dataset, test_dataset


# 2. 搭建神经网络
class ImageModel(nn.Module):
    def __init__(self):
        super().__init__()
        #搭建神经网络
        #第1个卷积层，输入3通道，输出32通道，卷积核大小3x3，步长1，填充1（padding=1保持尺寸不变）
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1)
        #第1个批归一化层，对32个通道做归一化，放在卷积后、激活前，稳定训练
        self.bn1 = nn.BatchNorm2d(32)
        #第1个池化层，池化核大小2x2，步长2，填充0
        self.pool1 = nn.MaxPool2d(2, 2, 0)

        #第2个卷积层，输入32通道，输出64通道，卷积核大小3x3，步长1，填充1（padding=1保持尺寸不变）
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        #第2个批归一化层，对64个通道做归一化
        self.bn2 = nn.BatchNorm2d(64)
        #第2个池化层，池化核大小2x2，步长2，填充0
        self.pool2 = nn.MaxPool2d(2, 2, 0)

        #新增：第3个卷积层，输入64通道，输出128通道，卷积核大小3x3，步长1，填充1
        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)
        #第3个批归一化层，对128个通道做归一化
        self.bn3 = nn.BatchNorm2d(128)
        #第3个池化层，池化核大小2x2，步长2，填充0
        self.pool3 = nn.MaxPool2d(2, 2, 0)

        #新增：Dropout层，训练时随机丢弃50%的神经元，防止全连接层过拟合
        self.dropout = nn.Dropout(0.5)

        #第1个全连接层，输入128*4*4 = 2048，输出256
        #（尺寸变化：32->16->8->4，所以是128*4*4=2048）
        self.linear1 = nn.Linear(2048, 256)
        #第2个全连接层 -> 输出层，输入256，输出10
        #（原第2、3个全连接层合并为一层，减少参数量）
        self.output = nn.Linear(256, 10)

    #前向传播
    def forward(self, x):
        #第1层，卷积层(加权求和) + 批归一化 + 激励层(激活函数) + 池化层(降维)
        x = self.conv1(x)   #卷积层
        x = self.bn1(x)     #批归一化
        x = torch.relu(x)   #激励层
        x = self.pool1(x)   #池化层，32->16

        # 第2层，结构同上：卷积 + 批归一化 + 激活 + 池化
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))  # 16->8

        # 第3层（新增），结构同上：卷积 + 批归一化 + 激活 + 池化
        x = self.pool3(torch.relu(self.bn3(self.conv3(x))))  # 8->4

        #细节：全连接层只能处理二维数据，所以要将数据进行拉平     (8,128,4,4) -> (8,2048)
        #参1：样本行数，参2：列数， -1表示自动生成
        x = x.reshape(x.size(0), -1)

        #第3层 加权求和 + 激活函数
        x = torch.relu(self.linear1(x))
        #新增：Dropout，随机丢弃部分神经元，防止过拟合
        x = self.dropout(x)

        #第5层（输出层）
        x = self.output(x)          #因为是多分类问题，所以输出层不加激活函数
        return x


# 3. 模型训练
def train(train_dataset):
    # 创建数据加载器
    dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    # 创建模型对象
    model = ImageModel().to(device)
    # 创建损失函数对象
    criterion = nn.CrossEntropyLoss()
    # 创建优化器对象
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    epochs = 10
    for epoch in range(epochs):
        total_loss,total_samples,total_correct,start = 0.0,0,0,time.time()
        for x,y in dataloader:
            x,y = x.to(device),y.to(device)
            model.train()
            y_pred = model(x)
            loss = criterion(y_pred,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #统计预测正确的样本个数
            total_correct += (torch.argmax(y_pred,dim=-1) == y).sum()
            #统计当前批次的总损失
            #loss.item():此次的损失平均值
            total_loss += loss.item() * len(y)
            #统计当前批次的总样本数
            total_samples += len(y)

        #打印这一轮的训练信息
        print(f'Epoch: {epoch+1}, Loss: {total_loss/total_samples:.4f}, Accuracy: {total_correct/total_samples:.4f}, Time: {time.time()-start:.2f}s')


    #保存模型
    torch.save(model.state_dict(), './model/image_model.pth')

# 4. 模型测试
def evaluate(test_dataset):
    dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    model = ImageModel().to(device)
    model.load_state_dict(torch.load('./model/image_model.pth'))
    total_correct,total_sample = 0,0
    for x,y in dataloader:
        x,y = x.to(device),y.to(device)
        model.eval()
        y_pred = model(x)
        y_pred = torch.argmax(y_pred,dim=-1)

        total_correct += (y_pred == y).sum()
        total_sample += len(y)

    print(f'Accuracy: {total_correct/total_sample:.2f}')

if __name__ == '__main__':
    #1. 获取数据集
    train_dataset,test_dataset = create_dataset()
    # print(f'训练集：{train_dataset.data.shape}')
    # print(f'测试集：{test_dataset.data.shape}')
    # # ('airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3, 'deer': 4, 'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9)
    # print(f'数据集类别：{train_dataset.class_to_idx}')
    #
    # #图像展示
    # plt.figure(figsize=(2,2))
    # plt.imshow(train_dataset.data[110])
    # plt.title(train_dataset.targets[110])
    # plt.show()

    #2. 搭建神经网络
    # model = ImageModel()

    #3. 模型训练
    # train(train_dataset)

    #4. 模型测试
    evaluate(test_dataset)