"""

卷积层；
        负责提取图像中的局部特征,结合卷积核(每个卷积核 = 1个神经元)实现，处理后的结果叫：特征图

特征图大小计算
    特征图大小 = (输入图像大小 - 卷积核大小 + 2*填充大小) / 步长 + 1

"""



#导包
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import matplotlib.pyplot as plot


#1. 加载RGB真彩图
img = plt.imread('picture/a.jpg')
#打印读取到的图像信息
# print(f'img:{img},shape:{img.shape}')       #HWC(高、宽、通道) = (1126, 2000, 3)

#2. 把图像形状从HWC -> CHW, 思路：img -> 张量 -> 转换维度
img1 = torch.tensor(img,dtype=torch.float)
img2 = img1.permute(2,0,1)
# print(f'img2:{img2},shape:{img2.shape}')     #CHW = (3, 1126, 2000)

#3. 因为只有1张图，所以我们给它再增加一个维度，从CHW -> (1,C,H,W),即1张3通道的1126*2000像素的图
img3 = img2.unsqueeze(0)
# print(f'img3:{img3},shape:{img3.shape}')     # (1, 3, 1126, 2000)

#4. 创建卷积层对象，提取特征图
#参1：输入特征数=图片的通道数，参2：输出特征数(几个卷积核)，参3：卷积核大小，参4：步长，参5：填充大小
conv = nn.Conv2d(3,4,3,2,0)

#5. 具体的卷积核计算
conv_img = conv(img3)
# print(f'conv_img:{conv_img},shape:{conv_img.shape}')        # (1, 4, 562, 999)

#6. 查看提取到的4个特征图
img4 = conv_img[0]
# print(f'img4:{img4},shape:{img4.shape}')                    # (4, 562, 999)

#7. 把上述图从CHW -> HWC
img5 = img4.permute(1,2,0)
# print(f'img5:{img5},shape:{img5.shape}')                    # (562, 999, 4)

#8. 可视化特征图
feature1 = img5[:,:,3].detach().numpy()
plot.imshow(feature1)
plot.show()
