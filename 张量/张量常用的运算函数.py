"""

涉及到的函数：
    sum(),max(),min(),mean()        有dim参数，0表示列，1表示行
    pow(),sqrt(),exp(),log(),log2() 无dim参数

需要掌握的函数：
    sum(),max(),min(),mean(),pow()
"""

import torch

t1 = torch.tensor([
    [1,2,3,4],
    [5,6,7,8]
],dtype=torch.float)

#有dim参数的函数
#计算和
print(t1.sum(dim=0))        #按列求和
print(t1.sum(dim=1))        #按行求和
print(t1.sum())             #整体求和
print('-'*30)
#计算最大值
print(t1.max(dim=0))        #按列求最大值
print(t1.max(dim=1))        #按行求最大值
print(t1.max())             #整体求最大值
print('-'*30)
#计算平均值
print(t1.mean(dim=0))        #按列求平均值
print(t1.mean(dim=1))        #按行求平均值
print(t1.mean())             #整体求平均值
print('-'*30)

#无dim参数的函数
print(t1.pow(2))            #元素n次幂
print(t1.sqrt())            #元素平方根
print(t1.exp())            #元素e的n次幂
print(t1.log())            #元素自然对数
print(t1.log2())           #元素二进制对数
