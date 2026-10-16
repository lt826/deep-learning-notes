"""

点乘：
    要求：两个张量的维度保持一致，对应的元素直接做相应的操作
    API：
        torch.mul(t1, t2)
        t1 * t2

矩阵乘法：
    要求：两个张量，第一个张量的列数必须等于第二个张量的行数(A列 = B行)
    API：
        torch.matmul(t1, t2)
        t1 @ t2

"""

import torch

#定义函数，演示点乘
def dm01():
    t1 = torch.tensor([[1,2,3],[4,5,6]])
    t2 = torch.tensor([[1,2,3],[4,5,6]])

    print(torch.mul(t1, t2))

#定义函数，演示矩阵乘法
def dm02():
    t1 = torch.tensor([[1,2,3,1],[4,5,6,3],[7,8,9,10]])     #3行4列
    t2 = torch.tensor([[1,2],[4,5],[7,8],[10,11]])          #4行2列

    print(torch.matmul(t1, t2))

if __name__ == '__main__':
    dm01()
    dm02()

