"""

涉及到的API：
    reshape():  在不改变张量内容的前提下，对其形状做改变
    unsqueeze():在指定的轴上增加一个维度，等价于：升维
    squeeze():  删除所有为1的维度，等价于：降维
    transpose():一次只能交换2个维度
    unsqueeze():一次可以同时交换多个维度
    view():     只能修改连续的张量形状，    连续张量 = 内存中的存储顺序和在张量中显示的顺序相同
    contiguous():把不连续的张量 -> 连续的张量，即：基于张量中显示的顺序，修改内存中的存储顺序
    is_contiguous():判断张量是否是连续的

需要掌握的函数：
    reshape()，unsqueeze()，unsqueeze()，view()
"""
import torch

#1.定义函数，演示：reshape()函数
def dm01():
    t1 = torch.randint(1,10,size=(2,3))
    print(f't1:{t1},shape:{t1.shape},row:{t1.shape[0]},columns:{t1.shape[1]}')

    #通过reshape函数把t1变成1行6列
    t2 = t1.reshape(1,6)
    print(f't2:{t2},shape:{t2.shape},row:{t2.shape[0]},columns:{t2.shape[1]}')

#2.定义函数，演示：squeeze()函数, unsqueeze()函数
def dm02():
    t1 = torch.randint(1,10,size=(2,3))
    print(f't1:{t1},shape:{t1.shape}')              # (2,3)

    #在0维上添加一个维度
    t2 = t1.unsqueeze(0)
    print(f't2:{t2},shape:{t2.shape}')              # (1,2,3)

    #在1维上添加一个维度
    t3 = t1.unsqueeze(1)
    print(f't3:{t3},shape:{t3.shape}')              # (2,1,3)

    #在2维上添加一个维度
    t4 = t1.unsqueeze(2)
    print(f't4:{t4},shape:{t4.shape}')              # (2,3,1)


    #删除所有为1的维度
    t5 = torch.randint(1,10,size=(2,1,3,1,1))
    print(f't5:{t5},shape:{t5.shape}')              # (2,1,3,1,1)
    t6 = t5.squeeze()
    print(f't6:{t6},shape:{t6.shape}')              # (2,3)


#3.定义函数，演示：transpose()函数, permute()函数
def dm03():
    t1 = torch.randint(1,10,size=(2,3,4))
    print(f't1:{t1},shape:{t1.shape}')
    print("-"*20)

    #改变维度从(2,3,4)变成(2,4,3)
    t2 = t1.transpose(1,2)
    print(f't2:{t2},shape:{t2.shape}')

    #改变维度从(2,3,4)变成(4,2,3)
    t3 = t1.permute(2,0,1)
    print(f't3:{t3},shape:{t3.shape}')

#4.定义函数，演示：view()函数, contiguous()函数, is_contiguous()函数
def dm04():
    t1 = torch.randint(1,10,size=(2,3))
    print(f't1:{t1},shape:{t1.shape}')

    #通过view()函数，修改上述张量的形状，从(2,3)变成(3,2)
    t2 = t1.view(3,2)
    print(f't2:{t2},shape:{t2.shape}')
    #判断t2是否是连续张量
    print(t2.is_contiguous())

    #通过transpose()交换维度后不连续了
    t3 = t1.transpose(0,1)
    print(f't3:{t3},shape:{t3.shape}')
    print(t3.is_contiguous())

    #尝试把t3变成连续张量
    t4 = t3.contiguous().view(2,3)
    print(f't4:{t4},shape:{t4.shape}')
    print(t4.is_contiguous())


if __name__ == '__main__':
    # dm01()
    # dm02()
    # dm03()
    dm04()