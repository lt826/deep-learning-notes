"""

池化层：
    目的：
        用来大幅降低参数量级(降维)
    思路：
        最大池化
        平均池化
    特点：
        池化不会改变数据的通道数
"""

import torch
import torch.nn as nn


#1. 定义函数，演示单通道池化
def dm01():
    #1. 创建一个1通道3*3的二维矩阵
    inputs = torch.tensor([     #1 通道C
        [                       #3 高度H
            [0,1,2],            #3 宽度W
            [3,4,5],
            [6,7,8]
        ]
    ])
    print(f'inputs:{inputs},shape:{inputs.shape}')          # (1, 3, 3)

    #2. 创建最大池化层
    #参1：池化核(池化窗口)大小，参2：步长，参3：填充大小
    pool1 = nn.MaxPool2d(2,1,0)
    outputs1 = pool1(inputs)
    print(f'outputs:{outputs1},shape:{outputs1.shape}')        # (1, 2, 2)

    #3. 创建平均池化层
    pool2 = nn.AvgPool2d(2,1,0)
    outputs2 = pool2(inputs)
    print(f'outputs:{outputs2},shape:{outputs2.shape}')        # (1, 2, 2)


#2. 定义函数，演示多通道池化
def dm02():
    #1. 创建一个3通道3*3的二维矩阵
    inputs = torch.tensor([     #3 通道C
        [                       # 通道1，H,W = 3,3
            [0,1,2],
            [3,4,5],
            [6,7,8]
        ],

        [                       # 通道2，H,W = 3,3
            [10,20,30],
            [40,50,60],
            [70,80,90]
        ],

        [                       # 通道3，H,W = 3,3
            [11,22,33],
            [44,55,66],
            [77,88,99]
        ]
    ])

    print(f'inputs:{inputs},shape:{inputs.shape}')          # (3, 3, 3)

    #2. 创建最大池化层
    #参1：池化核(池化窗口)大小，参2：步长，参3：填充大小
    pool1 = nn.MaxPool2d(2,1,0)
    outputs1 = pool1(inputs)
    print(f'outputs:{outputs1},shape:{outputs1.shape}')        # (3, 2, 2)

    #3. 创建平均池化层
    pool2 = nn.AvgPool2d(2,1,0)
    outputs2 = pool2(inputs)
    print(f'outputs:{outputs2},shape:{outputs2.shape}')        # (3, 2, 2)


if __name__ == '__main__':
    # dm01()
    dm02()