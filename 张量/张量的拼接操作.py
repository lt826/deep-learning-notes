"""

涉及到的API：
    cat()       不改变维度数，拼接张量，除了拼接的那个维度外，其它维度必须保持一致
    stack()     改变维度数，拼接张量，所有维度必须保持一致

"""

import torch

t1 = torch.randint(1,10,(2,3))
print(f't1:{t1},shape:{t1.shape}')

t2 = torch.randint(1,10,(2,3))
print(f't2:{t2},shape:{t2.shape}')

#展示cat()拼接
t3 = torch.cat((t1,t2),dim=0)           # (2,3) + (2,3) = (4,3)
print(f't3:{t3},shape:{t3.shape}')
t4 = torch.cat((t1,t2),dim=1)           # (2,3) + (2,3) = (2,6)
print(f't4:{t4},shape:{t4.shape}')


#展示stack()拼接,维度必须保持一致
t5 = torch.stack((t1,t2),dim=0)          # (2,3) + (2,3) = (2,2,3)
print(f't5:{t5},shape:{t5.shape}')

t6 = torch.stack((t1,t2),dim=1)          # (2,3) + (2,3) = (2,2,3)
print(f't6:{t6},shape:{t6.shape}')

t7 = torch.stack((t1,t2),dim=2)          # (2,3) + (2,3) = (2,3,2)
print(f't7:{t7},shape:{t7.shape}')
