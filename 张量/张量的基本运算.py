"""

涉及到的API:
    add(), sub(), mul(), div(), neg()      ->加减乘除取反
    add_( ), sub_( ), mul_( ), div_( )     ->修改原数据，类似于Pandas部分的 inplace = True

"""



import torch

t1 = torch.tensor([11,22,33])

t2 = t1.add(10)
#t2 = t1.add_(10)


print(t1)
print(t2)
