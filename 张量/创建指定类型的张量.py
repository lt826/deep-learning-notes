"""

涉及到的函数：
    type(torch支持的数据类型）
    half()/double()/float()/short()/int()/long()

需要掌握的函数：
    type()
"""

import torch


#场景1：直接创建指定类型的张量
t1 = torch.tensor([1,2,3,4,5,6], dtype=torch.float)     #默认创建的张量类型是float32
print(f"t1:{t1}, (元素)类型:{t1.dtype}, (张量)类型:{t1.type()}")

#场景2：创建好张量后 -> 类型转换
t2 = t1.type(torch.int16)
print(f"t2:{t2}, (元素)类型:{t2.dtype}, (张量)类型:{t2.type()}")
print('-' * 30)

#half()/double()/float()/short()/int()/long()
print(t2.half())        # torch.float16
print(t2.float())       # torch.float32，默认
print(t2.double())      # torch.float64
print(t2.short())       # torch.int16
print(t2.int())         # torch.int32
print(t2.long())        # torch.int64，默认



