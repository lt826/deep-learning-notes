import torch
print(torch.__version__)      # 2.13.0
print(torch.version.cuda)     # None ← CPU 版铁证
print(torch.cuda.is_available())  # False