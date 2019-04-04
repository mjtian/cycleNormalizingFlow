import torch
a = torch.randn(2 ,4)
b = torch.randn(4 ,2)

c = torch.matmul(a, b)

print (a)
print (b)
print (c)
