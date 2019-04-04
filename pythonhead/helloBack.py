
import torch

x = torch.ones(2 ,2, requires_grad=True)
y = 5 * x
out = y.mean()
out.backward(x)
print (x)
print (y)
print (x.grad)


