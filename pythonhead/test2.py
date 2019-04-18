import torch
import test
from test import SimpleMLP
import torch.nn as nn

net1 = SimpleMLP([28*28,100,50,1])
net2 = SimpleMLP([1,50,100,28*28])

test = torch.randn(10,28*28)
result1 = net1(test)
result2 = net2(result1)

assert result1.shape[0] == 10
assert result2.shape[0] == 10

assert result1.shape[1] == 1
assert result2.shape[1] == 28*28
