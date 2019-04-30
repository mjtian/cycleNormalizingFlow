import torch
from torch import nn
import os
import sys
sys.path.append(os.getcwd())
import utils



def test_mlp():
    net = utils.SimpleMLP([28*28,100,50,1])
    test = torch.randn(10,28*28)
    result = net.forward(test)
    assert result.shape[0] == 10
    assert result.shape[1] == 1
