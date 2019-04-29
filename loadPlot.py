import torch
import numpy as np
import torch
from torch import nn
from torchvision import transforms

from utils import load_MNIST, random_draw ,SimpleMLP,ScalableTanh
from realnvp import Realnvp
from gaussian import Gaussian

depth =10

tList = [SimpleMLP([392,392*2,392,392*2,392],[nn.ELU(),nn.ELU(),nn.ELU(),nn.Tanh()]) for _ in range(depth)]
sList = [SimpleMLP([392,392*2,392,392*2,392],[nn.ELU(),nn.ELU(),nn.ELU(),ScalableTanh(392)]) for _ in range(depth)]

maskList = []
for i in range(len(tList)//2):
    b = torch.zeros(1,28*28).byte()
    i = torch.randperm(b.numel()).narrow(0, 0, b.numel() // 2)
    b.zero_()[:,i] = 1
    b_=1-b
    maskList.append(b)
    maskList.append(b_)
maskList = torch.cat(maskList,0)

p = Gaussian([28*28])
f = Realnvp(sList,tList,p,maskList)

f.load_state_dict(torch.load("./"+f.name+"_1"+".saving"))

from matplotlib import pyplot as plt

sampleBatch = 10

samples = f.sample(sampleBatch).detach().numpy().reshape(sampleBatch,28,28)
for k in range(sampleBatch):
    a = plt.matshow(samples[k].reshape(28,28),cmap="gray")
    plt.colorbar(a)

plt.show()