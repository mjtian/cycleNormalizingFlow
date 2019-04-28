from __future__ import division
import numpy as np
import torch
from torchvision import transforms

from utils import load_MNIST, random_draw ,SimpleMLP
from realnvp import Realnvp
from gaussian import Gaussian
import math

def train():
    train_data, train_label, test_data, test_label = load_MNIST()
    lr = 0.5
    Epoch = 10
    Batchsize_test = 10
    Batchsize_train = 100
    Iteration = len(train_data) // Batchsize_train
    # an epoch means running through the training set roughly once



    x_test = random_draw(test_data, Batchsize_test)
    x_test1 = torch.from_numpy(x_test).to(torch.float32)

    x_test11 = x_test1.reshape(-1,28*28)
    tList =[SimpleMLP([392, 196, 392]), SimpleMLP([392, 196, 392]),SimpleMLP([392, 196, 392]), SimpleMLP([392, 196, 392])]
    sList =[SimpleMLP([392, 196, 392]), SimpleMLP([392, 196, 392]),SimpleMLP([392, 196, 392]), SimpleMLP([392, 196, 392])]
    p = Gaussian([28*28])
    f = Realnvp(sList,tList,prior=p)
    # import pdb
    # pdb.set_trace()
    logp1 = f.logProbability(x_test11)
    loss1= -logp1.mean()
    print('Before Training.\nTest loss = %.4f' %loss1)

    params = list(Realnvp.parameters())
    params = list(filter(lambda p: p.requires_grad, params))
    nparams = sum([np.prod(p.size()) for p in params])
    print ('total nubmer of trainable parameters:', nparams)
    optimizer = torch.optim.Adam(params, lr=lr)

    for epoch in range(Epoch):
        for j in range(Iteration):
           x_train= random_draw(train_data,Batchsize_train)
           x_train1 = torch.from_numpy(x_train).to(torch.float32)
           x_train11 = x_train1.reshape(-1,28*28)
           logp = f.logProbability(x_train11)
           loss = -logp.mean

           f.zero_grad()
           loss.backward()
           optimizer.step()

        print("epoch = %d/%d, loss = %.4f" %(epoch, loss))

    x = random_draw(test_data, Batchsize_test)
    x1 = torch.from_numpy(x).to(torch.float32)
    x11 = x1.reshape(-1,28*28)
    logp2 = f.logProbability(x11)
    loss2= -logp2.mean
    print('After Training.\nTest loss = %.4f' %loss2)



if __name__ == "__main__":
    train()
