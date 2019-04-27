from __future__ import division
import numpy as np

from utils import load_MNIST, random_draw
from realnvp import Realnvp
import math

def train():
    lr = 0.5
    Epoch = 10
    Iteration = len(train_data) // batch_size
    Batchsize_test = 10
    Batchsize_train = 100
    # an epoch means running through the training set roughly once

    params = list(Realnvp.parameters())
    params = list(filter(lambda p: p.requires_grad, params))
    nparams = sum([np.prod(p.size()) for p in params])
    print ('total nubmer of trainable parameters:', nparams)
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

    train_data, train_label, test_data, test_label = load_MNIST()
    x_test = random_draw(test_data, Batchsize_test)
    tList =[utils.SimpleMLP([4, 10, 4]), utils.SimpleMLP([4, 10, 4]),utils.SimpleMLP([4, 10, 4]), utils.SimpleMLP([4, 10, 4])]
    sList =[utils.SimpleMLP([4, 10, 4]), utils.SimpleMLP([4, 10, 4]),utils.SimpleMLP([4, 10, 4]), utils.SimpleMLP([4, 10, 4])]
    p = Gaussian([28,28])
    f = Realnvp(sList,tList,prior=p)
    logp1 = f.logProbability(x_test)
    loss1= - sum(logp1)/Batchsize_test
    print('Before Training.\nTest loss = %.4f' %loss1)

    for epoch in range(Epoch):
        for j in range(Iteration):
           x_trian= random_draw(train_data,Batchsize_train)
           logp = f.logProbability(x_trian)
           loss = - sum(logp)/Batchsize_train

           f.zero_grad()
           loss.backward()
           optimizer.step()

        print("epoch = %d/%d, loss = %.4f" %(epoch, loss))

    x = random_draw(test_data, Batchsize_test)
    logp2 = f.logProbability(x_test)
    loss2= - sum(logp2)/Batchsize_test
    print('After Training.\nTest loss = %.4f' %loss2)



if __name__ == "__main__":
    train()
