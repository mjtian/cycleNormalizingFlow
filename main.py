from __future__ import division
import numpy as np
import torch
from torch import nn
from utils import load_MNIST, random_draw ,SimpleMLP,ScalableTanh
from flow import RNVP
from gaussian import Gaussian


def train():
    train_data, train_label, test_data, test_label = load_MNIST()
    lr = 1e-3
    Epoch = 30
    Batchsize_test = 20
    Batchsize_train = 600
    Iteration = len(train_data) // Batchsize_train
    depth = 10
    sampleBatch = 10
    # an epoch means running through the training set roughly once

    x_test = random_draw(test_data, Batchsize_test)
    x_test1 = torch.from_numpy(x_test).to(torch.float32)

    x_test11 = x_test1.reshape(-1,28*28)
    tList = [SimpleMLP([784,392*2,392,392*2,784],[nn.ELU(),nn.ELU(),nn.ELU(),nn.Tanh()]) for _ in range(depth)]
    sList = [SimpleMLP([784,392*2,392,392*2,784],[nn.ELU(),nn.ELU(),nn.ELU(),ScalableTanh(784)]) for _ in range(depth)]

    maskList = []
    '''
    b = torch.zeros(1,28*28).byte()
    b[:,:28*28//2] = 1
    for i in range(len(tList)):
        maskList.append(b)
        b = 1-b
    '''
    for i in range(len(tList)//2):
        b = torch.zeros(1,28*28)
        i = torch.randperm(b.numel()).narrow(0, 0, b.numel() // 2)
        b.zero_()[:,i] = 1
        b_=1-b
        maskList.append(b)
        maskList.append(b_)
    maskList = torch.cat(maskList,0)

    p = Gaussian([28*28])
    f = RNVP(maskList,sList,tList,p)
    # import pdb
    # pdb.set_trace()
    logp1 = f.logProbability(x_test11)
    loss1= -logp1.mean()
    print('Before Training.\nTest loss = %.4f' %loss1)

    params = list(f.parameters())
    params = list(filter(lambda p: p.requires_grad, params))
    nparams = sum([np.prod(p.size()) for p in params])
    print ('total nubmer of trainable parameters:', nparams)
    optimizer = torch.optim.Adam(params, lr=lr)

    TrainLOSS = []
    TestLOSS = []
    for epoch in range(Epoch):

        for j in range(Iteration):
           x_train= random_draw(train_data,Batchsize_train)
           x_train1 = torch.from_numpy(x_train).to(torch.float32)
           x_train11 = x_train1.reshape(-1,28*28)
           logp = f.logProbability(x_train11)
           loss = -logp.mean()

           TrainLOSS.append(loss.item())

           f.zero_grad()
           loss.backward()
           optimizer.step()

           x = random_draw(test_data, Batchsize_test)
           x1 = torch.from_numpy(x).to(torch.float32)
           x11 = x1.reshape(-1,28*28)
           logp2 = f.logProbability(x11)
           loss2= -logp2.mean()

           TestLOSS.append(loss2.item())

        print("epoch = %d, loss = %.4f, test loss = %.4f" %(epoch, loss, loss2))

    torch.save(f.state_dict(),f.name+"_"+str(epoch)+".saving")
    trainLoss = np.array(TrainLOSS)
    testLoss = np.array(TestLOSS)

    x = random_draw(test_data, Batchsize_test)
    x1 = torch.from_numpy(x).to(torch.float32)
    x11 = x1.reshape(-1,28*28)
    logp2 = f.logProbability(x11)
    loss2= -logp2.mean()
    print('After Training.\nTest loss = %.4f' %loss2)

    from matplotlib import pyplot as plt

    #samples = np.tanh(f.sample(sampleBatch).detach().numpy().reshape(sampleBatch,28,28))
    import pdb
    pdb.set_trace()
    samples = (f.sample(sampleBatch)[0]).detach().numpy().reshape(sampleBatch,28,28)
    for k in range(sampleBatch):
        a = plt.matshow(samples[k].reshape(28,28),cmap="gray")
        plt.colorbar(a)
    plt.figure()
    plt.plot(trainLoss,label="Training")
    plt.plot(testLoss,label="Test")
    plt.legend()

    plt.show()

    y = x11.detach().numpy().reshape(Batchsize_test,28,28)
    for k in range(Batchsize_test):
        a = plt.matshow(y[k].reshape(28,28),cmap="gray")
        plt.colorbar(a)

    plt.figure()

    plt.legend()

    plt.show()
    # gua = p.sample(Batchsize_test)
    # x11 = x11 + gua
    trans = f.inverse(x11)
    trans = f.forward(trans[0])
    trans_= trans[0].detach().numpy().reshape(Batchsize_test,28,28)
    for k in range(Batchsize_test):
        a = plt.matshow(trans_[k].reshape(28,28),cmap="gray")
        plt.colorbar(a)

    plt.figure()

    plt.legend()

    plt.show()

    import pdb
    pdb.set_trace()

if __name__ == "__main__":
    train()
