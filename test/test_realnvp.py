import torch
from numpy.testing import assert_array_almost_equal
import os
import sys
sys.path.append(os.getcwd())
import utils
from torch import nn
from utils import ScalableTanh,SimpleMLP

from realnvp import Realnvp
from gaussian import Gaussian

def test_bijective():
    p = Gaussian([28*28])
    tList = [SimpleMLP([392,392*2,392,392*2,392],[nn.ELU(),nn.ELU(),nn.ELU(),nn.Tanh()]) for _ in range(4)]
    sList = [SimpleMLP([392,392*2,392,392*2,392],[nn.ELU(),nn.ELU(),nn.ELU(),ScalableTanh(392)]) for _ in range(4)]

    #tList =[utils.SimpleMLP([28*28/2, 28*28, 28*28/2]) for _ in range(4)]
    #sList =[utils.SimpleMLP([28*28/2, 28*28, 28*28/2]) for _ in range(4)]
    maskList = []
    '''
    for i in range(len(tList)//2):
        b = torch.zeros(1,28*28).byte()
        i = torch.randperm(b.numel()).narrow(0, 0, b.numel() // 2)
        b.zero_()[:,i] = 1
        b_=1-b
        maskList.append(b)
        maskList.append(b_)
    '''
    b = torch.zeros(1,28*28).byte()
    b[:,:28*28//2] = 1
    for i in range(len(tList)):
        maskList.append(b)
        b = 1-b
    maskList = torch.cat(maskList,0)
    #x = torch.randn(1,8)
    #  # Build your NICE net here, may take multiply lines.
    # import pdb
    # pdb.set_trace()
    f = Realnvp(sList,tList,p,maskList)
    x = f.sample(10)    #print 10行 1*8矩阵
    op = f.logProbability(x)  #print 1*10矩阵
    y,pi = f.inverse(x)
    pp = f.prior.logProbability(y)
    yx,pf = f.forward(y)
    yxy,pfi = f.inverse(yx)

    assert_array_almost_equal(x.detach().numpy(),yx.detach().numpy(), decimal=5)
    assert_array_almost_equal(y.detach().numpy(),yxy.detach().numpy(), decimal=5)

    assert_array_almost_equal(pi.detach().numpy(),-pf.detach().numpy(), decimal=5)
    assert_array_almost_equal(pf.detach().numpy(),-pfi.detach().numpy(), decimal=5)

    assert_array_almost_equal((op+pi).detach().numpy(),pp.detach().numpy(), decimal=5)
    assert_array_almost_equal((pp-pfi).detach().numpy(),op.detach().numpy(),decimal=5)

if __name__ == "__main__":
    test_bijective()
