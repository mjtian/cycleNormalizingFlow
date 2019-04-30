import torch
from torch import nn

from .flow import Flow
from utils import checkNan
from utils import o4Builder,o4invBuilder

class O4transformation(Flow):
    def __init__(self,thetaList,orderList,prior = None, name = "O4transformation"):
        assert thetaList.shape[0] == len(orderList)
        assert len(thetaList.shape) == 1
        super(O4transformation,self).__init__(prior,name)
        self.theta = nn.Parameter(thetaList,requires_grad=True)
        self.order = orderList

    def inverse(self,y):
        shape = y.shape
        inverseLogjac = y.new_zeros(y.shape[0])
        m = o4invBuilder(self.theta,self.order)
        y = y.reshape(-1,*shape[2:])
        y = y.reshape(y.shape[0],-1)
        y = torch.matmul(y,m)
        return y.reshape(shape),inverseLogjac

    def forward(self,x):
        shape = x.shape
        forwardLogjac = x.new_zeros(x.shape[0])
        m = o4Builder(self.theta,self.order)
        x = x.reshape(-1,*shape[2:])
        x = x.reshape(x.shape[0],-1)
        x = torch.matmul(x,m)
        return x.reshape(shape),forwardLogjac