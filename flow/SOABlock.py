import torch
from torch import nn

from .flow import Flow
from .AmpSinh import AmpSinh
from .O4transformation import O4transformation
from .scaling import Scaling
from utils import checkNan

class SOA(Flow):
    '''
    '''
    def __init__(self,thetas,order,prior = None, name = "SOABlocks"):
        super(SOA,self).__init__(prior,name)
        self.models = torch.nn.ModuleList([Scaling(4),O4transformation(thetas,order),AmpSinh()])

    def inverse(self,y):
        logjacs = y.new_zeros(y.shape[0])
        for net in reversed(self.models):
            y,logjac = net.inverse(y)
            logjacs = logjacs + logjac
        return y,logjacs

    def forward(self,x):
        logjacs = x.new_zeros(x.shape[0])
        for net in self.models:
            x,logjac = net.forward(x)
            logjacs = logjacs + logjac
        return x,logjacs