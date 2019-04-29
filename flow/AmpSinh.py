import numpy as np
import torch
from torch import nn

from .flow import Flow
import torch
import utils

def det(x):
    return (torch.abs(torch.sinh(2*(x[:,0]**2+x[:,1]**2)**0.5)/(2*(x[:,0]**2+x[:,1]**2)**0.5))).reshape(-1,1)

class AmpSinh(Flow):
    '''
    For now, only support x's dimension of [b,2,l]
    '''
    def __init__(self,prior = None, name = "AmpSinh"):
        super(AmpSinh,self).__init__(prior,name)
    def inverse(self,y):
        shape = y.shape
        y = y.reshape(shape[0],2,-1)
        l = y.shape[-1]
        y = y.transpose(1,2).reshape(-1,2)
        y = utils.ampAsinh(y)
        inverseLogjac = -det(y)
        y = y.reshape(-1,l,2).transpose(1,2)
        inverseLogjac = inverseLogjac.reshape(-1,l,1).transpose(1,2).sum(-1)
        return y.reshape(shape),inverseLogjac

    def forward(self,x):
        shape = x.shape
        x = x.reshape(shape[0],2,-1)
        l = x.shape[-1]
        x = x.transpose(1,2).reshape(-1,2)
        forwardLogjac = det(x)
        x = utils.ampSinh(x)
        x = x.reshape(-1,l,2).transpose(1,2)
        forwardLogjac = forwardLogjac.reshape(-1,l,1).transpose(1,2).sum(-1)
        return x.reshape(shape),forwardLogjac