from .flow import Flow
import torch
from torch import nn
import utils

class Scaling(Flow):
    "Channels are treat equally"
    def __init__(self, numVar, exp = True, parameters = None, prior = None, name = "Scaling"):
        super(Scaling,self).__init__(prior,name)
        if parameters is None:
            self.scale = nn.Parameter(torch.randn(numVar))
        else:
            assert parameters.shape[0] == numVar
            self.scale = nn.Parameter(parameters)
        self.expFlag = exp

    def inverse(self,y):
        shape = y.shape
        y = y.reshape(shape[0],shape[1],-1)
        y = y*torch.exp(-self.scale)
        inverseLogjac = -y.new_ones(y.shape[0])*sum(self.scale)*y.shape[1]
        return y.reshape(shape),inverseLogjac

    def forward(self,x):
        shape = x.shape
        x = x.reshape(shape[0],shape[1],-1)
        x = x*torch.exp(self.scale)
        forwardLogjac = x.new_ones(x.shape[0])*sum(self.scale)*x.shape[1]
        return x.reshape(shape),forwardLogjac