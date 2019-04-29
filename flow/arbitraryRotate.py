
import torch
from torch import nn
import numpy as np
import scipy

from .flow import Flow

class ArbitraryRotate(Flow):
    def __init__(self, c,prior = None, name = "ArbitraryRotate"):
        super(ArbitraryRotate,self).__init__(prior,name)
        q,_ = np.linalg.qr(np.random.randn(c,c))
        self.w = nn.Parameter(torch.tensor(q).to(torch.float32))

    def inverse(self,y):
        assert torch.det(self.w)>0.5
        assert torch.det(self.w)<1.5
        inverseLogjac = torch.slogdet(self.w)[1]*y.shape[-1]*y.shape[-2]*torch.ones(y.shape[0])
        y = torch.matmul(y.permute([0,2,3,1]),self.w.reshape(1,1,*self.w.shape)).permute(0,3,1,2)
        return y,inverseLogjac

    def forward(self,z):
        w_ = torch.inverse(self.w)
        assert torch.det(w_)>0.5
        assert torch.det(w_)<1.5
        forwardLogjac = torch.slogdet(w_)[1]*z.shape[-1]*z.shape[-2]*torch.ones(z.shape[0])
        z = torch.matmul(z.permute([0,2,3,1]),w_.reshape(1,1,*w_.shape)).permute(0,3,1,2)
        return z,forwardLogjac
