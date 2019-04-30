import torch
from torch import nn

from .flow import Flow
#from utils import checkNan

class RNVP(Flow):
    def __init__(self, maskList, tList, sList, prior = None, name = "RNVP"):
        super(RNVP,self).__init__(prior,name)

        assert len(tList) == len(sList)
        assert len(tList) == len(maskList)

        self.maskList = nn.Parameter(maskList,requires_grad=False)
        self.maskListR = nn.Parameter(1-maskList,requires_grad=False)

        self.tList = torch.nn.ModuleList(tList)
        self.sList = torch.nn.ModuleList(sList)

    def inverse(self,y):
        inverseLogjac = y.new_zeros(y.shape[0])
        for i in range(len(self.tList)):
            y_ = y*self.maskList[i]
            s = self.sList[i](y_)*self.maskListR[i]
            t = self.tList[i](y_)*self.maskListR[i]
            y = y_ + self.maskListR[i] * (y * torch.exp(s) + t)
            for _ in y.shape[1:]:
                s = s.sum(dim=-1)
            inverseLogjac += s
        return y,inverseLogjac

    def forward(self,z):
        forwardLogjac = z.new_zeros(z.shape[0])
        for i in reversed(range(len(self.tList))):
            z_ = self.maskList[i]*z
            s = self.sList[i](z_)*self.maskListR[i]
            t = self.tList[i](z_)*self.maskListR[i]
            z = self.maskListR[i] * (z - t) * torch.exp(-s) + z_
            for _ in z.shape[1:]:
                s = s.sum(dim=-1)
            forwardLogjac -= s
        return z,forwardLogjac