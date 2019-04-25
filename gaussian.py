import torch
import numpy as np
import torch.nn as nn

class Gaussian(nn.Module):
    def __init__(self,shapeList,name="Gaussian"):
        super(Gaussian,self).__init__()
        self.name = name
        self.shapeList = shapeList


    def sample(self, batch_s):
        size = [batch_s] +self.shapeList
        z = torch.randn(size)
        return z

    def logProbability(self, z):
        return(-0.5*z**2 - 0.5*torch.log(2.*np.pi)).reshape(z.shape[0],-1).sum(dim=1)







