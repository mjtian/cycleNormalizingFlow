import torch

class Gaussian(nn.Module):
    def __init__(self,shapeList,name="Gaussian"):
      super(Gaussian,self).__init__()
        self.name = name
        self.shapeList = nn.ModuleList(shapeList)
    def sample(self,batch_s):
        z = torch.randn(batch_s,self.shapeList)
        return z

    def logProbability(self,shape):

