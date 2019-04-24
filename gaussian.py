import torch

class Gaussian(nn.Module):
    def __init__(self,shapeList,name="Gaussian"):
      super(Gaussian,self).__init__()
        self.name = name
        self.shapeList = nn.ModuleList(shapeList)
    def sample(self,batch_s):
        for i in range(len(self.shapeList)):
            z = torch.randn(batch_s,self.shapeList[i])
        return z

    def logProbability(self,shape):

