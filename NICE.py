import torch
from torch import nn
from utils import SimpleMLP

class NICE(nn.Module): # fill in the parent class
    def __init__(self,tList,name="NICE"):
        super(NICE,self).__init__()
        self.name = name
        self.tList =nn.ModuleList(tList) # init your inner layer list here, remember torch has it's own init method

    def inverse(self,z):
        x = z[:,:z.shape[1]//2]
        y = z[:,z.shape[1]//2:]
        for i in range(len(self.tList)-1,-1,-1): # write the transmission of variables here, may take multiply lines.
            if (i %2) ==0:
                f = self.tList[i]
                x = x - f(y )


            else:

                f = self.tList[i]
                y = y - f(x)

        z = torch.cat((x, y),1)
        return z

    def forward(self, z):
        x = z[:,:z.shape[1]//2]
        y = z[:,z.shape[1]//2:]
        for i in range(len(self.tList)):  # write the transmission of variables here, may take multiply lines.
            if (i %2) ==0:
                f = self.tList[i]
                x = x + f(y)
            else:
                f = self.tList[i]
                y = y + f(x)


        z = torch.cat((x, y),1)

        return z




