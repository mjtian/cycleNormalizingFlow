import torch
from torch import nn
from utils import SimpleMLP

class Realnvp(nn.Module): # fill in the parent class
    def __init__(self,sList,tList,name="NICE"):
        super(Realnvp,self).__init__()
        self.name = name
        self.tList =nn.ModuleList(tList) # init your inner layer list here, remember torch has it's own init method
        self.sList =nn.ModuleList(sList)
    def inverse(self,z):
        x = z[:,:z.shape[1]//2]
        y = z[:,z.shape[1]//2:]
        for i in range(len(self.tList)-1,-1,-1): # write the transmission of variables here, may take multiply lines.
            if (i %2) ==0:
                ft = self.tList[i]
                fs = self.sList[i]
                x = (x-ft(y))*torch.exp(-fs(y))
            else:

                ft = self.tList[i]
                fs = self.sList[i]
                y = (y-ft(x))*torch.exp(-fs(x))

        z = torch.cat((x, y),1)
        return z

    def forward(self, z):
        x = z[:,:z.shape[1]//2]
        y = z[:,z.shape[1]//2:]
        for i in range(len(self.tList)):  # write the transmission of variables here, may take multiply lines.
            if (i %2) ==0:
                ft = self.tList[i]
                fs = self.sList[i]
                x = torch.exp(fs(y))*x + ft(y)
            else:
                ft = self.tList[i]
                fs = self.sList[i]
                y = torch.exp(fs(x))*y + ft(y)

        z = torch.cat((x, y),1)

        return z




