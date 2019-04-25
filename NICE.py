import torch
from torch import nn


class NICE(nn.Module): # fill in the parent class
    def __init__(self,tList,prior,name="NICE"):
        super(NICE,self).__init__()
        self.name = name
        self.tList =nn.ModuleList(tList) # init your inner layer list here, remember torch has it's own init method
        self.prior =prior #<------note here!!
    def inverse(self,z):
        inverseLogjac = z.new_zeros(z.shape[0]) <------note here!!
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
        return z,inverseLogjac <------note here!!

    def forward(self, z):
        forwardLogjac = z.new_zeros(z.shape[0]) #<------note here!!
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

        return z,forwardLogjac <------note here!!

    def sample(self,batchSize):
        pass

    def logProbability(self,z):
        pass





