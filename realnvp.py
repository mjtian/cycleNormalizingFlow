import torch
from torch import nn


class Realnvp(nn.Module): # fill in the parent class
    def __init__(self,sList,tList,prior,name="Realnvp"):
        super(Realnvp,self).__init__()
        self.name = name
        self.tList =nn.ModuleList(tList) # init your inner layer list here, remember torch has it's own init method
        self.sList =nn.ModuleList(sList)
        self.prior= prior #<------note here!!
    def inverse(self,z):
        inverseLogjac = z.new_zeros(z.shape[0]) #<------note here!!
        x = z[:,:z.shape[1]//2]
        y = z[:,z.shape[1]//2:]
        for i in range(len(self.tList)-1,-1,-1): # write the transmission of variables here, may take multiply lines.
            if (i %2) ==0:
                ft = self.tList[i]
                fs = self.sList[i]
                x = (x-ft(y))*torch.exp(-fs(y))
                inverseLogjac -= fs(y)
            else:
                ft = self.tList[i]
                fs = self.sList[i]
                y = (y-ft(x))*torch.exp(-fs(x))
                inverseLogjac -= fs(x)

        z = torch.cat((x, y),1)
        return z,inverseLogjac #<------note here!!

    def forward(self, z):
        forwardLogjac = z.new_zeros(z.shape[0]) #<------note here!!
        x = z[:,:z.shape[1]//2]
        y = z[:,z.shape[1]//2:]
        for i in range(len(self.tList)):  # write the transmission of variables here, may take multiply lines.
            if (i %2) ==0:
                ft = self.tList[i]
                fs = self.sList[i]
                x = torch.exp(fs(y))*x + ft(y)
                forwardLogjac += fs(y)
            else:
                ft = self.tList[i]
                fs = self.sList[i]
                y = torch.exp(fs(x))*y + ft(x)
                forwardLogjac += fs(x)
        z = torch.cat((x, y),1)

        return z,forwardLogjac #<------note here!!

    def sample(self,batchSize):
        b = self.prior.sample(batchSize)
        a = self.forward(b)
        return a[0]

    def logProbability(self,z):
        a = self.inverse(z)
        pp =self.prior.logProbability(a[0])
        logp =pp - a[1]
        return logp



