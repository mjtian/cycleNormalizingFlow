import torch
from torch import nn


class Realnvp(nn.Module): # fill in the parent class
    def __init__(self,sList,tList,prior,maskList,name="Realnvp"):
        super(Realnvp,self).__init__()
        self.name = name
        self.tList =nn.ModuleList(tList) # init your inner layer list here, remember torch has it's own init method
        self.sList =nn.ModuleList(sList)
        self.prior= prior
        assert len(tList) == maskList.shape[0]
        self.maskList = nn.Parameter(maskList,requires_grad=False)
    def inverse(self,z):
        inverseLogjac = z.new_zeros(z.shape[0])
        for i in range(len(self.tList)-1,-1,-1): # write the transmission of variables here, may take multiply lines.
            y = torch.masked_select(z,self.maskList[i]).view(z.shape[0],-1)
            x = torch.masked_select(z,1-self.maskList[i]).view(z.shape[0],-1)
            if (i %2) ==0:
                ft = self.tList[i]
                fs = self.sList[i]
                x = (x-ft(y))*torch.exp(-fs(y))
                inverseLogjac = inverseLogjac - fs(y).reshape(z.shape[0],-1).sum(dim=1)
            else:
                ft = self.tList[i]
                fs = self.sList[i]
                y = (y-ft(x))*torch.exp(-fs(x))
                inverseLogjac = inverseLogjac - fs(x).reshape(z.shape[0],-1).sum(dim=1)
            output = torch.zeros(z.shape).to(z)
            output.masked_scatter_(self.maskList[i],y)
            output.masked_scatter_(1-self.maskList[i],x)
            z = output

        return z,inverseLogjac

    def forward(self, z):
        forwardLogjac = z.new_zeros(z.shape[0])
        x = z[:,:z.shape[1]//2]
        y = z[:,z.shape[1]//2:]
        for i in range(len(self.tList)):  # write the transmission of variables here, may take multiply lines.
            y = torch.masked_select(z,self.maskList[i]).view(z.shape[0],-1)
            x = torch.masked_select(z,1-self.maskList[i]).view(z.shape[0],-1)
            if (i %2) ==0:
                ft = self.tList[i]
                fs = self.sList[i]
                x = torch.exp(fs(y))*x + ft(y)
                forwardLogjac = forwardLogjac + fs(y).reshape(z.shape[0],-1).sum(dim=1)
            else:
                ft = self.tList[i]
                fs = self.sList[i]
                y = torch.exp(fs(x))*y + ft(x)
                forwardLogjac = forwardLogjac + fs(x).reshape(z.shape[0],-1).sum(dim=1)
            output = torch.zeros(z.shape).to(z)
            output.masked_scatter_(self.maskList[i],y)
            output.masked_scatter_(1-self.maskList[i],x)
            z = output

        z = torch.cat((x, y),1)

        return z,forwardLogjac

    def sample(self, batchSize):
        b = self.prior.sample(batchSize)
        a = self.forward(b)
        return a[0]

    def logProbability(self,z):
        a = self.inverse(z)
        pp =self.prior.logProbability(a[0])
        logp =pp - a[1]
        return logp



