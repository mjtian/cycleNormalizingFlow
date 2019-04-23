import torch
from torch import nn
from utils import SimpleMLP

class NICE(nn.Module): # fill in the parent class
    def __init__(self,tList,name="NICE"):
        super(NICE,self).__init__()
        self.name = name
        self.tList = tList # init your inner layer list here, remember torch has it's own init method

    def inverse(self,y):
        for i in range(len(self.tList)): # write the transmission of variables here, may take multiply lines.
            net = SimpleMLP(i + 1, 50, 25, 8)
            y = net.forward(y)
        return y

    def forward(self,z):
        for i in range(len(self.tList)):  # write the transmission of variables here, may take multiply lines.
            # net = SimpleMLP(i+1, 25, 50, 100)
            # z = net.forward(z)

        self.tList=[v,u]

        v_y = self.tList[0](y)
        u_x = self.tList[1](x)

        f0 =(x+v_y,y)
        f1 =(x, y + u_x)

        f = f0 * f1
        return z



if __name__=='__main__':

    net = NICE([100,8])

    result1 = net.inverse
    result2 = net.forward
    print (result1.shape)

