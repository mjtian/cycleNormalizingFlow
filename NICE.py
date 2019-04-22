import torch
from torch import nn

class NICE(<_>): # fill in the parent class
    def __init__(self,tList,name="NICE"):
        super(NICE,self).__init__()
        self.name = name
        self.tList = <_> # init your inner layer list here, remember torch has it's own init method

    def inverse(self,y):
        for i in range(len(self.tList)):
            <_> # write the transmission of variables here, may take multiply lines.

        return y

    def forward(self,z):
        for i in range(len(self.tList)):
            <_> # write the transmission of variables here, may take multiply lines.

        return z