import torch
import torch.nn as nn




class SimpleMLP(nn.Module):
    def __init__(self,dimsList,activation=None,name="SimpleMLP"):

        super(SimpleMLP,self).__init__()

        # This build up activation if it is not give, in this example two ways of set default value are given.
        if activation is None:
            activation = [nn.ReLU() for i in range(len(dimsList)-2)]
            activation.append(nn.Sigmoid())

        assert(len(dimsList) == len(activation)+1)


        layerList = []
        for no in range(len(activation)):
            layerList.append(nn.Linear(dimsList[no], dimsList[no +1]))
            layerList.append(activation[no])


        self.layerList = nn.ModuleList(layerList)
        self.name = name

    def forward(self,x):
        for layer in self.layerList:
            x =layer(x)
        return x


if __name__=='__main__':
    net = SimpleMLP([28*28,100,50,1])
    test = torch.randn(10,28*28)
    result = net.forward(test)
    assert result.shape[0] == 10
    assert result.shape[1] == 1




