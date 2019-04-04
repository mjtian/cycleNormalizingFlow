import <_>
from <_> import nn
# import all needed libs, fill all blanks.

class SimpleMLP(nn.<_>): # Fill this blank. Hint: https://pytorch.org/docs/stable/nn.html?highlight=forward#torch.nn.Module
    def __init__(self,dimsList,activation=None,name="SimpleMLP"):
        # __init__ gives the init of instances of this class, this function is called a method of this class.
        super(SimpleMLP,self).__init__()# This init the parent class of this class.

        # This build up activation if it is not give, in this example two ways of set default value are given.
        if activation is <_>: # Fill this blank.
            activation = [nn.ReLU() for _ in range(len(dimsList)-2)]
            activation.append(<_>) # The last activation function should be sigmoid, refer to pytorch doc to find its init and fill here.

        assert(len(dimsList) <_> len(activation)+1) # Simple sanity check. what relation does these two value have?


        layerList = []
        for no in <_>(len(activation)): # Fill this blank, create a typical python loop here.
            layerList.append(nn.Linear(<_>,<_>)) # Refer to pytorch to find torch.nn.Linear, see how to init it.
            layerList.append(activation[<_>]) # Fill this blank, so at end of every Linear layer there will be an activation function from list activation.

        self.layerList = <_>(layerList) # Here init sub module for our net, Hint: https://pytorch.org/docs/stable/nn.html#torch.nn.ModuleList
        self.name = name # This give a name to our net, it's a convention.

    def forward(self,x): # The default method to transform a set variable in pytorch, entrance of data.
        for layer in self.layerList:
            x = <_> # Fill this blank, Hint: https://pytorch.org/docs/stable/nn.html?highlight=forward#torch.nn.Module.forward
        return <_> # Fill this blank


if __name__=='__main__':
    net = SimpleMLP([28*28,100,50,1])
    test = torch.randn(10,28*28)
    result = net.forward(test)
    assert result.shape[0] == 10
    assert result.shape[1] == 1