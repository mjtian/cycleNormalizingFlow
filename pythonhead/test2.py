import <_>

net1 = <_>

net2 = <_>

test = torch.randn(10,28*28)
result1 = net1(test)
result2 = net2(result2)

assert result1.shape[0] == 10
assert result2.shape[0] == 10

assert result1.shape[1] == 1
assert result2.shape[1] == 28*28