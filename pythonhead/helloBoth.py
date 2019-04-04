
import numpy as np
import torch
from numpy.testing import assert_array_almost_equal

a = torch.randn([1 ,2],dtype=torch.float64)
b = np.random.randn(2,4)

b_tensor = torch.from_numpy(b)
a_numpy = a.numpy()

c = np.dot(a_numpy, b)
c_tensor = torch.matmul(a, b_tensor)

print (a)
print (b)
print (b_tensor)
print (a_numpy)
print (c)
print (c_tensor)
assert_array_almost_equal(c, c_tensor.numpy())
