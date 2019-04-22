import torch
import numpy as np
from numpy.testing import assert_array_almost_equal


def test_example1():
    assert 1+1 == 2

def test_example2():
    t = torch.tensor([1,1.5,2]).to(torch.float64)
    tt = np.array([1.0,1.5,2.0],dtype=np.double)