import torch
from numpy.testing import assert_array_almost_equal
import os
import sys
sys.path.append(os.getcwd())
import utils

from realnvp import Realnvp

def test_bijective():
    tList =[utils.SimpleMLP([4, 10, 4]), utils.SimpleMLP([4, 10, 4]),utils.SimpleMLP([4, 10, 4]), utils.SimpleMLP([4, 10, 4])]
    sList =[utils.SimpleMLP([4, 10, 4]), utils.SimpleMLP([4, 10, 4]),utils.SimpleMLP([4, 10, 4]), utils.SimpleMLP([4, 10, 4])]
    x = torch.randn(1,8)
     # Build your NICE net here, may take multiply lines.
    # import pdb
    # pdb.set_trace()
    f = Realnvp(sList,tList)
    y = f.inverse(x)
    yx = f.forward(y)
    yxy = f.inverse(yx)

    assert_array_almost_equal(x.detach().numpy(),yx.detach().numpy())
    assert_array_almost_equal(y.detach().numpy(),yxy.detach().numpy())

if __name__ == "__main__":
    test_bijective()
