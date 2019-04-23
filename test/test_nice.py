import torch
from numpy.testing import assert_array_almost_equal
import os
import sys
sys.path.append(os.getcwd())
import utils
from NICE import NICE

def test_bijective():
    tList =[utils.SimpleMLP([4, 10, 4]), utils.SimpleMLP([4, 10, 4]),utils.SimpleMLP([4, 10, 4]), utils.SimpleMLP([4, 10, 4])]
    x = torch.randn(100,8)
    f = NICE(tList)  # Build your NICE net here, may take multiply lines.
    y = f.inverse(x)
    yx = f.forward(y)
    yxy = f.inverse(yx)

    assert_array_almost_equal(x.detach().numpy(),yx.detach().numpy())
    assert_array_almost_equal(y.detach().numpy(),yxy.detach().numpy())
