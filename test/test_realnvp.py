import torch
from numpy.testing import assert_array_almost_equal
import os
import sys
sys.path.append(os.getcwd())
import utils

from realnvp import Realnvp
from gaussian import Gaussian

def test_bijective():
    p = Gaussian([8])
    tList =[utils.SimpleMLP([4, 10, 4]), utils.SimpleMLP([4, 10, 4]),utils.SimpleMLP([4, 10, 4]), utils.SimpleMLP([4, 10, 4])]
    sList =[utils.SimpleMLP([4, 10, 4]), utils.SimpleMLP([4, 10, 4]),utils.SimpleMLP([4, 10, 4]), utils.SimpleMLP([4, 10, 4])]
    #x = torch.randn(1,8)
    #  # Build your NICE net here, may take multiply lines.
    # import pdb
    # pdb.set_trace()
    f = Realnvp(sList,tList,prior=p)
    x = f.sample(10)
    op = f.logProbability(x)
    y,pi = f.inverse(x)
    pp = f.prior.logProbability(y)
    yx,pf = f.forward(y)
    yxy,pfi = f.inverse(yx)

    assert_array_almost_equal(x.detach().numpy(),yx.detach().numpy(), decimal=6)
    assert_array_almost_equal(y.detach().numpy(),yxy.detach().numpy(), decimal=6)

    assert_array_almost_equal(pi.detach().numpy(),-pf.detach().numpy(), decimal=6)
    assert_array_almost_equal(pf.detach().numpy(),-pfi.detach().numpy(), decimal=6)

    assert_array_almost_equal((op+pi).detach().numpy(),pp.detach().numpy(), decimal=6)
    assert_array_almost_equal((pp-pfi).detach().numpy(),op.detach().numpy())

if __name__ == "__main__":
    test_bijective()
