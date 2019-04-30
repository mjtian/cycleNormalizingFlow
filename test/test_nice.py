import torch
from numpy.testing import assert_array_almost_equal
import os
import sys
sys.path.append(os.getcwd())
import utils
from NICE import NICE
from gaussian import Gaussian

def test_bijective():
    p = Gaussian([8])
    tList =[utils.SimpleMLP([4, 10, 4]), utils.SimpleMLP([4, 10, 4]),utils.SimpleMLP([4, 10, 4]), utils.SimpleMLP([4, 10, 4])]
    #x = torch.randn(100,8) # Build your NICE net here, may take multiply lines.
    f = NICE(tList,prior=p)
    # import pdb
    # pdb.set_trace()
    x = f.sample(100)   #变换后的sample

    op = f.logProbability(x)   #变换后的logp
    y,pi = f.inverse(x)     #变换前sample
    pp = f.prior.logProbability(y)    #变换前logp
    yx,pf = f.forward(y)
    yxy,pfi = f.inverse(yx)

    assert_array_almost_equal(x.detach().numpy(),yx.detach().numpy())
    assert_array_almost_equal(y.detach().numpy(),yxy.detach().numpy())

    assert_array_almost_equal(pi.detach().numpy(),-pf.detach().numpy())
    assert_array_almost_equal(pf.detach().numpy(),-pfi.detach().numpy())

    assert_array_almost_equal((op+pi).detach().numpy(),pp.detach().numpy())
    assert_array_almost_equal((pp-pfi).detach().numpy(),op.detach().numpy())
if __name__ == "__main__":
    test_bijective()
