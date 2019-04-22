import torch
from numpy.testing import assert_array_almost_equal

def test_bijective():
    x = torch.randn(100,8)
    f = <_>  # Build your NICE net here, may take multiply lines.
    y = f.inverse(x)
    yx = f.forward(y)
    yxy = f.inverse(yx)

    assert_array_almost_equal(x.detach().numpy(),yx.detach().numpy())
    assert_array_almost_equal(y.detach().numpy(),yxy.detach().numpy())