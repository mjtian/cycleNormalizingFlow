import torch

import os
import sys
sys.path.append(os.getcwd())
from gaussian import Gaussian

def test_gaussian():
    p = Gaussian([3,2,2])
    x = p.sample(10)
    assert x.shape[0] ==10
    assert x.shape[1] ==3
    assert x.shape[2] ==2
    assert x.shape[3] ==2

    t = torch.tensor([i for i in range(3*2*2)]).to(torch.float32).reshape(1,3,2,2)

    logp = p.logProbability(t)

    assert logp.item() == -264.0273
