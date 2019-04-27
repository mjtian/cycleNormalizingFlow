from __future__ import division
import numpy as np

from utils import load_MNIST, random_draw #记得init
from realnvp import Realnvp

def train():
    np.random.seed(5)
    batch_size = 100
    learning_rate = 0.5
    num_epoch = 10
    num_iteration = len(train_data) // batch_size
    # an epoch means running through the training set roughly once

    train_data, train_label, test_data, test_label = load_MNIST()
    x, label = random_draw(test_data, test_ label, 1000)
    tList =[utils.SimpleMLP([4, 10, 4]), utils.SimpleMLP([4, 10, 4]),utils.SimpleMLP([4, 10, 4]), utils.SimpleMLP([4, 10, 4])]
    sList =[utils.SimpleMLP([4, 10, 4]), utils.SimpleMLP([4, 10, 4]),utils.SimpleMLP([4, 10, 4]), utils.SimpleMLP([4, 10, 4])]
    p =
    f = Realnvp(sList,tList,prior=p)
    result_= f.forward(x)
    result = result_[0]
    lgp =f.logProbability(x)
    loss = -lgp.sum/batchsize
    print('Before Training.\nTest loss = %.4f, correct rate = %.3f' % (loss, match_ratio(result, label)))

   for epoch in range(num_epoch):
        for j in range(num_iteration):
            x, label = random_draw(train_data, train_label, batch_size)
            result_ = f.forward( x)
            result = result_[0]

            f_backward(f)

            # update network parameters
            for node in net:
                for p, p_delta in zip(node.parameters, node.parameters_deltas):
                    p -= learning_rate * p_delta  # stochastic gradient descent

        print("epoch = %d/%d, loss = %.4f, corret rate = %.2f" %
              (epoch, num_epoch, loss, match_ratio(result, label)))

    x, label = random_draw(test_data, test_label, 1000)
    result1= f_forward(x)
    result2 = result1[0]
    lgp =f.logProbability(x)
    loss = -lgp.sum/batchsize
    print('After Training.\nTest loss = %.4f, correct rate = %.3f' % (loss, match_ratio(result2, label)))


if __name__ == "__main__":
    train()
