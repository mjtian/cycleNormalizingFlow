from __future__ import division
import numpy as np

import subprocess
import os
import struct




def load_MNIST():
    '''
    download and unpack MNIST data.
    Returns:
        tuple: length is 4. They are training set data, training set label,
            test set data and test set label.
    '''
    base = "http://yann.lecun.com/exdb/mnist/"
    objects = ['t10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte',
               'train-images-idx3-ubyte', 'train-labels-idx1-ubyte']
    end = ".gz"
    path = "data/raw/"
    cmd = ["mkdir", "-p", path]
    subprocess.check_call(cmd)
    print('Downloading MNIST dataset. Please do not stop the program \
during the download. If you do, remove `data` folder and try again.')
    for obj in objects:
        if not os.path.isfile(path + obj):
            cmd = ["wget", base + obj + end, "-P", path]
            subprocess.check_call(cmd)
            cmd = ["gzip", "-d", path + obj + end]
            subprocess.check_call(cmd)

    def unpack(filename):
        '''unpack a single file.'''
        with open(filename, 'rb') as f:
            _, _, dims = struct.unpack('>HBB', f.read(4))
            shape = tuple(struct.unpack('>I', f.read(4))
                          [0] for d in range(dims))
            data = np.fromstring(f.read(), dtype=np.uint8).reshape(shape)
            return data

    # load objects
    data = []
    for name in objects:
        name = path + name
        data.append(unpack(name))
    labels = np.zeros([data[1].shape[0], 10])
    for i, iterm in enumerate(data[1]):
        labels[i][iterm] = 1
    data[1] = labels
    labels = np.zeros([data[3].shape[0], 10])
    for i, iterm in enumerate(data[3]):
        labels[i][iterm] = 1
    data[3] = labels
    return data

def random_draw(data, batch_size):
    '''
    random draw a batch of data and label.
    Args:
        data (ndarray): dataset with the first axis the batch dimension.
        label (ndarray): one-hot label for dataset,
            for example, 3 is [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
        batch_size (int): size of batch, the number of data to draw.
    Returns:
        tuple: length is 2, They are drawed samples from dataset, and labels.
    '''
    perm = np.random.permutation(data.shape[0])
    data_b = data[perm[:batch_size]]
    return data_b.reshape([data_b.shape[0], -1]) / 255.0
