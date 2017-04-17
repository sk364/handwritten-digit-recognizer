"""
Script to collect MNIST data
"""

import os
import gzip
import pickle
import wget
from collections import Counter
from mnist import MNIST
from skimage.feature import hog
import numpy as np


def load_mnist():
    if not os.path.exists(os.path.join(os.curdir, 'data')):
        os.mkdir(os.path.join(os.curdir, 'data'))
        wget.download('http://deeplearning.net/data/mnist/mnist.pkl.gz',
                      out='data')

    data_file = gzip.open(os.path.join(os.curdir, 'data', 'mnist.pkl.gz'), 'rb')
    training_data, validation_data, test_data = pickle.load(data_file,
                                                            encoding="latin1")
    data_file.close()

    training_inputs = [np.reshape(x, (784, 1)) for x in training_data[0]]
    training_results = [vectorized_result(y) for y in training_data[1]]
    training_data = list(zip(training_inputs, training_results))

    validation_inputs = [np.reshape(x, (784, 1)) for x in validation_data[0]]
    validation_results = validation_data[1]
    validation_data = list(zip(validation_inputs, validation_results))

    test_inputs = [np.reshape(x, (784, 1)) for x in test_data[0]]
    test_data = list(zip(test_inputs, test_data[1]))

    return training_data, validation_data, test_data

'''
trying something

'''


def load_mnist2():
    mndata = MNIST('./data/')

    features, labels = mndata.load_training()

    # Extract the features and labels
    features = np.array(features, 'int16')
    labels = np.array(labels, 'int')

    # Extract the hog features
    list_hog_fd = []

    for feature in features:
        fd = hog(feature.reshape((28, 28)), orientations=9,
                 pixels_per_cell=(14, 14), cells_per_block=(1, 1),
                 visualise=False)

        list_hog_fd.append(fd)

    hog_features_training = np.array(list_hog_fd, 'float64')

    print ("Count of digits in dataset", Counter(labels))

    return hog_features_training


def vectorized_result(y):
    e = np.zeros((10, 1))
    e[y] = 1.0
    return e
