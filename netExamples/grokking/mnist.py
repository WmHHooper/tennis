import sys, numpy as np
import os
from numpy import nan
from nnet import NNet
from verbosePrint import vprint, vprintnn, demo, vIteration, stage
from keras.datasets import mnist

demo = 101
vIteration = -1
stage= 'a'

# download this file from
# https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
# and store it in $HOME/.keras/datasets/mnist.npz
# see: https://keras.io/api/datasets/mnist/#load_data-function
download = 'mnist.npz'
try:
    (x_train, y_train), (x_test, y_test) = mnist.load_data(download)
except:
    print('Failed to find %s.' % download)
    sys.exit(1)
images, labels = (x_train[0:1000].reshape(1000, 28 * 28) / 255, y_train[0:1000])

one_hot_labels = np.zeros((len(labels), 10))
for i, l in enumerate(labels):
    one_hot_labels[i][l] = 1
labels = one_hot_labels

test_images = x_test.reshape(len(x_test),28*28) / 255
test_labels = np.zeros((len(y_test),10))
for i,l in enumerate(y_test):
        test_labels[i][l] = 1
