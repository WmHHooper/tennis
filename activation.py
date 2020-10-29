import numpy as np
from numpy import nan
# from enum import Enum, auto
import sys

def linear(x, dummy=nan):
    return x  # the identity activation

def linear2deriv(output):
    return (output > 0) + (output <= 0)
    # returns 1 for every element of output

# Grokking, p. 125
def relu(x):
    return (x > 0) * x  # returns x if x > 0
    # return 0 otherwise

def relu2deriv(output):
    # return output > 0  # returns 1 for datain > 0
    # return 0 otherwise
    return output > 0  # returns 1 for datain >= 0

# these two functions reproduce a bug in
# Grokking, p. 156
def brelu(x):
    return (x > 0) * x  # returns x if x > 0
    # return 0 otherwise

def brelu2deriv(output):
    # return output > 0  # returns 1 for datain > 0
    # return 0 otherwise
    return output >= 0  # returns 1 for datain >= 0

def zeta(x):
    return (x > 0) * (x < 1) * x  # x if 0 < x < 1
    # return 0 otherwise

def zeta2deriv(output):
    return (output > 0) * (output < 1)  # 1 if 0 < y < 1
    # return 0 otherwise

# see Grokking, p. 173
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid2deriv(output):
    return output * (1 - output)

def tanh(x):
    return np.tanh(x)

def tanh2deriv(output):
    return 1 - (output ** 2)

# see Grokking, p. 173-174
def softmax(x):
    try:
        temp = np.exp(x)
        total = np.sum(temp, axis=1, keepdims=True)
        return temp / total
    except:
        print('SoftMax error: exp(%s)' % x.dtype)
        print(x)
        sys.exit(1)

def softmax2deriv(output):
    # This is not quite right, but it works because
    # a linear derivative pushes the deltas
    # in the proper direction
    # and approximately proper amount
    return (output > 0) + (output <= 0)  #i.e., all 1's

actMap = {
    'linear': [linear, linear2deriv],
    'relu': [relu, relu2deriv],
    'brelu': [brelu, brelu2deriv],
    'zeta': [zeta, zeta2deriv],
    'sigmoid': [sigmoid, sigmoid2deriv],
    'tanh': [tanh, tanh2deriv],
    'softmax': [softmax, softmax2deriv],
}

def getAct(s):
    return actMap[s.lower()][0]

def getDer(s):
    return actMap[s.lower()][1]

