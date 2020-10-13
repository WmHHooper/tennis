import numpy as np
from numpy import nan
from enum import Enum, auto

def linear(x, dummy=nan):
    return x  # the identity activation

def linear2deriv(output):
    return (output > 0) + (output <= 0)
    # returns 1 for every element of output

def relu(x):
    return (x > 0) * x  # returns x if x > 0
    # return 0 otherwise

def relu2deriv(output):
    # return output > 0  # returns 1 for datain > 0
    # return 0 otherwise
    return output > 0  # returns 1 for datain >= 0

# these two functions reproduce a bug in Grokking, p. 284
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
    return (output > 0) * (output < 1)   # 1 if 0 < y < 1
    # return 0 otherwise

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
    # see Grokking, p. 335

def sigmoid2deriv(output):
    return output * (1 - output)
    # see Grokking, p. 354

def tanh(x):
    return np.tanh(x)
    # see Grokking, p. 336

def tanh2deriv(output):
    return 1 - (output ** 2)
    # see Grokking, p. 354

def softmax(x):
    temp = np.exp(x)
    return temp / np.sum(temp, axis=1, keepdims=True)
    # see Grokking, p. 344

def softmax2deriv(output):
    raise NotImplementedError('Softmax is intended for final output, only!')

actMap = {
    'linear': linear,
    'relu': relu,
    'brelu': brelu,
    'zeta': zeta,
    'sigmoid': sigmoid,
    'tanh': tanh,
    'softmax': softmax
}

def getAct(s):
    return actMap[s.lower()]

derMap = {
    'linear': linear2deriv,
    'relu': relu2deriv,
    'brelu': brelu2deriv,
    'zeta': zeta2deriv,
    'sigmoid': sigmoid2deriv,
    'tanh': tanh2deriv,
    'softmax': softmax2deriv
}

def getDer(s):
    return derMap[s.lower()]

