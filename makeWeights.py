import numpy as np
import sys, os
from numpy import nan
from nnet import NNet
from verbosePrint import vprint
import verbosePrint

nn = NNet(sizes=[3, 2], batch_size=2)
# nn = NNet([[[0.258584, -0.770864],
#             [0.320592, 0.253271],
#             [0.718206, -0.475193],
#             [0.517670, 0.043095]]])
# nn.setActivations(['sigmoid'])

nn.printnn()