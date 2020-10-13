import numpy as np
from nnet import NNet

nn = NNet([[[0.1],
            [0.2],
            [-0.1]]])
nn.setAlpha(0.01)
nn.setVerbose(True)

datain = [[8.5, 0.65, 1.2]]
goal = [[1]]
for i in range(4):
    output = nn.fire(datain)
    print('Goal:    ' + str(goal))
    print(nn)
    nn.learn(datain, goal)
