import numpy as np
import sys, os
from numpy import nan
from nnet import NNet
from verbosePrint import vprint
import verbosePrint

inputData = [
 [0, 0, 0, 0, 0],
 [0, 0, 0, 0, 1],
 [0, 0, 0, 1, 0],
 [0, 0, 0, 1, 1],
 [0, 0, 1, 0, 0],
 [0, 0, 1, 0, 1],
 [0, 0, 1, 1, 0],
 [0, 0, 1, 1, 1],
 [0, 1, 0, 0, 0],
 [0, 1, 0, 0, 1],
 [0, 1, 0, 1, 0],
 [0, 1, 0, 1, 1],
 [0, 1, 1, 0, 0],
 [0, 1, 1, 0, 1],
 [0, 1, 1, 1, 0],
 [0, 1, 1, 1, 1],
 [1, 0, 0, 0, 0],
 [1, 0, 0, 0, 1],
 [1, 0, 0, 1, 0],
 [1, 0, 0, 1, 1],
 [1, 0, 1, 0, 0],
 [1, 0, 1, 0, 1],
 [1, 0, 1, 1, 0],
 [1, 0, 1, 1, 1],
 [1, 1, 0, 0, 0],
 [1, 1, 0, 0, 1],
 [1, 1, 0, 1, 0],
 [1, 1, 0, 1, 1],
 [1, 1, 1, 0, 0],
 [1, 1, 1, 0, 1],
 [1, 1, 1, 1, 0],
 [1, 1, 1, 1, 1],
]

targetData = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 1],
    [0, 0, 0],
    [0, 1, 0],
    [0, 1, 0],
    [0, 0, 0],
    [0, 0, 0],
    [1, 0, 0],
    [1, 0, 0],
    [1, 0, 1],
    [1, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 1],
    [0, 0, 0],
    [0, 1, 0],
    [0, 1, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 1],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
]

nn = NNet(sizes=[5, 3], bias=True)
nn.setActivations(['sigmoid'])

verbosePrint.vIteration = -1
verbosePrint.stage = ''

nn.checkup(inputData, targetData)

cycles = 40
report = cycles/10

for iteration in range(cycles + 1):
    vprint(iteration, '~~~~~~~~~~~ Iteration %d ~~~~~~~~~~~' % iteration)
    combinedError = 0
    for row_index in range(len(targetData)):
        datain = inputData[row_index:row_index + 1]
        goal_prediction = targetData[row_index:row_index + 1]
        prediction = nn.fire(datain)
        # print('Prediction:' + str(prediction))
        vprint(iteration, 'Goal:\n%s' % str(goal_prediction))
        vprint(iteration, nn)

        error = (goal_prediction - prediction) ** 2
        combinedError += error

        nn.learn(datain, goal_prediction)

    if iteration % report == 0:
        print('Iteration: %d Error: %s' % (iteration, str(combinedError)))
    vprint(iteration, '')
    vprint(iteration, '~~~~~~~~~~~~~~~ End ~~~~~~~~~~~~~~~~')
    vprint(iteration, nn, quit=True)

print()
nn.checkup(inputData, targetData)
