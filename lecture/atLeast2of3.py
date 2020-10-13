import numpy as np
import sys, os
from numpy import nan
from nnet import NNet
from verbosePrint import vprint
import verbosePrint

inputData = [
    [0, 0, 0, 1],
    [0, 0, 1, 1],
    [0, 1, 0, 1],
    [0, 1, 1, 1],
    [1, 0, 0, 1],
    [1, 0, 1, 1],
    [1, 1, 0, 1],
    [1, 1, 1, 1],
]

targetData = [
    [0, 0],
    [0, 0],
    [0, 0],
    [1, 0],
    [0, 0],
    [1, 0],
    [1, 0],
    [1, 1],
]

# nn = NNet(sizes=[4, 2])
nn = NNet([[[0.258584, -0.770864],
            [0.320592, 0.253271],
            [0.718206, -0.475193],
            [0.517670, 0.043095]]])
# nn.setActivations(['sigmoid'])

verbosePrint.vIteration = -1
verbosePrint.stage = ''

nn.printnn()

for row_index in range(len(targetData)):
    datain = inputData[row_index:row_index + 1]
    goal_prediction = targetData[row_index:row_index + 1]
    prediction = nn.fire(datain)
    print('Input:', datain, end=' ')
    print('Goal:', goal_prediction, end=' ')
    print('Prediction:', prediction)

cycles = 400
report = cycles/10

for iteration in range(cycles):
    vprint(iteration, '~~~~~~~~~~~ Iteration %d ~~~~~~~~~~~' % iteration)
    combinedError = 0
    for row_index in range(len(targetData)):
        datain = inputData[row_index:row_index + 1]
        goal_prediction = targetData[row_index:row_index + 1]
        prediction = nn.fire(datain)
        # print('Prediction:' + str(prediction))
        vprint(iteration, nn)

        error = (goal_prediction - prediction) ** 2
        combinedError += error

        nn.learn(datain, goal_prediction)

    if iteration % report == 0:
        print("Error:" + str(combinedError))
    vprint(iteration, '')
    vprint(iteration, '~~~~~~~~~~~~~~~ End ~~~~~~~~~~~~~~~~')
    vprint(iteration, nn, quit=True)

print("Error:" + str(combinedError))

for row_index in range(len(targetData)):
    datain = inputData[row_index:row_index + 1]
    goal_prediction = targetData[row_index:row_index + 1]
    prediction = nn.fire(datain)
    print('Input:', datain, end=' ')
    print('Goal:', goal_prediction, end=' ')
    print('Prediction:', prediction)

nn.printnn()