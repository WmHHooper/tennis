import numpy as np
import sys, os
from numpy import nan
from nnet import NNet
from verbosePrint import vp, vprint, vprintnn
import verbosePrint

demo = 6
verbosePrint.vIteration = 0
verbosePrint.stage= ''

vp();

if demo == 4:
    # weights = [[[0.1],
    #             [0.2],
    #             [-0.1]]]
    nn = NNet([[[0.1],
                [0.2],
                [-0.1]]])
    nn.setAlpha(0.01)

    datain = np.array([[8.5, 0.65, 1.2]])
    goal = np.array([[1]])
    for i in range(4):
        output = nn.fire(datain)
        print('Goal:    ' + str(goal))
        print(nn)
        nn.learn(datain, goal)

if demo == 6:
    nn = NNet([[[0.5],
                [0.48],
                [-0.7]]])
    nn.setAlpha(0.1)

    streetlights = np.array([[1, 0, 1],
                             [0, 1, 1],
                             [0, 0, 1],
                             [1, 1, 1],
                             [0, 1, 1],
                             [1, 0, 1]])

    walk_vs_stop = np.array([[0, 1, 0, 1, 1, 0]]).T

    datain = streetlights[0]  # [1,0,1]
    goal_prediction = walk_vs_stop[0]  # equals 0... i.e. "stop"

    for iteration in range(40):
        vprint(iteration, '~~~~~~~~~~~ Iteration %d ~~~~~~~~~~~' % iteration)
        error_for_all_lights = 0
        for row_index in range(len(walk_vs_stop)):
            datain = streetlights[row_index:row_index+1]
            goal_prediction = walk_vs_stop[row_index:row_index+1]
            prediction = nn.fire(datain)
            # print('Prediction:' + str(prediction))
            vprint(iteration, nn)

            error = (goal_prediction - prediction) ** 2
            error_for_all_lights += error

            nn.learn(datain, goal_prediction)

        print("Error:" + str(error_for_all_lights))
        vprint(iteration, '')
    vprint(iteration, '~~~~~~~~~~~~~~~ End ~~~~~~~~~~~~~~~~')
    vprint(iteration, nn, quit=True)

if demo == 7:
    np.random.seed(1)

    def relu(x):
        return (x > 0) * x  # returns x if x > 0
        # return 0 otherwise


    def relu2deriv(output):
        return output > 0  # returns 1 for datain > 0
        # return 0 otherwise


    streetlights = np.array([[1, 0, 1],
                             [0, 1, 1],
                             [0, 0, 1],
                             [1, 1, 1]])

    walk_vs_stop = np.array([[1, 1, 0, 0]]).T

    alpha = 0.2
    hidden_size = 4

    layer_0 = [nan] * 3
    layer_1 = [nan] * 4
    layer_2 = nan
    weights_0_1 = 2 * np.random.random((3, hidden_size)) - 1
    weights_1_2 = 2 * np.random.random((hidden_size, 1)) - 1

    nIterations = 60
    # verbose = True
    # nIterations = 2

    for iteration in range(nIterations):
        vprint(iteration, '~~~~~~~~~~~ Iteration %d ~~~~~~~~~~~' % iteration)
        layer_2_error = 0
        for i in range(len(streetlights)):
            layer_0 = streetlights[i:i + 1]
            layer_1 = relu(np.dot(layer_0, weights_0_1))
            layer_2 = np.dot(layer_1, weights_1_2)
            goal = walk_vs_stop[i:i + 1]
            vprint(iteration, 'Goal:   ' + str(goal))
            vprintnn(iteration, quit=True)

            layer_2_error += np.sum((layer_2 - goal) ** 2)

            layer_2_delta = (layer_2 - goal)
            derivative = relu2deriv(layer_1)
            layer_1_delta = layer_2_delta.dot(weights_1_2.T) * derivative

            weights_1_2 -= alpha * layer_1.T.dot(layer_2_delta)
            weights_0_1 -= alpha * layer_0.T.dot(layer_1_delta)

        if (iteration % 10 == 9):
            print("Error: " + str(layer_2_error))

if demo == 8:
    np.random.seed(1)

    streetlights = np.array([[1, 0, 1],
                             [0, 1, 1],
                             [0, 0, 1],
                             [1, 1, 1]])

    walk_vs_stop = np.array([[1, 1, 0, 0]]).T

    nn = NNet(sizes=[3, 4, 1])
    nn.setAlpha(0.2)
    nn.setActivations(['relu', 'linear'])
    vprint(0, nn)

    nIterations = 60
    # verbose = True
    # nIterations = 2

    for iteration in range(nIterations):
        vprint(iteration, '~~~~~~~~~~~ Iteration %d ~~~~~~~~~~~' % iteration)
        layer_2_error = 0
        for i in range(len(streetlights)):
            datain = streetlights[i:i + 1]
            prediction = nn.fire(datain)
            goal = walk_vs_stop[i:i+1]
            vprint(iteration, 'Goal:   ' + str(goal))
            # print('Prediction:' + str(prediction))
            vprint(iteration, nn, quit=True)

            error = (goal - prediction) ** 2
            layer_2_error += error

            nn.learn(datain, goal)

        if (iteration % 10 == 9):
            print("Error:" + str(layer_2_error))
