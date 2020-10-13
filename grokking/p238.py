import numpy as np
from nnet import NNet
from verbosePrint import vprint

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
        datain = streetlights[row_index:row_index + 1]
        goal_prediction = walk_vs_stop[row_index:row_index + 1]
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