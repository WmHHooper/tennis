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
[0, 0, 0, 0],
[0, 0, 0, 0],
[0, 0, 0, 0],
[0, 0, 0, 1],
[0, 0, 0, 0],
[0, 0, 0, 0],
[0, 0, 1, 0],
[0, 0, 0, 0],
[0, 0, 0, 0],
[0, 0, 1, 0],
[0, 0, 0, 0],
[0, 0, 0, 1],
[0, 1, 0, 0],
[0, 1, 0, 0],
[0, 0, 0, 0],
[0, 0, 0, 0],
[0, 0, 0, 0],
[0, 0, 0, 0],
[0, 1, 0, 0],
[0, 1, 0, 1],
[0, 0, 0, 0],
[0, 0, 0, 0],
[0, 0, 1, 0],
[0, 0, 0, 0],
[1, 0, 0, 0],
[1, 0, 1, 0],
[1, 0, 0, 0],
[1, 0, 0, 1],
[0, 0, 0, 0],
[0, 0, 0, 0],
[0, 0, 0, 0],
[0, 0, 0, 0],
]

# inputData = tuple([tuple([e for e in row])
#                    for row in inputData])
# targetData = tuple([tuple([e for e in row])
#                    for row in targetData])

inputTraining = inputData
targetTraining = targetData

from nnet import NNet
from verbosePrint import vprint
import verbosePrint

# nn = NNet(sizes=[5, 3], bias=True)
nn = NNet(sizes=[5, 10, 4], bias=True)
# nn = NNet([[[-0.829638, 0.164111, 0.398885],
#             [-0.603684, -0.603331, -0.819179],
#             [-0.080592, -0.386044, -0.931615],
#             [0.762514, -0.142887, -0.737862],
#             [0.175430, 0.790112, -0.267367],
#             [-0.732674, -0.825474, 0.232357]]], bias=True)
            # ]])
nn.setActivations(['relu', 'linear'])
# nn.setVerbose([])

verbosePrint.vIteration = -1
verbosePrint.stage = ''

cycles = 100
report = max(1, cycles/10)

if cycles > 0:
    nn.checkup(inputData, targetData)
    for iteration in range(cycles + 1):
        vprint(iteration, '~~~~~~~~~~~ Iteration %d ~~~~~~~~~~~' % iteration)
        combinedError = 0
        for row_index in range(len(targetTraining)):
            datain = inputTraining[row_index:row_index + 1]
            goal_prediction = targetTraining[row_index:row_index + 1]
            prediction = nn.fire(datain)
            # print('Prediction:' + str(prediction))
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
