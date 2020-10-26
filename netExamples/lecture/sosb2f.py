
from netExamples.lecture.sosb \
    import inputData, inputTraining, targetData, targetTraining

from nnet import NNet
from verbosePrint import vprint
import verbosePrint

nn = NNet(sizes=[12, 22, 4], bias=True)
nn.setActivations(['tanh', 'sigmoid'])

verbosePrint.vIteration = -1
verbosePrint.stage = ''

cycles = 20
report = max(1, cycles/10)
checkupParams = (inputData, targetData, inputTraining, 25)

if cycles > 0:
    nn.checkup(*checkupParams)
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
nn.checkup(*checkupParams)

