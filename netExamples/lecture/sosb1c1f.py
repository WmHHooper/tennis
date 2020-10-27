
from netExamples.lecture.sosb \
    import inputData, inputTraining, targetData, targetTraining
from cmatrix import ConvolutionMatrix

from nnet import NNet
from verbosePrint import vprint
import verbosePrint

nn = NNet(sizes=[12, 40, 4], bias=True)
cm = ConvolutionMatrix(rows=3, cols=4, bias=True, shapes=((1,12),(1,3)))
nn.replaceLayer(0, cm)
nn.setActivations(['softMax', 'sigmoid'])

verbosePrint.vIteration = -1
verbosePrint.stage = ''

cycles = 100
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

