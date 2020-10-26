from nnet import NNet
from verbosePrint import vprint
import verbosePrint

from netExamples.lecture.p3of5 import inputData
from netExamples.lecture.p3of5 import inputTraining
# from lecture.p3of5 import inputData as inputTraining
from netExamples.lecture.p3of5 import exactly as targetData
from netExamples.lecture.p3of5 import exactTraining as targetTraining
# from lecture.p3of5 import exactly as targetTraining

# nn = NNet(sizes=[6, 8, 2])
nn = NNet([
# weights 0-1 pre-trained on training data by atLeast3of5
# [[ 0.33895931,  0.3254088 ,  0.23082558],
#  [ 0.31869515,  0.37907859,  0.15820846],
#  [ 0.41052567,  0.34751307,  0.04301862],
#  [ 0.17284007,  0.52017019,  0.21712528],
#  [ 0.2754157 ,  0.386195  ,  0.13886435],
#  [ 0.14321084, -0.35544313, -0.43080433]],

# weights 0-1 pre-trained on all data by atLeast3of5
[[ 0.12585974,  0.17653591,  0.30195952],
 [ 0.13780732,  0.27187273,  0.33932816],
 [ 0.15811428,  0.31700224,  0.33515429],
 [ 0.17158333,  0.34341205,  0.32706608],
 [ 0.18033154,  0.35940584,  0.31899871],
 [ 0.32277643, -0.28375181, -0.53594695]],

# weights 1-2 are random
[[0.039424, -0.008154],
 [-0.537812, 0.364380],
 [-0.872521, 0.468537]],

], bias=[True, False])
# ])
nn.setActivations(['sigmoid', 'sigmoid'])

nn.checkup(inputData, targetData)

verbosePrint.vIteration = -1
verbosePrint.stage = ''

cycles = 400
report = cycles/10

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
