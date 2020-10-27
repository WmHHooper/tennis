from nnet import NNet
from verbosePrint import vprint
import verbosePrint

from netExamples.lecture.p3of5 import inputData
from netExamples.lecture.p3of5 import inputData as inputTraining
# from lecture.p3of5 import inputTraining
from netExamples.lecture.p3of5 import exactly as targetData
from netExamples.lecture.p3of5 import exactly as targetTraining
# from lecture.p3of5 import exactTraining as targetTraining

# nn = NNet(sizes=[6, 8, 2])
nn = NNet([
[[-0.809690, 0.529351, 0.130375, -0.668283, 0.607374, -0.560586, -0.631622, 0.770885],
 [0.772290, 0.701108, -0.449341, -0.654130, -0.313803, -0.156230, -0.912943, -0.920589],
 [0.285597, 0.646907, 0.003674, 0.279674, -0.764945, 0.966331, -0.000227, -0.537906],
 [0.966027, -0.940834, 0.268773, -0.683670, -0.772791, 0.159225, -0.624018, -0.483886],
 [0.418490, 0.493562, 0.056094, -0.845281, 0.681007, 0.868586, 0.402037, 0.880762],
 [0.270917, -0.058948, 0.545103, 0.637797, 0.012415, -0.676826, 0.995314, 0.577275]],
[[0.651401, 0.099612],
 [-0.519044, 0.414466],
 [0.030243, 0.950533],
 [0.594134, -0.812050],
 [-0.468415, -0.555548],
 [0.934102, -0.536747],
 [-0.005624, -0.327394],
 [-0.506726, -0.665108]],
], bias=[True, False])
# ])

nn.setActivations(['relu', 'linear'])

nn.checkup(inputData, targetData)

verbosePrint.vIteration = -1
verbosePrint.stage = ''

cycles = 2000
report = cycles/10
batch_size = 4

for iteration in range(cycles + 1):
    vprint(iteration, '~~~~~~~~~~~ Iteration %d ~~~~~~~~~~~' % iteration)
    combinedError = 0
    batch_limit = int(len(targetTraining) / batch_size)
    for batch_index in range(batch_limit):
        row_index = batch_index * batch_size
        batch = row_index + batch_size
        datain = inputTraining[row_index:batch]
        goal_prediction = targetTraining[row_index:batch]
        prediction = nn.fire(datain)
        # print('Prediction:' + str(prediction))
        vprint(iteration, nn)

        error = (goal_prediction - prediction) ** 2
        combinedError += sum(error)

        nn.learn(datain, goal_prediction)

    if iteration % report == 0:
        print('Iteration: %d Error: %s' % (iteration, str(combinedError)))
    vprint(iteration, '')
    vprint(iteration, '~~~~~~~~~~~~~~~ End ~~~~~~~~~~~~~~~~')
    vprint(iteration, nn, quit=True)

print()
nn.checkup(inputData, targetData)
