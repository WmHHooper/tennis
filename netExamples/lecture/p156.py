# NNet rendering of the neural net in Grokking, p. 156
import numpy as np
import sys
from netExamples.grokking.mnist import \
    images, labels, test_images, test_labels
from nnet import NNet

np.random.seed(1)
iterations = 300

nn = NNet(sizes=[784, 100, 10])
nn.setAlpha(0.005)
nn.setActivations(['relu', 'linear'])
nn.setMaskPr({1: 2})
nn.scale(0.1)

nn.fire(images[0:1])
nn.checkup(images[0:1], labels[0:1])
# vprint(0, nn)

for j in range(iterations):
    error, correct_cnt = (0.0, 0)
    for i in range(int(len(images))):
        # vprint(i, nn, quit=True)
        prediction = nn.learn(images[i:i + 1], labels[i:i + 1])
        # vprint(i, nn, suffix='b', quit=True)

        error += np.sum((labels[i:i + 1] - prediction) ** 2)
        correct_cnt += int(np.argmax(prediction)
                           == np.argmax(labels[i:i + 1]))

    if (j % 10 == 0):
        test_error = 0.0
        test_correct_cnt = 0

        for i in range(len(test_images)):
            tprediction = nn.fire(test_images[i:i + 1])

            test_error += np.sum((test_labels[i:i + 1] - tprediction) ** 2)
            test_correct_cnt += int(np.argmax(tprediction)
                                    == np.argmax(test_labels[i:i + 1]))

        sys.stdout.write("\n" + "I:" + str(j)
            + " Test-Err:" + str(test_error / float(len(test_images)))[0:5]
            + " Test-Acc:" + str(test_correct_cnt / float(len(test_images)))
            + " Train-Err:" + str(error / float(len(images)))[0:5]
            + " Train-Acc:" + str(correct_cnt / float(len(images)))
            )