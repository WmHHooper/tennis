# NNet rendering of the neural net in Grokking, p. 159
import numpy as np
import sys
from netExamples.grokking.mnist import \
    images, labels, test_images, test_labels
from nnet import NNet

np.random.seed(1)

batch_size = 100
iterations = 301

nn = NNet(sizes=[784, 100, 10], batch_size=batch_size)
nn.setActivations(['brelu', 'linear'])
nn.setMaskPr({1: 2})
nn.setAlpha(0.001)
nn.scale(0.1)

for j in range(iterations):
    error, correct_cnt = (0.0, 0)
    for i in range(int(len(images) / batch_size)):
        batch_start, batch_end = ((i * batch_size), ((i + 1) * batch_size))
        prediction = nn.learn(images[batch_start:batch_end],
                              labels[batch_start:batch_end])
        # vprint(i, nn, suffix='a', quit=True)
        # vprint(i, nn.dropout_masks[1], suffix='m', quit=True)
        # nn.train(labels[batch_start:batch_end])
        # vprint(i, nn, stage='b', quit=True)

        error += np.sum((labels[batch_start:batch_end] - prediction) ** 2)
        for k in range(batch_size):
            correct_cnt += int(
                np.argmax(prediction[k:k + 1])
                == np.argmax(labels[batch_start + k:batch_start + k + 1]))

    if (j % 10 == 0):
        test_error = 0.0
        test_correct_cnt = 0

        for i in range(len(test_images)):
            tprediction = nn.fire(test_images[i:i + 1])

            test_error += np.sum((test_labels[i:i + 1] - tprediction) ** 2)
            test_correct_cnt += int(np.argmax(tprediction)
                                    == np.argmax(test_labels[i:i + 1]))

        sys.stdout.write(
            "\n"
            + "I:" + str(j)
            + " Test-Err:" + str(test_error / float(len(test_images)))[0:5]
            + " Test-Acc:" + str(test_correct_cnt / float(len(test_images)))
            + " Train-Err:" + str(error / float(len(images)))[0:5]
            + " Train-Acc:" + str(correct_cnt / float(len(images)))
            )
