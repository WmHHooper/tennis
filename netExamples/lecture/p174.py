# NNet rendering of the neural net in Grokking, p. 174
import numpy as np
import sys
from netExamples.grokking.mnist import \
    images, labels, test_images, test_labels
from nnet import NNet

np.random.seed(1)

batch_size = 100
iterations = 301

nn = NNet(sizes=[784, 100, 10], batch_size=batch_size)
nn.setActivations(['tanh', 'softmax'])
nn.setMaskPr({1: 2})
nn.setAlpha(0.2)
nn.scale(0.1)

# vprint(0, nn, quit=True)

for j in range(iterations):
    correct_cnt = 0
    for i in range(int(len(images) / batch_size)):
        batch_start, batch_end = ((i * batch_size), ((i + 1) * batch_size))
        prediction = nn.learn(images[batch_start:batch_end],
                              labels[batch_start:batch_end])
        # vprint(i, nn, suffix='a', quit=True)
        # vprint(i, nn.dropout_masks[1], suffix='m', quit=True)

        for k in range(batch_size):
            correct_cnt += int(
                np.argmax(prediction[k:k + 1]) ==
                np.argmax(labels[batch_start + k:batch_start + k + 1]))

        # nn.train(labels[batch_start:batch_end])
        # vprint(i, nn, suffix='b', quit=True)

    test_correct_cnt = 0

    for i in range(len(test_images)):
        prediction = nn.fire(test_images[i:i + 1])

        test_correct_cnt += int(np.argmax(prediction)
                                == np.argmax(test_labels[i:i + 1]))
    if (j % 10 == 0):
        print("I:" + str(j) + \
              " Test-Acc:" + str(test_correct_cnt / float(len(test_images))) + \
              " Train-Acc:" + str(correct_cnt / float(len(images)))
              )
