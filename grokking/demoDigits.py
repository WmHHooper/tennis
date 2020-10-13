import sys, numpy as np
import os
from numpy import nan
from nnet import NNet
from verbosePrint import vprint, vprintnn, demo, vIteration, stage
from keras.datasets import mnist

demo = -1
vIteration = -1
stage= 'a'

download = os.getcwd() + '/../grokking-data/mnist.npz'
(x_train, y_train), (x_test, y_test) = mnist.load_data(download)

images, labels = (x_train[0:1000].reshape(1000, 28 * 28) / 255, y_train[0:1000])

one_hot_labels = np.zeros((len(labels), 10))
for i, l in enumerate(labels):
    one_hot_labels[i][l] = 1
labels = one_hot_labels

test_images = x_test.reshape(len(x_test),28*28) / 255
test_labels = np.zeros((len(y_test),10))
for i,l in enumerate(y_test):
        test_labels[i][l] = 1

if demo == 81:
    np.random.seed(1)

    relu = lambda x: (x >= 0) * x  # returns x if x > 0, return 0 otherwise
    relu2deriv = lambda x: x >= 0  # returns 1 for input > 0, return 0 otherwise
    alpha = 0.005
    iterations = 350
    hidden_size, pixels_per_image, num_labels = (40, 784, 10)

    weights_0_1 = 0.2 * np.random.random((pixels_per_image, hidden_size)) - 0.1
    weights_1_2 = 0.2 * np.random.random((hidden_size, num_labels)) - 0.1

    # this produces the same weights, except for the last decimal digit
    # weights_0_1 = 2 * np.random.random((pixels_per_image, hidden_size)) - 1
    # weights_1_2 = 2 * np.random.random((hidden_size, num_labels)) - 1
    # scale = 0.1
    # weights_0_1 *= scale
    # weights_1_2 *= scale

    i = 0
    layer_0 = images[i:i + 1]
    layer_1 = relu(np.dot(layer_0, weights_0_1))
    layer_2 = np.dot(layer_1, weights_1_2)

    vprintnn(0)

    for j in range(iterations):
        error, correct_cnt = (0.0, 0)

        for i in range(len(images)):
            vprintnn(i, quit=True)

            layer_0 = images[i:i + 1]
            layer_1 = relu(np.dot(layer_0, weights_0_1))
            layer_2 = np.dot(layer_1, weights_1_2)

            error += np.sum((labels[i:i + 1] - layer_2) ** 2)
            correct_cnt += int(np.argmax(layer_2) == \
                               np.argmax(labels[i:i + 1]))

            layer_2_delta = (labels[i:i + 1] - layer_2)
            deriv = relu2deriv(layer_1)
            layer_1_delta = layer_2_delta.dot(weights_1_2.T) \
                            * deriv
            l1_dot_l2d = layer_1.T.dot(layer_2_delta)
            weights_1_2 += alpha * l1_dot_l2d
            l0_dot_l1d = layer_0.T.dot(layer_1_delta)
            vprint(i, 'L0 * L1d: %s' % l0_dot_l1d,
                   quit=True, suffix='a')
            weights_0_1 += alpha * l0_dot_l1d

            vprintnn(i, quit=True, suffix='b')

            sys.stdout.write("\r I:" + str(j) + \
                         " Train-Err:" + str(error / float(len(images)))[0:5] + \
                         " Train-Acc:" + str(correct_cnt / float(len(images))))

        if (j % 10 == 0 or j == iterations - 1):
            error, correct_cnt = (0.0, 0)

            for i in range(len(test_images)):
                layer_0 = test_images[i:i + 1]
                layer_1 = relu(np.dot(layer_0, weights_0_1))
                layer_2 = np.dot(layer_1, weights_1_2)

                error += np.sum((test_labels[i:i + 1] - layer_2) ** 2)
                correct_cnt += int(np.argmax(layer_2) == \
                                   np.argmax(test_labels[i:i + 1]))
            sys.stdout.write(" Test-Err:" + str(error / float(len(test_images)))[0:5] + \
                             " Test-Acc:" + str(correct_cnt / float(len(test_images))))
            print()

if demo == 82:
    np.random.seed(1)

    iterations = 351

    nn = NNet(sizes=[784, 40, 10])
    nn.setActivations(['brelu', 'linear'])
    nn.setAlpha(0.005)
    nn.scale(0.1)

    nn.fire(images[0:1])
    vprint(0, nn)

    for j in range(iterations):
        error, correct_cnt = (0.0, 0)
        for i in range(len(images)):
            vprint(i, nn, quit=True)
            prediction = nn.learn(images[i:i+1], labels[i:i+1])
            vprint(i, nn, suffix='b', quit=True)

            error += np.sum((labels[i:i+1] - prediction) ** 2)
            correct_cnt += int(np.argmax(prediction) == \
                   np.argmax(test_labels[i:i + 1]))

            sys.stdout.write("\r I:" + str(j) + \
                         " Train-Err:" + str(error / float(len(images)))[0:5] + \
                         " Train-Acc:" + str(correct_cnt / float(len(images))))

        if (j % 10 == 0):
            test_error = 0.0
            test_correct_cnt = 0

            for i in range(len(test_images)):
                tprediction = nn.fire(test_images[i:i + 1])

                test_error += np.sum((test_labels[i:i + 1] - tprediction) ** 2)
                test_correct_cnt += int(np.argmax(tprediction) == np.argmax(test_labels[i:i + 1]))

            sys.stdout.write(" Test-Err:" + str(test_error / float(len(test_images)))[0:5] + \
                             " Test-Acc:" + str(test_correct_cnt / float(len(test_images))))
            print()

if demo == 83:
    np.random.seed(1)

    def relu(x):
        return (x >= 0) * x  # returns x if x > 0
        # returns 0 otherwise

    def relu2deriv(output):
        return output >= 0  # returns 1 for input > 0

    alpha, iterations, hidden_size = (0.005, 300, 100)
    pixels_per_image, num_labels = (784, 10)

    weights_0_1 = 0.2 * np.random.random((pixels_per_image, hidden_size)) - 0.1
    weights_1_2 = 0.2 * np.random.random((hidden_size, num_labels)) - 0.1

    for j in range(iterations):
        error, correct_cnt = (0.0, 0)
        for i in range(len(images)):
            layer_0 = images[i:i + 1]
            layer_1 = relu(np.dot(layer_0, weights_0_1))
            dropout_mask = np.random.randint(2, size=layer_1.shape)
            layer_1 *= dropout_mask * 2
            layer_2 = np.dot(layer_1, weights_1_2)

            error += np.sum((labels[i:i + 1] - layer_2) ** 2)
            correct_cnt += int(np.argmax(layer_2) == np.argmax(labels[i:i + 1]))
            layer_2_delta = (labels[i:i + 1] - layer_2)
            layer_1_delta = layer_2_delta.dot(weights_1_2.T) * relu2deriv(layer_1)
            layer_1_delta *= dropout_mask

            weights_1_2 += alpha * layer_1.T.dot(layer_2_delta)
            weights_0_1 += alpha * layer_0.T.dot(layer_1_delta)

        if (j % 10 == 0):
            test_error = 0.0
            test_correct_cnt = 0

            for i in range(len(test_images)):
                layer_0 = test_images[i:i + 1]
                layer_1 = relu(np.dot(layer_0, weights_0_1))
                layer_2 = np.dot(layer_1, weights_1_2)

                test_error += np.sum((test_labels[i:i + 1] - layer_2) ** 2)
                test_correct_cnt += int(np.argmax(layer_2) == np.argmax(test_labels[i:i + 1]))

            sys.stdout.write("\n" + \
                             "I:" + str(j) + \
                             " Test-Err:" + str(test_error / float(len(test_images)))[0:5] + \
                             " Test-Acc:" + str(test_correct_cnt / float(len(test_images))) + \
                             " Train-Err:" + str(error / float(len(images)))[0:5] + \
                             " Train-Acc:" + str(correct_cnt / float(len(images))))

if demo == 84:
    np.random.seed(1)
    iterations = 300

    nn = NNet(sizes=[784, 100, 10])
    nn.setAlpha(0.005)
    nn.setActivations(['brelu', 'linear'])
    nn.scale(0.1)

    nn.fire(images[0:1])
    vprint(0, nn)

    for j in range(iterations):
        error, correct_cnt = (0.0, 0)
        for i in range(int(len(images))):
            vprint(i, nn, quit=True)
            masks = { 1:2 } # pass 1/2 the neurons from layer 1
            prediction = nn.learn(images[i:i+1], labels[i:i+1], masks)
            vprint(i, nn, suffix='b', quit=True)

            error += np.sum((labels[i:i + 1] - prediction) ** 2)
            correct_cnt += int(np.argmax(prediction) == np.argmax(labels[i:i + 1]))

            # sys.stdout.write("\r I:" + str(j) + \
            #              " Train-Err:" + str(error / float(len(images)))[0:5] + \
            #              " Train-Acc:" + str(correct_cnt / float(len(images))))

        if (j % 10 == 0):
            test_error = 0.0
            test_correct_cnt = 0

            for i in range(len(test_images)):
                tprediction = nn.fire(test_images[i:i + 1])

                test_error += np.sum((test_labels[i:i + 1] - tprediction) ** 2)
                test_correct_cnt += int(np.argmax(tprediction) == np.argmax(test_labels[i:i + 1]))

            # sys.stdout.write(" Test-Err:" + str(test_error / float(len(test_images)))[0:5] + \
            #                  " Test-Acc:" + str(test_correct_cnt / float(len(test_images))))
            # print()

            sys.stdout.write("\n" + \
                             "I:" + str(j) + \
                             " Test-Err:" + str(test_error / float(len(test_images)))[0:5] + \
                             " Test-Acc:" + str(test_correct_cnt / float(len(test_images))) + \
                             " Train-Err:" + str(error / float(len(images)))[0:5] + \
                             " Train-Acc:" + str(correct_cnt / float(len(images))))
if demo == 85:
    np.random.seed(1)

    def relu(x):
        return (x >= 0) * x  # returns x if x > 0

    def relu2deriv(output):
        return output >= 0  # returns 1 for input > 0

    batch_size = 100
    alpha, iterations = (0.001, 300)
    pixels_per_image, num_labels, hidden_size = (784, 10, 100)

    weights_0_1 = 0.2 * np.random.random((pixels_per_image, hidden_size)) - 0.1
    weights_1_2 = 0.2 * np.random.random((hidden_size, num_labels)) - 0.1

    for j in range(iterations):
        error, correct_cnt = (0.0, 0)
        for i in range(int(len(images) / batch_size)):
            batch_start, batch_end = ((i * batch_size), ((i + 1) * batch_size))

            layer_0 = images[batch_start:batch_end]
            layer_1 = relu(np.dot(layer_0, weights_0_1))
            dropout_mask = np.random.randint(2, size=layer_1.shape)
            vprint(i, dropout_mask, suffix='m', quit=True)
            layer_1 *= dropout_mask * 2
            layer_2 = np.dot(layer_1, weights_1_2)
            vprintnn(i, suffix='a', quit=True)

            error += np.sum((labels[batch_start:batch_end] - layer_2) ** 2)
            for k in range(batch_size):
                correct_cnt += int(np.argmax(layer_2[k:k + 1]) ==
                                   np.argmax(labels[batch_start + k:batch_start + k + 1]))

                layer_2_delta = (labels[batch_start:batch_end] - layer_2) / batch_size
                layer_1_delta = layer_2_delta.dot(weights_1_2.T) * relu2deriv(layer_1)
                layer_1_delta *= dropout_mask

                weights_1_2 += alpha * layer_1.T.dot(layer_2_delta)
                weights_0_1 += alpha * layer_0.T.dot(layer_1_delta)
                vprintnn(i, suffix='b', quit=True)

        if (j % 10 == 0):
            test_error = 0.0
            test_correct_cnt = 0

            for i in range(len(test_images)):
                layer_0 = test_images[i:i + 1]
                layer_1 = relu(np.dot(layer_0, weights_0_1))
                layer_2 = np.dot(layer_1, weights_1_2)

                test_error += np.sum((test_labels[i:i + 1] - layer_2) ** 2)
                test_correct_cnt += int(np.argmax(layer_2) == np.argmax(test_labels[i:i + 1]))

            sys.stdout.write("\n" + \
                             "I:" + str(j) + \
                             " Test-Err:" + str(test_error / float(len(test_images)))[0:5] + \
                             " Test-Acc:" + str(test_correct_cnt / float(len(test_images))) + \
                             " Train-Err:" + str(error / float(len(images)))[0:5] + \
                             " Train-Acc:" + str(correct_cnt / float(len(images))))

if demo == 86:
    np.random.seed(1)

    batch_size = 100
    iterations = 301

    nn = NNet(sizes=[784, 100, 10], batch_size=batch_size)
    nn.setActivations(['brelu', 'linear'])
    nn.setAlpha(0.001)
    nn.scale(0.1)

    for j in range(iterations):
        error, correct_cnt = (0.0, 0)
        for i in range(int(len(images) / batch_size)):
            batch_start, batch_end = ((i * batch_size), ((i + 1) * batch_size))
            masks = { 1:2 } # pass 1/2 the neurons from layer 1
            prediction = nn.fire(images[batch_start:batch_end], masks)
            vprint(i, nn, suffix='a', quit=True)
            vprint(i, nn.dropout_masks[1], suffix='m', quit=True)
            # nn.train(labels[batch_start:batch_end])
            # vprint(i, nn, stage='b', quit=True)

            error += np.sum((labels[batch_start:batch_end] - prediction) ** 2)
            for k in range(batch_size):
                correct_cnt += int(np.argmax(prediction[k:k + 1]) ==
                                   np.argmax(labels[batch_start + k:batch_start + k + 1]))

        if (j % 10 == 0):
            test_error = 0.0
            test_correct_cnt = 0

            for i in range(len(test_images)):
                tprediction = nn.fire(test_images[i:i + 1])

                test_error += np.sum((test_labels[i:i + 1] - tprediction) ** 2)
                test_correct_cnt += int(np.argmax(tprediction) == np.argmax(test_labels[i:i + 1]))

            sys.stdout.write("\n" + \
                             "I:" + str(j) + \
                             " Test-Err:" + str(test_error / float(len(test_images)))[0:5] + \
                             " Test-Acc:" + str(test_correct_cnt / float(len(test_images))) + \
                             " Train-Err:" + str(error / float(len(images)))[0:5] + \
                             " Train-Acc:" + str(correct_cnt / float(len(images))))

if demo == 91:
    np.random.seed(1)

    def tanh(x):
        return np.tanh(x)

    def tanh2deriv(output):
        return 1 - (output ** 2)

    def softmax(x):
        temp = np.exp(x)
        return temp / np.sum(temp, axis=1, keepdims=True)

    alpha, iterations, hidden_size = (2, 300, 100)
    pixels_per_image, num_labels = (784, 10)
    batch_size = 100

    weights_0_1 = 0.02 * np.random.random((pixels_per_image, hidden_size)) - 0.01
    weights_1_2 = 0.2 * np.random.random((hidden_size, num_labels)) - 0.1

    for j in range(iterations):
        correct_cnt = 0
        for i in range(int(len(images) / batch_size)):
            batch_start, batch_end = ((i * batch_size), ((i + 1) * batch_size))
            layer_0 = images[batch_start:batch_end]
            layer_1 = tanh(np.dot(layer_0, weights_0_1))
            dropout_mask = np.random.randint(2, size=layer_1.shape)
            vprint(i, dropout_mask, suffix='m', quit=True)
            layer_1 *= dropout_mask * 2
            layer_2 = softmax(np.dot(layer_1, weights_1_2))
            vprintnn(i, suffix='a', quit=True)

            for k in range(batch_size):
                correct_cnt += int(
                    np.argmax(layer_2[k:k + 1]) == np.argmax(labels[batch_start + k:batch_start + k + 1]))

            layer_2_delta = (labels[batch_start:batch_end] - layer_2) / (batch_size * layer_2.shape[0])
            layer_1_delta = layer_2_delta.dot(weights_1_2.T) * tanh2deriv(layer_1)
            layer_1_delta *= dropout_mask

            weights_1_2 += alpha * layer_1.T.dot(layer_2_delta)
            weights_0_1 += alpha * layer_0.T.dot(layer_1_delta)
            vprintnn(i, suffix='b', quit=True)

        test_correct_cnt = 0

        for i in range(len(test_images)):
            layer_0 = test_images[i:i + 1]
            layer_1 = tanh(np.dot(layer_0, weights_0_1))
            layer_2 = np.dot(layer_1, weights_1_2)

            test_correct_cnt += int(np.argmax(layer_2) == np.argmax(test_labels[i:i + 1]))
        if (j % 10 == 0):
            sys.stdout.write("\n" + \
                             "I:" + str(j) + \
                             " Test-Acc:" + str(test_correct_cnt / float(len(test_images))) + \
                             " Train-Acc:" + str(correct_cnt / float(len(images))))

if demo == 92:
    np.random.seed(1)

    batch_size = 100
    iterations = 3501

    nn = NNet(sizes=[784, 100, 10], batch_size=batch_size)
    nn.setActivations(['tanh', 'softmax'])
    nn.setAlpha(0.2)
    nn.scale(0.1)

    masks = {1: 2}  # pass 1/2 the neurons from layer 1
    # nn.fire(images[0:batch_size])
    # vprint(0, nn, quit=True)

    for j in range(iterations):
        correct_cnt = 0
        for i in range(int(len(images) / batch_size)):
            batch_start, batch_end = ((i * batch_size), ((i + 1) * batch_size))
            prediction = nn.fire(images[batch_start:batch_end], masks)
            vprint(i, nn, suffix='a', quit=True)
            vprint(i, nn.dropout_masks[1], suffix='m', quit=True)

            for k in range(batch_size):
                correct_cnt += int(
                    np.argmax(prediction[k:k + 1]) ==
                    np.argmax(labels[batch_start + k:batch_start + k + 1]))

            nn.train(labels[batch_start:batch_end])
            vprint(i, nn, suffix='b', quit=True)

        test_correct_cnt = 0

        for i in range(len(test_images)):
            prediction = nn.fire(test_images[i:i + 1])

            test_correct_cnt += int(np.argmax(prediction) == np.argmax(test_labels[i:i + 1]))
        if (j % 10 == 0):
            print("I:" + str(j) + \
                  " Test-Acc:" + str(test_correct_cnt / float(len(test_images))) + \
                  " Train-Acc:" + str(correct_cnt / float(len(images)))
                  )

if demo == 101:

    np.random.seed(1)


    def tanh(x):
        return np.tanh(x)


    def tanh2deriv(output):
        return 1 - (output ** 2)


    def softmax(x):
        temp = np.exp(x)
        return temp / np.sum(temp, axis=1, keepdims=True)


    alpha, iterations = (2, 300)
    pixels_per_image, num_labels = (784, 10)
    batch_size = 128

    input_rows = 28
    input_cols = 28

    kernel_rows = 3
    kernel_cols = 3
    num_kernels = 16

    hidden_size = ((input_rows - kernel_rows) *
                   (input_cols - kernel_cols)) * num_kernels

    # weights_0_1 = 0.02*np.random.random((pixels_per_image,hidden_size))-0.01
    kernels = 0.02 * np.random.random((kernel_rows * kernel_cols,
                                       num_kernels)) - 0.01

    weights_1_2 = 0.2 * np.random.random((hidden_size,
                                          num_labels)) - 0.1


    def get_image_section(layer, row_from, row_to, col_from, col_to):
        section = layer[:, row_from:row_to, col_from:col_to]
        return section.reshape(-1, 1, row_to - row_from, col_to - col_from)


    for j in range(iterations):
        correct_cnt = 0
        for i in range(int(len(images) / batch_size)):
            batch_start, batch_end = ((i * batch_size), ((i + 1) * batch_size))
            layer_0 = images[batch_start:batch_end]
            layer_0 = layer_0.reshape(layer_0.shape[0], 28, 28)
            # layer_0.shape

            sects = list()
            for row_start in range(layer_0.shape[1] - kernel_rows):
                for col_start in range(layer_0.shape[2] - kernel_cols):
                    sect = get_image_section(layer_0,
                                             row_start,
                                             row_start + kernel_rows,
                                             col_start,
                                             col_start + kernel_cols)
                    sects.append(sect)

            expanded_input = np.concatenate(sects, axis=1)
            es = expanded_input.shape
            flattened_input = expanded_input.reshape(es[0] * es[1], -1)

            kernel_output = flattened_input.dot(kernels)
            layer_1 = tanh(kernel_output.reshape(es[0], -1))
            dropout_mask = np.random.randint(2, size=layer_1.shape)
            layer_1 *= dropout_mask * 2
            layer_2 = softmax(np.dot(layer_1, weights_1_2))

            for k in range(batch_size):
                labelset = labels[batch_start + k:batch_start + k + 1]
                _inc = int(np.argmax(layer_2[k:k + 1]) ==
                           np.argmax(labelset))
                correct_cnt += _inc

            layer_2_delta = (labels[batch_start:batch_end] - layer_2) \
                            / (batch_size * layer_2.shape[0])
            layer_1_delta = layer_2_delta.dot(weights_1_2.T) * \
                            tanh2deriv(layer_1)
            layer_1_delta *= dropout_mask
            weights_1_2 += alpha * layer_1.T.dot(layer_2_delta)
            l1d_reshape = layer_1_delta.reshape(kernel_output.shape)
            k_update = flattened_input.T.dot(l1d_reshape)
            kernels -= alpha * k_update

        test_correct_cnt = 0

        for i in range(len(test_images)):

            layer_0 = test_images[i:i + 1]
            #         layer_1 = tanh(np.dot(layer_0,weights_0_1))
            layer_0 = layer_0.reshape(layer_0.shape[0], 28, 28)
            layer_0.shape

            sects = list()
            for row_start in range(layer_0.shape[1] - kernel_rows):
                for col_start in range(layer_0.shape[2] - kernel_cols):
                    sect = get_image_section(layer_0,
                                             row_start,
                                             row_start + kernel_rows,
                                             col_start,
                                             col_start + kernel_cols)
                    sects.append(sect)

            expanded_input = np.concatenate(sects, axis=1)
            es = expanded_input.shape
            flattened_input = expanded_input.reshape(es[0] * es[1], -1)

            kernel_output = flattened_input.dot(kernels)
            layer_1 = tanh(kernel_output.reshape(es[0], -1))
            layer_2 = np.dot(layer_1, weights_1_2)

            test_correct_cnt += int(np.argmax(layer_2) ==
                                    np.argmax(test_labels[i:i + 1]))
        if (j % 1 == 0):
            sys.stdout.write("\n" + \
                             "I:" + str(j) + \
                             " Test-Acc:" + str(test_correct_cnt / float(len(test_images))) + \
                             " Train-Acc:" + str(correct_cnt / float(len(images))))