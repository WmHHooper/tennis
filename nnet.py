import sys, math
import numpy as np
from numpy import nan
from weightmatrix import WeightMatrix
from mask import nomask, nomasks, Mask

class NNet:
    layers = []
    grids = []
    verbose = []
    maskPr = {}

    def __init__(self, weights=False, layerList=False, sizes=False,
                 bias=False, batch_size=1):
        '''
        Create a multi-layered neural network.
        :param weights: A list of weights between each layer.
        The weights are
        arranged from input to output,
        e.g.: [[weights 0-1],[weights 1-2],...,[weights 8-9]]
        Each weight is an N x M grid, where N is the input size and
        M is the output size.
        :param layerList: A list of layer values.
        The layers are arranged from input to output,
        e.g.: [[layer0],[layer1],...,[layer9]]
        If these are unspecified, layers with an appropriate number of
        columns will be created and filled with NaN's.
        :param sizes: An alternate way of specifying weights.  A list
        of integers, e.g. [9, 8, 7, ..., 3, 2] is used to create NumPy
        arrays
        [[9 x 8], [8 x 7], ..., [3 x 2]] with random coefficients
        in the range [-1..1], and layers to match.
        :param batch_size: The number of rows in each layer.
        '''
        if isinstance(weights, (list, tuple)):
            weights = np.array(weights)
        if isinstance(weights, np.ndarray):
            m = len(weights)
            n = m + 1
            self.grids = [None] * m
            self.layers = [None] * n
            if isinstance(bias, (tuple, list)):
                pass
            else:
                bias = [bias] * m
            for i in range(m):
                w = weights[i]
                lw = WeightMatrix(w, bias=bias[i])
                self.grids[i] = lw
            if isinstance(layerList, (list, np.ndarray)):
                assert n == len(layerList)
                for i in range(n):
                    self.layers[i] = layerList[i]
            else:
                for i in range(m):
                    rows = len(w)
                    cols = len(w[0])
                    self.layers[i] = np.array([[nan] * rows] * batch_size)
                self.layers[m] = np.array([[nan] * cols] * batch_size)
        else:
            n = len(sizes)
            m = n - 1
            assert n >= 2
            self.grids = [None] * m
            self.layers = [None] * n
            if isinstance(bias, (tuple, list)):
                pass
            else:
                bias = [bias] * m
            for i in range(m):
                self.grids[i] = WeightMatrix(
                    rows=sizes[i], cols=sizes[i + 1], bias=bias[i])
            for i in range(n):
                self.layers[i] = np.array([[nan] * sizes[i]] * batch_size)
        assert len(self.layers) == (len(self.grids) + 1)

    def __str__(self):
        n = len(self.layers)
        s = ''
        for i in range(n - 1, -1, -1):
            if i < (n - 1):
                s += 'Weights %d-%d:\n' % (i, i+1)
                s += str(self.grids[i])
            s += 'Layer %d:\n' % i
            s += np.array2string(self.layers[i], threshold=1e5)
            s += '\n'
        return s

    def printnn(self, fp=sys.stdout):
        s = str(self)
        n = len(s)
        chunkSize = 100000
        chunks = math.ceil(n / chunkSize)
        for i in range(chunks):
            first = i * chunkSize
            last = min((i+1) * chunkSize, n)
            substring = s[first:last]
            fp.write(substring)
        fp.write('\n')

    def indices(self, index):
        '''
        :param index: indicates which layers are to be selected.
        If the index is a single number, that layer is the only one selected.
        If the index is a list, (e.g. [0,2]), the corresponding layers
        (e.g., grids[0] and grids[2] are scaled.
        :return: A list of indices, by default [0, ... , # of layers - 1]
        '''
        if isinstance(index, (tuple, list)):
            return index
        if isinstance(index, int):
            return [index]
        # if index == 'All':
        m = len(self.grids)
        return range(m)

    def scale(self, s, index='All'):
        '''
        Set the range of positive and negative weight values in a matrix.
        :param scaleFactor: all current weights in the layer are multiplied
        by this number
        :param index: indicates which layers are to be scaled.
        See indices() for more info.
        '''
        for i in self.indices(index):
            self.grids[i].scale(s)

    def setVerbose(self, v, index='All'):
        '''
        :param v: indicates which parts of the layer will be
        printed when the layer fires, trains or str(layer) is called.
        :param index: indicates which layers are to be set.
        See indices() for more info.
        '''
        for i in self.indices(index):
            self.grids[i].setVerbose(v)

    def setAlpha(self, alpha, index='ALL'):
        '''
        Set alpha value of one or more weight matrices.
        :param alpha: The mount by which weights that contribute
        to erroneus results will be adjusted.  The default is 0.1.
        :param index: indicates which layers are to be set.
        See indices() for more info.
        '''
        for i in self.indices(index):
            self.grids[i].setAlpha(alpha)

    def setActivations(self, lList):
        '''
        :param lList: A list of the activation functions for each
        layer.
        '''
        n = len(lList)
        assert n == len(self.grids)
        lList.insert(0, 'linear')
        for i in range(n):
            self.grids[i].setActivation(lList[i], lList[i + 1])

    def setMaskPr(self, maskPr):
        '''
        Set masking probabilities.
        :param maskPr: a map of the form { i:pR, ... },
            where i is the layer and pR is the retention parameter
            as described in masks.py .  These probabilities are used
            to generate a new random mask each time the network
            trains on a different input vector
        '''
        self.maskPr = maskPr

    def replaceLayer(self, i, wm):
        '''
        Replace one weight matrix  in a neural network with
        a custom weight matrix.
        :param i: the index of the grid to replace.
        :param wm: a WeightMatrix that
        on firing, transforms input layer[i] to output layer[i+1]
        on training, transforms output delta[i+1] to input delta[i]
        '''

        llen = len(self.layers[i][0])
        wlen = wm.inputLength()
        assert llen == wlen
        assert len(self.layers[i+1][0]) == wm.outputLength()
        self.grids[i] = wm

    def fire(self, layer0, masks={}):
        '''
        Fire the multiple layers of a neural network.
        :param layer0: the input layer.
        :param masks: a map of the form { i:M, ... }
        where i is the layer index and M is a Mask object,
        as defined in mask.py
        :return: the final (output) layer of the network
        '''
        if isinstance(layer0, np.ndarray):
            self.layers[0] = layer0
        else:
            self.layers[0] = np.array(layer0)

        m = len(self.grids)
        n = m + 1

        for i in range(m):
            j = i + 1
            self.layers[j] = self.grids[i].fire(self.layers[i])
            if j in masks:
                omask = masks[j].omask()
                self.layers[j] *= omask
        return self.layers[m]

    def train(self, goal, masks={}):
        '''
        Adjust weights to converge on goal
        :param goal: the desired output of the final layer
        :param masks: a map of the form { i:M, ... }
        where i is the layer index and M is a Mask object,
        as defined in mask.py
        '''
        if isinstance(goal, np.ndarray):
            pass # expected value
        else:
            goal = np.array(goal)

        m = len(self.grids)
        n = m + 1
        goals = [None] * m
        goals.append(goal)
        for j in range(m, 0, -1):
            i = j - 1
            iDelta = self.grids[i].train(
                self.layers[i], self.layers[j], goals[j])
            if i in masks:
                imask = masks[i].imask()
                iDelta *= imask
            goals[i] = self.layers[i] + iDelta

    def learn(self, layer0, goal):
        '''
        Train the network on a single batch.
        :param layer0: the input batch
        :param goal: the desired output of the final layer
        '''
        masks = {}
        for i in self.maskPr:
            layer = self.layers[i]
            pR = self.maskPr[i]
            maskI = Mask(layer, pR)
            masks[i] = maskI

        self.fire(layer0, masks)
        self.train(goal, masks)
        return self.layers[-1]

    def checkup(self,
                inputData,
                targetData,
                trainingData=None,
                maxPrint = 100):
        if trainingData == None:
            trainingData = inputData
        print('Structure:')
        self.printnn()
        total_count = (len(targetData) * len(targetData[0]))
        training_error_count = 0
        training_count = len(trainingData) * len(targetData[0])
        testing_error_count = 0
        testing_count = total_count - training_count

        printCount = 0
        print('Performance:')
        for row_index in range(len(targetData)):
            datain = inputData[row_index:row_index + 1]
            goal_prediction = targetData[row_index:row_index + 1]
            # https://stackoverflow.com/a/10873843/2619926
            rgoal = np.rint(goal_prediction).astype(int)
            rgoal = [[max(0,(min(1, x))) for x in row]
                     for row in rgoal]
            prediction = self.fire(datain)
            rpred = np.rint(prediction).astype(int)
            rpred = [[max(0,(min(1, x))) for x in row]
                     for row in rpred]
            row_error = np.sum(np.abs(np.array(rgoal) - rpred))
            if datain[0] in trainingData:
                training_error_count += row_error
                marker = 'x '
            else:
                testing_error_count += row_error
                marker = '* '

            if row_index < maxPrint:
                print('Goal(r):', rgoal,
                      'Prediction(r):', rpred,
                      marker if row_error > 0 else '  ',
                      '(actual G:', goal_prediction,
                      'P:', prediction, ')'
                      'Input:', datain,
                )
            elif row_index == maxPrint:
                print('...')
        print('Summary - training: %d errors of %d, %.2f%%' %
              (training_error_count, training_count,
               100 * training_error_count/training_count),
              '- testing: %d errors of %d, %.2f%%' %
              (testing_error_count, testing_count,
               100 * testing_error_count/testing_count)
              if testing_count > 0 else '')
        print()