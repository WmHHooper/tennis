import sys, numpy as np
from activation import getAct, getDer
from mask import nomask, nomasks # , mask
from pprint import pformat
from numbers import Number

allVerbose = ['activation', 'alpha', 'odelta', 'idelta', 'wdelta']
def inVerbose(s):
    return s.lower() in allVerbose

# def deepArray(weights):
#     if isinstance(weights, (np.ndarray, Number)):
#         return weights
#     return np.array([deepArray(w) for w in weights])

class WeightMatrix:
    weights = False   # A 2-dimensional array of coefficients.
    alpha = 0.1
    verbose = []
    bias = False

    def __init__(self,
                 initialWeights=[],
                 rows=0, cols=0,
                 bias=False,
                 ):
        """
        Create a matrix of weights, fully connecting an input layer
        with an output layer.
        :param initialWeights: if you want exact weights,
                specify one 2-D grid for each pair of layers
        :param rows, cols: alternately, specify the layer sizes
                and weights will be generated randomly
        :param bias: True iff you want one column to be a bias.
        If initialWeights are specified, the last column of weights
        is expected to be the bias.
        If rows and cols are specified, the new matrix will have
        (cols + 1) columns.
        """
        self.bias = bias
        if len(initialWeights) > 0:
            # self.weights = deepArray(initialWeights)
            if isinstance(initialWeights, np.ndarray):
                self.weights = initialWeights
            else:
                self.weights = np.array(initialWeights)
            rows = len(self.weights)
        else:
            self.weights = 2 * np.random.rand(rows + bias, cols) - 1
        self.setActivation('linear', 'linear')

    def __str__(self):
        s = ''
        if self.verbose:
            s += 'activation: %s\n' % self.actFunction.__name__
            s += 'derivative: %s\n' % self.derFunction.__name__
            s += 'Alpha:      %f\n' % self.alpha
        # https://docs.python.org/3/library/pprint.html
        astr = pformat(self.weights).split('\n')
        # remove 'array(' and ')'
        lstr = [row[6:] for row in astr]
        s += '\n'.join(lstr)[:-1]
        s += ','
        s += '\n'
        return s

    def scale(self, scaleFactor):
        self.weights *= scaleFactor

    def getWeights(self):
        return self.weights

    def setVerbose(self, t):
        if t == True:
            self.verbose = allVerbose
        else:
            self.verbose = t

    def setAlpha(self, a):
        self.alpha = a

    def getAlpha(self):
        return self.alpha

    def inputLength(self):
        rows = len(self.weights)
        if self.bias:
            rows -= 1
        return rows

    def outputLength(self):
        cols = len(self.weights[0])
        return cols

    def setActivation(self, derLabel, actLabel):
        """
        Set the functions for activation and back propagation.
        :param derLabel: The derivative for back-propagation
        to the previous layer
        :param actLabel: The activation to apply
        to the weighted sum
        :return:
        """
        self.derFunction = getDer(derLabel)
        self.actFunction = getAct(actLabel)

    def fire(self, datain
             # , pR=nomask
             ):
        '''
        Fire the neural network.
        :param datain: the input layer, one row of values for each
        member of the input batch.
        :return: the next layer of the grid.
        '''
        # if not isinstance(datain, np.ndarray):
        #     datain = np.array(datain)
        assert isinstance(datain, np.ndarray)
        lengthin = len(datain[0])
        assert (lengthin + self.bias) \
               == len(self.weights)
        if self.bias:
            biasColumn = np.ones((len(datain),1))
            datain = np.block([datain, biasColumn])
        sums = np.dot(datain, self.weights)
        output = self.actFunction(sums)
        return output

    def train(self
              , datain
              , prediction
              , goal
              # , imask=nomask
              ):
        """
        One iteration of the learning algorithm
        described in Trask, 227.
        :param datain: one dimensional array of input data
        :param goal: one dimensional array of data tags
        :param masks: a pair of masks, mask[0] for the input layer,
            and mask[1] for the output layer.
        :param imask: an input mask.  Each value in the mask
            is multiplied by the corresponding delta for the input.
        :return: amount to change input.
        Used for back-propagation.
        """
        assert isinstance(datain, np.ndarray)
        assert isinstance(prediction, np.ndarray)
        assert isinstance(goal, np.ndarray)

        batch_size = len(prediction)
        assert batch_size == len(goal)
        oDelta = (goal - prediction) / batch_size
        # if inVerbose('oDelta'):
        #     print("oDelta: ", oDelta)

        # if not isinstance(datain,np.ndarray):
        #     datain = np.array(datain)

        try:
            error = np.sum(oDelta ** 2)
        except:
            print('oDelta * 2 fails:', oDelta)
            sys.exit()

        derivs = self.derFunction(datain)
        iDelta = oDelta.dot(self.weights.T)
        # if inVerbose('iDelta'):
        #     print("iDelta: ", iDelta)
        if self.bias:
            iDelta = np.delete(iDelta, -1, 1)
            biasColumn = np.ones((len(datain),1))
            datain = np.block([datain, biasColumn])
        iDelta *= derivs
        # if isinstance(imask, np.ndarray):
        #     sm = imask.shape
        #     sd = iDelta.shape
        #     assert sm == sd
        #     iDelta *= imask
        wDelta = datain.T.dot(oDelta)
        # print("LI * LOD:", dotIO)
        wDelta *= self.alpha
        # if inVerbose('wDelta'):
        #     print("wDelta: ", wDelta)
        self.weights += wDelta
        # print("wts:", self.weights)
        return iDelta
