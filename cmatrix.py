import sys
import numpy as np
import weightmatrix

class ConvolutionMatrix(weightmatrix.WeightMatrix):
    weights = False   # A 2-dimensional array of coefficients.
    alpha = 0.1
    verbose = []
    bias = False

    def __init__(self,
                 initialWeights=[],
                 rows=0, cols=0,
                 bias=False,
                 shapes=((1,1),(1,1)),
                 ):
        '''
        Create a matrix of weights, connecting
        a subset of the input layer with
        a subset of the output layer.
        :param initialWeights: if you want exact weights,
                specify one 2-D grid for each pair of layers
        :param rows, cols: alternately, specify the layer sizes
                and weights will be generated randomly
        :param bias: True iff you want one column to be a bias.
        If initialWeights are specified, the last column of weights
        is expected to be the bias.
        If rows and cols are specified, the new matrix will have
        (cols + 1) columns.
        :param shapes: a nested list.  The first element describes
        the shape of the input layer.  The second element describes
        the shape of each window.
        '''
        super().__init__(initialWeights, rows, cols, bias)
        kernel_rows, kernel_cols = shapes[1]
        assert kernel_rows * kernel_cols == super().inputLength()
        self.shapes = shapes

    def inputLength(self):
        input_rows, input_cols = self.shapes[0]
        return input_rows * input_cols

    def outputLength(self):
        oRows, oCols = self.outputShape()
        return oRows * oCols

    def outputShape(self):
        input_rows, input_cols = self.shapes[0]
        kernel_rows, kernel_cols = self.shapes[1]
        rLimit = input_rows - kernel_rows + 1
        cLimit = input_cols - kernel_cols + 1
        return rLimit * cLimit, super().outputLength()


    def fire(self, datain):
        '''
        Fire the neural network.
        :param datain: the input layer, one row of values for each
        member of the input batch.
        :return: the next layer of the grid.
        '''
        input_rows, input_cols = self.shapes[0]
        batch_size = len(datain)
        assert self.inputLength() == len(datain[0])
        dataGrid = datain.reshape(batch_size, input_rows, input_cols)

        kernel_rows, kernel_cols = self.shapes[1]
        rLimit = input_rows - kernel_rows + 1
        cLimit = input_cols - kernel_cols + 1

        sects = []
        for rs in range(rLimit):
            rf = rs + kernel_rows
            for cs in range(cLimit):
                cf = cs + kernel_cols
                sect = dataGrid[:, rs:rf, cs:cf]
                sects.append(sect)

        expanded_input = np.concatenate(sects, axis=1)
        es = expanded_input.shape
        flattened_input = expanded_input.reshape(es[0] * es[1], -1)
        if self.bias:
            biasColumn = np.ones((len(flattened_input), 1))
            flattened_input = np.block([flattened_input, biasColumn])
        self.flattened_input = flattened_input
        output = self.flattened_input.dot(self.weights)
        output = self.actFunction(output)
        output = output.reshape(batch_size, -1)
        # output *= self.masks[0]
        return output

    def train(self
          , datain
          , prediction
          , goal
          ):
        batch_size = len(prediction)
        assert batch_size == len(goal)
        oDelta = (goal - prediction) / batch_size
        oShape = self.outputShape()
        odReshape = oDelta.reshape(*oShape)
        # This is not right, but we don't need this until
        # we stack convolutional layers
        # iDelta = odReshape.dot(self.weights.T)
        # iDelta = iDelta.reshape(batch_size, self.inputLength())
        iDelta = np.ones((batch_size, self.inputLength()))
        iDelta *= np.nan
        derivs = self.derFunction(datain)
        iDelta *= derivs

        wDelta = self.flattened_input.T.dot(odReshape)
        wDelta *= self.alpha
        self.weights += wDelta
        return iDelta