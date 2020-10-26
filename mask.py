import numpy as np
from numpy import nan

nomask = 1

class Mask:
    input_coefficients = nomask
    output_coefficients = nomask

    def __init__(self, layer, pR=2):
        '''
        Create a pair of input/output masks, to match a layer architecture.
        :param layer: an output layer from a set of neurons, e.g.
            or a tuple describing the shape of the layer,
            or simply a int representing the shape.
        :param pR:
            The fraction of neurons retained in the mask, 0 < pR < 1,
            or the integer reciprocal of that value, e.g. 3 converts to 0.333...
        '''
        if isinstance(layer, tuple):
            shape = layer
        elif isinstance(layer, np.ndarray):
            shape = layer.shape
        elif layer == 0:
            return
        elif isinstance(layer, int):
            shape = (1, layer)
        else:
            raise Exception('Layer must be a Numpy array, a tuple, or an int.')

        assert pR > 0
        if pR < 1:
            floats = np.random.rand(*shape)
            self.input_coefficients = (floats > pR) * 1.0
            self.output_coefficients = self.input_coefficients / pR
        else:
            assert isinstance(pR, int)
            cutoff = pR - 2
            ints = np.random.randint(pR, size=shape)
            self.input_coefficients = (ints > cutoff) * 1
            self.output_coefficients = self.input_coefficients * pR

    def __str__(self):
        s = ''
        # s += 'iMask:\n'
        s += np.array2string(self.input_coefficients, threshold=1e5)
        # s += '\noMask:\n'
        # s += np.array2string(self.output_coefficients, threshold=1e5)
        return s

    def imask(self):
        return self.input_coefficients

    def omask(self):
        return self.output_coefficients

nomasks = Mask(0)
assert nomask == nomasks.imask()
assert nomask == nomasks.omask()

testing = False
# testing = True
if testing:
    # layer_1 = np.array([[0] * 30] * 2)
    # layer_1 = (3, 10)
    layer_1 = 20
    np.random.seed(1)
    mask1 = Mask(layer_1)
    np.random.seed(1)
    # imask2 = np.random.randint(2, size=layer_1.shape)
    # omask2 = imask2 * 2
    imask1 = mask1.imask()
    print(imask1, imask1.sum())
    # print(imask2)
    print(mask1.omask())
    # print(omask2)
