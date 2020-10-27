
from netExamples.lecture.sosb \
    import inputData, inputTraining, targetData, targetTraining
from weightmatrix import WeightMatrix
from cmatrix import ConvolutionMatrix

from nnet import NNet
from verbosePrint import vprint
import verbosePrint

nn = NNet(sizes=[12, 40, 4], bias=True)

weights_0_1 = [
    [ -1.67058053, -31.88964805, -22.62960613, 175.50044075],
    [ -9.44243163, 113.81472505, -66.92167979,  61.20256674],
    [-16.84458954,  69.75625936,  57.28839568, -20.36714292],
    [ -0.1817625 ,  25.48485469,  22.55888487,  25.30124179]]
cm = ConvolutionMatrix(weights_0_1, bias=True, shapes=((1,12),(1,3)))
nn.replaceLayer(0, cm)

weights_1_2 = [
    [-0.53760496, -0.50491136, -0.80651041,  0.19903322],
    [ 0.06139362, -0.83799228, -1.12469754, -0.34619998],
    [ 2.13319875, -1.01278365, -0.60101967, -1.32352224],
    [-3.87630816, -0.85575407,  0.66372222, -0.1069061 ],
    [-0.03494623,  0.19946813, -0.06985775,  0.94311043],
    [ 0.09994165, -1.32276598, -1.11279273, -0.06587269],
    [-0.61605472,  2.19865815,  0.81997844, -0.51609196],
    [-3.24488518, -2.66263305, -0.76680333,  0.13544236],
    [-0.5195168 , -0.03477992, -0.28119573, -0.34124789],
    [ 3.17216247,  0.4045901 , -1.61294996, -1.34893918],
    [-2.88202726,  0.85740532,  2.05525156, -0.14483095],
    [-1.7404537 , -3.44601327, -3.0483553 , -0.62157819],
    [-0.70939291, -0.27563795, -0.25757858,  0.5512526 ],
    [-3.59388209,  4.25797461,  1.45625797, -1.30262422],
    [ 0.48876496, -2.97221355,  0.58654931,  3.08511689],
    [ 0.90566781, -2.58510198, -4.03744114, -4.77640605],
    [ 0.47739193, -0.30211966, -0.04852575,  0.09719012],
    [-3.03153206, -3.66641968,  3.14190399,  1.02406116],
    [-1.55011669,  0.86393419, -3.42326163,  1.47414747],
    [ 2.01875802,  1.1890159 , -3.00100893, -4.17975272],
    [-0.00752651,  0.80878279, -0.15970456,  0.1719537 ],
    [-2.3609116 , -2.09156277, -2.44053594,  3.30822731],
    [-2.53829866, -2.14236091,  0.08922379, -2.74865442],
    [ 3.31944907,  0.59780841, -0.14390473, -2.84930608],
    [ 0.31776421, -0.58653336,  0.38737669,  0.84336997],
    [ 0.05284319, -3.18366088, -3.62821349, -2.60397406],
    [ 1.23561288, -2.20991739, -1.82335855,  0.7110058 ],
    [-3.77706054,  4.17151034,  2.51407694,  0.5453451 ],
    [ 0.53870221,  0.91373462,  0.71841289, -0.43130623],
    [-1.39230794, -0.59656183, -3.06075963, -2.33341487],
    [ 1.20933192,  1.17955314, -3.29600719, -1.80380998],
    [-1.96890573, -4.10730626,  3.33177446,  1.91903193],
    [-0.8858574 ,  0.05881214,  0.33295324,  0.47227012],
    [ 1.68368021, -2.16028071, -0.49761511, -2.0199788 ],
    [-1.20030509,  0.26128166,  2.88815594, -1.60763975],
    [-2.27999708, -1.43537472, -4.40890801,  2.78249789],
    [-0.54611297, -0.38348355,  0.6507966 , -0.13915858],
    [-0.43032027,  1.99742958,  0.19046502, -1.75774803],
    [-0.85919565, -1.99212299,  1.90655059,  0.97828721],
    [ 0.04083744, -2.04338197, -3.88257489, -1.77983968],
    [-2.87640294, -2.36045511, -2.19081092, -1.82161738]]
wm = WeightMatrix(weights_1_2, bias=True)
nn.replaceLayer(1, wm)

nn.setActivations(['softMax', 'sigmoid'])

inputData = [[0,0,0,0,0,1,1,1,0,0,0,0]]
targetData = [[0,0,1,0]]

nn.fire(inputData)
nn.checkup(inputData, targetData)
