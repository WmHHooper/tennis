import sys, pickle, numpy as np
from nnet import NNet

# These variables determine the demo number, the iteration,
# and the stage within the iteration where the objects are printed
demo = -1
vIteration = -1
stage= 'a'

def vp():
    print(vIteration)

def vprint(i, *s, **kwargs):
    if i != vIteration:
        return

    kwsuffix = ''
    if 'stage' in kwargs:
        kwsuffix = kwargs['stage']
    if kwsuffix != stage:
        return

    prefix='nne'
    if 'prefix' in kwargs:
        prefix = kwargs['prefix']

    exeFilename = sys.argv[0]
    base = exeFilename[:-3] # remove '.py'
    if demo > 0:
        print("demo = %d, i = %d%s" % (demo, vIteration, stage))
    else:
        print("i = %d%s" % (vIteration, stage))
    printed = False
    for nn in s:
        if hasattr(nn, 'printnn'):
            nn.printnn()
            printed = True
        elif isinstance(nn, np.ndarray):
            print(np.array2string(nn, threshold=1e5))
            printed = True
    if not printed:
        print(*s)

    if 'quit' in kwargs and kwargs['quit'] == True:
        basename = '%s%di%d%s' % (base, demo, i, stage)
        pfp = open(basename + '.pkl', "wb")
        pickle.dump(s, file=pfp)
        tfp = open(basename + '.txt', "w")
        printed = False
        for nn in s:
            if hasattr(nn, 'printnn'):
                nn.printnn(tfp)
                printed = True
            elif isinstance(nn, np.ndarray):
                print(np.array2string(nn, threshold=1e5), file=tfp)
                printed = True
        if not printed:
            print(*s, file=tfp)
        sys.exit()


layer_0 = [[[]]]
layer_1 = [[[]]]
layer_2 = [[[]]]
weights_0_1 = [[[]]]
weights_1_2 = [[[]]]
alpha = 0.01

def vprintnn(i, quit=False, suffix=''):
    ll = [ layer_0, layer_1, layer_2 ]
    ww = [ weights_0_1, weights_1_2]
    nn = NNet(weights=ww, layerList=ll)
    nn.setAlpha(alpha)
    vprint(i, nn, quit=quit, prefix='gro', suffix=suffix)
