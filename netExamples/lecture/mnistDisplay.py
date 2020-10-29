from netExamples.grokking.mnist import x_test, x_train, y_test, y_train
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import sys

while True:
    try:
        mindex = int(input('Image Number (or "q" to quit): '))
    except:
        sys.exit(0)

    x = x_train if mindex < 60000 else x_test
    y = y_train if mindex < 60000 else y_test
    l = 'train' if mindex < 60000 else 'test'
    index = mindex % 60000
    print('x%s[%d] = %d' % (l, index, y[index]))
    grayScale = x[index]
    rgb = [[[255-g, 255-g, 255-g] for g in row]
           for row in grayScale]
    color = np.array(rgb, dtype=np.uint8)
    img = Image.fromarray(color, 'RGB')
    plt.figure(1)
    plt.imshow(img)
    plt.pause(0.1)
