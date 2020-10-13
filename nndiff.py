#!/usr/bin/env python3

import sys, pickle, numpy as np
from nnet import NNet

argc = len(sys.argv)
# if argc < 2:
#     print('''usage: python nndiff.py file1.pkl file2.pkl [maxerrs]
#     file1.pkl: a neural net saved in Python Pickle format.
#     file2.pkl: a neural net saved in Python Pickle format.
#     maxerrs: the maximum number of errors to print before quitting.
#     '''[:-5])
#     exit()

lprint = True
gprint = True
# lprint = False
# gprint = False
threshold = 1e-8

filename1 = 'demo85i0b.pkl'
if argc > 1:
    filename1 = sys.argv[1]
# print(filename1)
f1 = open(filename1, "rb")
p1 = pickle.load(f1)

filename2 = 'demo86i0b.pkl'
if argc > 2:
    filename2 = sys.argv[2]
# print(filename2)
f2 = open(filename2, "rb")
p2 = pickle.load(f2)

errCount = 0
maxerrs = 10
if argc > 3:
    maxerrs = int(sys.argv[3])

if p1 == p2:
    exit()

if isinstance(p1, tuple):
    nn1 = p1[0]
    nn2 = p2[0]

if lprint:
    ll1 = nn1.layers
    ll2 = nn2.layers
    n = len(ll1)
    assert n == len(ll2)

    for li in range(n):
        l1 = ll1[li]
        l2 = ll2[li]
        ln = len(l1)
        assert ln == len(l2)
        nrows = len(l1)
        assert nrows == len(l2)
        for ri in range(nrows):
            row1 = l1[ri]
            row2 = l2[ri]
            # if row1 == row2:
            #     continue
            ncols = len(row1)
            assert ncols == len(row2)
            for ci in range(ncols):
                cell1 = row1[ci]
                cell2 = row2[ci]
                if abs(cell1 - cell2) < threshold:
                    continue
                print('%s.layer[%d][%d][%d] = %.20g' % (filename1, li, ri, ci, cell1))
                print('%s.layer[%d][%d][%d] = %.20g' % (filename2, li, ri, ci, cell2))
                errCount += 1
                if errCount >= maxerrs:
                    exit(1)

if gprint:
    g1 = nn1.grids
    g2 = nn2.grids
    m = len(g1)
    assert m == len(g2)
    for gi in range(m):
        alpha1 = g1[gi].getAlpha()
        alpha2 = g2[gi].getAlpha()
        if alpha1 != alpha2:
            print('%s.grid[%d].alpha = %.20g' % (filename1, gi, alpha1))
            print('%s.grid[%d].alpha = %.20g' % (filename2, gi, alpha2))
            errCount += 1
            if errCount >= maxerrs:
                exit(1)

        m1 = g1[gi].getWeights()
        m2 = g2[gi].getWeights()
        # if m1 == m2:
        #     continue
        nrows = len(m1)
        assert nrows == len(m2)
        for ri in range(nrows):
            row1 = m1[ri]
            row2 = m2[ri]
            # if row1 == row2:
            #     continue
            ncols = len(row1)
            assert ncols == len(row2)
            # errCount = 0
            for ci in range(ncols):
                cell1 = row1[ci]
                cell2 = row2[ci]
                if abs(cell1 - cell2) < threshold:
                    continue
                print('%s.grid[%d][%d][%d] = %.20g' % (filename1, gi, ri, ci, cell1))
                print('%s.grid[%d][%d][%d] = %.20g' % (filename2, gi, ri, ci, cell2))
                # print('difference          = %.20g' % (cell1 - cell2))
                errCount += 1
                if errCount >= maxerrs:
                    exit(1)
            # if errCount > 0:
            #     print('%s.grid[%d][%d][:] has errors.' % (filename1, gi, ri))

if errCount == 0:
    print('%s and %s appear to be equal.' % (filename1, filename2))