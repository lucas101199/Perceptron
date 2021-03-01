import numpy as np
from pylab import rand

def perceptronBinary(dataset, n):
    w = np.zeros(1)
    for i in range(1, n):
        for data in dataset:
            if data[i][1]


def genererDonnees(n) :
    x1b = (rand(n)*2-1)/2-0.5
    x2b = (rand(n) * 2 - 1) / 2 + 0.5
    x1r = (rand(n) * 2 - 1) / 2 + 0.5
    x2r = (rand(n) * 2 - 1) / 2 - 0.5
    donnees = []
    for i in range(len(x1b)) :
        donnees.append(((x1b[i], x2b[i]), False))
        donnees.append(((x1r[i], x2r[i]), True))
    return donnees