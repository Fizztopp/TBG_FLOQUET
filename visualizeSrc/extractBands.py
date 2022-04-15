import numpy as np
def extractNLowEnergyBands(numberOfBands, mat):

    assert(np.ndim(mat) == 3)
    assert(mat.shape[1] == mat.shape[2])

    totalNumberOfBands = mat.shape[1]
    lBIndex = totalNumberOfBands//2 - numberOfBands//2
    uBIndex = totalNumberOfBands//2 + numberOfBands//2
    outMatrix = mat[:, lBIndex : uBIndex, lBIndex : uBIndex]

    return outMatrix