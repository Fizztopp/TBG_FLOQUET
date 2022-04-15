import numpy as np
import h5py
import matplotlib.pyplot as plt
from readInData import readInData

linAtom = 31

[hk0, hkA, hkAA, hkExp, kPoints] = readInData(linAtom)

bandStep = 1

for k in range(hkA.shape[0]//bandStep):

    #decide which matrix to plot
    realParthk0 = hkA[bandStep*k, :, :].real
    imagParthk0 = hkA[bandStep*k, :, :].imag


    fig, ax1 = plt.subplots(nrows=1, ncols=1)
    colormesh1 = ax1.pcolormesh(realParthk0)
    fig.colorbar(colormesh1, ax=ax1)
    plt.show()

    #fig, ax1 = plt.subplots(nrows=1, ncols=1)
    #colormesh1 = ax1.pcolormesh(imagPart)
    #fig.colorbar(colormesh1, ax=ax1)
    #plt.show()
