import numpy as np
import h5py
import matplotlib.pyplot as plt

fileName = 'HkExpCoupling.hdf5'
file = h5py.File('../Data/' + fileName, 'r')
hkARealPart = file['Real'][()]
hkAImagPart = file['Imag'][()]
print('hkARealPart.shape = ' + str(hkARealPart.shape))


for k in range(hkARealPart.shape[0]//10):

    #decide which matrix to plot
    realPart = hkARealPart[10*k, :, :]
    imagPart = hkAImagPart[10*k, :, :]


    fig, ax1 = plt.subplots(nrows=1, ncols=1)
    colormesh1 = ax1.pcolormesh(realPart[117:127, 117:127])
    fig.colorbar(colormesh1, ax=ax1)
    plt.show()

    #fig, ax1 = plt.subplots(nrows=1, ncols=1)
    #colormesh1 = ax1.pcolormesh(imagPart)
    #fig.colorbar(colormesh1, ax=ax1)
    #plt.show()
