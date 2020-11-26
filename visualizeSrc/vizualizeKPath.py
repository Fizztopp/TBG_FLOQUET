import numpy as np
import h5py
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

linAtom = 4

fileName = 'HkA_' + str(linAtom) + '.hdf5'
file = h5py.File('../Data/' + fileName, 'r')
hkARealPart = file['Real'][()]
hkAImagPart = file['Imag'][()]
print('hkARealPart.shape = ' + str(hkARealPart.shape))

fileName = 'HkAA_' + str(linAtom) + '.hdf5'
file = h5py.File('../Data/' + fileName, 'r')
hkAARealPart = file['Real'][()]
hkAAImagPart = file['Imag'][()]
print('hkAARealPart.shape = ' + str(hkAARealPart.shape))

fileName = 'HkExpCoupling_' + str(linAtom) + '.hdf5'
file = h5py.File('../Data/' + fileName, 'r')
hkExpCouplingRealPart = file['Real'][()]
hkExpCouplingImagPart = file['Imag'][()]
print('hkExpCouplingRealPart.shape = ' + str(hkExpCouplingRealPart.shape))

fileName = 'Hk0_' + str(linAtom) + '.hdf5'
file = h5py.File('../Data/' + fileName, 'r')
hk0RealPart = file['Real'][()]
hk0ImagPart = file['Imag'][()]
print('hk0RealPart.shape = ' + str(hk0RealPart.shape))


fileNameKPath = 'KSetKPoints_' + str(linAtom) + '.hdf5'
fileKPoints = h5py.File('../Data/' + fileNameKPath, 'r')
kPointsArr = fileKPoints['Real'][()]
print('kPoints.shape = ' + str(kPointsArr.shape))

assert(hkARealPart.shape[0] == kPointsArr.shape[0])
assert(hkAARealPart.shape[0] == kPointsArr.shape[0])
assert(hkExpCouplingRealPart.shape[0] == kPointsArr.shape[0])
assert(hk0RealPart.shape[0] == kPointsArr.shape[0])

numberOfBands = hk0RealPart.shape[1]

matrixElementsHkA = np.array([
    [numberOfBands // 2 - 3, numberOfBands // 2 - 3],
    [numberOfBands // 2 - 2, numberOfBands // 2 - 2],
    [numberOfBands // 2 - 1, numberOfBands // 2 - 1],
    [numberOfBands // 2, numberOfBands // 2],
    [numberOfBands // 2 + 1, numberOfBands // 2 + 1],
    [numberOfBands // 2 + 2, numberOfBands // 2 + 2],
])

matrixElementsHk0 = np.array([
    [numberOfBands // 2 - 3, numberOfBands // 2 - 3],
    [numberOfBands // 2 - 2, numberOfBands // 2 - 2],
    [numberOfBands // 2 - 1, numberOfBands // 2 - 1],
    [numberOfBands // 2, numberOfBands // 2],
    [numberOfBands // 2 + 1, numberOfBands // 2 + 1],
    [numberOfBands // 2 + 2, numberOfBands // 2 + 2],
])

hkAData = np.zeros((matrixElementsHkA.shape[0], hkARealPart.shape[0]), dtype='complex')
hkAAData = np.zeros((matrixElementsHkA.shape[0], hkAARealPart.shape[0]), dtype='complex')
hkExpCouplingData = np.zeros((matrixElementsHkA.shape[0], hkExpCouplingRealPart.shape[0]), dtype='complex')
hk0Data = np.zeros((matrixElementsHk0.shape[0], hk0RealPart.shape[0]), dtype='complex')

for ind in range(len(matrixElementsHkA)):
    hkAData[ind, :] = hkARealPart[:, matrixElementsHkA[ind, 0], matrixElementsHkA[ind, 1]] + 1j * hkAImagPart[:, matrixElementsHkA[ind, 0], matrixElementsHkA[ind, 1]]
    hkAAData[ind, :] = hkAARealPart[:, matrixElementsHkA[ind, 0], matrixElementsHkA[ind, 1]] + 1j * hkAAImagPart[:, matrixElementsHkA[ind, 0], matrixElementsHkA[ind, 1]]
    hkExpCouplingData[ind, :] = hkExpCouplingRealPart[:, matrixElementsHkA[ind, 0], matrixElementsHkA[ind, 1]] + 1j * hkExpCouplingImagPart[:, matrixElementsHkA[ind, 0], matrixElementsHkA[ind, 1]]

for ind in range(len(matrixElementsHk0)):
    hk0Data[ind, :] = hk0RealPart[:, matrixElementsHk0[ind, 0], matrixElementsHk0[ind, 1]] + 1j * hk0ImagPart[:, matrixElementsHk0[ind, 0], matrixElementsHk0[ind, 1]]

#hk0Data = hkExpCouplingData - hkAData - hkAAData - hk0Data

fig, ax = plt.subplots(nrows=1, ncols=1)
cmap1 = plt.cm.get_cmap('terrain')
cmap2 = plt.cm.get_cmap('jet')
for ind in range(len(matrixElementsHkA)):
    ax.plot(hk0Data[ind, :].real, color='mediumseagreen')
for ind in range(len(matrixElementsHkA)):
    ax.plot(hkAAData[ind, :].real, color='orange', linestyle = '--')

plt.show()




