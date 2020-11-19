import numpy as np
import h5py
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fileName = 'HkAInHk0BasisFullBZ.hdf5'
file = h5py.File('../Data/' + fileName, 'r')
keys = list(file.keys())
print('Keys in Dataset are ' + str(keys))
realPart = file['Real'][()]
imagPart = file['Imag'][()]
print('realPart.shape = ' + str(realPart.shape))

fileNameIBZ = 'FullBZKPoints.hdf5'
fileKPoints = h5py.File('../Data/' + fileNameIBZ, 'r')
kPointsArr = fileKPoints['Real'][()]
print('kPoints.shape = ' + str(kPointsArr.shape))

assert(realPart.shape[0] == kPointsArr.shape[0])

kX = kPointsArr[:, 0]
kY = kPointsArr[:, 1]

# take specific matrix element
n = 125
m = 125
plotDataReal = realPart[:, n, m]
plotDataImag = imagPart[:, n, m]

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(kX, kY, plotDataReal)
plt.show()

# fig, ax1 = plt.subplots(nrows=1, ncols=1)
# colormesh1 = ax1.pcolormesh(imagPart)
# fig.colorbar(colormesh1, ax=ax1)
# plt.show()
