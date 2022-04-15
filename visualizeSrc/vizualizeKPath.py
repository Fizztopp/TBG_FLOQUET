import numpy as np
import h5py
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from readInData import readInData
from extractBands import extractNLowEnergyBands


linAtom = 28

[hk0, hkA, hkAA, hkExp, kPoints] = readInData(linAtom)
totalNumberOfBands = hk0.shape[1]

desiredNumberOfBands = 20

hk0Data = extractNLowEnergyBands(desiredNumberOfBands, hk0)
hkAData = extractNLowEnergyBands(desiredNumberOfBands, hkA)
hkAAData = extractNLowEnergyBands(desiredNumberOfBands, hkAA)
hkExpData = extractNLowEnergyBands(desiredNumberOfBands, hkExp)

exactCouplingData = hkExpData - hk0Data - hkAData - hkAAData

fig, ax = plt.subplots(nrows=1, ncols=1)
cmap1 = plt.cm.get_cmap('terrain')
cmap2 = plt.cm.get_cmap('jet')
color = cmap1(0.)
colors = [0.0, 0.15, 0.3, 0.55, 0.85]
#colors = [1.0, 1.0, 0.3, 1.0, 1.0]
for ind in range(desiredNumberOfBands):
    color = cmap1(colors[ind // 4])

    ax.plot(np.arange(len(kPoints)), hk0Data[:, ind, ind].real, color=color)
    ax.plot(np.arange(len(kPoints)), hkAData[:, ind, ind].real, color=color, linestyle = '--')

#for ind in range(4):
#    band = 8
#    ax.plot(np.arange(len(kPoints)), hkAData[:, ind + 8, band].real, color='black', linestyle = '-', linewidth = 2)
#    ax.plot(np.arange(len(kPoints)), hkAData[:, ind + 8, band + 4].real, color='gray', linestyle = '-', linewidth = 2)
#    ax.plot(np.arange(len(kPoints)), hkAData[:, ind + 8, band + 8].real, color='lightgray', linestyle = '-', linewidth = 2)

#for ind in range(desiredNumberOfBands):
#    ax.plot(hkAAData[:, ind, ind].real, color='dodgerblue', linestyle = '--')
#for ind in range(desiredNumberOfBands):
#    ax.plot(exactCouplingData[:, ind, ind].real, color='indianred', linestyle = '--')

ax.xaxis.set_ticks = [0, 1]
ax.xaxis.set_labels = ['0', '1']

plt.show()




