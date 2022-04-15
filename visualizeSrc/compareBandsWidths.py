import numpy as np
import matplotlib.pyplot as plt
from readInData import readInData
from extractBands import extractNLowEnergyBands
from calcualteTwistAngle import calculateAngle


linAtomArr = [4, 8, 12, 16, 20, 24, 28, 31]

angles = calculateAngle(np.array(linAtomArr), np.array(linAtomArr) + 1) * 180. / np.pi

hk0BandWidth = np.zeros((0), dtype='double')
hkABandWidth = np.zeros((0), dtype='double')
hkAABandWidth = np.zeros((0), dtype='double')
hkExpBandWidth = np.zeros((0), dtype='double')

hkAInterBand = np.zeros((0), dtype='double')

for linAtom in linAtomArr:

    [hk0, hkA, hkAA, hkExp, kPoints] = readInData(linAtom)

    numberOfBands = 12

    hk0LEBands = extractNLowEnergyBands(numberOfBands, hk0)
    hkALEBands = extractNLowEnergyBands(numberOfBands, hkA)
    hkAALEBands = extractNLowEnergyBands(numberOfBands, hkAA)
    hkEXPLEBands = extractNLowEnergyBands(numberOfBands, hkExp)

    b1 = 4
    b2 = 7

    hk0BandDiffArr = np.abs(hk0LEBands[:, b1, b1] - hk0LEBands[:, b2, b2])
    hkABandDiffArr = np.abs(hkALEBands[:, b1, b1])
    hkAABandDiffArr = np.abs(hkAALEBands[:, b1, b1])
    hkAInterBandDiffArr = np.abs(hkALEBands[:, b1, b2 + 2])

    #hkExpBandDiffArr = np.abs(hkEXPLEBands[:, b1, b1] - hkEXPLEBands[:, b2, b2])

    #fig, ax = plt.subplots(nrows=1, ncols=1)
    #ax.plot(hkABandDiffArr[:].real, color='red')
    #ax.plot(hkALEBands[:, 0, 0].real, color='blue')
    #plt.show()

    print("maximum of hk0BandDiffArr = {}".format(np.amax(hk0BandDiffArr)))

    hk0BandWidth = np.append(hk0BandWidth, [np.amax(hk0BandDiffArr)], axis=0)
    hkABandWidth = np.append(hkABandWidth, [np.amax(hkABandDiffArr)], axis=0)
    hkAABandWidth = np.append(hkAABandWidth, [np.amax(hkAABandDiffArr)], axis=0)
    hkAInterBand = np.append(hkAInterBand, [np.amax(hkAInterBandDiffArr)], axis=0)
    #hkExpBandWidth = np.append(hkExpBandWidth, [np.amax(hkExpBandDiffArr)], axis=0)

print("hk0 Bandswidths.shape = {}".format(hk0BandWidth.shape))

angles = angles[3:]
hk0BandWidth = hk0BandWidth[3:]
hkABandWidth = hkABandWidth[3:]
hkAABandWidth = hkAABandWidth[3:]
hkAInterBand = hkAInterBand[3:]

fig, ax = plt.subplots(nrows=1, ncols=1)
ax.plot(angles, hk0BandWidth * 2 * 1e1, color='mediumseagreen', linestyle = '', marker = 'X', markersize = 12, label='Bandwidth x 20')
ax.plot(angles, hkABandWidth * 1e2, color='darkcyan', linestyle = '', marker = 'X', markersize = 12, label='A Intraband Coupling x 1e2')
ax.plot(angles, hkAABandWidth * 1e4, color='sienna', linestyle = '', marker = 'X', markersize = 12, label='AA Intraband Coupling x 1e4')
ax.plot(angles, hkAInterBand * 1e2, color='lightcoral', linestyle = '', marker = 'X', markersize = 12, label='Interband Coupling x 1e2')
#ax.plot(angles, hkExpBandWidth, color='red', linestyle = '', marker = 'x')
plt.legend()
labels = ['{:.3}°'.format(angle) for angle in angles]
plt.xticks(angles, labels)
ax.tick_params(axis='both', which='major', labelsize=14)
ax.set_xlabel('angle', fontsize=14)
ax.set_ylabel('energy[eV]', fontsize=14)
for angle in angles:
    plt.axvline(x=angle, color = 'gray', linestyle = '--', linewidth = .5)
plt.axhline(y=0., color = 'gray', linestyle = '-', linewidth = .5)


plt.show()

