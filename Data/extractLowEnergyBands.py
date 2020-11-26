import numpy as np
import h5py
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

linAtom = 16

fileName = 'HkA_' + str(linAtom) + '.hdf5'
file = h5py.File('N' + str(linAtom)+ '/' + fileName, 'r')
hkARealPart = file['Real'][()]
hkAImagPart = file['Imag'][()]
print('hkARealPart.shape = ' + str(hkARealPart.shape))

fileName = 'HkAA_' + str(linAtom) + '.hdf5'
file = h5py.File('N' + str(linAtom)+ '/' + fileName, 'r')
hkAARealPart = file['Real'][()]
hkAAImagPart = file['Imag'][()]
print('hkAARealPart.shape = ' + str(hkAARealPart.shape))

fileName = 'HkExpCoupling_' + str(linAtom) + '.hdf5'
file = h5py.File('N' + str(linAtom)+ '/' + fileName, 'r')
hkExpCouplingRealPart = file['Real'][()]
hkExpCouplingImagPart = file['Imag'][()]
print('hkExpCouplingRealPart.shape = ' + str(hkExpCouplingRealPart.shape))

fileName = 'Hk0_' + str(linAtom) + '.hdf5'
file = h5py.File('N' + str(linAtom)+ '/' + fileName, 'r')
hk0RealPart = file['Real'][()]
hk0ImagPart = file['Imag'][()]
print('hk0RealPart.shape = ' + str(hk0RealPart.shape))


fileNameKPath = 'KSetKPoints_' + str(linAtom) + '.hdf5'
fileKPoints = h5py.File('N' + str(linAtom)+ '/' + fileNameKPath, 'r')
kPointsArr = fileKPoints['Real'][()]
print('kPoints.shape = ' + str(kPointsArr.shape))

assert(hkARealPart.shape[0] == kPointsArr.shape[0])
assert(hkAARealPart.shape[0] == kPointsArr.shape[0])
assert(hkExpCouplingRealPart.shape[0] == kPointsArr.shape[0])
assert(hk0RealPart.shape[0] == kPointsArr.shape[0])

numberOfBands = hk0RealPart.shape[1]


extractBands = 20


hk0Extracted = np.zeros((hk0RealPart.shape[0]) * extractBands * extractBands, dtype='complex')
hkAExtracted = np.zeros((hkARealPart.shape[0]) * extractBands * extractBands, dtype='complex')
hkAAExtracted = np.zeros((hkAARealPart.shape[0]) * extractBands * extractBands, dtype='complex')
hkExpCouplingExtracted = np.zeros((hkExpCouplingRealPart.shape[0]) * extractBands * extractBands, dtype='complex')

for band in range(extractBands):
    bandToCopy = numberOfBands // 2 - extractBands//2 + band
    Hk0Extracted = hk0RealPart[:, bandToCopy, bandToCopy] + 1j * hk0ImagPart[:, bandToCopy, bandToCopy]
    HkAExtracted = hkARealPart[:, bandToCopy, bandToCopy] + 1j * hkAImagPart[:, bandToCopy, bandToCopy]
    HkAAExtracted = hkAARealPart[:, bandToCopy, bandToCopy] + 1j * hkAAImagPart[:, bandToCopy, bandToCopy]
    HkExpCouplingExtracted = hkExpCouplingRealPart[:, bandToCopy, bandToCopy] + 1j * hkExpCouplingImagPart[:, bandToCopy, bandToCopy]

outFile = h5py.File('Hk0_' + str(linAtom) + '_ext_' + str(extractBands) + '.hdf5', 'w')
outFile.create_dataset('Real', data=Hk0Extracted.real, dtype='double')
outFile.create_dataset('Imag', data=Hk0Extracted.imag, dtype='double')
outFile.close()

outFile = h5py.File('HkA_' + str(linAtom) + '_ext_' + str(extractBands) + '.hdf5', 'w')
outFile.create_dataset('Real', data=HkAExtracted.real, dtype='double')
outFile.create_dataset('Imag', data=HkAExtracted.imag, dtype='double')
outFile.close()

outFile = h5py.File('HkAA_' + str(linAtom) + '_ext_' + str(extractBands) + '.hdf5', 'w')
outFile.create_dataset('Real', data=HkAAExtracted.real, dtype='double')
outFile.create_dataset('Imag', data=HkAAExtracted.imag, dtype='double')
outFile.close()

outFile = h5py.File('HkExpCoupling_' + str(linAtom) + '_ext_' + str(extractBands) + '.hdf5', 'w')
outFile.create_dataset('Real', data=HkExpCouplingExtracted.real, dtype='double')
outFile.create_dataset('Imag', data=HkExpCouplingExtracted.imag, dtype='double')
outFile.close()

