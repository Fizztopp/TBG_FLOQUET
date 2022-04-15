import h5py

def readInData(linAtom):

    folderPrefix = '../Data/N' + str(linAtom) + 'Ext/'
    fileSuffix = '_ext_20.hdf5'

    fileName = 'HkA_' + str(linAtom)
    file = h5py.File(folderPrefix + fileName + fileSuffix, 'r')
    hkA = file['Real'][()] + 1j * file['Imag'][()]
    print('hkA.shape = ' + str(hkA.shape))

    fileName = 'HkAA_' + str(linAtom)
    file = h5py.File(folderPrefix + fileName + fileSuffix, 'r')
    hkAA = file['Real'][()] + 1j * file['Imag'][()]
    print('hkAA.shape = ' + str(hkAA.shape))

    fileName = 'HkExpCoupling_' + str(linAtom)
    file = h5py.File(folderPrefix + fileName + fileSuffix, 'r')
    hkExp = file['Real'][()] + 1j * file['Imag'][()]
    print('hkExp.shape = ' + str(hkExp.shape))

    fileName = 'Hk0_' + str(linAtom)
    file = h5py.File(folderPrefix + fileName + fileSuffix, 'r')
    hk0 = file['Real'][()] + 1j + file['Imag'][()]
    print('hk0.shape = ' + str(hk0.shape))

    fileNameKPath = 'KSetKPoints_' + str(linAtom) + '.hdf5'
    fileKPoints = h5py.File(folderPrefix + fileNameKPath, 'r')
    kPoints = fileKPoints['Real'][()]
    print('kPoints.shape = ' + str(kPoints.shape))

    assert(hkA.shape[0] == kPoints.shape[0])
    assert(hkAA.shape[0] == kPoints.shape[0])
    assert(hkExp.shape[0] == kPoints.shape[0])
    assert(hk0.shape[0] == kPoints.shape[0])

    return [hk0, hkA, hkAA, hkExp, kPoints]
