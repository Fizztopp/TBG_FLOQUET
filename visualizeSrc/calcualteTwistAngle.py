import numpy as np

def calculateAngle(n, m):
    return np.arccos((n**2 + 4 * n * m + m**2)/(2 * (n**2 + n * m + m**2)))

def calculateAtomsInUnitCell(n, m):
    return 4 * (n**2 + m * n + m**2) 

def main():
    #nArr = np.array([4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 ,16 , 17, 18])
    nArr = np.arange(32)

    for n in nArr:
        m = n + 1
        angle = calculateAngle(n, m)
        print('Twist angle for n={}, m={} = {}rad'.format(n, m, angle))
        angleDeg = angle * 180. / np.pi
        print('Twist angle for n={}, m={} = {}deg'.format(n, m, angleDeg))

        numberAtoms = calculateAtomsInUnitCell(n, m)
        print('Number of atoms in unit-cell = {}'.format(numberAtoms))

        print('\n')
main()
