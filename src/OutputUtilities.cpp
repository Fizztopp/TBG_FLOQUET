#include <vector>
#include <string>
#include <complex>
#include <sstream>
#include <iostream>

#include "FileHandling.h"
#include "OutputUtilities.h"
#include "Hk0.h"
#include "HkA.h"

/**
 *
 * @param kVector Vector of 3D vectors of k-points
 * @param fileName file name for output
 */
void flattenAndOutputKArray(const std::vector<std::vector<double>> kVector, const std::string fileName) {
    std::vector<double> flatKArray(3 * kVector.size(), 0.0);
    for (unsigned long k = 0; k < kVector.size(); ++k) {
        flatKArray[3 * k + 0] = kVector[k][0];
        flatKArray[3 * k + 1] = kVector[k][1];
        flatKArray[3 * k + 2] = kVector[k][2];
    }
    writeReal2DArrayToHdf5(flatKArray, fileName, kVector.size(), 3ul);

}

/**
 *
 * @param kSet vector of 3-d kpoint vectors for which matricies should be calculated
 * @param name for naming the outputted file
 * @param lvec super-lattice vector
 * @param UNIT_CELL atomic positions in unit-cell
 */
void generateMatrixOutputForKSet(const std::vector<std::vector<double>> &kSet,
                                 const std::string name,
                                 const std::vector<double> &lvec,
                                 const std::vector<std::vector<double>> &UNIT_CELL) {
    std::vector<double> tempKVec(3, 0.0);
    std::vector<double> eValsHk0Temp(NATOM, 0.0);
    std::vector<std::complex<double>> HkAInHk0BasisTemp(NATOM * NATOM, std::complex<double>(0.0, 0.0));
    std::vector<std::complex<double>> basisVectors(NATOM * NATOM, std::complex<double>(0.0, 0.0));

    std::vector<std::complex<double>> HkAInHk0BasisKPath;
    std::vector<std::complex<double>> HkAAInHk0BasisKPath;
    std::vector<std::complex<double>> HkExpCouplingInHk0BasisKPath;
    std::vector<std::complex<double>> Hk0InHk0BasisKPath;
    for (auto k = 0ul; k < kSet.size(); ++k) {
        std::cout << "Processing at k-point # " << k << '\n';
        tempKVec = kSet[k];
        eValsHk0Temp = Hk0DiagonalWithBasis(basisVectors, tempKVec, lvec, UNIT_CELL);
        //calculate HKA
        HkAInHk0BasisTemp = HkAInGivenBasis(basisVectors, tempKVec, lvec, UNIT_CELL);
        for (auto matrixElem = 0ul; matrixElem < NATOM * NATOM; ++matrixElem) {
            HkAInHk0BasisKPath.push_back(HkAInHk0BasisTemp[matrixElem]);
        }
        //calculate HKAA
        HkAInHk0BasisTemp = HkAAInGivenBasis(basisVectors, tempKVec, lvec, UNIT_CELL);
        for (auto matrixElem = 0ul; matrixElem < NATOM * NATOM; ++matrixElem) {
            HkAAInHk0BasisKPath.push_back(HkAInHk0BasisTemp[matrixElem]);
        }
        //calculate HkExpCoupling
        HkAInHk0BasisTemp = HkExpCouplingInGivenBasis(basisVectors, tempKVec, lvec, UNIT_CELL);
        for (auto matrixElem = 0ul; matrixElem < NATOM * NATOM; ++matrixElem) {
            HkExpCouplingInHk0BasisKPath.push_back(HkAInHk0BasisTemp[matrixElem]);
        }
        //fill Hk0 matrix
        for (auto dim1 = 0ul; dim1 < NATOM; ++dim1) {
            for (auto dim2 = 0ul; dim2 < NATOM; ++dim2) {
                if (dim1 == dim2) {
                    Hk0InHk0BasisKPath.emplace_back(std::complex<double>(eValsHk0Temp[dim1], 0.0));
                } else {
                    Hk0InHk0BasisKPath.emplace_back(std::complex<double>(0.0, 0.0));
                }
            }
        }
    }
    std::cout << std::endl;

    std::string nameHk0 = createOutputString("Data/Hk0.hdf5");
    std::string nameHkA = createOutputString("Data/HkA.hdf5");
    std::string nameHkAA = createOutputString("Data/HkAA.hdf5");
    std::string nameHkExpCoupling = createOutputString("Data/HkExpCoupling.hdf5");

    writeComplex3DArrayToHdf5(Hk0InHk0BasisKPath, nameHk0, kSet.size(), NATOM, NATOM);
    writeComplex3DArrayToHdf5(HkAInHk0BasisKPath, nameHkA, kSet.size(), NATOM, NATOM);
    writeComplex3DArrayToHdf5(HkAAInHk0BasisKPath, nameHkAA, kSet.size(), NATOM, NATOM);
    writeComplex3DArrayToHdf5(HkExpCouplingInHk0BasisKPath, nameHkExpCoupling, kSet.size(), NATOM, NATOM);

    flattenAndOutputKArray(kSet, "Data/KSetKPoints.hdf5");
}
