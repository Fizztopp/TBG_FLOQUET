#include <vector>
#include <string>
#include <complex>
#include <sstream>
#include <iostream>

#include "FileHandling.h"
#include "OutputUtilities.h"
#include "Hk0.h"
#include "HkA.h"

#include <cassert>

/**
 *
 * @param matrix matrix in which diagonal is written
 * @param diagonal real since we take it from diagonalization of hermitian matrix
 * @param dimension of matrix = length of diagonal
 */
void diagonalToMatrix(std::vector<std::complex<double>> &matrix,
                      const std::vector<double> &diagonal,
                      const unsigned long dimension);


/**
 * calculate matricies of interes in Hk0-basis
 * @param HkAInHk0BasisKPath - matrix HkA in Hk0 basis for given set of k-points
 * @param HkAAInHk0BasisKPath - matrix HkAA in Hk0 basis for given set of k-points
 * @param HkExpCouplingInHk0BasisKPath - matrix HkExpCoupling in Hk0 basis for given set of k-points
 * @param Hk0InHk0BasisKPath - matrix Hk0 in Hk0 basis for given set of k-points
 * @param kSet - for which to calculate matricies
 * @param lvec - supercell vectors needed to calculate matricies
 * @param UNIT_CELL - atomic positions in unit cell needed to calculate matricies
 */
void matriciesInHk0Basis(std::vector<std::complex<double>> &HkAInHk0BasisKPath,
                         std::vector<std::complex<double>> &HkAAInHk0BasisKPath,
                         std::vector<std::complex<double>> &HkExpCouplingInHk0BasisKPath,
                         std::vector<std::complex<double>> &Hk0InHk0BasisKPath,
                         const std::vector<std::vector<double>> &kSet,
                         const std::vector<double> &lvec,
                         const std::vector<std::vector<double>> &UNIT_CELL);

/**
 *
 * @param kVector Vector of 3D vectors of k-points
 * @param fileName file name for output
 */
void flattenAndOutputKArray(const std::vector<std::vector<double>> kVector, const std::string fileName);


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


    std::vector<std::complex<double>> HkAInHk0BasisKPath;
    std::vector<std::complex<double>> HkAAInHk0BasisKPath;
    std::vector<std::complex<double>> HkExpCouplingInHk0BasisKPath;
    std::vector<std::complex<double>> Hk0InHk0BasisKPath;

    matriciesInHk0Basis(HkAInHk0BasisKPath,
                        HkAAInHk0BasisKPath,
                        HkExpCouplingInHk0BasisKPath,
                        Hk0InHk0BasisKPath,
                        kSet,
                        lvec,
                        UNIT_CELL);

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

/**
 * calculate matricies of interes in Hk0-basis
 * @param HkAInHk0BasisKPath - matrix HkA in Hk0 basis for given set of k-points
 * @param HkAAInHk0BasisKPath - matrix HkAA in Hk0 basis for given set of k-points
 * @param HkExpCouplingInHk0BasisKPath - matrix HkExpCoupling in Hk0 basis for given set of k-points
 * @param Hk0InHk0BasisKPath - matrix Hk0 in Hk0 basis for given set of k-points
 * @param kSet - for which to calculate matricies
 * @param lvec - supercell vectors needed to calculate matricies
 * @param UNIT_CELL - atomic positions in unit cell needed to calculate matricies
 */
void matriciesInHk0Basis(std::vector<std::complex<double>> &HkAInHk0BasisKPath,
                         std::vector<std::complex<double>> &HkAAInHk0BasisKPath,
                         std::vector<std::complex<double>> &HkExpCouplingInHk0BasisKPath,
                         std::vector<std::complex<double>> &Hk0InHk0BasisKPath,
                         const std::vector<std::vector<double>> &kSet,
                         const std::vector<double> &lvec,
                         const std::vector<std::vector<double>> &UNIT_CELL) {

    std::vector<double> eValsHk0Temp(NATOM, 0.0);
    std::vector<std::complex<double>> MatInHk0BasisTemp(NATOM * NATOM, std::complex<double>(0.0, 0.0));
    std::vector<std::complex<double>> basisVectorsTemp(NATOM * NATOM, std::complex<double>(0.0, 0.0));

    HkAInHk0BasisKPath.reserve(kSet.size() * NATOM * NATOM);
    HkAAInHk0BasisKPath.reserve(kSet.size() * NATOM * NATOM);
    HkExpCouplingInHk0BasisKPath.reserve(kSet.size() * NATOM * NATOM);
    Hk0InHk0BasisKPath.reserve(kSet.size() * NATOM * NATOM);

    for (auto k = 0ul; k < kSet.size(); ++k) {
        std::cout << "Processing at k-point # " << k << '\n';
        eValsHk0Temp = Hk0DiagonalWithBasis(basisVectorsTemp, kSet[k], lvec, UNIT_CELL);

        MatInHk0BasisTemp = HkAInGivenBasis(basisVectorsTemp, kSet[k], lvec, UNIT_CELL);
        HkAInHk0BasisKPath.insert(HkAInHk0BasisKPath.end(), MatInHk0BasisTemp.begin(), MatInHk0BasisTemp.end());

        MatInHk0BasisTemp = HkAAInGivenBasis(basisVectorsTemp, kSet[k], lvec, UNIT_CELL);
        HkAAInHk0BasisKPath.insert(HkAAInHk0BasisKPath.end(), MatInHk0BasisTemp.begin(), MatInHk0BasisTemp.end());

        MatInHk0BasisTemp = HkExpCouplingInGivenBasis(basisVectorsTemp, kSet[k], lvec, UNIT_CELL);
        HkExpCouplingInHk0BasisKPath.insert(HkExpCouplingInHk0BasisKPath.end(), MatInHk0BasisTemp.begin(),
                                            MatInHk0BasisTemp.end());

        diagonalToMatrix(MatInHk0BasisTemp, eValsHk0Temp, NATOM);
        Hk0InHk0BasisKPath.insert(Hk0InHk0BasisKPath.end(), MatInHk0BasisTemp.begin(), MatInHk0BasisTemp.end());
    }
    std::cout << std::endl;
}

/**
 *
 * @param matrix matrix in which diagonal is written
 * @param diagonal real since we take it from diagonalization of hermitian matrix
 * @param dimension of matrix = length of diagonal
 */
void diagonalToMatrix(std::vector<std::complex<double>> &matrix,
                      const std::vector<double> &diagonal,
                      const unsigned long dimension) {

    assert(diagonal.size() == dimension);
    assert(matrix.size() == dimension * dimension);

    std::fill(matrix.begin(), matrix.end(), std::complex<double>(0.0, 0.0));
    for (auto ind = 0ul; ind < dimension; ++ind) {
        matrix[dimension * ind + ind] = diagonal[ind];
    }
}

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
