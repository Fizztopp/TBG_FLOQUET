#ifndef TBG_OUTPUTUTILITIES_H
#define TBG_OUTPUTUTILITIES_H


#include <vector>
#include <string>
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
                                 const std::vector<std::vector<double>> &UNIT_CELL);

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


#endif //TBG_OUTPUTUTILITIES_H
