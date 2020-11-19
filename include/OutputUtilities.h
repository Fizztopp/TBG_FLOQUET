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



#endif //TBG_OUTPUTUTILITIES_H
