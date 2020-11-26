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

/**
 *
 * @param rank rank on which one wants to know the chunksize
 * @param numprocs total number of processes
 * @param totalSize total size of to be calculated k-array
 * @return size of the chunk saved on that task
 */
int mapRankToChunksize(const int rank, const int numprocs, const int totalSize);

/**
 *
 * @param numprocs number of mpi processes
 * @param kSetSize number of k-points to calculate
 * @return k-point indices at which each task starts its local k-chunk
 */
std::vector<unsigned long> startChunkIndiciesMPI(const int numprocs, const unsigned long kSetSize);


/**
 *
 * @param collectMat matrix of all k-points that is to be collected from all ranks
 * @param localMat matrix that was previously calculated on each rank locally
 */
void collectOnMaster(std::vector<std::complex<double>> &collectMat,
                     const std::vector<std::complex<double>> &localMat,
                     const size_t totalSize);


/**
 *
 * @param matrix matrix in which diagonal is written
 * @param diagonal real since we take it from diagonalization of hermitian matrix
 * @param dimension of matrix = length of diagonal
 */
void diagonalToMatrix(std::vector<std::complex<double>> &matrix,
                      const std::vector<double> &diagonal,
                      const unsigned long dimension);

void splitKSetAndObtainMatriciesOnMaster(const std::vector<std::vector<double>> &kSet, const std::vector<double> &lvec,
                                         const std::vector<std::vector<double>> &UNIT_CELL,
                                         std::vector<std::complex<double>> &HkAInHk0BasisKPathComplete,
                                         std::vector<std::complex<double>> &HkAAInHk0BasisKPathComplete,
                                         std::vector<std::complex<double>> &HkExpCouplingInHk0BasisKPathComplete,
                                         std::vector<std::complex<double>> &Hk0InHk0BasisKPathComplete);


/**
 *
 * @param kVector Vector of 3D vectors of k-points
 * @param fileName file name for output
 */
void flattenAndOutputKArray(const std::vector<std::vector<double>> kVector, const std::string fileName);

/**
 *
 * @param kSetTotal the complete k-set for which output should be created
 * @return the part of the k-set to be calculated on current task
 */
std::vector<std::vector<double>> createLocalKset(const std::vector<std::vector<double>> &kSetTotal);

void printMatricies(const std::vector<std::vector<double>> &kSet,
                            const std::vector<std::complex<double>> &HkAInHk0BasisKPathComplete,
                            const std::vector<std::complex<double>> &HkAAInHk0BasisKPathComplete,
                            const std::vector<std::complex<double>> &HkExpCouplingInHk0BasisKPathComplete,
                            const std::vector<std::complex<double>> &Hk0InHk0BasisKPathComplete);


#endif //TBG_OUTPUTUTILITIES_H
