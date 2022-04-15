#include <vector>
#include <string>
#include <complex>
#include <cassert>

#include "FileHandling.h"
#include "OutputUtilities.h"
#include "Hk0.h"
#include "HkA.h"

#include "mpi.h"


/**
 *
 * @param kSet vector of 3-d kpoint vectors for which matricies should be calculated
 * @param name for naming the outputted file
 * @param lvec super-lattice vector
 * @param UNIT_CELL atomic positions in unit-cell
 */
void generateMatrixOutputForKSet(const std::vector<std::vector<double>> &kSet,
                                 const std::vector<double> &lvec,
                                 const std::vector<std::vector<double>> &UNIT_CELL) {

    std::vector<std::complex<double>> HkAInHk0BasisKPathComplete;
    std::vector<std::complex<double>> HkAAInHk0BasisKPathComplete;
    std::vector<std::complex<double>> HkExpCouplingInHk0BasisKPathComplete;
    std::vector<std::complex<double>> Hk0InHk0BasisKPathComplete;
    splitKSetAndObtainMatriciesOnMaster(kSet, lvec, UNIT_CELL, HkAInHk0BasisKPathComplete, HkAAInHk0BasisKPathComplete,
                                        HkExpCouplingInHk0BasisKPathComplete, Hk0InHk0BasisKPathComplete);

    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    if (myrank == 0) {
        printMatricies(kSet, HkAInHk0BasisKPathComplete, HkAAInHk0BasisKPathComplete,
                       HkExpCouplingInHk0BasisKPathComplete,
                       Hk0InHk0BasisKPathComplete);
    }
}

void splitKSetAndObtainMatriciesOnMaster(const vector<std::vector<double>> &kSet, const vector<double> &lvec,
                                         const vector<std::vector<double>> &UNIT_CELL,
                                         vector<std::complex<double>> &HkAInHk0BasisKPathComplete,
                                         vector<std::complex<double>> &HkAAInHk0BasisKPathComplete,
                                         vector<std::complex<double>> &HkExpCouplingInHk0BasisKPathComplete,
                                         vector<std::complex<double>> &Hk0InHk0BasisKPathComplete) {

    vector<complex<double>> HkAInHk0BasisKPath;
    vector<complex<double>> HkAAInHk0BasisKPath;
    vector<complex<double>> HkExpCouplingInHk0BasisKPath;
    vector<complex<double>> Hk0InHk0BasisKPath;

    vector<vector<double>> localKSet = createLocalKset(kSet);

    matriciesInHk0Basis(HkAInHk0BasisKPath,
                        HkAAInHk0BasisKPath,
                        HkExpCouplingInHk0BasisKPath,
                        Hk0InHk0BasisKPath,
                        localKSet,
                        lvec,
                        UNIT_CELL);
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    if (myrank == 0) {
        Hk0InHk0BasisKPathComplete.reserve(kSet.size() * NATOM * NATOM);
        HkAInHk0BasisKPathComplete.reserve(kSet.size() * NATOM * NATOM);
        HkAAInHk0BasisKPathComplete.reserve(kSet.size() * NATOM * NATOM);
        HkExpCouplingInHk0BasisKPathComplete.reserve(kSet.size() * NATOM * NATOM);
    }

    collectOnMaster(Hk0InHk0BasisKPathComplete, Hk0InHk0BasisKPath, kSet.size());
    collectOnMaster(HkAInHk0BasisKPathComplete, HkAInHk0BasisKPath, kSet.size());
    collectOnMaster(HkAAInHk0BasisKPathComplete, HkAAInHk0BasisKPath, kSet.size());
    collectOnMaster(HkExpCouplingInHk0BasisKPathComplete, HkExpCouplingInHk0BasisKPath, kSet.size());
}


/**
 *
 * @param kSetTotal the complete k-set for which output should be created
 * @return the part of the k-set to be calculated on current task
 */
std::vector<std::vector<double>> createLocalKset(const std::vector<std::vector<double>> &kSetTotal) {
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    int numprocs;
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

    std::vector<unsigned long> startIndices = startChunkIndiciesMPI(numprocs, kSetTotal.size());
    std::vector<std::vector<double>> localKSet;
    localKSet.insert(localKSet.end(), kSetTotal.begin() + startIndices[myrank],
                     kSetTotal.begin() + startIndices[myrank + 1]);

    return localKSet;
}


/**
 *
 * @param collectMat matrix of all k-points that is to be collected from all ranks
 * @param localMat matrix that was previously calculated on each rank locally
 */
void collectOnMaster(std::vector<std::complex<double>> &collectMat,
                     const std::vector<std::complex<double>> &localMat,
                     const size_t totalSize) {

    MPI_Status status;
    int numprocs;
    int myrank;
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    for (int rank = 0; rank < numprocs; ++rank) {
        if (rank == 0) {
            if (myrank == 0) {
                collectMat.insert(collectMat.end(), localMat.begin(), localMat.end());
            }
        } else {
            if (myrank == 0) {
                int chunkSizeToReceive = mapRankToChunksize(rank, numprocs, totalSize) * NATOM * NATOM;
                std::vector<std::complex<double>> tempVec (chunkSizeToReceive, std::complex<double> (0.0, 0.0));
                MPI_Recv(&tempVec[0], chunkSizeToReceive, MPI_DOUBLE_COMPLEX, rank, rank, MPI_COMM_WORLD,
                         &status);
                collectMat.insert(collectMat.end(), tempVec.begin(), tempVec.end());
            } else {
                if (myrank == rank) {
                    assert(localMat.size() ==
                           mapRankToChunksize(rank, numprocs, totalSize) * NATOM * NATOM);
                    MPI_Send(&localMat[0], localMat.size(), MPI_DOUBLE_COMPLEX, 0, rank, MPI_COMM_WORLD);
                }
            }
        }
    }


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

    std::vector<double> ones (NATOM, 1.0);
    std::vector<std::complex<double>> unitMatrix (NATOM * NATOM, std::complex<double> (0.0, 0.0));
    diagonalToMatrix(unitMatrix, ones, NATOM);

    for (auto k = 0ul; k < kSet.size(); ++k) {
        eValsHk0Temp = Hk0DiagonalWithBasis(basisVectorsTemp, kSet[k], lvec, UNIT_CELL);

        //check if difference is in basis vectors
        //basisVectorsTemp = unitMatrix;

        //for(auto kk = 0ul; kk < NATOM; ++kk){
        //    std::cout << eValsHk0Temp[kk] << '\n';
        //    std::cout << basisVectorsTemp[kk * NATOM] << '\n';
        //}

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
}

void printMatricies(const vector<std::vector<double>> &kSet,
                    const vector<std::complex<double>> &HkAInHk0BasisKPathComplete,
                    const vector<std::complex<double>> &HkAAInHk0BasisKPathComplete,
                    const vector<std::complex<double>> &HkExpCouplingInHk0BasisKPathComplete,
                    const vector<std::complex<double>> &Hk0InHk0BasisKPathComplete) {

    string nameHk0 = createOutputString("Data/Hk0");
    string nameHkA = createOutputString("Data/HkA");
    string nameHkAA = createOutputString("Data/HkAA");
    string nameHkExpCoupling = createOutputString("Data/HkExpCoupling");

    writeComplex3DArrayToHdf5(Hk0InHk0BasisKPathComplete, nameHk0, kSet.size(), NATOM, NATOM);
    writeComplex3DArrayToHdf5(HkAInHk0BasisKPathComplete, nameHkA, kSet.size(), NATOM, NATOM);
    writeComplex3DArrayToHdf5(HkAAInHk0BasisKPathComplete, nameHkAA, kSet.size(), NATOM, NATOM);
    writeComplex3DArrayToHdf5(HkExpCouplingInHk0BasisKPathComplete, nameHkExpCoupling, kSet.size(), NATOM, NATOM);

    flattenAndOutputKArray(kSet, "Data/KSetKPoints.hdf5");
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

/**
 *
 * @param numprocs number of mpi processes
 * @param kSetSize number of k-points to calculate
 * @return k-point indices at which each task starts its local k-chunk
 */
std::vector<unsigned long> startChunkIndiciesMPI(const int numprocs, const unsigned long kSetSize) {

    std::vector<unsigned long> startChunkIndicies;
    int pos = 0;
    for (int proc = 0; proc < numprocs; ++proc) {
        startChunkIndicies.push_back(pos);
        pos += mapRankToChunksize(proc, numprocs, int(kSetSize));
    }
    startChunkIndicies.push_back(pos);
    assert(*(startChunkIndicies.end() - 1) == kSetSize);

    return startChunkIndicies;
}

int mapRankToChunksize(const int rank, const int numprocs, const int totalSize) {
    int rest = totalSize % numprocs;
    if (rank < rest) {
        return int(totalSize / numprocs) + 1;
    } else {
        return int(totalSize / numprocs);
    }
}