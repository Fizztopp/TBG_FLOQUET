#include <vector>
#include <string>
#include <fstream>

#include "Constants.h"
#include "Hk0.h"
#include "gtest/gtest.h"
#include "FileHandling.h"
#include "OutputUtilities.h"
#include "TestInitialization.h"
#include "mpi.h"
#include "mkl.h"

bool fileExists(const std::string &name);

TEST(NoChange, DataIsConsistentWithPreviousResuls) {

    //int myrank;
    //MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    //if (myrank != 0) {
    //    GTEST_SKIP();
    //}

    mkl_set_num_threads(1);

    EXPECT_EQ(SC, 4);

    std::vector<double> lvec(4ul, 0.0);
    std::vector<std::vector<double>> UNIT_CELL;

    testInitialization(lvec, UNIT_CELL);

    vector<dvec> K_PATH;
    ReadIn(K_PATH, "testInputFiles/k_path.dat");

    std::vector<std::vector<double>> reducedKPath;
    reducedKPath.insert(reducedKPath.end(), K_PATH.begin(), K_PATH.begin() + 10);
    K_PATH = reducedKPath;

    std::vector<std::complex<double>> HkAInHk0BasisKPath;
    std::vector<std::complex<double>> HkAAInHk0BasisKPath;
    std::vector<std::complex<double>> HkExpCouplingInHk0BasisKPath;
    std::vector<std::complex<double>> Hk0InHk0BasisKPath;

    matriciesInHk0Basis(HkAInHk0BasisKPath,
                        HkAAInHk0BasisKPath,
                        HkExpCouplingInHk0BasisKPath,
                        Hk0InHk0BasisKPath,
                        K_PATH,
                        lvec,
                        UNIT_CELL);

    std::string nameHk0 = createOutputString("testComparisonData/Hk0");
    std::string nameHkA = createOutputString("testComparisonData/HkA");
    std::string nameHkAA = createOutputString("testComparisonData/HkAA");
    std::string nameHkExpCoupling = createOutputString("testComparisonData/HkExpCoupling");

    if (fileExists(nameHk0) && fileExists(nameHkA) && fileExists(nameHkAA) && fileExists(nameHkExpCoupling)) {

        std::cout << "--- DATA FOR COMPARISON FOUND! ---" << '\n' << std::endl;

        std::vector<std::complex<double>> HkAInHk0BasisKPathRead(K_PATH.size() * NATOM * NATOM,
                                                                 std::complex<double>(0.0, 0.0));
        readInComplex3DArray(HkAInHk0BasisKPathRead, nameHkA);
        std::vector<std::complex<double>> HkAAInHk0BasisKPathRead(K_PATH.size() * NATOM * NATOM,
                                                                  std::complex<double>(0.0, 0.0));
        readInComplex3DArray(HkAAInHk0BasisKPathRead, nameHkAA);
        std::vector<std::complex<double>> Hk0InHk0BasisKPathRead(K_PATH.size() * NATOM * NATOM,
                                                                 std::complex<double>(0.0, 0.0));
        readInComplex3DArray(Hk0InHk0BasisKPathRead, nameHk0);
        std::vector<std::complex<double>> HkExpCouplingInHk0BasisKPathRead(K_PATH.size() * NATOM * NATOM,
                                                                           std::complex<double>(0.0, 0.0));
        readInComplex3DArray(HkExpCouplingInHk0BasisKPathRead, nameHkExpCoupling);

        for (auto ind = 0ul; ind < Hk0InHk0BasisKPath.size(); ++ind) {
            EXPECT_NEAR(Hk0InHk0BasisKPath[ind].real(), Hk0InHk0BasisKPathRead[ind].real(), 1e-15);
            EXPECT_NEAR(Hk0InHk0BasisKPath[ind].imag(), Hk0InHk0BasisKPathRead[ind].imag(), 1e-15);
            EXPECT_NEAR(HkAInHk0BasisKPath[ind].real(), HkAInHk0BasisKPathRead[ind].real(), 1e-15);
            EXPECT_NEAR(HkAInHk0BasisKPath[ind].imag(), HkAInHk0BasisKPathRead[ind].imag(), 1e-15);
            EXPECT_NEAR(HkAAInHk0BasisKPath[ind].real(), HkAAInHk0BasisKPathRead[ind].real(), 1e-15);
            EXPECT_NEAR(HkAAInHk0BasisKPath[ind].imag(), HkAAInHk0BasisKPathRead[ind].imag(), 1e-15);
            EXPECT_NEAR(HkExpCouplingInHk0BasisKPath[ind].real(), HkExpCouplingInHk0BasisKPathRead[ind].real(), 1e-15);
            EXPECT_NEAR(HkExpCouplingInHk0BasisKPath[ind].imag(), HkExpCouplingInHk0BasisKPathRead[ind].imag(), 1e-15);
        }

    } else {
        std::cout << "--- NO DATA FOR COMPARISON FOUND! - WILL WRITE DATA FOR THE FUTURE! ---" << '\n' << std::endl;
        writeComplex3DArrayToHdf5(HkAInHk0BasisKPath, nameHkA, K_PATH.size(), NATOM, NATOM);
        writeComplex3DArrayToHdf5(HkAAInHk0BasisKPath, nameHkAA, K_PATH.size(), NATOM, NATOM);
        writeComplex3DArrayToHdf5(HkExpCouplingInHk0BasisKPath, nameHkExpCoupling, K_PATH.size(), NATOM, NATOM);
        writeComplex3DArrayToHdf5(Hk0InHk0BasisKPath, nameHk0, K_PATH.size(), NATOM, NATOM);
    }

}


TEST(NoChange, MPICalculationAndDataDistributionIsConsistent) {

    mkl_set_num_threads(1);

    int numprocs;
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    if (myrank == 0) {
        std::cout << "Performing MPI-Test on " << numprocs << " processes!" << std::endl;
    }
    EXPECT_EQ(SC, 4);

    std::vector<double> lvec(4ul, 0.0);
    std::vector<std::vector<double>> UNIT_CELL;

    testInitialization(lvec, UNIT_CELL);

    vector<dvec> K_PATH;
    ReadIn(K_PATH, "testInputFiles/k_path.dat");

    std::vector<std::vector<double>> reducedKPath;
    reducedKPath.insert(reducedKPath.end(), K_PATH.begin(), K_PATH.begin() + 10);
    K_PATH = reducedKPath;

    std::vector<std::complex<double>> HkAInHk0BasisKPath;
    std::vector<std::complex<double>> HkAAInHk0BasisKPath;
    std::vector<std::complex<double>> HkExpCouplingInHk0BasisKPath;
    std::vector<std::complex<double>> Hk0InHk0BasisKPath;

    splitKSetAndObtainMatriciesOnMaster(K_PATH, lvec, UNIT_CELL, HkAInHk0BasisKPath, HkAAInHk0BasisKPath,
                                        HkExpCouplingInHk0BasisKPath, Hk0InHk0BasisKPath);

    if (myrank == 0) {

        std::string nameHk0 = createOutputString("testComparisonData/Hk0MPI");
        std::string nameHkA = createOutputString("testComparisonData/HkAMPI");
        std::string nameHkAA = createOutputString("testComparisonData/HkAAMPI");
        std::string nameHkExpCoupling = createOutputString("testComparisonData/HkExpCouplingMPI");

        if (fileExists(nameHk0) && fileExists(nameHkA) && fileExists(nameHkAA) && fileExists(nameHkExpCoupling)) {

            std::cout << "--- DATA FOR COMPARISON FOUND! ---" << '\n' << std::endl;

            std::vector<std::complex<double>> HkAInHk0BasisKPathRead(K_PATH.size() * NATOM * NATOM,
                                                                     std::complex<double>(0.0, 0.0));
            readInComplex3DArray(HkAInHk0BasisKPathRead, nameHkA);
            std::vector<std::complex<double>> HkAAInHk0BasisKPathRead(K_PATH.size() * NATOM * NATOM,
                                                                      std::complex<double>(0.0, 0.0));
            readInComplex3DArray(HkAAInHk0BasisKPathRead, nameHkAA);
            std::vector<std::complex<double>> Hk0InHk0BasisKPathRead(K_PATH.size() * NATOM * NATOM,
                                                                     std::complex<double>(0.0, 0.0));
            readInComplex3DArray(Hk0InHk0BasisKPathRead, nameHk0);
            std::vector<std::complex<double>> HkExpCouplingInHk0BasisKPathRead(K_PATH.size() * NATOM * NATOM,
                                                                               std::complex<double>(0.0, 0.0));
            readInComplex3DArray(HkExpCouplingInHk0BasisKPathRead, nameHkExpCoupling);

            for (auto ind = 0ul; ind < Hk0InHk0BasisKPath.size(); ++ind) {
                EXPECT_NEAR(Hk0InHk0BasisKPath[ind].real(), Hk0InHk0BasisKPathRead[ind].real(), 1e-12);
                EXPECT_NEAR(Hk0InHk0BasisKPath[ind].imag(), Hk0InHk0BasisKPathRead[ind].imag(), 1e-12);
                EXPECT_NEAR(HkAInHk0BasisKPath[ind].real(), HkAInHk0BasisKPathRead[ind].real(), 1e-12);
                EXPECT_NEAR(HkAInHk0BasisKPath[ind].imag(), HkAInHk0BasisKPathRead[ind].imag(), 1e-12);
                EXPECT_NEAR(HkAAInHk0BasisKPath[ind].real(), HkAAInHk0BasisKPathRead[ind].real(), 1e-12);
                EXPECT_NEAR(HkAAInHk0BasisKPath[ind].imag(), HkAAInHk0BasisKPathRead[ind].imag(), 1e-12);
                EXPECT_NEAR(HkExpCouplingInHk0BasisKPath[ind].real(), HkExpCouplingInHk0BasisKPathRead[ind].real(),
                            1e-15);
                EXPECT_NEAR(HkExpCouplingInHk0BasisKPath[ind].imag(), HkExpCouplingInHk0BasisKPathRead[ind].imag(),
                            1e-15);
            }

        } else {
            std::cout << "--- NO DATA FOR COMPARISON FOUND! - WILL WRITE DATA FOR THE FUTURE! ---" << '\n' << std::endl;
            writeComplex3DArrayToHdf5(HkAInHk0BasisKPath, nameHkA, K_PATH.size(), NATOM, NATOM);
            writeComplex3DArrayToHdf5(HkAAInHk0BasisKPath, nameHkAA, K_PATH.size(), NATOM, NATOM);
            writeComplex3DArrayToHdf5(HkExpCouplingInHk0BasisKPath, nameHkExpCoupling, K_PATH.size(), NATOM, NATOM);
            writeComplex3DArrayToHdf5(Hk0InHk0BasisKPath, nameHk0, K_PATH.size(), NATOM, NATOM);
        }
    }
    EXPECT_TRUE(true);

}


TEST(NoChange, DiagonalizationOfHk0IsConsistent) {

    //int myrank;
    //MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    //if (myrank != 0) {
    //    GTEST_SKIP();
    //}

    mkl_set_num_threads(1);

    EXPECT_EQ(SC, 4);

    std::vector<double> lvec(4ul, 0.0);
    std::vector<std::vector<double>> UNIT_CELL;

    testInitialization(lvec, UNIT_CELL);

    std::vector<double> testKPoint(3, 0.0);

    std::vector<std::complex<double>> basisVectors(NATOM * NATOM, 0.0);
    std::vector<double> eVals = Hk0DiagonalWithBasis(basisVectors, testKPoint, lvec, UNIT_CELL);

    //std::string nameEvals = createOutputString("testComparisonData/eValsHk0");
    std::string nameBasis = createOutputString("testComparisonData/basisHk0");

    if (fileExists(nameBasis)) {

        std::cout << "--- DATA FOR COMPARISON FOUND! ---" << '\n' << std::endl;

        //std::vector<std::complex<double>> eValsRead(NATOM, 0.0);
        //readInComplex1DArray(eValsRead, nameEvals);

        std::vector<std::complex<double>> basisRead(NATOM * NATOM, std::complex<double>(0.0, 0.0));
        readInComplex2DArray(basisRead, nameBasis);

        for (auto ind1 = 0ul; ind1 < NATOM; ++ind1) {
            for (auto ind2 = 0ul; ind2 < NATOM; ++ind2) {
                EXPECT_NEAR(basisVectors[ind1 * NATOM + ind2].real(), basisRead[ind1 * NATOM + ind2].real(), 1e-12);
                EXPECT_NEAR(basisVectors[ind1 * NATOM + ind2].imag(), basisRead[ind1 * NATOM + ind2].imag(), 1e-12);
            }
        }
    } else {
        std::cout << "--- NO DATA FOR COMPARISON FOUND! - WILL WRITE DATA FOR THE FUTURE! ---" << '\n'
                  << std::endl;
        writeComplex2DArrayToHdf5(basisVectors, nameBasis, NATOM, NATOM);
    }

}

bool fileExists(const std::string &name) {
    ifstream f(name.c_str());
    return f.good();
}