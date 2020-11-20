#include <vector>
#include <string>
#include <fstream>

#include "gtest/gtest.h"
#include "FileHandling.h"
#include "OutputUtilities.h"
#include "TestInitialization.h"

bool fileExists(const std::string &name);

TEST(NoChange, DataIsConsistentWithPreviousResuls) {

    EXPECT_EQ(SC, 4);

    std::vector<double> lvec(4ul, 0.0);
    std::vector<std::vector<double>> UNIT_CELL;

    testInitialization(lvec, UNIT_CELL);

    vector<dvec> K_PATH;
    ReadIn(K_PATH, "testInputFiles/k_path.dat");


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

        for(auto ind = 0ul; ind < Hk0InHk0BasisKPath.size(); ++ind){
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

bool fileExists(const std::string &name) {
    ifstream f(name.c_str());
    return f.good();
}