#include <vector>
#include <string>

#include "gtest/gtest.h"
#include "FileHandling.h"

#include "mpi.h"

TEST(FileHandling, writingFileThrowsNoException) {

    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    if (myrank == 0) {

        std::string testFileName("testOutputFile.hdf5");
        std::vector<double> testData{1.0, 2.0, 3.0, 4.0};
        EXPECT_NO_THROW(writeReal1DArrayToHdf5(testData, testFileName));
    }

    MPI_Barrier(MPI_COMM_WORLD);
}

TEST(FileHandling, writingArrayAndReadingItAgainIsConsistent) {

    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    if (myrank == 0) {

        std::string testFileName("testOutputFile.hdf5");
        unsigned long testDimension = 27;
        std::vector<std::complex<double>> testData(testDimension, std::complex<double>(0.0, 0.0));
        for (auto ind = 0ul; ind < testDimension; ++ind) {
            testData[ind] = 3.6 * double(ind) + II * 7.3 * std::complex<double>(double(ind - 3ul));
        }
        writeComplex3DArrayToHdf5(testData, testFileName, 3ul, 3ul, 3ul);

        std::vector<std::complex<double>> readInData(testDimension, std::complex<double>(0.0, 0.0));

        readInComplex3DArray(readInData, testFileName);
        for (auto ind = 0ul; ind < testDimension; ++ind) {
            EXPECT_NEAR(testData[ind].real(), readInData[ind].real(), 1e-15);
            EXPECT_NEAR(testData[ind].imag(), readInData[ind].imag(), 1e-15);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

}
