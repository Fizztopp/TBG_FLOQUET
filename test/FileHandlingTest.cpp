#include <vector>
#include <string>

#include "gtest/gtest.h"
#include "FileHandling.h"

TEST(FileHandling, writingFileThrowsNoException) {

    std::string testFileName("testOutputFile.hdf5");
    std::vector<double> testData{1.0, 2.0, 3.0, 4.0};
    EXPECT_NO_THROW(writeReal1DArrayToHdf5(testData, testFileName););
}

TEST(FileHandling, writingArrayAndReadingItAgainIsConsistent) {
    std::string testFileName("testOutputFile.hdf5");
    unsigned long testDimension = 16;
    std::vector<std::complex<double>> testData(testDimension, std::complex<double>(0.0, 0.0));
    for (auto ind = 0ul; ind < testDimension; ++ind) {
        testData[ind] = 3.6 * double(ind) + II * 7.3 * std::complex<double>(double(ind - 3ul));
    }
    writeComplex2DArrayToHdf5(testData, testFileName, 4ul, 4ul);

    std::vector<std::complex<double>> readInData(testDimension, std::complex<double> (0.0, 0.0));

    readInComplex2DArray(readInData, testFileName);
    for(auto ind = 0ul; ind < testDimension; ++ind){
        EXPECT_NEAR(testData[ind].real(), readInData[ind].real(), 1e-15);
        EXPECT_NEAR(testData[ind].imag(), readInData[ind].imag(), 1e-15);
    }


}
