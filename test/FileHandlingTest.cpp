#include <vector>
#include <string>

#include "gtest/gtest.h"
#include "FileHandling.h"

TEST(FileHandling, writingFileThrowsNoException) {

    std::string testFileName("testOutputFile.hdf5");

    std::vector<double> testData{1.0, 2.0, 3.0, 4.0};

    EXPECT_NO_THROW(writeReal1DArrayToHdf5(testData, testFileName););
}
