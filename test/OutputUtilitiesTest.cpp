#include <vector>
#include <complex>

#include "gtest/gtest.h"
#include "OutputUtilities.h"



TEST(OutputUtilityTests, MapRankToChunksizeWorks){

    std::vector<int> myranks{0, 0, 1, 0, 0, 1};
    std::vector<int> numprocs{1, 2, 2, 2, 2, 2};
    std::vector<int> totalSize{197, 197, 197, 2, 1, 1};
    std::vector<int> expectedResult{197, 99, 98, 1, 1, 0};

    EXPECT_EQ(myranks.size(), numprocs.size()) << "Input wrong";
    EXPECT_EQ(myranks.size(), totalSize.size()) << "Input wrong";

    for(auto ind = 0ul; ind < myranks.size(); ++ind){
        int actualResultTemp = mapRankToChunksize(myranks[ind], numprocs[ind], totalSize[ind]);
        EXPECT_EQ(expectedResult[ind], actualResultTemp);
    }
}

TEST(OuputUtilitiyTests, startChunkIndicesMPIWorks){

    std::vector<int> numprocs{1, 2, 2};
    std::vector<int> totalSize{197, 197, 2};

    std::vector<std::vector<unsigned long>> expectedResult{{0, 197}, {0, 99, 197}, {0, 1, 2}};

    for(auto ind = 0ul; ind < numprocs.size(); ++ind){
        std::vector<unsigned long> actualResultTemp = startChunkIndiciesMPI(numprocs[ind], totalSize[ind]);
        EXPECT_EQ(expectedResult[ind], actualResultTemp);
    }
}

