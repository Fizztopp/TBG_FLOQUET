#include <vector>
#include <complex>

#include "gtest/gtest.h"
#include "OutputUtilities.h"
#include "Constants.h"

#include "mpi.h"


TEST(OutputUtilityTests, MapRankToChunksizeWorks) {

    std::vector<int> myranks{0, 0, 1, 0, 0, 1};
    std::vector<int> numprocs{1, 2, 2, 2, 2, 2};
    std::vector<int> totalSize{197, 197, 197, 2, 1, 1};
    std::vector<int> expectedResult{197, 99, 98, 1, 1, 0};

    EXPECT_EQ(myranks.size(), numprocs.size()) << "Input wrong";
    EXPECT_EQ(myranks.size(), totalSize.size()) << "Input wrong";

    for (auto ind = 0ul; ind < myranks.size(); ++ind) {
        int actualResultTemp = mapRankToChunksize(myranks[ind], numprocs[ind], totalSize[ind]);
        EXPECT_EQ(expectedResult[ind], actualResultTemp);
    }
}

TEST(OuputUtilitiyTests, startChunkIndicesMPIWorks) {

    std::vector<int> numprocs{1, 2, 2};
    std::vector<int> totalSize{197, 197, 2};

    std::vector<std::vector<unsigned long>> expectedResult{{0, 197},
                                                           {0, 99, 197},
                                                           {0, 1,  2}};

    for (auto ind = 0ul; ind < numprocs.size(); ++ind) {
        std::vector<unsigned long> actualResultTemp = startChunkIndiciesMPI(numprocs[ind], totalSize[ind]);
        EXPECT_EQ(expectedResult[ind], actualResultTemp);
    }
}

TEST(OutputUtilityTests, creatLocalKSetWorks) {

    int numprocs;
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    const size_t kSetSize = 13;
    std::vector<std::vector<double>> kSet;
    for (auto ind = 0ul; ind < kSetSize; ++ind) {
        kSet.push_back({double(ind), double(ind), double(ind)});
    }
    std::vector<std::vector<double>> localKSet = createLocalKset(kSet);

    switch (numprocs) {
        case 1:
            EXPECT_EQ(localKSet.size(), kSetSize);
            break;
        case 2:
            switch (myrank) {
                case 0:
                    EXPECT_EQ(localKSet.size(), kSetSize / 2ul + 1ul);
                    break;
                case 1:
                    EXPECT_EQ(localKSet.size(), kSetSize / 2ul);
                    break;
            }
            break;
        case 3:
            switch (myrank) {
                case 0:
                    EXPECT_EQ(localKSet.size(), kSetSize / 3ul + 1ul);
                    break;
                case 1:
                    EXPECT_EQ(localKSet.size(), kSetSize / 3ul);
                    break;
                case 2:
                    EXPECT_EQ(localKSet.size(), kSetSize / 3ul);
                    break;
            }
            break;
        default:
            EXPECT_TRUE(true);
    }
}

TEST(OutputUtilityTests, collectOnMasterWorks) {

    const size_t totalSize = 13;
    int numprocs;
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    size_t localSize = mapRankToChunksize(myrank, numprocs, totalSize);
    std::vector<std::complex<double>> localVec(localSize * NATOM * NATOM, std::complex<double> (0.0, 0.0));

    for (auto k = 0ul; k < localSize; ++k) {
        for (auto band = 0ul; band < NATOM * NATOM; ++band) {
            localVec[k * NATOM * NATOM + band] = std::complex<double>(double(k), 0.0);
        }
    }

    std::vector<std::complex<double>> totalVec;
    if (myrank == 0) {
        totalVec.reserve(totalSize * NATOM * NATOM);
    }

    collectOnMaster(totalVec, localVec, totalSize);

    if (myrank == 0) {
        if (numprocs == 1) {
            for (auto k = 0ul; k < totalSize; ++k) {
                EXPECT_EQ(totalVec[k * NATOM * NATOM], std::complex<double>(double(k), 0.0));
            }
        }
        if (numprocs == 2) {
            for (auto k = 0ul; k < totalSize; ++k) {
                if (k < 7) {
                    EXPECT_EQ(totalVec[k * NATOM * NATOM], std::complex<double>(double(k), 0.0));
                } else {
                    EXPECT_EQ(totalVec[k * NATOM * NATOM], std::complex<double>(double(k - 7), 0.0));
                }
            }
        } else {
            EXPECT_TRUE(true);
        }
    }
}


