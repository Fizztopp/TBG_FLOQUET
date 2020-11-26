#include "gtest/gtest.h"

#include "mpi.h"
int main(int argc, char** argv)
{

    //initialization of test library
    ::testing::InitGoogleTest(&argc, argv);

    MPI_Init(&argc, &argv);
    //perform all tests
    int returnVal = RUN_ALL_TESTS();

    MPI_Finalize();

    return returnVal;
}
