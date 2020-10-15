#include "gtest/gtest.h"

int main(int argc, char** argv)
{

    //initialization of test library
    ::testing::InitGoogleTest(&argc, argv);

    //perform all tests
    return RUN_ALL_TESTS();
}
