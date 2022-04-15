project(UnitTests)

add_subdirectory(googletest)
include_directories(googletest/include googletest)
include_directories(../include)

set(project_sources
        ../src/Diagonalization.cpp
        ../src/Hk0.cpp
        ../src/EqBerryCurvature.cpp
        ../src/FloquetBerryCurvature.cpp
        ../src/FileHandling.cpp
        ../src/Hk.cpp
        ../src/HkA.cpp
        ../src/OutputUtilities.cpp
        )

set(test_sources
        runAllTests.cpp
        trivialTest.cpp
        LinATest.cpp
				#DiagonalizationTest.cpp
        FileHandlingTest.cpp
				#NoChangeTest.cpp
        TestInitialization.cpp
        OutputUtilitiesTest.cpp
        )

add_executable(runAllTests.out ${project_sources} ${test_sources})
target_link_libraries(runAllTests.out gtest gtest_main)
target_link_libraries(runAllTests.out ${HDF5_LIBRARIES})
target_link_libraries(runAllTests.out ${MPI_CXX_LIBRARIES})


add_test(NAME allTests COMMAND runAllTests.out)
