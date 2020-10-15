project(UnitTests)

add_subdirectory(googletest)
include_directories(googletest/include googletest)

add_executable(runAllTests.out runAllTests.cpp trivialTest.cpp LinATest.cpp)
target_link_libraries(runAllTests.out gtest gtest_main)

add_test(NAME allTests COMMAND runAllTests.out)