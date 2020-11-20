#specefiy required version - this is the newest one
cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
#using intel compiler
set(CMAKE_CXX_COMPILER "mpiicpc")
set(CMAKE_C_COMPILER "mpiicc")

enable_language(CXX)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

enable_testing()

find_package(HDF5 REQUIRED COMPONENTS CXX)
find_package(MPI REQUIRED)

#set(ENV{LD_LIBRARY_PATH} ENV{MKL_ROOT}/lib/intel64_lin)
#set(ENV{LIBRARY_PATH} ENV{MKL_ROOT}/lib/intel64_lin)

set(project_sources
        src/TBG.cpp
        src/Diagonalization.cpp
        src/Hk0.cpp
        src/EqBerryCurvature.cpp
        src/FloquetBerryCurvature.cpp
        src/FileHandling.cpp
        src/Hk.cpp
        src/HkA.cpp
        src/OutputUtilities.cpp
        )

set(header_files
        include/Constants.h
        include/Diagonalization.h
        include/InlineFunctions.h
        include/MatrixMultiplication.h
        include/Hk0.h
        include/EqBerryCurvature.h
        include/FloquetBerryCurvature.h
        include/FileHandling.h
        include/Hk.h
        include/HkA.h
        include/OutputUtilities.h
        include/TestInitialization.h)

include_directories("./include")
include_directories(SYSTEM ${MPI_INCLUDE_PATH})
include_directories(${HDF5_INCLUDE_DIRS})
include_directories(${MKL_INCLUDE_DIRS})


#add compiler flags
set(C_flags "-Ofast -Wall -Wextra -pedantic -mkl -qopenmp")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${C_flags}")

#set project name
project(TBG)
#add an executable - using variables
add_executable(${PROJECT_NAME}.out ${project_sources} ${header_files})

target_link_libraries(${PROJECT_NAME}.out ${HDF5_LIBRARIES})
target_link_libraries(${PROJECT_NAME}.out ${MPI_CXX_LIBRARIES})



add_subdirectory(test)
