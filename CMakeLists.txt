#specefiy required version - this is the newest one
cmake_minimum_required(VERSION 3.11)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
#using intel compiler
set(CMAKE_CXX_COMPILER "icpc")
set(CMAKE_C_COMPILER "icc")

enable_language(CXX)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

enable_testing()

find_package(HDF5 REQUIRED COMPONENTS CXX)

set(project_sources
        src/TBG.cpp
        src/Diagonalization.cpp
        src/Hk0.cpp
        src/EqBerryCurvature.cpp
        src/FloquetBerryCurvature.cpp
        src/FileHandling.cpp
        src/Hk.cpp
        src/HkA.cpp
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
        )

include_directories("./include")
#include MKL
include_directories("/home/chris/intel/compilers_and_libraries_2020.1.217/linux/mpi/intel64/include")
include_directories(${HDF5_INCLUDE_DIRS})

#add compiler flags
set(C_flags "-Ofast -Wall -Wextra -pedantic -mkl -qopenmp -D 'NO_MPI'")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${C_flags}")

#set project name
project(TBG)
#add an executable - using variables
add_executable(${PROJECT_NAME}.out ${project_sources} ${header_files})

target_link_libraries(${PROJECT_NAME}.out ${HDF5_LIBRARIES})

add_subdirectory(test)