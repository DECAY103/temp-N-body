cmake_minimum_required(VERSION 3.14)
project(AL-BH LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# -------------------------------------------------------------------
# Build options
# -------------------------------------------------------------------
option(ENABLE_MPI    "Enable MPI support"    ON)
option(ENABLE_OPENMP "Enable OpenMP support" ON)

# -------------------------------------------------------------------
# Include paths
# -------------------------------------------------------------------
include_directories(${CMAKE_SOURCE_DIR}/include)

# -------------------------------------------------------------------
# Source files — Core (serial + Morton + Octree + Partition + Load Balancer + Comm)
# -------------------------------------------------------------------
set(CORE_SOURCES
    src/particle_generator.cpp
    src/config.cpp
    src/morton.cpp
    src/octree.cpp
    src/octree_pointer.cpp
    src/force.cpp
    src/integrator.cpp
    src/metrics.cpp
    src/partition.cpp
    src/load_balancer.cpp
    src/communication.cpp
    src/simulation.cpp
)

# -------------------------------------------------------------------
# MPI configuration
# -------------------------------------------------------------------
if(ENABLE_MPI)
    find_package(MPI REQUIRED)
endif()

# -------------------------------------------------------------------
# Main executable
# -------------------------------------------------------------------
add_executable(al_bh src/main.cpp ${CORE_SOURCES})

# OpenMP
if(ENABLE_OPENMP)
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(al_bh PUBLIC OpenMP::OpenMP_CXX)
        message(STATUS "OpenMP found: ${OpenMP_CXX_FLAGS}")
    else()
        message(WARNING "OpenMP not found — building without thread parallelism")
    endif()
endif()

# MPI
if(ENABLE_MPI)
    target_include_directories(al_bh PRIVATE ${MPI_CXX_INCLUDE_DIRS})
    target_link_libraries(al_bh PUBLIC MPI::MPI_CXX)
    target_compile_definitions(al_bh PRIVATE ENABLE_MPI USE_MPI)
    message(STATUS "MPI enabled")
endif()

# -------------------------------------------------------------------
# Test executables
# -------------------------------------------------------------------
# Test Morton encoding
add_executable(test_morton tests/test_morton.cpp
    src/morton.cpp src/particle_generator.cpp)
target_include_directories(test_morton PRIVATE ${CMAKE_SOURCE_DIR}/include)

# Test Octree
add_executable(test_octree tests/test_octree.cpp
    src/octree.cpp src/octree_pointer.cpp src/morton.cpp src/particle_generator.cpp)
target_include_directories(test_octree PRIVATE ${CMAKE_SOURCE_DIR}/include)

# Test Force computation
add_executable(test_force tests/test_force.cpp
    src/force.cpp src/octree.cpp src/octree_pointer.cpp
    src/morton.cpp src/particle_generator.cpp)
target_include_directories(test_force PRIVATE ${CMAKE_SOURCE_DIR}/include)
if(ENABLE_OPENMP AND OpenMP_CXX_FOUND)
    target_link_libraries(test_force PUBLIC OpenMP::OpenMP_CXX)
endif()

# Test Domain Decomposition & Partitioning (Milestone 3)
add_executable(test_partition tests/test_partition.cpp
    src/partition.cpp src/morton.cpp src/particle_generator.cpp)
target_include_directories(test_partition PRIVATE ${CMAKE_SOURCE_DIR}/include)

# Test Dynamic Load Balancer (Milestone 6)
add_executable(test_load_balancer tests/test_load_balancer.cpp
    src/load_balancer.cpp src/partition.cpp src/morton.cpp src/particle_generator.cpp)
target_include_directories(test_load_balancer PRIVATE ${CMAKE_SOURCE_DIR}/include)

# Test MPI Communication (Milestone 5)
if(ENABLE_MPI)
    add_executable(test_mpi_comm tests/test_mpi_comm.cpp
        src/communication.cpp src/particle_generator.cpp)
    target_include_directories(test_mpi_comm PRIVATE ${CMAKE_SOURCE_DIR}/include ${MPI_CXX_INCLUDE_DIRS})
    target_link_libraries(test_mpi_comm PUBLIC MPI::MPI_CXX)
    target_compile_definitions(test_mpi_comm PRIVATE ENABLE_MPI USE_MPI)
    target_compile_options(test_mpi_comm PRIVATE -O2 -Wall)
endif()

# -------------------------------------------------------------------
# CTest Registration
# -------------------------------------------------------------------
enable_testing()
add_test(NAME test_morton COMMAND test_morton)
add_test(NAME test_octree COMMAND test_octree)
add_test(NAME test_force COMMAND test_force)
add_test(NAME test_partition COMMAND test_partition)
add_test(NAME test_load_balancer COMMAND test_load_balancer)
if(ENABLE_MPI)
    if(MPIEXEC_EXECUTABLE)
        add_test(NAME test_mpi_comm COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 $<TARGET_FILE:test_mpi_comm>)
    else()
        add_test(NAME test_mpi_comm COMMAND mpirun -np 2 $<TARGET_FILE:test_mpi_comm>)
    endif()
endif()

# -------------------------------------------------------------------
# Compiler flags
# -------------------------------------------------------------------
target_compile_options(al_bh PRIVATE -O2 -Wall -Wextra)
target_compile_options(test_morton PRIVATE -O2 -Wall)
target_compile_options(test_octree PRIVATE -O2 -Wall)
target_compile_options(test_force PRIVATE -O2 -Wall)
target_compile_options(test_partition PRIVATE -O2 -Wall)
target_compile_options(test_load_balancer PRIVATE -O2 -Wall)

# Ensure assertions are never stripped in tests
target_compile_options(test_morton PRIVATE -UNDEBUG)
target_compile_options(test_octree PRIVATE -UNDEBUG)
target_compile_options(test_force PRIVATE -UNDEBUG)
target_compile_options(test_partition PRIVATE -UNDEBUG)
target_compile_options(test_load_balancer PRIVATE -UNDEBUG)
if(ENABLE_MPI)
    target_compile_options(test_mpi_comm PRIVATE -UNDEBUG)
endif()

message(STATUS "AL-BH configured: MPI=${ENABLE_MPI}, OpenMP=${ENABLE_OPENMP}")
