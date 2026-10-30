cmake_minimum_required(VERSION 3.14)
project(AL-BH LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# -------------------------------------------------------------------
# Build options
# -------------------------------------------------------------------
option(ENABLE_MPI    "Enable MPI support"    OFF)
option(ENABLE_OPENMP "Enable OpenMP support" ON)

# -------------------------------------------------------------------
# Include paths
# -------------------------------------------------------------------
include_directories(${CMAKE_SOURCE_DIR}/include)

# -------------------------------------------------------------------
# Source files — Milestone 1+2 core (serial + Morton + Octree)
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
    src/simulation.cpp
)

# -------------------------------------------------------------------
# MPI sources (Milestones 3, 5, 6) — added when ENABLE_MPI is ON
# -------------------------------------------------------------------
if(ENABLE_MPI)
    find_package(MPI REQUIRED)
    # Future: add partition.cpp, communication.cpp, load_balancer.cpp
    set(MPI_SOURCES
        # src/partition.cpp
        # src/communication.cpp
        # src/load_balancer.cpp
    )
    list(APPEND CORE_SOURCES ${MPI_SOURCES})
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
    target_compile_definitions(al_bh PRIVATE USE_MPI)
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

# -------------------------------------------------------------------
# Compiler flags
# -------------------------------------------------------------------
target_compile_options(al_bh PRIVATE -O2 -Wall -Wextra)
target_compile_options(test_morton PRIVATE -O2 -Wall)
target_compile_options(test_octree PRIVATE -O2 -Wall)
target_compile_options(test_force PRIVATE -O2 -Wall)
target_compile_options(test_partition PRIVATE -O2 -Wall)
target_compile_options(test_load_balancer PRIVATE -O2 -Wall)

message(STATUS "AL-BH configured: MPI=${ENABLE_MPI}, OpenMP=${ENABLE_OPENMP}")
