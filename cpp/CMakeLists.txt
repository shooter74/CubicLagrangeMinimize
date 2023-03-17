cmake_minimum_required(VERSION 3.0)

project(CubicLagrangeMinimize)

# Set the C++ standard to C++11
set(CMAKE_CXX_STANDARD 11)

# Find the Eigen library
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIRS})

# Add the source files
set(SOURCES
    src/tests.cpp
    src/CubicLagrangeMinimize.cpp
)

# Add an executable target
add_executable(CubicLagrangeMinimize ${SOURCES})

# Link against the Eigen library
target_link_libraries(CubicLagrangeMinimize Eigen3::Eigen)