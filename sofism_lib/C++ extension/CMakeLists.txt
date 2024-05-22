cmake_minimum_required(VERSION 3.12)
project(correlation_project LANGUAGES CXX)

add_executable(correlations corr.cpp)
# Find Python
find_package(Python3 COMPONENTS Development NumPy REQUIRED)

# Find Eigen
include_directories(C:/Users/ngtua/Desktop/Corr/eigen)
include_directories(C:/Users/ngtua/Desktop/Corr/build/pybind11/include)
include_directories(C:/Users/ngtua/AppData/Local/Programs/Python/Python312/include)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development NumPy)
include_directories(${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})
find_package(Eigen3 REQUIRED)

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add the corr.cpp file
#add_subdirectory(pybind11)
find_package(pybind11 REQUIRED)
pybind11_add_module(correlations corr.cpp)

# Link Eigen
target_link_libraries(correlations PRIVATE Eigen3::Eigen)

# Link Python and NumPy

target_include_directories(correlations PRIVATE ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})
target_link_libraries(correlations PRIVATE Python3::Python Python3::NumPy)
