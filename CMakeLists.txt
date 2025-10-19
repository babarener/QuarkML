cmake_minimum_required(VERSION 3.20)
project(QuarkML VERSION 0.1.0 LANGUAGES CXX)

# Require C++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Output directories
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Add source and include directories
include_directories(${PROJECT_SOURCE_DIR}/include)

# Gather sources (currently empty but ready)
file(GLOB_RECURSE QUARKML_SOURCES
    src/core/*.cpp
    src/linear/*.cpp
    src/cluster/*.cpp
    src/tree/*.cpp
    src/io/*.cpp
    src/preprocessing/*.cpp
)

# Build static library
add_library(QuarkML STATIC ${QUARKML_SOURCES})

target_include_directories(QuarkML PUBLIC ${PROJECT_SOURCE_DIR}/include)

# Example: linear_regression_basic  
#add_executable(example_linear examples/linear_regression_basic.cpp)
#target_link_libraries(example_linear PRIVATE QuarkML)

# Example: linear_regression_ridge_demo
add_executable(linear_regression_ridge_demo examples/linear_regression_ridge_demo.cpp)
target_link_libraries(linear_regression_ridge_demo PRIVATE QuarkML)