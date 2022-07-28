cmake_minimum_required(VERSION 3.22.3)

project(MyNeuralNetwork)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_subdirectory(libs)

add_executable(MyNeuralNetwork src/main.cpp)

target_link_libraries(MyNeuralNetwork MyNNLib)