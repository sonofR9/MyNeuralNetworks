cmake_minimum_required(VERSION 3.22.3)

project(MyNeuralNetwork)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_subdirectory(libs build/libs)

add_executable($(PROJECT_NAME) src/main.cpp)

set (INCLUDE_MyNN libs/MyNN/include)

target_link_libraries($(PROJECT_NAME) PUBLIC MyNNLib)