cmake_minimum_required(VERSION 3.22.3)

project(MyNeuralNetwork)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set (TESTING ON)

add_subdirectory(libs)


if (TESTING)
    enable_testing()
    add_subdirectory(googletest)
    add_subdirectory(tests)
    
else ()

endif()

set(Headers

)

set(Sources
    src/main.cpp
)

add_executable(MyNeuralNetwork ${Sources})

target_link_libraries(MyNeuralNetwork MyNNLib)