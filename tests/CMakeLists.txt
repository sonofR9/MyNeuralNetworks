add_executable(hello_test
    TestActivations.cpp
    TestNeurons.cpp
    TestModel.cpp
  )

target_link_libraries(hello_test PUBLIC
    GTest::gtest_main
    MyNNLib
  )

include(GoogleTest)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

gtest_discover_tests(hello_test)