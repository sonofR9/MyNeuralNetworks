file (GLOB_RECURSE SRC_FILES_MyNN libs/MyNN/src/*.cpp)
set (INCLUDE_MyNN libs/lib1/include)

add_library(MyNNLib STATIC $(SRC_FILES_MyNN))

target_include_directories($(PROJECT_NAME) PUBLIC $(INCLUDE_MyNN))