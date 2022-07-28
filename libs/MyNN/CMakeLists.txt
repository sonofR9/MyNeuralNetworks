#file (GLOB SRC_FILES_MyNN src/*.cpp)
set (INCLUDE_MyNN include)
file (GLOB HEADERS_MyNN include/*.hpp)

add_library(MyNNLib INTERFACE)

# target_sources(MyNNLib INTERFACE
#     FILE_SET HEADERS
#     BASE_DIRS include
#     FILES ${HEADERS_MyNN}
# )

target_include_directories(MyNNLib INTERFACE ${INCLUDE_MyNN})