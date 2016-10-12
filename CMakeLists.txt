cmake_minimum_required(VERSION 2.8.11)
project(aircam)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -march=native")

include_directories(fmtlib)

file(GLOB fmt_sources fmtlib/fmt/*.cc)
add_library(fmt ${fmt_sources})

set(sources devils_checkerboard.cc)
add_executable(devils_checkerboard
               ${sources})
target_link_libraries(devils_checkerboard fmt)

add_custom_target(format
                  COMMAND clang-format-3.6 -i -style=file
                  ${sources})

