cmake_minimum_required(VERSION 3.25...4.2)

project(
  top.lbm
  LANGUAGES CXX
  VERSION 0.1.0
  DESCRIPTION "Hybrid MPI+OpenMP parallel D2Q9 lattice Boltzmann solver for Kármán vortex street flow"
  HOMEPAGE_URL "https://dssgabriel.github.io/CHP203-TOP/project"
)

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-builtin-synchronization")

find_package(MPI 3.0 REQUIRED)
include(cmake/third-party.cmake)

add_library(top.lbm-lib STATIC)
add_executable(top.lbm-exe)
add_executable(top.display)

add_subdirectory(src/lbm)
add_subdirectory(src/bin)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(top.lbm-exe PRIVATE -Wall -Wextra -g -DDEBUG -fno-inline-functions)
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(top.lbm-exe PRIVATE -g -O3 -march=native)
endif()

