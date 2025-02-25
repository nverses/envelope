cmake_minimum_required(VERSION 3.14)
project(overall)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Debug")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_EXPORT_COMPILE_COMMANDS True)
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -g -Wall -Werror -fPIC")

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

if(CMAKE_BUILD_TYPE MATCHES "Debug")
  set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -fno-omit-frame-pointer")
endif()

set(THIRD_PARTY_DIR ${CMAKE_BINARY_DIR}/third_party)

find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED)

option(ENVELOPE_ENABLE_BENCHMARKS "Add catch2 benchmarks" OFF)

include(cmake/rapidjson.cmake)
include(cmake/catch2.cmake)
include(cmake/eigen.cmake)
include(cmake/lbfgs.cmake)
include(cmake/pybind11.cmake)

message("-- CMAKE Build Type is ${CMAKE_BUILD_TYPE}")

# use, i.e. don't skip the full RPATH for the build tree
set(CMAKE_SKIP_BUILD_RPATH FALSE)

# when building, don't use the install RPATH already
# (but later on when installing)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

# add the automatically determined parts of the RPATH
# which point to directories outside the build tree to the install RPATH
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

enable_testing()
 
add_subdirectory(envlp)
add_subdirectory(pyenvlp)
