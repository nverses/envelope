include(FetchContent)

FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11
    GIT_TAG        v2.13.6
)

FetchContent_GetProperties(pybind11)
if(NOT pybind11_POPULATED)
    FetchContent_Populate(pybind11)
    add_subdirectory(${pybind11_SOURCE_DIR} ${pybind11_BINARY_DIR})
endif()
FetchContent_MakeAvailable(pybind11)


set(pybind11_DIR "${pybind11_SOURCE_DIR}")
option(PYBIND11_FINDPYTHON, ON)
message("-- pybind11_DIR=${pybind11_DIR} pybind11_FOUND=${pybind11_FOUND} pybind11_VERSION=${pybind11_VERSION}") 
