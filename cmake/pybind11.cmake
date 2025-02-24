include(ExternalProject)

project(pybind11)

ExternalProject_Add(${PROJECT_NAME}-external
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG "v2.13.6"
        GIT_SHALLOW TRUE
        SOURCE_DIR ${THIRD_PARTY_DIR}/${PROJECT_NAME}
        BUILD_COMMAND ""
        INSTALL_COMMAND ""
        CONFIGURE_COMMAND "")
add_library(${PROJECT_NAME} INTERFACE)
add_dependencies(${PROJECT_NAME} ${PROJECT_NAME}-external)
target_include_directories(${PROJECT_NAME} SYSTEM INTERFACE ${THIRD_PARTY_DIR}/${PROJECT_NAME}/include)


set(pybind11_DIR ${CMAKE_BINARY_DIR}/third_party/pybind11)
option(PYBIND11_FINDPYTHON, ON)

find_package(Python COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG)

