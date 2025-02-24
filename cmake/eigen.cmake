include(ExternalProject)

project(Eigen)

ExternalProject_Add(${PROJECT_NAME}-external
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG "3.4.0"
        GIT_SHALLOW TRUE
        SOURCE_DIR ${THIRD_PARTY_DIR}/${PROJECT_NAME}
        BUILD_COMMAND ""
        INSTALL_COMMAND ""
        CONFIGURE_COMMAND "")
add_library(${PROJECT_NAME} INTERFACE)
add_dependencies(${PROJECT_NAME} ${PROJECT_NAME}-external)
target_include_directories(${PROJECT_NAME} SYSTEM INTERFACE ${THIRD_PARTY_DIR}/${PROJECT_NAME})
