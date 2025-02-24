include(ExternalProject)

project(rapidjson)

ExternalProject_Add(${PROJECT_NAME}-external
        GIT_REPOSITORY https://github.com/Tencent/rapidjson.git
        GIT_TAG "v1.1.0"
        GIT_SHALLOW TRUE
        SOURCE_DIR ${THIRD_PARTY_DIR}/${PROJECT_NAME}
        BUILD_COMMAND ""
        INSTALL_COMMAND ""
        CONFIGURE_COMMAND "")
add_library(${PROJECT_NAME} INTERFACE)
add_dependencies(${PROJECT_NAME} ${PROJECT_NAME}-external)
target_include_directories(${PROJECT_NAME} SYSTEM INTERFACE ${THIRD_PARTY_DIR}/${PROJECT_NAME}/include)
