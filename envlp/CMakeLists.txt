project(envlp)

add_library(${PROJECT_NAME} STATIC)

target_link_libraries(${PROJECT_NAME} rapidjson)
target_link_libraries(${PROJECT_NAME} Eigen)
target_link_libraries(${PROJECT_NAME} Catch2)
target_link_libraries(${PROJECT_NAME} pthread)
target_link_libraries(${PROJECT_NAME} gomp)

target_include_directories(${PROJECT_NAME} PUBLIC include)

add_subdirectory(src)
add_subdirectory(tests)
