project(envlp_tests)

add_executable(${PROJECT_NAME}
  main.cpp
  envopt_test.cpp
  lau_test.cpp
  lbfgs_test.cpp
  regression_env_test.cpp
  test_util.hpp
)
target_link_libraries(${PROJECT_NAME} envlp lbfgs_lib Catch2::Catch2)
target_compile_definitions(${PROJECT_NAME} PUBLIC TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../../data")

add_test(${PROJECT_NAME} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${PROJECT_NAME})
