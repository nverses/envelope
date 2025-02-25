# python_test(
#   PACKAGE
#     pyenvlp_modules
#   MODULES
#     test_envlp.py
#   LINK_PACKAGES
#     pyenvlp
# )

enable_testing()
add_test(NAME pyenvlp_tests
  COMMAND python3 -m pytest ${CMAKE_CURRENT_SOURCE_DIR}
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../../lib
)
set_property(TEST pyenvlp_tests
             PROPERTY ENVIRONMENT TEST_DATA_DIR=${TEST_DATA_DIR})

