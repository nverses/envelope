find_package(Python COMPONENTS Interpreter Development)
# find_package(pybind11 CONFIG)

# pybind11 method:
pybind11_add_module(_pyenvlp
  ${CMAKE_CURRENT_LIST_DIR}/pyenvlp/pyenvlp.hpp
  ${CMAKE_CURRENT_LIST_DIR}/pyenvlp/pyenvlp.cpp
  ${CMAKE_CURRENT_LIST_DIR}/pyenvlp/pyenvlp_module.cpp
  ${PYTHON_FILES}
)
target_link_libraries(_pyenvlp PRIVATE pybind11::module envlp Eigen rapidjson)
set_target_properties(_pyenvlp PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib/pyenvlp"
  INTERPROCEDURAL_OPTIMIZATION ON
  CXX_VISIBILITY_PRESET default
  VISIBILITY_INLINES_HIDDEN ON)

file (GLOB PYTHON_FILES "${CMAKE_CURRENT_LIST_DIR}/pyenvlp/*.py")
if(PYTHON_FILES)
  add_custom_target(pyenvlp ALL VERBATIM)
  add_custom_command(TARGET pyenvlp PRE_BUILD
    COMMAND
      ${CMAKE_COMMAND} -E copy ${PYTHON_FILES} $<TARGET_FILE_DIR:_pyenvlp>
    DEPENDS ${PYTHON_FILES}
  )
endif()

# install(TARGETS _pyenvlp
#   COMPONENT python
#   LIBRARY DESTINATION "${PYTHON_LIBRARY_DIR}/pyenvlp"
# )
