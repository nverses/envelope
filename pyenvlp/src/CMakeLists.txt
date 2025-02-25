find_package(Python COMPONENTS Interpreter Development)
# find_package(pybind11 CONFIG)

# pybind11 method:
pybind11_add_module(_pyenvlp
  ${CMAKE_CURRENT_LIST_DIR}/pyenvlp/pyenvlp.hpp
  ${CMAKE_CURRENT_LIST_DIR}/pyenvlp/pyenvlp.cpp
  ${CMAKE_CURRENT_LIST_DIR}/pyenvlp/pyenvlp_module.cpp
)
target_link_libraries(_pyenvlp PRIVATE pybind11::module envlp Eigen rapidjson)
set_target_properties(_pyenvlp PROPERTIES
                                INTERPROCEDURAL_OPTIMIZATION ON
                                CXX_VISIBILITY_PRESET default
                                VISIBILITY_INLINES_HIDDEN ON)
