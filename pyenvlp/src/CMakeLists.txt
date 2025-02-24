# find_package(Python COMPONENTS Interpreter Development)
# find_package(pybind11 CONFIG)

# pybind11 method:
# pybind11_add_module(_pyenvlp
#   ${CMAKE_CURRENT_LIST_DIR}/pyenvlp/pyenvlp_module.cpp
#   ${CMAKE_CURRENT_LIST_DIR}/pyenvlp/pyenvlp.cpp
# )
# target_link_libraries(${PROJECT_NAME} Eigen pthread OpenMP::OpenMP_CXX lbfgs cppenvlp)

# Python method:
# Python_add_library(_pyenvlp src2.cpp)
# target_link_libraries(_pyenvlp PUBLIC pybind11::headers)
# set_target_properties(_pyenvlp PROPERTIES
#                                 INTERPROCEDURAL_OPTIMIZATION ON
#                                 CXX_VISIBILITY_PRESET ON
#                                 VISIBILITY_INLINES_HIDDEN ON)

# pybind_package(
#   PACKAGE
#     pyenvlp
#   MODULE
#     _pyenvlp
#   DEP_FILE
#     ${CMAKE_CURRENT_LIST_DIR}/pyenvlp/pyenvlp.dep
#   SRCS
#     WITHOUT_SOABI
#     ${CMAKE_CURRENT_LIST_DIR}/pyenvlp/pyenvlp_module.cpp
#     ${CMAKE_CURRENT_LIST_DIR}/pyenvlp/pyenvlp.cpp
#     ${CMAKE_CURRENT_LIST_DIR}/pyenvlp/__init__.py
#     ${CMAKE_CURRENT_LIST_DIR}/pyenvlp/lau.py
#     ${CMAKE_CURRENT_LIST_DIR}/pyenvlp/opt.py
#     ${CMAKE_CURRENT_LIST_DIR}/pyenvlp/fit.py
#   LINK_TARGETS
#     ${KINETIC_ENVELOPE_TARGET}
#   LOPTS
#     "-lgomp"
# )
