pybind11_add_module(_pyenvlp
    src/pyenvlp.cpp
    src/pyenvlp_module.cpp
)

# Link against the C++ library
target_link_libraries(_pyenvlp PRIVATE envlp)

# Copy the Python package python files
add_custom_target(pyenvlp ALL VERBATIM)
file(GLOB PYTHON_SOURCES
    "python/*.py"
  )
file(COPY ${PYTHON_SOURCES} DESTINATION ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
