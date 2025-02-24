target_sources(${PROJECT_NAME} PRIVATE
  envopt.cpp
  la_util.cpp
  gram.cpp
  regression.cpp
  regression_env.cpp
)
target_link_libraries(${PROJECT_NAME} Eigen pthread OpenMP::OpenMP_CXX lbfgs_lib)
