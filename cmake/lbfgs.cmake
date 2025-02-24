include(FetchContent)

FetchContent_Declare(
  lbfgs
  GIT_REPOSITORY https://github.com/chokkan/liblbfgs.git
  GIT_TAG "master"
  GIT_SHALLOW TRUE
)

FetchContent_GetProperties(lbfgs)
if (NOT lbfgs_POPULATED)
    FetchContent_Populate(lbfgs)
endif()
add_subdirectory(${lbfgs_SOURCE_DIR})
