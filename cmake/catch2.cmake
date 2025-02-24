Include(FetchContent)

FetchContent_Declare(
        Catch2
        GIT_SHALLOW TRUE
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG v2.13.7)
FetchContent_MakeAvailable(Catch2)

if (DAGGY_ENABLE_BENCHMARKS)
  SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -DCATCH_CONFIG_ENABLE_BENCHMARKING")
endif()
