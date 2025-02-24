#include <iostream>

#include "envelope/envopt.hpp"

#define CATCH_CONFIG_MAIN

#include <catch2/catch.hpp>

TEST_CASE("Sanity tests", "[sanity]")
{
  REQUIRE(1 == 1);
}

// compile and run
// g++ -std=c++17 -o test test.cpp  && ./test
