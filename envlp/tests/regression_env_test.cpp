#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wfloat-equal"

#include "envelope/regression_env.hpp"

#include <Eigen/Dense>
#include <catch2/catch.hpp>
#include <envelope/eigen_types.hpp>
#include <envelope/gram.hpp>
#include <envelope/la_util.hpp>
#include <envelope/mat_util.hpp>
#include <iostream>

#include "test_util.hpp"

using namespace envelope;

TEST_CASE("regression env", "[regression][envlp]")
{
  // another dataset
  auto [X, Y] = load_csv("waterstrider.csv", {}, {0});

  int n                = X.rows();
  auto [xtx, xty, yty] = GramUtil::dot(X, Y, true);
  std::cout << "xtx:\n" << xtx << std::endl;

  RidgeEnvlp re(true, 0.001);
  re.fitGram(xtx, xty, yty, n);

  DMat betabase = re.coefsBase();  // ridge cannonical coefs
  DMat betaenv  = re.coefs();      // envelope projected coefs
  // doing this since MatUtils::hstack is broken! fix that function.
  DMat betas(betabase.rows(), betabase.cols() + betaenv.cols());
  betas.block(0, 0, betabase.rows(), betabase.cols())             = betabase;
  betas.block(0, betabase.cols(), betaenv.rows(), betaenv.cols()) = betaenv;

  std::cout << "js:\n" << re.toJsonString() << std::endl;
  std::cout << "selected u dim=" << re.getEnvelopeDim() << std::endl;
  std::cout << "gamma:\n" << re.getGamma() << std::endl;
  std::cout << "beta base vs envlp:\n" << betas << std::endl;

  REQUIRE(betabase.rows() == X.cols());
  REQUIRE(betaenv.rows() == X.cols());
  DMat yh = re.predict(X);
  std::cout << "yh:\n" << yh << std::endl;
}

#pragma GCC diagnostic pop
