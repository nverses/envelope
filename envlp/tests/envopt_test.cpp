#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wfloat-equal"

#include "envelope/envopt.hpp"

#include <Eigen/Dense>
#include <catch2/catch.hpp>
#include <envelope/eigen_types.hpp>
#include <envelope/la_util.hpp>
#include <envelope/mat_util.hpp>
#include <iostream>

#include "test_util.hpp"

using namespace envelope;

TEST_CASE("opt solve1", "[envopt][solve1]")
{
  auto [X, Y] = load_csv("wheatprotein.csv", {7}, {});
  int n       = Y.rows();
  int r       = Y.cols();
  std::cout << "Y (" << n << " x " << r << "):\n" << Y << std::endl;
  double nadj  = (n - 1) / double(n);
  DMat sigY    = LAU::cov(Y) * nadj;
  DMat sigYX   = LAU::cov(Y, X).transpose() * nadj;
  DMat sigX    = LAU::cov(X) * nadj;
  DMat betaOLS = LAU::solve(sigX, sigYX);
  std::cout << "sigY:\n" << sigY << std::endl;
  std::cout << "sigYX:\n" << sigYX << std::endl;
  std::cout << "sigX:\n" << sigX << std::endl;
  std::cout << "betaOLS:\n" << betaOLS << std::endl;

  DMat U = betaOLS.transpose() * sigYX;
  DMat M = sigY - U;
  std::cout << "U:\n" << U << std::endl;
  std::cout << "M:\n" << M << std::endl;

  int u                        = 1;
  auto [gamma, gamma0, objval] = EnvOpt::solveEnvelope(M, U, u);
  std::cout << "gamma:\n" << gamma << std::endl;
  std::cout << "gamma0:\n" << gamma0 << std::endl;
  std::cout << "objval:\n" << objval << std::endl;

  DMat refgamma(r, 1);
  refgamma << -0.12286716, 0.50248022, 0.41406779, -0.67501756, 0.07128729,
      -0.17681534, 0.26262312;

  DMat refgamma0(r, r - 1);
  refgamma0 << 0.50248022, 0.41406779, -0.67501756, 0.07128729, -0.17681534,
      0.26262312, 0.77514137, -0.18529429, 0.30206865, -0.03190088, 0.07912442,
      -0.11752318, -0.18529429, 0.84730862, 0.24891905, -0.02628786, 0.06520231,
      -0.09684474, 0.30206865, 0.24891905, 0.59420961, 0.04285473, -0.10629348,
      0.15787729, -0.03190088, -0.02628786, 0.04285473, 0.9954742, 0.01122545,
      -0.01667311, 0.07912442, 0.06520231, -0.10629348, 0.01122545, 0.97215729,
      0.04135467, -0.11752318, -0.09684474, 0.15787729, -0.01667311, 0.04135467,
      0.93857608;
  REQUIRE((gamma - refgamma).cwiseAbs().mean() < 1e-6);
  REQUIRE((gamma0 - refgamma0).cwiseAbs().mean() < 1e-6);
  REQUIRE(fabs(objval - 13.558295279515212) < 1e-7);
}

TEST_CASE("opt solven", "[envopt][solven]")
{
  auto [X, Y] = load_csv("wheatprotein.csv", {7}, {});
  std::cout << "Y:\n" << Y << std::endl;
  int n        = Y.rows();
  int r        = Y.cols();
  double nadj  = (n - 1) / double(n);
  DMat sigY    = LAU::cov(Y) * nadj;
  DMat sigYX   = LAU::cov(Y, X).transpose() * nadj;
  DMat sigX    = LAU::cov(X) * nadj;
  DMat betaOLS = LAU::solve(sigX, sigYX);
  std::cout << "sigY:\n" << sigY << std::endl;
  std::cout << "sigYX:\n" << sigYX << std::endl;
  std::cout << "sigX:\n" << sigX << std::endl;
  std::cout << "betaOLS:\n" << betaOLS << std::endl;

  DMat U = betaOLS.transpose() * sigYX;
  DMat M = sigY - U;
  std::cout << "U:\n" << U << std::endl;
  std::cout << "M:\n" << M << std::endl;

  // solve direct using the static function
  int u                        = 2;
  auto [gamma, gamma0, objval] = EnvOpt::solveEnvelope(M, U, u);
  std::cout << "gamma:\n" << gamma << std::endl;
  std::cout << "gamma0:\n" << gamma0 << std::endl;
  std::cout << "objval:\n" << objval << std::endl;

  // also solve using the EnvOpt class
  EnvOpt eo(M, U);
  eo.solve(u);  // 2
  auto [g, g0] = eo.getGammas();

  // solution from python impl for u=2
  DMat refgamma(r, u);
  refgamma << -0.08107961, -0.34900047, 0.52338564, -0.24263913, 0.43792073,
      -0.24288698, -0.65026385, -0.26284341, 0., 0.81339985, -0.1884047,
      -0.17199183, 0.26340006, 0.00660829;

  DMat refgamma0(r, r - u);
  refgamma0 << 0.20044414, -0.68296989, 0.50269053, -0.25710899, 0.21493308,
      -0.48130063, 0.27772642, 0.57003109, 0.01330487, -0.18248013, 0.81420354,
      0.26224955, 0.01779116, 0.07388688, -0.10841271, 0.10616012, 0.58721307,
      0.33286041, -0.15881754, 0.126341, 0.22141964, 0.03049463, 0.53129882,
      0.06405887, 0.04518989, 0.0046702, -0.12319209, 0.15166505, 0.94643731,
      0.03128113, -0.0701855, 0.16346232, -0.07728834, 0.05646718, 0.94327555;

  REQUIRE((gamma - refgamma).cwiseAbs().mean() < 1e-7);
  REQUIRE((gamma0 - refgamma0).cwiseAbs().mean() < 1e-7);
  REQUIRE(fabs(objval - 13.551665837322698) < 1e-7);
  REQUIRE((g - refgamma).cwiseAbs().mean() < 1e-7);
  REQUIRE((g0 - refgamma0).cwiseAbs().mean() < 1e-7);
  REQUIRE(fabs(eo.getObjValue() - 13.551665837322698) < 1e-7);
}

TEST_CASE("opt sweep", "[envopt][sweep]")
{
  auto [X, Y] = load_csv("wheatprotein.csv", {7}, {});
  int n       = Y.rows();
  int r       = Y.cols();
  std::cout << "Y (" << n << " x " << r << "):\n" << Y << std::endl;
  double nadj      = (n - 1) / double(n);
  DMat sigY        = LAU::cov(Y) * nadj;
  DMat sigYX       = LAU::cov(Y, X).transpose() * nadj;
  DMat sigX        = LAU::cov(X) * nadj;
  DMat betaOLS     = LAU::solve(sigX, sigYX);
  auto [ev, E]     = LAU::eig(sigY);
  double ov_offset = ev.array().log().sum();
  // on multi-Y, U is tot var of (y'x)' (x'x)^-1 y'x, M is residual var of Y
  DMat U = betaOLS.transpose() * sigYX;
  DMat M = sigY - U;

  EnvOpt eo(M, U);
  auto [loglik, best_u] = eo.sweep({1, 2, 3, 4, 5, 6, 7}, n, ov_offset, 0);
  std::cout << "best_u=" << best_u << " log-likelihood:\n"
            << loglik << std::endl;
  REQUIRE(best_u == 7);

  // another dataset
  auto [X2, Y2] = load_csv("waterstrider.csv", {}, {0});
  DMat sigY2    = LAU::cov(Y2);
  DMat sigYX2   = LAU::cov(Y2, X2);
  DMat sigXY2   = LAU::cov(X2, Y2);
  DMat sigX2    = LAU::cov(X2);
  std::cout << "sigY2\n" << sigY2 << std::endl;
  std::cout << "sigYX2\n" << sigYX2 << std::endl;
  std::cout << "sigXY2\n" << sigXY2 << std::endl;
  std::cout << "sigX2\n" << sigX2 << std::endl;
  DMat betaOLS2 = LAU::solve(sigX2, sigXY2);
  std::cout << "betaOLS2:\n" << betaOLS2 << std::endl;

  // on multi-X, U is totvar of (x'y) (y'y)^-1 (x'y)', U is residual var of X
  DMat U2 = sigXY2 * LAU::cholinv(sigY2) * sigXY2.transpose();
  DMat M2 = sigX2 - U2;
  std::cout << "U2:\n" << U2 << std::endl;
  std::cout << "M2:\n" << M2 << std::endl;

  eo.setData(M2, U2);
  auto [loglik2, best_u2] = eo.sweep({7, 6, 3, 2}, X2.rows(), ov_offset, 0);
  std::cout << "best_u2=" << best_u2 << " log-likelihood:\n"
            << loglik2 << std::endl;
  REQUIRE(best_u2 == 7);
}
#pragma GCC diagnostic pop
