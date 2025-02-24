#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wfloat-equal"
#pragma GCC diagnostic ignored "-Wzero-as-null-pointer-constant"

#include <Eigen/Dense>
#include <catch2/catch.hpp>
#include <envelope/eigen_types.hpp>
#include <envelope/mat_util.hpp>
#include <iostream>

#include "envelope/la_util.hpp"
#include "test_util.hpp"

using namespace envelope;

// MatrixView class comes from dataset.h
TEST_CASE("cov test", "[lau][cov]")
{
  DMat a     = DMat::Random(10, 5);
  DMat b     = DMat::Random(10, 3);
  DMat cova  = LAU::cov(a);
  DMat covab = LAU::cov(a, b);
  // demean by hand and demean using eigen short syntax
  DMat amu = a;
  double n = a.rows();
  for (int j = 0; j < a.cols(); j++) {
    double cmu = 0.0;
    for (int i = 0; i < a.rows(); i++) {
      cmu += a(i, j);
    }
    cmu /= n;
    // each col
    amu.col(j).array() -= cmu;
  }
  DMat amu2 = a;
  amu2.rowwise() -= amu2.colwise().mean();

  DMat cov_hand = (amu.transpose() * amu) / (n - 1);

  std::cout << "\n\ncov:\n" << cova << std::endl;
  std::cout << "\n\ncov_hand:\n" << cov_hand << std::endl;
  REQUIRE((amu - amu2).cwiseAbs().sum() < 1e-7);
  REQUIRE((cova - cov_hand).cwiseAbs().sum() < 1e-7);
  REQUIRE(cova.rows() == 5);
  REQUIRE(covab.cols() == 3);
}

TEST_CASE("eig test", "[lau][eig]")
{
  DMat a    = DMat::Random(100, 5);
  DMat cova = LAU::cov(a);

  auto [eval, E] = LAU::eig(cova);
  std::cout << "eigen values:\n" << eval << "\n";
  std::cout << "\neigen vectors:\n" << E << "\n";

  auto [X, Y] = load_csv("wheatprotein.csv", {7}, {});
  std::cout << "\nX:\n" << X << std::endl;
  std::cout << "\nY:\n" << Y << std::endl;
}

TEST_CASE("qr test", "[lau][qr]")
{
  DMat a = DMat::Random(100, 5);

  auto [X, Y] = load_csv("wheatprotein.csv", {7}, {});
  DMat ycov   = LAU::cov(Y);
  std::cout << "ycov:\n" << ycov << "\n";

  auto [Q, R] = LAU::qr(ycov, true);
  std::cout << "\nQ:\n" << Q << std::endl;
  std::cout << "\nR:\n" << R << std::endl;

  auto [Qc, Rc] = LAU::qr(Y, false);
  std::cout << "\nQc:\n" << Qc.block(0, 0, 3, 5) << std::endl;
  std::cout << "\nRc:\n" << Rc << std::endl;
  std::cout << "\nQc shape: " << Qc.rows() << " x " << Qc.cols() << std::endl;
}

TEST_CASE("cholinv", "[lau][cholinv]")
{
  auto [X, Y] = load_csv("wheatprotein.csv", {7}, {});
  DMat ycov   = LAU::cov(Y);
  std::cout << "ycov:\n" << ycov << "\n";

  DMat ycovinv  = LAU::cholinv(ycov);
  DMat ycovinv2 = LAU::pinv(ycov);
  // reference pre-computed from numpy for comparison
  DMat refinv(7, 7);
  refinv << 1.40838508, -0.18534271, -1.07148106, -0.63423625, 0.08720229,
      0.35743367, 1.27078579, -0.18534271, 1.85389834, -2.15880188, 0.69912856,
      0.0005755, -0.20309453, 1.49475936, -1.07148106, -2.15880188, 5.61242634,
      -1.88457107, -0.04229011, -0.42144399, -10.28053127, -0.63423625,
      0.69912856, -1.88457107, 1.84616356, -0.07313971, 0.05001442, 6.04459242,
      0.08720229, 0.0005755, -0.04229011, -0.07313971, 0.0100055, 0.01421021,
      -0.06233033, 0.35743367, -0.20309453, -0.42144399, 0.05001442, 0.01421021,
      0.21232239, 1.46502726, 1.27078579, 1.49475936, -10.28053127, 6.04459242,
      -0.06233033, 1.46502726, 30.51966933;

  std::cout << "ycovinv:\n" << ycovinv << std::endl;
  std::cout << "ycovinv2:\n" << ycovinv2 << std::endl;
  std::cout << "refinv:\n" << refinv << std::endl;
  REQUIRE((refinv - ycovinv).cwiseAbs().mean() < 1e-7);
  REQUIRE((refinv - ycovinv2).cwiseAbs().mean() < 1e-7);

  // check related solve
  DMat I  = DMat::Identity(ycov.rows(), ycov.cols());
  DMat xx = LAU::solve(ycov, I);
  std::cout << "xx:\n" << xx << std::endl;
}

TEST_CASE("geindex", "[lau][geindex]")
{
  auto [X, Y] = load_csv("wheatprotein.csv", {7}, {});
  DMat ycov   = LAU::cov(Y);
  std::cout << "ycov:\n" << ycov << "\n";

  IArr idx = LAU::geindex(ycov);
  std::cout << "idx: " << idx.transpose() << std::endl;
  // should be the same as reference solution
  int i = 0;
  for (int val : std::vector<int>{4, 1, 3, 5, 2, 0, 6}) {
    REQUIRE(idx(i++) == val);
  }
}

#pragma GCC diagnostic pop
