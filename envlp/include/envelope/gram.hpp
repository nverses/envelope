#pragma once

#include <rapidjson/document.h>

#include <Eigen/Dense>

#include "envelope/eigen_types.hpp"

namespace envelope {

  // Properties of Gram
  //
  //   X   : (n x k)
  //   X'X : (k x k)
  //   X'X = xtx
  //
  // X data matrix with intercept (vector of 1's):
  //
  //   X   = [ 1, x1, x2, ... xn ]
  //
  // X'X dot prooduct yields the following:
  //
  //   xtx = [ 1'1   1'x1   1'x2      ..   ]
  //         [ 1'x1  x1'x1  x1'x2     ..   ]
  //         [ 1'x2  x2'x1  x2'x2     ..   ]
  //         [       .....           xn'xn ]
  //
  // Looking into the individual items of gram matrix:
  //
  //   xtx(0,0) = 1'1 = sum( 1*1 + 1*1 + ... ) = n
  //   xtx(0,1) = 1'x1 = sum( 1*x1[0] + 1*x1[1] + ... ) = sum(x1)
  //   xtx(1,1) = x1'x1 = sum( x1[0]*x1[0] + x1[1]*x1[1] + ... ) = sum(x1*x1)
  //   mu = xtx(0,1) / xtx(0,0)
  //   var = (xtx(1,1) - xtx(0,1)**2) / xtx(0,0)
  //
  // Gram matrix that have been computed with intercept column can be
  // Zscore-ed without the need to use the original data matrix.

  // return type containing xtx, xty
  using GramTwin = std::tuple<DMat, DMat>;
  // return type containing xtx, xty, yty
  using GramTriplet = std::tuple<DMat, DMat, DMat>;

  class GramUtil
  {
  public:
    // compute xtx, xty gram matrix from x,y data matrix
    static GramTriplet dot(const DMatV& x, const DMatV& y,
                           bool with_intercept = false);

    // weighted version of dot product for xtx and xty
    static GramTriplet dotw(const DMatV& x, const DMatV& y, const DMatV& w,
                            bool with_intercept = false);

    // apply ridge shinkage to the diags of xtx
    static void addRidge(DMat& xtx, double ridge = 0.0,
                         bool has_intercept = false);

    // standardize xtx, xty as if z(X), Y were used
    static GramTwin standardize(const DMatV& xtx, const DMatV& xty,
                                const DMatV& xsum, const DMatV& xsqsum,
                                int nrows);

    static GramTwin standardize_intrinsic(const DMatV& xtx, const DMatV& xty);

    // standardizes xtx, xty as if z(X) , z(Y) were used
    static GramTwin standardize2(const DMatV& xtx, const DMatV& xty,
                                 const DMatV& xsum, const DMatV& xsqsum,
                                 const DMatV& ysum, const DMatV& ysqsum,
                                 int nrows);

    // NOTE: standardizing xty also requires yty matrix!
    static GramTwin standardize_intrinsic2(const DMatV& xtx, const DMatV& xty,
                                           const DMatV& yty);

    // demean xtx, xty given gram matrix containing intercept column
    // `denom` is the sum of sample weights. In the simplest case, this
    // is equal to `nrows`
    static GramTwin demean(const DMatV& xtx, const DMatV& xty,
                           const DMatV& xsum, const DMatV& ysum,
                           double denominator);

    static GramTwin demean_intrinsic(const DMatV& xtx, const DMatV& xty);
  };

  // least-squares gramian matrix solver
  class GramLsq
  {
  public:
    GramLsq();
    virtual ~GramLsq();

    GramLsq& addGram(const DMatV& xtx, const DMatV& xty);
    GramLsq& subGram(const DMatV& xtx, const DMatV& xty);

    virtual DMat solve();
    // if xtx, xty storage is not needed, solve directly
    virtual DMat solveThru(const DMatV& xtx, const DMatV& xty);

  protected:
    DMat m_xtx;
    DMat m_xty;
  };
}  // namespace envelope
