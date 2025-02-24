#include <cmath>
#include <iostream>

#include "envelope/regression.hpp"

using namespace envelope;

GramTriplet GramUtil::dot(const DMatV& x, const DMatV& y, bool with_intercept)
{
  // If we fit with intercept, we need to modify our gram matrices.
  // We do this instead of copying potentially large x matrix with
  // extra columns of all 1's.
  DMat xtx, xty, yty;
  // NOTE: make an explicit copy to get around undefined behavior on -O2 -g
  DMat xt = x.transpose();
  DMat yt = y.transpose();
  if (with_intercept) {
    // addition of top row and left-most column
    xtx.setZero(x.cols() + 1, x.cols() + 1);
    xtx.block(1, 1, x.cols(), x.cols()) = xt * x;
    xtx(0, 0)                           = x.rows();
    xtx.block(0, 1, 1, x.cols())        = x.colwise().sum();
    xtx.block(1, 0, x.cols(), 1)        = x.colwise().sum().transpose();
    // addition of top row
    xty.setZero(x.cols() + 1, y.cols());
    xty.block(1, 0, x.cols(), y.cols()).noalias() = xt * y;
    xty.block(0, 0, 1, y.cols())                  = y.colwise().sum();
  }
  else {
    xtx = xt * x;
    xty = xt * y;
  }
  // yty is simple
  yty = yt * y;
  return {xtx, xty, yty};
}

GramTriplet GramUtil::dotw(const DMatV& x, const DMatV& y, const DMatV& w,
                           bool with_intercept)
{
  if (w.cols() != 1) {
    throw std::invalid_argument(
        "GramUtil::dotw(): weight input w should be matrix with 1 column "
        "vector");
  }
  // We need to be careful with mem usage here. Simple approach would be
  // to create a copy of x and multiply w to every column of x.
  // This would double the memory of x, and with large datasets, will
  // create memory spike to more than double the memory usage prior to
  // calling this function.
  // Instead, we roll up the matrices by using ephemeral w.asDiagonal() call
  // to weight and call matrix mult operation that reduce to efficient
  // linear algebra operations without creating large mem footprint.
  DMat xtx, xty, yty;
  DMat xt = x.transpose();
  DMat yt = y.transpose();

  if (!with_intercept) {
    xtx = xt * w.asDiagonal() * x;
    xty = xt * w.asDiagonal() * y;
    yty = yt * w.asDiagonal() * y;
    return {xtx, xty, yty};
  }

  // construct xtx
  xtx.setZero(x.cols() + 1, x.cols() + 1);
  xtx.block(1, 1, x.cols(), x.cols()) = xt * w.asDiagonal() * x;
  xtx(0, 0)                           = w.sum();
  xtx.block(0, 1, 1, x.cols())        = (w.asDiagonal() * x).colwise().sum();
  xtx.block(1, 0, x.cols(), 1) = xtx.block(0, 1, 1, x.cols()).transpose();

  // construct xty
  xty.setZero(x.cols() + 1, y.cols());
  xty.block(1, 0, x.cols(), y.cols()).noalias() = xt * w.asDiagonal() * y;
  xty.block(0, 0, 1, y.cols()).noalias() = (w.asDiagonal() * y).colwise().sum();

  // yty
  yty = yt * w.asDiagonal() * y;
  return {xtx, xty, yty};
}

void GramUtil::addRidge(DMat& xtx, double ridge, bool has_intercept)
{
  int n = xtx.rows();
  int m = xtx.cols();
  if (n != m) {
    throw std::invalid_argument(
        "GramUtil::applyRidge(): xtx matrix do not have symmetric "
        "dimensions");
  }
  if (!has_intercept) {
    xtx += DMat::Identity(n, m) * ridge;
  }
  else {
    // apply only on the diags of x, not the intercept (top-left corner)
    xtx.block(1, 1, n - 1, m - 1) += DMat::Identity(n - 1, m - 1) * ridge;
  }
}

/* unused
static DMat standardize_yty(int nrows, const DMatV& yty, const DMatV& ysum,
                            const DMatV& ysqsum)
{
  double n  = nrows;
  DMat ytyz = yty;
  for (int j = 0; j < yty.cols(); j++) {
    double muj   = ysum(0, j) / n;
    double sdj   = sqrt(ysqsum(0, j) / n - pow(muj, 2));
    double yjsum = yty(0, j);
    // yty is symmetric
    for (int i = j; i < yty.rows(); i++) {
      double mui    = ysum(0, i) / n;
      double sdi    = sqrt(ysqsum(0, i) / n - pow(mui, 2));
      double yisum  = yty(0, i);
      double yijsum = yty(i, j);
      double var    = sdi * sdj;
      double cell =
          (var > 0) ? (yijsum - mui * yjsum - muj * yisum + n * mui * muj) / var
                    : 0.0;
      ytyz(i, j) = cell;
      if (i != j) {
        ytyz(j, i) = cell;
      }
    }
  }
  return ytyz;
}
*/

static DMat standardize_xtx(int nrows, const DMatV& xtx, const DMatV& xsum,
                            const DMatV& xsqsum)
{
  double n = nrows;
  // make the copy first
  DMat xtxz = xtx;
  // top row, left column are all zeros
  xtxz.block(0, 1, 1, xtx.cols() - 1).array() = 0.0;
  xtxz.block(1, 0, xtx.rows() - 1, 1).array() = 0.0;
  // NOTE: there might be more fancy way to do this with Eigen API..
  //       but better to be correct than sorry.
  for (int j = 1; j < xtx.cols(); j++) {
    double muj   = xsum(0, j - 1) / n;
    double sdj   = sqrt(xsqsum(0, j - 1) / n - pow(muj, 2));
    double xjsum = xtx(0, j);
    // just compute the upper triangle, since matrix is symmetric
    for (int i = j; i < xtx.rows(); i++) {
      double mui    = xsum(0, i - 1) / n;
      double sdi    = sqrt(xsqsum(0, i - 1) / n - pow(mui, 2));
      double xisum  = xtx(0, i);
      double xijsum = xtx(i, j);
      double var    = sdi * sdj;
      double cell =
          (var > 0) ? (xijsum - mui * xjsum - muj * xisum + n * mui * muj) / var
                    : 0.0;
      // symmetric
      xtxz(i, j) = cell;
      if (i != j) {
        xtxz(j, i) = cell;
      }
    }
  }
  return xtxz;
}

static DMat standardize_xty(int nrows, const DMatV& xty, const DMatV& xsum,
                            const DMatV& xsqsum, const DMatV& ysum = DMatV(),
                            const DMatV& ysqsum = DMatV())
{
  double n = nrows;
  // make a copy first, and update inplace
  DMat xtyz = xty;

  if (!ysum.size() && !ysqsum.size()) {
    for (int j = 0; j < xty.cols(); j++) {
      double ysum = xty(0, j);
      for (int i = 1; i < xty.rows(); i++) {
        double mui   = xsum(0, i - 1) / n;
        double sdi   = sqrt(xsqsum(0, i - 1) / n - pow(mui, 2));
        double xysum = xty(i, j);
        xtyz(i, j)   = (sdi > 0) ? (xysum - ysum * mui) / sdi : 0.0;
      }
    }
  }
  else {
    // sum(zy0*zx0)
    //  = sum( (y0 - E[y0]) / sd[y0] * (x0 - E[x0]) / sd[x0] )
    //  = sum( (y0*x0 - y0*E[x0] - x0*E[y0] + E[y0]*E[x0]) / (sd[y0]*sd[x0]) )
    //

    // xty first row is zero since sum of zscore should be zero
    for (int k = 0; k < xtyz.cols(); k++) {
      xtyz(0, k) = 0.0;
    }
    int t = xty.cols();  // number of targets
    for (int j = 0; j < t; j++) {
      double ysumj = ysum(0, j);
      double ymu   = ysumj / n;
      double ysd   = sqrt(ysqsum(0, j) / n - pow(ymu, 2));
      for (int i = 1; i < xty.rows(); i++) {
        double xsumi = xsum(0, i - 1);
        double xmu   = xsumi / n;
        double xysum = xty(i, j);
        double xsd   = sqrt(xsqsum(0, i - 1) / n - pow(xmu, 2));
        double denom = ysd * xsd;
        xtyz(i, j)   = (denom > 0)
                           ? (xysum - ysumj * xmu - xsumi * ymu + n * xmu * ymu) /
                               (ysd * xsd)
                           : 0.0;
      }
    }
  }
  return xtyz;
}

GramTwin GramUtil::standardize(const DMatV& xtx, const DMatV& xty,
                               const DMatV& xsum, const DMatV& xsqsum,
                               int nrows)
{
  // Standardization on XtX and XtY:
  //
  // zscoring involves modifying the original data:
  //
  //  zsc(x) = (x - x.mean()) /  x.std()
  //
  // However, xtx can be directly modified provided that XtX and XtY
  // contains the dot product computed with the intercept. With the
  // intercept, the top-left corner as well as top-row, left-column
  // has specific meaning such that:
  //
  //   xtx(0,0) = n                        number of samples
  //   xtx(0,1) = sum(x1)                  sum of x1 column
  //   xtx(1,1) = sum(x1*x1)               squared sum of x1 columns
  //   xtx(0,1) / xtx(0,0) = mu            mean computed from the parts
  //   (xtx(1,1)/n - xtx(0,1)/n**2) = var  variance computed from the parts
  //
  // Thus, modifying the gram matrices for standardization do not require
  // the original data. It requires:
  //
  // * XtX, XtY computed with intercept
  // * xsums estimate, xsums_square estimate, number of rows
  //
  // zcs(XtX) :
  //
  //  | sum(z0^2) sum(z0*z1) ... sum(z0*zn) |
  //  | ...       sum(z1^2)  ...            |
  //  | sum(z0*zn)           ... sum(zn^2)  |
  //
  //  E[x0], var[x0] are constants: can be multiplied outside of summation
  //
  // sum(z0^2) = sum( ((x0 - E[x0]) / sd[x0])^2 )
  //           = sum( (x0^2 - 2x0 E[x0] + E[x0]^2) / var[x0] )
  //           = sum( x0^2 ) / var[x0]
  //              - 2 E[x0] sum( x0 ) / var[x0]
  //              + n*E[x0]^2 / var[x0]
  //
  // sum(z0*z1) = sum( ((x0 - E[x0]) / sd[x0]) * ((x1 - E[x1]) / sd[x1]) )
  //            = sum( (x0 x1 - x0 E[x1] - x1 E[x0] + E[x0] E[x1]) /
  //            (sd[x0]*sd[x1]) )
  // separate the sums
  //            =    sum( x0 x1 / sd[x0] sd[x1] )
  //               - sum( E[x1] x0 / sd[x0] sd[x1] )
  //               - sum( E[x0] x1 / sd[x0] sd[x1] )
  //               + sum( E[x0] E[x1] / sd[x0] sd[x1] )
  // constants outside the sum
  //            =    sum( x0 x1 ) / sd[x0] sd[x1]
  //               - sum( x0 ) * (E[x1] / sd[x0] sd[x1])
  //               - sum( x1 ) * (E[x0] / sd[x0] sd[x1])
  //               + n * (E[x0] E[x1] / sd[x0] sd[x1])
  // with intercept, top row, first col of standardized xtx is all zeros
  //
  // zcs(XtY):
  //
  //   | sum(y0)    sym(y1)    .. |
  //   | sum(y0*z0) sym(y1*z0) .. |
  //   | ...        ...           |
  //   | sum(y0*zn) sym(y1*zn) .. |
  //
  // sum(y0*z0) =   sum( y0 * (x0 - E[x0]) / sd[x0]) )
  //            =   sum( (y0 * x0) / sd[x0] ) - sum( y0 E[x0] / sd[x0] )
  // constants outside the sum
  //            =   sum( y0 * x0 ) / sd[x0]
  //              - sum( y0 ) * E[x0] / sd[x0]

  if (xsum.rows() != 1 || xsqsum.rows() != 1) {
    throw std::invalid_argument(
        "GramUtil::standardize(): xsum, xsqsum requires single row matrix");
  }
  if (xsum.cols() + 1 != xtx.cols() || xsqsum.cols() + 1 != xtx.cols()) {
    throw std::invalid_argument(
        "GramUtil::standardize(): dimensions of matrices passed in are "
        "incompatible. XtX and XtY matrices need to be constructed with "
        "intercept, where as xsum, xsqsum sizes match number of original "
        "features");
  }
  DMat xtxz = standardize_xtx(nrows, xtx, xsum, xsqsum);
  DMat xtyz = standardize_xty(nrows, xty, xsum, xsqsum);

  return {xtxz, xtyz};
}

GramTwin GramUtil::standardize_intrinsic(const DMatV& xtx, const DMatV& xty)
{
  int k       = xtx.cols();
  DMat xsum   = xtx.block(0, 1, 1, k - 1);
  DMat xsqsum = xtx.block(1, 1, k - 1, k - 1).diagonal().transpose();
  int n       = static_cast<int>(xtx(0, 0));
  return GramUtil::standardize(xtx, xty, xsum, xsqsum, n);
}

GramTwin GramUtil::standardize2(const DMatV& xtx, const DMatV& xty,
                                const DMatV& xsum, const DMatV& xsqsum,
                                const DMatV& ysum, const DMatV& ysqsum,
                                int nrows)
{
  // Standardization on XtX and XtY:
  //
  // zscoring involves modifying the original data:
  //
  //  zsc(x) = (x - x.mean()) /  x.std()
  //  zsc(y) = (y - y.mean()) /  y.std()
  //
  // Standardizing XtX follows the same procedure above, but now we
  // also need to consider zsc(y). This requires squared sum of y, which
  // can be obtained from YtY - ysqsum is essentially diag(YtY).
  //
  // Recall:
  //
  //   xtx(0,0) = n                        number of samples
  //   xtx(0,1) = sum(x1)                  sum of x1 column
  //   xtx(1,1) = sum(x1*x1)               squared sum of x1 columns
  //   xtx(0,1) / xtx(0,0) = mu            mean computed from the parts
  //   (xtx(1,1)/n - xtx(0,1)/n**2) = var  variance computed from the parts
  //
  //   xty(0,0) = sum(y1)
  //   xty(1,k) = sum(x1*yk)
  //
  //   yty(0,0) = sum(y1*y1)
  //
  // zcs(XtY):
  //
  //   | sum(zy0)        sym(zy1)     |
  //   | sum(zy0*zx0)    sym(zy1*zx0) |
  //   | ...             ...          |
  //   | sum(zy0*zxn)    sym(zy1*zxn) |
  //
  // sum(zy0*zx0)
  //  = sum( (y0 - E[y0]) / sd[y0] * (x0 - E[x0]) / sd[x0] )
  //  = sum( (y0*x0 - y0*E[x0] - x0*E[y0] + E[y0]*E[x0]) / (sd[y0]*sd[x0]) )

  if (xsum.rows() != 1 || xsqsum.rows() != 1) {
    throw std::invalid_argument(
        "GramUtil::standardize2(): xsum, xsqsum requires single row matrix");
  }
  if (xsum.cols() + 1 != xtx.cols() || xsqsum.cols() + 1 != xtx.cols()) {
    throw std::invalid_argument(
        "GramUtil::standardize2(): dimensions of matrices passed in are "
        "incompatible. XtX and XtY matrices need to be constructed with "
        "intercept, where as xsum, xsqsum sizes match number of original "
        "features");
  }
  if (ysum.rows() != 1 || ysqsum.rows() != 1) {
    throw std::invalid_argument(
        "GramUtil::standardize2(): ysum, ysqsum requires single row matrix");
  }
  if (xty.cols() != ysum.cols()) {
    throw std::invalid_argument(
        "GramUtil::standardize2(): xty columns (no. targets) needs to match "
        "the number of ysum elements");
  }
  int n     = nrows;
  DMat xtxz = standardize_xtx(n, xtx, xsum, xsqsum);
  DMat xtyz = standardize_xty(n, xty, xsum, xsqsum, ysum, ysqsum);

  return {xtxz, xtyz};
}

GramTwin GramUtil::standardize_intrinsic2(const DMatV& xtx, const DMatV& xty,
                                          const DMatV& yty)
{
  int k       = xtx.cols();
  DMat xsum   = xtx.block(0, 1, 1, k - 1);
  DMat xsqsum = xtx.block(1, 1, k - 1, k - 1).diagonal().transpose();
  int t       = xty.cols();
  DMat ysum   = xty.block(0, 0, 1, t);
  DMat ysqsum = yty.diagonal().transpose();
  int n       = static_cast<int>(xtx(0, 0));
  return GramUtil::standardize2(xtx, xty, xsum, xsqsum, ysum, ysqsum, n);
}

GramTwin GramUtil::demean(const DMatV& xtx, const DMatV& xty, const DMatV& xsum,
                          const DMatV& ysum, double n)
{
  // Demean on XtX and XtY:
  //
  // demean involves modifying the original data:
  //
  //  demean(x) = (x - x.mean())
  //
  // Inputs xsum, ysum contains the sums of column features and targets.
  //
  // demean(XtY):
  //
  //   | sum(y0-E[y0])              sym(y1-E[y1])              .. |
  //   | sum((y0-E[y0])*(x0-E[x0])) sym((y1-E[y1]*(x0-E[x0]))) .. |
  //   | ...                                       ...            |
  //   | sum((y0-E[y0])*(xn-E[xn])) sym((y1-E[y1])*(xn-E[xn])) .. |
  //
  // sum((y0-E[y0])*(x0-E[x0]))
  //    =   sum( (y0 x0) - (y0 E[x0]) - (x0 E[y0]) + E[y0] E[x0] )
  // constants outside the sum
  //    =   sum( y0 * x0 )
  //      - sum( y0 ) E[x0]
  //      - sum( x0 ) E[y0]
  //      + n E[x0] E[y0]

  if (xsum.rows() != 1 || ysum.rows() != 1) {
    throw std::invalid_argument(
        "GramUtil::demean(): xsum, xsqsum requires single row matrix");
  }
  if (xsum.cols() + 1 != xtx.cols() || ysum.cols() != xty.cols()) {
    throw std::invalid_argument(
        "GramUtil::demean(): dimensions of matrices passed in are "
        "incompatible. XtX and XtY matrices need to be constructed with "
        "intercept, where as xsum, xsqsum sizes match number of original "
        "features");
  }
  // double n = xtx(0, 0);
  // make the copy first
  DMat xtxz = xtx;
  DMat xtyz = xty;

  // xtx top row, left column are all zeros, except top-left cell
  for (int j = 1; j < xtxz.cols(); j++) {
    xtxz(0, j) = 0.0;
    xtxz(j, 0) = 0.0;
  }
  // xty first row is zero
  for (int k = 0; k < xtyz.cols(); k++) {
    xtyz(0, k) = 0.0;
  }

  // xtxz.block(0, 1, 1, xtx.cols() - 1).array() = 0.0;
  // xtxz.block(1, 0, xtx.rows() - 1, 1).array() = 0.0;

  // NOTE: there might be more fancy way to do this with Eigen API..
  //       but better to be correct than sorry.
  int xn = xtx.cols();
  for (int j = 1; j < xn; j++) {
    double muj   = xsum(0, j - 1) / n;
    double xjsum = xtx(0, j);
    // just compute the lower triangle, since matrix is symmetric
    for (int i = j; i < xn; i++) {
      double mui    = xsum(0, i - 1) / n;
      double xisum  = xtx(0, i);
      double xijsum = xtx(i, j);
      double cell   = xijsum - (mui * xjsum) - (muj * xisum) + (n * mui * muj);
      // symmetric
      xtxz(i, j) = cell;
      if (i != j) {
        xtxz(j, i) = cell;
      }
    }
  }
  //    =   sum( y0 * x0 )
  //      - sum( y0 ) E[x0]
  //      - sum( x0 ) E[y0]
  //      + n E[x0] E[y0]

  // sum( y * zscore(1) ) = 0.0
  // xtyz.block(0, 0, 1, xtyz.cols()).array() = 0.0;
  for (int k = 0; k < xty.cols(); k++) {
    double yksum = xty(0, k);
    double ymu   = ysum(0, k) / n;
    for (int i = 1; i < xty.rows(); i++) {
      double xsumi = xtx(0, i);
      double xmu   = xsum(0, i - 1) / n;
      double xysum = xty(i, k);
      xtyz(i, k)   = (xysum - yksum * xmu - xsumi * ymu + n * ymu * xmu);
    }
  }
  return {xtxz, xtyz};
}

GramTwin GramUtil::demean_intrinsic(const DMatV& xtx, const DMatV& xty)
{
  int k     = xtx.cols();
  DMat xsum = xtx.block(0, 1, 1, k - 1);
  DMat ysum = xty.block(0, 0, 1, xty.cols());
  int n     = static_cast<int>(xtx(0, 0));
  return GramUtil::demean(xtx, xty, xsum, ysum, n);
}

GramLsq::GramLsq() {}
GramLsq::~GramLsq() {}

GramLsq& GramLsq::addGram(const DMatV& xtx, const DMatV& xty)
{
  if (m_xtx.size()) {
    m_xtx += xtx;
  }
  else {
    m_xtx = xtx;
  }
  if (m_xty.size()) {
    m_xty += xty;
  }
  else {
    m_xty = xty;
  }
  return *this;
}

GramLsq& GramLsq::subGram(const DMatV& xtx, const DMatV& xty)
{
  if (m_xtx.size()) {
    m_xtx -= xtx;
  }
  if (m_xty.size()) {
    m_xty -= xty;
  }
  return *this;
}

DMat GramLsq::solve()
{
  return solveThru(m_xtx, m_xty);
}

DMat GramLsq::solveThru(const DMatV& xtx, const DMatV& xty)
{
  // check for xtx invertibility
  // Eigen::FullPivLU<DMat> lu;
  // lu.compute(xtx);
  // if (!lu.isInvertible()) {
  //   throw std::runtime_error("GramLsq::solveThru(): xtx not invertible!");
  // }
  // https://eigen.tuxfamily.org/dox/group__LeastSquares.html
  // return xtx.llt().solve(xty);
  return xtx.ldlt().solve(xty);
}
