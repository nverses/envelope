#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wzero-as-null-pointer-constant"

#include "envelope/regression_env.hpp"

#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>

#include <iostream>

#include "envelope/gram.hpp"
#include "envelope/la_util.hpp"
#include "envelope/mat_util.hpp"

using namespace envelope;

RidgeEnvlp::RidgeEnvlp(bool fit_intercept, double l2_lambda, bool zs_x,
                       bool zs_y, bool do_coef_scale, int u_step)
  : Ridge(fit_intercept, l2_lambda)
  , m_zscore_x(zs_x)
  , m_zscore_y(zs_y)
  , m_do_coef_scale(do_coef_scale)
  , m_coef_scale(1.0)
  , m_u_step(u_step)
  , m_u(-1)
  , m_objval(0.0)
  , m_objval_offset(0.0)
  , m_loglik(0.0)
{
  if (!zs_x && zs_y) {
    std::cerr << "RidgeEnvlp::RidgeEnvlp(): zscore_y depends on "
                 "zscore_x to be turne on. Setting zscore_x to true."
              << std::endl;
    m_zscore_x = true;
  }
  if (!m_intercept && m_zscore_x) {
    std::cerr << "RidgeEnvlp::RidgeEnvlp(): setting fit_intercept "
                 "to true, since zscore_x is turned on"
              << std::endl;
    m_intercept = true;
  }
}

void RidgeEnvlp::setEnvelopeDim(int u)
{
  m_u = u;
}

int RidgeEnvlp::getEnvelopeDim() const
{
  return m_u;
}

DMat RidgeEnvlp::coefsBase() const
{
  if (m_intercept && m_coef_base.size()) {
    return m_coef_base.block(1, 0, m_coef_base.rows() - 1, m_coef_base.cols());
  }
  return m_coef_base;
}

DMat RidgeEnvlp::getGamma() const
{
  // does not consider intercept
  return m_gamma;
}

DMat RidgeEnvlp::getGamma0() const
{
  // does not consider intercept
  return m_gamma0;
}

DMat RidgeEnvlp::getOmega() const
{
  return m_omega;
}

double RidgeEnvlp::getObjValue() const
{
  return m_objval;
}

double RidgeEnvlp::getLogLikelihood() const
{
  return m_loglik;
}

DMat RidgeEnvlp::getDimList() const
{
  return m_u_vec;
}

DMat RidgeEnvlp::getBICList() const
{
  return m_bic_vec;
}

void RidgeEnvlp::setXZScoring(bool val)
{
  m_zscore_x = val;
}

void RidgeEnvlp::setYZScoring(bool val)
{
  m_zscore_y = val;
}

bool RidgeEnvlp::getXZScoring() const
{
  return m_zscore_x;
}

bool RidgeEnvlp::getYZScoring() const
{
  return m_zscore_y;
}

DMat RidgeEnvlp::getXSqSum() const
{
  return m_xsqsum;
}

DMat RidgeEnvlp::getYSqSum() const
{
  return m_ysqsum;
}

void RidgeEnvlp::computeXEnvGammas(const DMatV& xtx, const DMatV& xty,
                                   const DMatV& yty, int nrows)
{
  // compute the M and U matrices
  DMat U = xty * LAU::cholinv(yty) * xty.transpose();
  DMat M = xtx - U;
  // further divide by n - this makes the scale of BIC more ammenable
  // but should not change the gammahat, gamma0hat matrices.
  U /= double(nrows);
  M /= double(nrows);

  // std::cout << "U:\n" << U << std::endl;
  // std::cout << "M:\n" << M << std::endl;
  m_env.setData(M, U);

  // create linspace from 1 to r
  int p = xtx.cols();  // predictor dim
  int r = xty.cols();  // response dim
  // if dim u has not been determined
  if (m_u <= 0) {
    double ov_offset = 0.0;
    // yty can be zero if z-scored
    if (fabs(yty.sum()) > 1e-4) {
      auto [ev, E] = LAU::eig(yty);
      ov_offset    = ev.array().log().sum();
    }
    // step size of spacing out the list of dims
    int step = std::min(std::max(1, int(p / 10)), p);
    if (m_u_step > 0) {
      step = std::min(m_u_step, step);
    }
    IArr uarr = MatUtil::range(1, p, step);
    std::cout << "RidgeEnvlp::computeXEnvGammas: tot_dim=" << p
              << " u_step=" << step << " u_list=" << uarr.transpose()
              << std::endl;
    // now compute the best u
    auto [bicvec, best_u] = m_env.sweepArr(uarr, nrows, ov_offset, r);
    std::cout << "RidgeEnvlp::computeXEnvGammas: computed best BIC u=" << best_u
              << std::endl;
    m_u = best_u;
    // save the data
    m_u_vec   = uarr.cast<double>();
    m_bic_vec = bicvec;
  }
  else if (m_u >= p) {
    // pick the one quarter of the way
    m_u = std::max(1, int(p / 4));
  }
  // compute the objective value offset using demeaned y'y
  if (fabs(yty.sum()) > 1e-4) {
    auto [ev, E]    = LAU::eig(yty);
    m_objval_offset = ev.array().log().sum();
  }
  // solve for gammahat, gamma0hat, includes M, U that considers intercept
  m_env.solve(m_u);
  auto [g, g0] = m_env.getGammas();
  // save the projection matrices
  m_gamma  = g;
  m_gamma0 = g0;
  m_objval = m_env.getObjValue();
  m_loglik = EnvOpt::calcLogLikelihood(m_objval + m_objval_offset, nrows, p, r);
}

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

GramTriplet RidgeEnvlp::prepareGramZ3(const DMatV& xtx, const DMatV& xty,
                                      const DMatV& yty, double nrows)
{
  if (fabs(nrows) < 1) {
    throw std::invalid_argument(
        "RidgeEnvlp::prepareGramZ3(): requires nrows != 0");
  }
  if (m_intercept && (0.5 < fabs(xtx(0, 0) - nrows))) {
    // throw std::invalid_argument(
    //     "RidgeEnvlp::prepareGramZ3(): intercept turned on, but  xtx does "
    //     "not seem to contain intercept vector of 1's: nrows=" +
    //     std::to_string(nrows) + " xtx(0,0)=" + std::to_string(xtx(0, 0)));
    std::cerr
        << "RidgeEnvlp::prepareGramZ3(): intercept turned on, but  xtx does "
           "not seem to contain intercept vector of 1's: nrows="
        << nrows << " xtx(0,0)=" << xtx(0, 0) << std::endl;
  }
  int k = xtx.cols();
  if (m_intercept || m_zscore_x) {
    m_xsum  = xtx.block(0, 1, 1, k - 1);
    m_ysum  = xty.block(0, 0, 1, xty.cols());
    m_nrows = static_cast<int>(xtx(0, 0));
  }
  // zscore need the squared sums
  if (m_zscore_x) {
    m_xsqsum = xtx.block(1, 1, k - 1, k - 1).diagonal().transpose();
    if (m_zscore_y) {
      if (!m_ysum.size()) {
        throw std::invalid_argument(
            "RidgeEnvlp::prepareGramZ3(): cannot zscore yty withou m_ysum! "
            "(perhaps missing intercept?)");
      }
      m_ysqsum = yty.diagonal().transpose();
      // zscore both x and y requires sums and squared sums for both x,y
      auto [xtxz, xtyz] = GramUtil::standardize2(xtx, xty, m_xsum, m_xsqsum,
                                                 m_ysum, m_ysqsum, nrows);
      // zscored yty gets used in envelope optimization. Fortunately, we have
      // ysum (from xty with intercept), and ysqsum (from yty diagonals)
      DMat ytyz = standardize_yty(nrows, yty, m_ysum, m_ysqsum);

      return {std::move(xtxz), std::move(xtyz), std::move(ytyz)};
    }
    // only zscore x, not y
    auto [xtxz, xtyz] =
        GramUtil::standardize(xtx, xty, m_xsum, m_xsqsum, nrows);
    return {std::move(xtxz), std::move(xtyz), yty};
  }
  if (m_intercept) {
    // no zscore, fallback to the same behavior asRegression::prepareGram
    auto [xtxd, xtyd] = GramUtil::demean(xtx, xty, m_xsum, m_ysum, m_nrows);
    return {std::move(xtxd), std::move(xtyd), yty};
  }
  return {xtx, xty, yty};
}

void RidgeEnvlp::fitGram(const DMatV& xtx, const DMatV& xty, const DMatV& yty,
                         int nrows)
{
  if (!xtx.size() || !xty.size() || !yty.size() || nrows == 0) {
    throw std::invalid_argument(
        "RidgeEnvlp::fitGram(): xtx, xty, yty cannot be empty, nrows > 0!");
  }
  // first compute baseline ridge coef
  auto [xtxp, xtyp, ytyp] = prepareGramZ3(xtx, xty, yty, nrows);
  if (m_l2_lambda > 0) {
    GramUtil::addRidge(xtxp, m_l2_lambda, m_intercept);
  }
  m_coef_base = m_gram.solveThru(xtxp, xtyp);

  DMat xtxpni = xtxp.block(1, 1, xtxp.rows() - 1, xtxp.cols() - 1);
  DMat xtypni = xtyp.block(1, 0, xtyp.rows() - 1, xtyp.cols());
  // if intercept exists, we remove intercept in gram matrices for envelope
  if (m_intercept) {
    computeXEnvGammas(xtxpni, xtypni, ytyp, nrows);
  }
  else {
    computeXEnvGammas(xtxp, xtyp, ytyp, nrows);
  }

  // now create the envelope projected coefficient
  if (m_gamma.size()) {
    // gamma matrix does not consider intercept
    if (m_intercept) {
      // the coef member contains intercept value, so we slice
      DMat m_coef_base_ni =
          m_coef_base.block(1, 0, m_coef_base.rows() - 1, m_coef_base.cols());
      std::cout << "m_coef_base_ni: (" << m_coef_base_ni.rows() << " x "
                << m_coef_base_ni.cols() << ")"
                << " gamma: (" << m_gamma.rows() << " x " << m_gamma.cols()
                << ")" << std::endl;
      if (m_gamma.rows() != m_coef_base_ni.rows()) {
        throw std::runtime_error(
            "RidgeEnvlp::fitGram(): gamma.cols() do not match"
            " m_coef_base_ni.rows()!");
      }
      // NOTE:
      // For y-env projection, coef is simpler
      //
      // DMat etahat = m_gamma.transpose() * m_coef_base_ni;
      // DMat coef   = m_gamma * etahat;
      //
      // For x-env projection, coef is computed this way
      //
      // beta_env = g * inv( g' * x'x * g ) * g' * x'y
      //          = g * inv( g' * x'x * g ) * g' * x'x * beta_ols
      //          = g * inv( g' * x'x * g ) * g' * x'x * inv(x'x) * x'y
      //          = g * inv( g' * x'x * g ) * g' * x'y
      //          since x'x * inv(x'x) = I
      // DMat etahat_simple = m_gamma.transpose() * xtypni;
      DMat etahat       = m_gamma.transpose() * xtxpni * m_coef_base_ni;
      m_omega           = m_gamma.transpose() * xtxpni * m_gamma;  // u x u
      DMat omegahat_inv = LAU::cholinv(m_omega);  // may not invert?
      DMat coef         = m_gamma * omegahat_inv * etahat;

      // first intercept term is zero
      m_coef.setZero(m_coef_base.rows(), m_coef_base.cols());
      m_coef.block(1, 0, m_coef.rows() - 1, m_coef.cols()) = coef;
    }
    else {
      DMat etahat       = m_gamma.transpose() * xtxpni * m_coef_base;
      m_omega           = m_gamma.transpose() * xtxpni * m_gamma;  // u x u
      DMat omegahat_inv = LAU::cholinv(m_omega);
      m_coef            = m_gamma * omegahat_inv * etahat;
    }
  }
  else {
    m_coef = m_coef_base;
  }

  m_coef_scale = 1.0;
  if (m_do_coef_scale) {
    m_coef_scale = m_coef_base.lpNorm<1>() / m_coef.lpNorm<1>();
  }
}

DMat RidgeEnvlp::getInterceptOffsetZ(bool include_offset) const
{
  // NOTE:
  // if the intercept exists, we center the x and y, this is consistent with
  // sklearn/linear_model/_base.py:_preprocess_data
  // This will also affect the predictions, check:
  //
  // def _set_intercept(self, X_offset, y_offset, X_scale):
  //   """Set the intercept_"""
  //   if self.fit_intercept:
  //       self.coef_ = self.coef_ / X_scale
  //       self.intercept_ = y_offset - np.dot(X_offset, self.coef_.T)
  //   else:
  //       self.intercept_ = 0.0

  if (!include_offset or !m_intercept) {
    return Eigen::MatrixXd::Constant(m_ysum.rows(), m_ysum.cols(), 0.0);
  }
  int t = m_coef.size() ? m_coef.cols() : m_ysum.size() ? m_ysum.cols() : 0;
  if (!t) {
    return Eigen::MatrixXd::Constant(m_ysum.rows(), m_ysum.cols(), 0.0);
  }
  // intercept_offset = E[y] - E[x] * b
  Eigen::MatrixXd offset = -((m_xsum / m_nrows) * coefs());
  offset += (m_ysum / m_nrows);
  return offset;
}

DMat RidgeEnvlp::predict(const DMatV& x, bool include_intercept)
{
  if (x.cols() + 1 == m_coef.rows()) {
    if (!m_intercept) {
      std::cerr << "RidgeEnvlp::predict(): WARN intercept parameter not "
                   "specified "
                   "but coefficient matrix seem to contain intercept row"
                << std::endl;
    }
  }
  int k = x.cols();
  int n = x.rows();
  if (m_zscore_x && (k != m_xsum.cols() || k != m_xsqsum.cols())) {
    throw std::invalid_argument(
        "RidgeEnvlp::predict(): zscore_x turned on, but xsum, xsqsum do "
        "not contain the same number of items as x.cols()");
  }
  if (m_zscore_y && (!m_ysum.size() || !m_ysqsum.size())) {
    throw std::invalid_argument(
        "RidgeEnvlp::predict(): zscore_y turned on, but ysum, ysqsum do "
        "not contain any values");
  }
  if (m_zscore_x && m_nrows == 0) {
    throw std::invalid_argument(
        "RidgeEnvlp::predict(): zscore_x turned on, m_nrows is zero!");
  }

  DMat offset = getInterceptOffsetZ(include_intercept);
  DMat coef   = m_coef;
  // apply the scale to coef
  if (m_do_coef_scale && m_coef_scale > 0) {
    coef *= m_coef_scale;
  }

  DMat yh;
  // size matches, we can do simple dot product
  if (k == coef.rows()) {
    // first convert x into zscore if we have zscore turned on
    yh = x * coef;
    if (offset.size() && offset.cols() == yh.cols()) {
      yh.rowwise() += offset.row(0);
    }
    return yh;
  }
  // coef contains extra row for intercept term (always true for zscore)
  else if (x.cols() + 1 == coef.rows()) {
    DMat coefz = coef.block(1, 0, coef.rows() - 1, coef.cols());
    DMat xz    = x;
    // if we had zscore_x turned on, standardize x into zscore using previously
    // serialized xsum, xsqsum (mean, stddev) estimates.
    if (m_zscore_x) {
      for (int j = 0; j < k; j++) {
        double xmu  = m_xsum(0, j) / m_nrows;
        double xvar = m_xsqsum(0, j) / m_nrows - xmu * xmu;
        double xsd  = xvar > 0 ? sqrt(xvar) : 0;
        for (int i = 0; i < n; i++) {
          xz(i, j) -= xmu;
          if (xsd > 0) {
            xz(i, j) /= xsd;
          }
        }
      }
    }
    // now dot product
    yh = xz * coefz;
    // if we had zscore_y turned on, we need to scale it by the stddev
    // estimate of the target from the fit.
    if (m_zscore_y) {
      int t = coefz.cols();
      for (int j = 0; j < t; j++) {
        double ymu  = m_ysum(0, j) / m_nrows;
        double yvar = m_ysqsum(0, j) / m_nrows - ymu * ymu;
        double ysd  = yvar > 0 ? sqrt(yvar) : 1.0;
        std::cout << "RidgeEnvlp::predict(): target col=" << j
                  << " apply yh scaling by ysd=" << ysd << std::endl;
        yh.col(j).array() *= ysd;
      }
    }
    // contribution from intercept - when x, y are demeaned, the offset
    // is contribution of x-mean, y-mean from train dataset.
    // NOTE: offset is added AFTER the yh is scaled out by stddev(y)
    if (offset.size() && offset.cols() == yh.cols()) {
      yh.rowwise() += offset.row(0);
    }
    return yh;
  }
  throw std::invalid_argument(
      "RidgeEnvlp::predict(): provided x matrix dims do not match. If "
      "intercept option was turned on, do not provide columns with 1's");
}

rapidjson::Value RidgeEnvlp::toJson(rapidjson::Document::AllocatorType& a) const
{
  namespace rj = rapidjson;
  rj::Value v  = Ridge::toJson(a);

  if (v.HasMember("name")) {
    v["name"].Swap(rj::Value("RidgeEnvlp", a).Move());
  }
  else {
    v.AddMember("name", "RidgeEnvlp", a);
  }

  v.AddMember("coef_scale", rj::Value(m_coef_scale).Move(), a);
  v.AddMember("do_coef_scale", rj::Value(m_do_coef_scale).Move(), a);
  v.AddMember("zscore_x", rj::Value(m_zscore_x).Move(), a);
  v.AddMember("zscore_y", rj::Value(m_zscore_y).Move(), a);

  if (m_xsqsum.size()) {
    rj::Value xsqsum(rj::kArrayType);
    for (int j = 0; j < m_xsqsum.cols(); j++) {
      xsqsum.PushBack(rj::Value(m_xsqsum(0, j)).Move(), a);
    }
    v.AddMember("xsqsum", xsqsum.Move(), a);
  }
  if (m_ysqsum.size()) {
    rj::Value ysqsum(rj::kArrayType);
    for (int j = 0; j < m_ysqsum.cols(); j++) {
      ysqsum.PushBack(rj::Value(m_ysqsum(0, j)).Move(), a);
    }
    v.AddMember("ysqsum", ysqsum.Move(), a);
  }

  if (m_coef_base.size()) {
    rj::Value coef(rj::kArrayType);
    for (int j = 0; j < m_coef_base.cols(); j++) {
      rj::Value coefcol(rj::kArrayType);
      for (int i = 0; i < m_coef_base.rows(); i++) {
        coefcol.PushBack(rj::Value(m_coef_base(i, j)).Move(), a);
      }
      coef.PushBack(coefcol.Move(), a);
    }
    v.AddMember("coef_base", coef.Move(), a);
  }

  v.AddMember("u", rj::Value(m_u).Move(), a);
  v.AddMember("objval", rj::Value(m_objval).Move(), a);
  v.AddMember("loglik", rj::Value(m_loglik).Move(), a);

  if (m_gamma.size()) {
    rj::Value g(rj::kArrayType);
    // col-major serialization
    for (int j = 0; j < m_gamma.cols(); j++) {
      rj::Value gcol(rj::kArrayType);
      for (int i = 0; i < m_gamma.rows(); i++) {
        gcol.PushBack(rj::Value(m_gamma(i, j)).Move(), a);
      }
      g.PushBack(gcol.Move(), a);
    }
    v.AddMember("gamma", g.Move(), a);
  }

  if (m_gamma0.size()) {
    rj::Value g0(rj::kArrayType);
    // col-major serialization
    for (int j = 0; j < m_gamma0.cols(); j++) {
      rj::Value gcol(rj::kArrayType);
      for (int i = 0; i < m_gamma0.rows(); i++) {
        gcol.PushBack(rj::Value(m_gamma0(i, j)).Move(), a);
      }
      g0.PushBack(gcol.Move(), a);
    }
    v.AddMember("gamma0", g0.Move(), a);
  }

  if (m_omega.size()) {
    rj::Value o(rj::kArrayType);
    // col-major serialization
    for (int j = 0; j < m_omega.cols(); j++) {
      rj::Value ocol(rj::kArrayType);
      for (int i = 0; i < m_omega.rows(); i++) {
        ocol.PushBack(rj::Value(m_omega(i, j)).Move(), a);
      }
      o.PushBack(ocol.Move(), a);
    }
    v.AddMember("omega", o.Move(), a);
  }

  if (m_u_vec.size()) {
    rj::Value uvec(rj::kArrayType);
    for (int i = 0; i < m_u_vec.rows(); i++) {
      uvec.PushBack(rj::Value(m_u_vec(i, 0)).Move(), a);
    }
    v.AddMember("uvec", uvec.Move(), a);
  }
  if (m_bic_vec.size()) {
    rj::Value bicvec(rj::kArrayType);
    for (int i = 0; i < m_bic_vec.rows(); i++) {
      bicvec.PushBack(rj::Value(m_bic_vec(i, 0)).Move(), a);
    }
    v.AddMember("bicvec", bicvec.Move(), a);
  }

  return v;
}

void RidgeEnvlp::fromJson(rapidjson::Value& v)
{
  namespace rj = rapidjson;
  // run the base class deserialize method
  Ridge::fromJson(v);

  if (v.HasMember("coef_scale")) {
    const rj::Value& cs = v["coef_scale"];
    m_coef_scale        = cs.GetDouble();
  }
  if (v.HasMember("do_coef_scale")) {
    const rj::Value& tf = v["do_coef_scale"];
    m_do_coef_scale     = tf.GetBool();
  }
  if (v.HasMember("zscore_x")) {
    const rj::Value& tf = v["zscore_x"];
    m_zscore_x          = tf.GetBool();
  }
  if (v.HasMember("zscore_y")) {
    const rj::Value& tf = v["zscore_y"];
    m_zscore_y          = tf.GetBool();
  }
  if (v.HasMember("xsqsum")) {
    const rj::Value& iv = v["xsqsum"];
    m_xsqsum.setZero(1, iv.GetArray().Size());
    int j = 0;
    for (auto& xval : iv.GetArray()) {
      m_xsqsum(0, j++) = xval.GetDouble();
    }
  }
  if (v.HasMember("ysqsum")) {
    const rj::Value& iv = v["ysqsum"];
    m_ysqsum.setZero(1, iv.GetArray().Size());
    int j = 0;
    for (auto& yval : iv.GetArray()) {
      m_ysqsum(0, j++) = yval.GetDouble();
    }
  }

  if (v.HasMember("coef_base")) {
    const rj::Value& coef = v["coef_base"];
    int f                 = coef.GetArray().Size();
    int k = -1, i = 0, j = 0;
    for (auto& coefcol : coef.GetArray()) {
      if (coefcol.GetType() != rj::kArrayType) {
        throw std::invalid_argument("inner json object not an array");
      }
      if (k == -1) {
        k = coefcol.GetArray().Size();
        // initialize the size of our m_coef
        m_coef_base.setZero(k, f);
      }
      i = 0;
      for (auto& cval : coefcol.GetArray()) {
        m_coef_base(i++, j) = cval.GetDouble();
      }
      j++;
    }
  }

  if (v.HasMember("u")) {
    const rj::Value& cs = v["u"];
    m_u                 = cs.GetInt();
  }
  if (v.HasMember("objval")) {
    const rj::Value& cs = v["objval"];
    m_objval            = cs.GetDouble();
  }
  if (v.HasMember("loglik")) {
    const rj::Value& cs = v["loglik"];
    m_loglik            = cs.GetDouble();
  }

  if (v.HasMember("gamma")) {
    const rj::Value& g = v["gamma"];
    int f              = g.GetArray().Size();
    int k = -1, i = 0, j = 0;
    for (auto& gcol : g.GetArray()) {
      if (gcol.GetType() != rj::kArrayType) {
        throw std::invalid_argument("inner json object not an array");
      }
      if (k == -1) {
        k = gcol.GetArray().Size();
        m_gamma.setZero(k, f);
      }
      i = 0;
      for (auto& val : gcol.GetArray()) {
        m_gamma(i++, j) = val.GetDouble();
      }
      j++;
    }
  }

  if (v.HasMember("gamma0")) {
    const rj::Value& g = v["gamma0"];
    int f              = g.GetArray().Size();
    int k = -1, i = 0, j = 0;
    for (auto& gcol : g.GetArray()) {
      if (gcol.GetType() != rj::kArrayType) {
        throw std::invalid_argument("inner json object not an array");
      }
      if (k == -1) {
        k = gcol.GetArray().Size();
        m_gamma0.setZero(k, f);
      }
      i = 0;
      for (auto& val : gcol.GetArray()) {
        m_gamma0(i++, j) = val.GetDouble();
      }
      j++;
    }
  }

  if (v.HasMember("omega")) {
    const rj::Value& o = v["omega"];
    int f              = o.GetArray().Size();
    int k = -1, i = 0, j = 0;
    for (auto& ocol : o.GetArray()) {
      if (ocol.GetType() != rj::kArrayType) {
        throw std::invalid_argument("inner json object not an array");
      }
      if (k == -1) {
        k = ocol.GetArray().Size();
        m_omega.setZero(k, f);
      }
      i = 0;
      for (auto& val : ocol.GetArray()) {
        m_omega(i++, j) = val.GetDouble();
      }
      j++;
    }
  }

  if (v.HasMember("uvec")) {
    const rj::Value& l = v["uvec"];
    m_u_vec.setZero(l.GetArray().Size(), 1);
    int i = 0;
    for (auto& val : l.GetArray()) {
      m_u_vec(i++, 0) = val.GetDouble();
    }
  }
  if (v.HasMember("bicvec")) {
    const rj::Value& l = v["bicvec"];
    m_bic_vec.setZero(l.GetArray().Size(), 1);
    int i = 0;
    for (auto& val : l.GetArray()) {
      m_bic_vec(i++, 0) = val.GetDouble();
    }
  }
}
