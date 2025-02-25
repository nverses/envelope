#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wzero-as-null-pointer-constant"

#include "envelope/regression.hpp"

#include "rapidjson/document.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"
#include <chrono>
#include <iostream>

using namespace envelope;

Regression::Regression(bool intercept)
  : m_intercept(intercept)
  , m_nrows(0)
{
}

Regression::~Regression() {}

void Regression::fit(const DMatV& x, const DMatV& y, const DMatV& raw_w)
{
  if (x.rows() != y.rows() || (raw_w.rows() > 0 && x.rows() != raw_w.rows())) {
    throw std::invalid_argument(
        "Regression::fit(): passed in matrices do not have same length");
  }

  // Scale weights to sum to nrows
  DMat w;
  if (raw_w.rows()) {
    w = raw_w * (x.rows() / raw_w.sum());
  }

  // differentiate between weighted and non-weighted
  auto [xtx, xty, yty] = w.size() ? GramUtil::dotw(x, y, w, m_intercept)
                                  : GramUtil::dot(x, y, m_intercept);

  if (m_intercept) {
    m_xsum = xtx.block(0, 1, 1, xtx.cols() - 1);
    m_ysum = xty.block(0, 0, 1, xty.cols());
  }

  m_nrows = x.rows();
  // additionally center the x, y (or update xtx, xty)
  fitGram(xtx, xty, yty, x.rows());
}

std::pair<DMat, DMat> Regression::prepareGram(const DMatV& xtx,
                                              const DMatV& xty)
{
  // NOTE:
  // In the presence of intercept, we center our inputs and save the
  // corresponding xsums, ysums to be able to reconstruct the offset
  // in prediction generation. But this is an annoying unavoidable copy,
  // unless you trust the user to prepare the data prior to fitting...

  if (m_intercept) {
    if (m_xsum.size() == 0) {
      m_xsum = xtx.block(0, 1, 1, xtx.cols() - 1);
    }
    if (m_ysum.size() == 0) {
      m_ysum = xty.block(0, 0, 1, xty.cols());
    }

    if (m_nrows == 0) {
      m_nrows = xtx(0, 0);
    }

    double x_offset_diff = fabs(xtx.block(1, 1, xtx.rows() - 1, xtx.cols() - 1)
                                    .colwise()
                                    .sum()
                                    .cwiseAbs()
                                    .sum());
    if (1e-9 < x_offset_diff) {
      auto [xtxd, xtyd] = GramUtil::demean(xtx, xty, m_xsum, m_ysum, xtx(0, 0));
      return {xtxd, xtyd};
    }
  }
  return {xtx, xty};
}

void Regression::fitGram(const DMatV& xtx, const DMatV& xty, const DMatV&, int)
{
  auto [xtxp, xtyp] = prepareGram(xtx, xty);
  m_coef            = m_gram.solveThru(xtxp, xtyp);
}

DMat Regression::coefs() const
{
  // if we have the intercept, we have extra row, just return the
  // coefs for the original data.
  if (m_intercept && m_coef.size()) {
    return m_coef.block(1, 0, m_coef.rows() - 1, m_coef.cols());
  }
  return m_coef;
}

int Regression::getNRows() const
{
  return m_nrows;
}

DMat Regression::getXSum() const
{
  return m_xsum;
}

DMat Regression::getYSum() const
{
  return m_ysum;
}

DMat Regression::getInterceptOffset(bool include_offset) const
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
    return DMat::Constant(m_ysum.rows(), m_ysum.cols(), 0.0);
  }
  int t = m_coef.size() ? m_coef.cols() : m_ysum.size() ? m_ysum.cols() : 0;
  if (!t) {
    return DMat::Constant(m_ysum.rows(), m_ysum.cols(), 0.0);
  }

  // intercept_offset = E[y] - E[x] * b
  DMat offset = -((m_xsum / m_nrows) * coefs());
  offset += (m_ysum / m_nrows);
  return offset;
}

DMat Regression::predict(const DMatV& x, bool include_intercept)
{
  if (x.cols() + 1 == m_coef.rows()) {
    if (!m_intercept) {
      std::cerr
          << "Regression::predict(): WARN intercept parameter not specified "
             "but coefficient matrix seem to contain intercept row"
          << std::endl;
    }
  }
  DMat offset = getInterceptOffset(include_intercept);
  return Regression::doPredict(x, m_coef, offset);
}

DMat Regression::doPredict(const DMatV& x, const DMatV& coef,
                           const DMatV& offset)
{
  // if intercept exists, the predict function needs this offset:
  // y.mean() - x.mean(axis=0) @ sklcv.coef_

  DMat yh;
  // size matches, we can do simple dot product
  if (x.cols() == coef.rows()) {
    yh = x * coef;
    if (offset.size() && offset.cols() == yh.cols()) {
      yh.rowwise() += offset.row(0);
    }
    return yh;
  }
  // coefficient contains extra row for intercept term
  else if (x.cols() + 1 == coef.rows()) {
    yh = x * coef.block(1, 0, coef.rows() - 1, coef.cols());
    // contribution from intercept - when x, y are demeaned, the offset
    // is contribution of x-mean, y-mean from train dataset.
    if (offset.size() && offset.cols() == yh.cols()) {
      yh.rowwise() += offset.row(0);
    }
    return yh;
  }
  throw std::invalid_argument(
      "Regression::doPredict(): provided x matrix dimensions do not match. If "
      "intercept option was turned on, do not provide columns with 1's");
}

DMat Regression::calcMSE(const DMatV& y, const DMatV& yh)
{
  // MSE = 1/n sum( (y-yh)^2 )
  return (y - yh).colwise().squaredNorm() / y.rows();
}

DMat Regression::calcRsq(const DMatV& y, const DMatV& yh)
{
  // Rsq = 1.0 - sum( (y-yh)^2 ) / sum( (y - E[y])^2 )
  DMat ymu   = y.colwise().mean();
  DMat sse   = (y - yh).colwise().squaredNorm();
  DMat sst   = (y.rowwise() - ymu.row(0)).colwise().squaredNorm();
  DMat ratio = sse.array() / sst.array();
  return DMat::Ones(ratio.rows(), ratio.cols()) - ratio;
}

rapidjson::Value Regression::toJson(rapidjson::Document::AllocatorType& a) const
{
  namespace rj = rapidjson;
  rj::Value v(rj::kObjectType);
  int k = m_coef.rows();  // number of features
  int f = m_coef.cols();  // number of y columns
  rj::Value coef(rj::kArrayType);
  for (int j = 0; j < f; j++) {
    rj::Value coefcol(rj::kArrayType);
    for (int i = 0; i < k; i++) {
      coefcol.PushBack(rj::Value(m_coef(i, j)).Move(), a);
    }
    coef.PushBack(coefcol.Move(), a);
  }

  if (m_xsum.size()) {
    rj::Value xsum(rj::kArrayType);
    for (int j = 0; j < m_xsum.cols(); j++) {
      xsum.PushBack(rj::Value(m_xsum(0, j)).Move(), a);
    }
    v.AddMember("xsum", xsum.Move(), a);
  }
  if (m_ysum.size()) {
    rj::Value ysum(rj::kArrayType);
    for (int j = 0; j < m_ysum.cols(); j++) {
      ysum.PushBack(rj::Value(m_ysum(0, j)).Move(), a);
    }
    v.AddMember("ysum", ysum.Move(), a);
  }

  v.AddMember("coef", coef.Move(), a);
  v.AddMember("nrows", rj::Value(m_nrows).Move(), a);
  v.AddMember("fit_intercept", rj::Value(m_intercept).Move(), a);
  v.AddMember("name", "Regression", a);
  return v;
}

std::string Regression::toJsonString()
{
  namespace rj = rapidjson;
  rj::Document d;
  d.SetObject();
  rj::Document::AllocatorType& a = d.GetAllocator();
  // calls the virtual function
  rj::Value js = this->toJson(a);
  js.Swap(d);
  rapidjson::StringBuffer buffer;
  rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
  d.Accept(writer);
  return buffer.GetString();
}

void Regression::fromJsonString(const std::string& js)
{
  namespace rj = rapidjson;
  rj::Document d;
  d.Parse(js.c_str());

  this->fromJson(d);
}

void Regression::fromJson(rapidjson::Value& v)
{
  namespace rj = rapidjson;
  if (v.IsNull() || !v.HasMember("coef") ||
      v["coef"].GetType() != rj::kArrayType) {
    throw std::invalid_argument(
        "json object do not have 'coef' key with array values");
  }
  const rj::Value& coef = v["coef"];

  int f = coef.GetArray().Size();
  int k = -1;
  int i = 0;
  int j = 0;
  for (auto& coefcol : coef.GetArray()) {
    if (coefcol.GetType() != rj::kArrayType) {
      throw std::invalid_argument("inner json object not an array");
    }
    if (k == -1) {
      k = coefcol.GetArray().Size();
      // initialize the size of our m_coef
      m_coef.setZero(k, f);
    }
    i = 0;
    for (auto& cval : coefcol.GetArray()) {
      m_coef(i++, j) = cval.GetDouble();
    }
    j++;
  }
  // m_xsum, m_ysum can be optional
  if (v.HasMember("xsum")) {
    const rj::Value& iv = v["xsum"];
    m_xsum.setZero(1, iv.GetArray().Size());
    int j = 0;
    for (auto& xval : iv.GetArray()) {
      m_xsum(0, j++) = xval.GetDouble();
    }
  }
  if (v.HasMember("ysum")) {
    const rj::Value& iv = v["ysum"];
    m_ysum.setZero(1, iv.GetArray().Size());
    int j = 0;
    for (auto& yval : iv.GetArray()) {
      m_ysum(0, j++) = yval.GetDouble();
    }
  }
  if (v.HasMember("nrows")) {
    const rj::Value& nval = v["nrows"];
    m_nrows               = nval.GetInt();
  }
  if (v.HasMember("fit_intercept")) {
    const rj::Value& intcpt = v["fit_intercept"];
    m_intercept             = intcpt.GetBool();
  }
}

Ridge::Ridge(bool fit_intercept, double l2_lambda)
  : Regression(fit_intercept)
  , m_l2_lambda(l2_lambda)
{
}

Ridge::~Ridge() {}

void Ridge::fitGram(const DMatV& xtx, const DMatV& xty, const DMatV&, int)
{
  // incurs copy since we need to add to diag
  auto [xtxp, xtyp] = prepareGram(xtx, xty);
  if (m_l2_lambda > 0) {
    GramUtil::addRidge(xtxp, m_l2_lambda, m_intercept);
  }
  m_coef = m_gram.solveThru(xtxp, xtyp);
}

void Ridge::setL2Lambda(double l2)
{
  m_l2_lambda = l2;
}

double Ridge::getL2Lambda() const
{
  return m_l2_lambda;
}

rapidjson::Value Ridge::toJson(rapidjson::Document::AllocatorType& a) const
{
  namespace rj = rapidjson;
  rj::Value v  = Regression::toJson(a);
  v.AddMember("l2_lambda", rj::Value(m_l2_lambda).Move(), a);
  if (v.HasMember("name")) {
    v["name"].Swap(rj::Value("Ridge", a).Move());
  }
  else {
    v.AddMember("name", "Ridge", a);
  }
  return v;
}

void Ridge::fromJson(rapidjson::Value& v)
{
  namespace rj = rapidjson;
  Regression::fromJson(v);
  if (v.HasMember("l2_lambda")) {
    const rj::Value& l2 = v["l2_lambda"];

    m_l2_lambda = l2.GetDouble();
  }
}

//
// ZScoreTransformer
//
ZScoreTransformer::ZScoreTransformer()
  : m_nrows(0)
{
}

void ZScoreTransformer::fit(const DMatV& x)
{
  m_nrows = x.rows();
  m_sum   = x.colwise().sum();
  m_sqsum = x.colwise().squaredNorm();
}

void ZScoreTransformer::fitGram(const DMatV& xtx, int nrows)
{
  if (!xtx.size()) {
    throw std::invalid_argument(
        "ZScoreTransformer::fitGram(): xtx matrix is empty");
  }
  if (static_cast<int>(xtx(0, 0)) != nrows) {
    throw std::invalid_argument(
        "ZScoreTransformer::fitGram(): xtx matrix needs to constructed with "
        "an "
        "intercept");
  }
  int n   = xtx.rows() - 1;
  m_nrows = nrows;
  m_sum   = xtx.block(0, 1, 1, n);
  m_sqsum = xtx.block(1, 1, n, n).diagonal().transpose();
}

DMat ZScoreTransformer::transform(const DMatV& x)
{
  DMat xz = x;
  for (int j = 0; j < x.cols(); j++) {
    double mu = m_sum(0, j) / m_nrows;
    double sd = sqrt(m_sqsum(0, j) / m_nrows - pow(mu, 2));
    xz.col(j).array() -= mu;
    xz.col(j).array() /= sd;
  }
  return xz;
}

std::pair<DMat, DMat> ZScoreTransformer::transformGram(const DMatV& xtx,
                                                       const DMatV& xty,
                                                       bool retain_intercept)
{
  // Standardize returns new copy of xtx, xty with intercept strips intact
  // however, the top row and left column gets all zeros except the top-left
  // corner that contains the number of samples. Solving the regression
  // with these intercept dot products yields first row coefficients having
  // exactly the average value of the target Y.
  auto [xtxz, xtyz] = GramUtil::standardize(xtx, xty, m_sum, m_sqsum, m_nrows);
  if (!retain_intercept) {
    // We strip the xtxz, xtyz to contain values pertaining to X cols only.
    // We avoid aliasing issues in Eigen by calling .eval()
    xtxz = xtxz.block(1, 1, xtxz.rows() - 1, xtxz.cols() - 1).eval();
    xtyz = xtyz.block(1, 0, xtyz.rows() - 1, xtyz.cols()).eval();
  }
  return {xtxz, xtyz};
}

int ZScoreTransformer::getSize() const
{
  return m_nrows;
}

std::pair<DMat, DMat> ZScoreTransformer::getMoments() const
{
  return {m_sum, m_sqsum};
}

rapidjson::Value ZScoreTransformer::toJson(
    rapidjson::Document::AllocatorType& a) const
{
  namespace rj = rapidjson;
  rj::Value v(rj::kObjectType);
  if (!m_sum.size()) {
    return v;
  }
  if (m_sum.cols() != m_sqsum.cols()) {
    throw std::invalid_argument(
        "ZScoreTransfomer::toJson(): m_sum and m_sqsum do not have the same "
        "size");
  }
  int k = m_sum.cols();  // number of features
  rj::Value sumarr(rj::kArrayType);
  rj::Value sqsumarr(rj::kArrayType);
  for (int j = 0; j < k; j++) {
    sumarr.PushBack(rj::Value(m_sum(0, j)).Move(), a);
    sqsumarr.PushBack(rj::Value(m_sqsum(0, j)).Move(), a);
  }
  v.AddMember("nrows", rj::Value(m_nrows).Move(), a);
  v.AddMember("sum", sumarr.Move(), a);
  v.AddMember("sqsum", sqsumarr.Move(), a);
  v.AddMember("name", "ZScoreTransformer", a);
  return v;
}

void ZScoreTransformer::fromJson(rapidjson::Value& v)
{
  namespace rj = rapidjson;
  if (v.IsNull() || !v.HasMember("sum") ||
      v["sum"].GetType() != rj::kArrayType) {
    throw std::invalid_argument(
        "json object do not have 'sum' key with array values");
  }
  if (!v.HasMember("nrows") || !v.HasMember("sqsum") ||
      v["sqsum"].GetType() != rj::kArrayType) {
    throw std::invalid_argument(
        "json object do not have 'nrows' key or 'sqsum' key with array "
        "values");
  }
  const rj::Value& sumarr   = v["sum"];
  const rj::Value& sqsumarr = v["sqsum"];

  int k = sumarr.GetArray().Size();
  m_sum.setZero(1, k);
  m_sqsum.setZero(1, k);
  int j = 0;
  for (auto& item : sumarr.GetArray()) {
    m_sum(0, j++) = item.GetDouble();
  }
  j = 0;
  for (auto& item : sqsumarr.GetArray()) {
    m_sqsum(0, j++) = item.GetDouble();
  }
  if (v.HasMember("nrows")) {
    const rj::Value& rowsv = v["nrows"];
    m_nrows                = rowsv.GetInt();
  }
}

#pragma GCC diagnostic pop
