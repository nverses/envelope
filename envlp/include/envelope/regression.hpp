#pragma once

#include <rapidjson/document.h>

#include <Eigen/Dense>

#include "envelope/gram.hpp"

namespace envelope {

  //
  // linear regression suite based on least-square sum loss function
  //
  class Regression
  {
  public:
    Regression(bool fit_intercept = true);
    virtual ~Regression();

    // fit using the full data
    virtual void fit(const DMatV& x, const DMatV& y, const DMatV& w = DMatV());

    // solve xtx, xty directly, yty and nrows are optional
    virtual void fitGram(const DMatV& xtx, const DMatV& xty,
                         const DMatV& yty = DMatV(), int nrows = 0);

    // retrieve the solved coef
    virtual DMat coefs() const;

    // predict using the coef (requires fit() call)
    virtual DMat predict(const DMatV& x, bool include_intercept = true);

    // export fit result to json
    virtual rapidjson::Value toJson(
        rapidjson::Document::AllocatorType& a) const;

    // take in json to load coefs
    virtual void fromJson(rapidjson::Value& val);

    // convenience to return mode as json string
    std::string toJsonString();
    void fromJsonString(const std::string& jsstr);

    // return a copy to keep it simple
    int getNRows() const;
    DMat getXSum() const;
    DMat getYSum() const;

    static DMat calcMSE(const DMatV& y, const DMatV& yh);

    static DMat calcRsq(const DMatV& y, const DMatV& yh);

    DMat getInterceptOffset(bool include_offset = true) const;

  protected:
    std::pair<DMat, DMat> prepareGram(const DMatV& xtx, const DMatV& xty);
    // if we demeaned the x, y we will have offset to be used in predict
    static DMat doPredict(const DMatV& x, const DMatV& coef,
                          const DMatV& offset = DMatV());
    bool m_intercept;
    int m_nrows;
    DMat m_coef;  // k (features) x f (targets)
    DMat m_xsum;  // 1 x k (features)
    DMat m_ysum;  // 1 x f (targets)
    GramLsq m_gram;
  };

  class Ridge : public Regression
  {
  public:
    Ridge(bool fit_intercept = true, double l2_lambda = 0.0);

    // solve xtx, xty directly, yty and nrows are optional
    void fitGram(const DMatV& xtx, const DMatV& xty, const DMatV& yty = DMatV(),
                 int nrows = 0) override;

    void setL2Lambda(double l2);
    double getL2Lambda() const;

    // export fit result to json
    rapidjson::Value toJson(
        rapidjson::Document::AllocatorType& a) const override;

    // take in json to load coefs
    void fromJson(rapidjson::Value& val) override;

  protected:
    double m_l2_lambda;
  };

  class ZScoreTransformer
  {
  public:
    ZScoreTransformer();
    // compute the mean and variance of x
    void fit(const DMatV& x);

    // extract the mean and variance from xtx
    void fitGram(const DMatV& xtx, int nrows);

    // transform original data X matrix with extracted mu, var
    DMat transform(const DMatV& x);

    // transform xtx and xty matrix with extracted mu, var, requires
    // xtx and xty generated with intercept.
    // (see: GramUtil::dot(x,y,with_intercept))
    std::pair<DMat, DMat> transformGram(const DMatV& xtx, const DMatV& xty,
                                        bool retain_intercept = true);

    // get the size
    int getSize() const;
    // return the estimated values
    std::pair<DMat, DMat> getMoments() const;

    // export fit result to json
    rapidjson::Value toJson(rapidjson::Document::AllocatorType& a) const;
    // take in json to load coefs
    void fromJson(rapidjson::Value& val);

  private:
    // sums from X
    int m_nrows;
    DMat m_sum;    // 1 x k
    DMat m_sqsum;  // 1 x k
  };

}  // namespace envelope
