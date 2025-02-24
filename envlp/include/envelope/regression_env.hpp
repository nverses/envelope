#pragma once

#include <rapidjson/document.h>

#include <Eigen/Dense>

#include "envelope/eigen_types.hpp"
#include "envelope/envopt.hpp"
#include "envelope/regression.hpp"

namespace envelope {

  // performs X-envelope model based on ridge regression
  class RidgeEnvlp : public Ridge
  {
  public:
    RidgeEnvlp(bool fit_intercept = true, double l2_lambda = 0.0,
               bool zscore_x = false, bool zscore_y = false,
               bool scale_coef = false, int u_step = 0);

    // envelope model can be solved with grammian matrices
    void fitGram(const DMatV& xtx, const DMatV& xty, const DMatV& yty = DMatV(),
                 int nrows = 0) override;

    // predict considers possible zscoring
    DMat predict(const DMatV& x, bool include_intercept = true) override;

    // export fit result to json
    rapidjson::Value toJson(
        rapidjson::Document::AllocatorType& a) const override;

    // take in json to load coefs
    void fromJson(rapidjson::Value& val) override;

    void setXZScoring(bool val);
    void setYZScoring(bool val);
    bool getXZScoring() const;
    bool getYZScoring() const;

    // accessor for the estimates
    DMat getXSqSum() const;
    DMat getYSqSum() const;

  public:
    DMat coefsBase() const;
    void setEnvelopeDim(int u);
    int getEnvelopeDim() const;
    DMat getGamma() const;
    DMat getGamma0() const;
    DMat getOmega() const;
    double getObjValue() const;
    double getLogLikelihood() const;
    // sweep resutls
    DMat getDimList() const;
    DMat getBICList() const;

  protected:
    GramTriplet prepareGramZ3(const DMatV& xtx, const DMatV& xty,
                              const DMatV& yty, double denominator);
    DMat getInterceptOffsetZ(bool include_yoffset) const;

    void computeXEnvGammas(const DMatV& xtx, const DMatV& xty, const DMatV& yty,
                           int nrows);

    bool m_zscore_x;
    bool m_zscore_y;
    bool m_do_coef_scale;
    double m_coef_scale;
    DMat m_xsqsum;  // used for standardizing
    DMat m_ysqsum;  // used for standardizing

    DMat m_coef_base;  // cannonical ridge solution
    int m_u_step;      // step size of u dimension sweep
    int m_u;
    double m_objval;
    double m_objval_offset;
    double m_loglik;
    DMat m_gamma;
    DMat m_gamma0;
    DMat m_omega;  // gamma' * xtx * gamma (u x u)
    DMat m_u_vec;
    DMat m_bic_vec;

    EnvOpt m_env;  // envelope optimizer
  };

}  // namespace envelope
