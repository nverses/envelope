#pragma once

#include <Eigen/Dense>
#include <tuple>

#include "envelope/eigen_types.hpp"
#include "envelope/la_util.hpp"

namespace envelope {

  class EnvOpt
  {
  public:
    EnvOpt();
    EnvOpt(DMatV M, DMatV U, DMatV init = DMatV());

    struct RunStats_t
    {
      double elapsed = 0.0;  // total time taken to solve
      int iter       = 0;    // combines max_iter x coord descent iter
      int lbfgs_iter = 0;    // total iterations taken under lbfgs opt
    };

    void setData(DMatV M, DMatV U, DMatV init = DMatV());

    // solve for envelope subspace dimension u
    void solve(int u);

    // solve in parallel, return log-lik values for u list, req no. samples n
    // Inputs:
    //  objval_offset : for xenv np.log(np.eig(sigY)[0]).sum(), zero otherwise
    //  degree        : for xenv with single target r=1, zero otherwise
    std::pair<DMat, int> sweep(const std::vector<int>& uvec, int n_samples,
                               double objval_offset = 0.0, int degree = 1);
    std::pair<DMat, int> sweepArr(const IArr& uarr, int n_samples,
                                  double objval_offset = 0.0, int degree = 1);

    DMat getGammaHat() const;
    DMat getGamma0Hat() const;
    std::pair<DMat, DMat> getGammas() const;

    double getObjValue() const;
    static double calcLogLikelihood(double objval, int n, int t,
                                    int degree = 0);

    void setMaxIter(int maxiter);
    RunStats_t getStats() const;

  public:
    static std::tuple<DMat, DMat, double> solveEnvelope(
        DMatV M, DMatV U, int u, DMatV init = DMatV(), int maxiter = 100,
        double ftol = 1e-3, RunStats_t* stats = nullptr);

    // returns {init, invMU, eigvals_of_MU, objval}
    static std::tuple<DMat, DMat, DMat, double> computeInitial(
        DMatV M, DMatV U, int u, DMatV init = DMatV());

  private:
    static double calcObjective(DMatV gx, DMatV sigma1, DMatV sigma2);

    static DMat pickInitial(DMatV E, DMatV sigma1, unsigned ncols);

    // returns {gammahat (r x u), gamma0hat (r x r-u), objval}
    static std::tuple<DMat, DMat, double> solve1(DMatV M, DMatV U,
                                                 DMatV init        = DMatV(),
                                                 int maxiter       = 100,
                                                 double ftol       = 1e-3,
                                                 RunStats_t* stats = nullptr);
    // returns {gammahat (r x u), gamma0hat (r x r-u), objval}
    static std::tuple<DMat, DMat, double> solven(
        DMatV M, DMatV U, int u, DMatV init = DMatV(), bool lastonly = false,
        int maxiter = 100, double ftol = 1e-3, RunStats_t* stats = nullptr);

    DMat m_M;
    DMat m_U;
    DMat m_init;
    int m_maxiter;
    double m_ftol;

    double m_objval;
    DMat m_gammahat;
    DMat m_gamma0hat;

    RunStats_t m_st;
  };

}  // namespace envelope
