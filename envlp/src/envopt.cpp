#include "envelope/envopt.hpp"

#include <lbfgs.h>
#include <math.h>
#include <omp.h>

#include <chrono>
#include <iostream>

#include "envelope/la_util.hpp"
#include "envelope/mat_util.hpp"

using namespace envelope;

EnvOpt::EnvOpt()
  : m_maxiter(100)
  , m_ftol(1e-3)
  , m_objval(0.0)
{
}

EnvOpt::EnvOpt(DMatV M, DMatV U, DMatV init)
  : EnvOpt()
{
  setData(M, U, init);
}

void EnvOpt::setData(DMatV M, DMatV U, DMatV init)
{
  // do validation checks here
  if (M.rows() != M.cols()) {
    throw std::invalid_argument("M is not a square matrix");
  }
  if (U.rows() != U.cols()) {
    throw std::invalid_argument("U is not a square matrix");
  }
  if (M.rows() != U.rows() || M.cols() != U.cols()) {
    throw std::invalid_argument("M and U dimensions needs to match");
  }
  Eigen::FullPivLU<DMat> lu(M);
  if (!lu.isInvertible()) {
    throw std::invalid_argument("M is not positive definite, not invertible");
  }
  m_M = M;  // copies
  m_U = U;  // copies
  if (init.size()) {
    m_init = init;
  }
}

void EnvOpt::solve(int u)
{
  if (!m_M.size() || !m_U.size()) {
    throw std::invalid_argument("M or U empty, call EnvOpt::setData() first");
  }
  // this is ugly. conversion from empty DMat() to DMatV() causes issues
  DMatV init0;
  if (m_init.size()) {
    init0 = DMatV(m_init);
  }
  auto [gamma, gamma0, objval] =
      EnvOpt::solveEnvelope(m_M, m_U, u, init0, m_maxiter, m_ftol, &m_st);

  m_gammahat  = std::move(gamma);
  m_gamma0hat = std::move(gamma0);
  m_objval    = objval;
}

double EnvOpt::calcLogLikelihood(double objval, int n, int t, int degree)
{
  double loglik = -double(n) * (t + degree) / 2.0 * (log(2.0 * M_PI) + 1.0) -
                  double(n) / 2.0 * (objval);
  return loglik;
}

std::pair<DMat, int> EnvOpt::sweep(const std::vector<int>& uvec, int n,
                                   double objval_offset, int degree)
{
  // the total dim is usually: r for response, p for predictor
  int t = m_M.cols();  // total dimension of the problem
  int k = uvec.size();
  DMat bicv(k, 1);
  DMatV init0;  // empty

  // optimize in parallel across list of u envelope subspace dims
  int i;
#pragma omp parallel for schedule(dynamic), private(i)
  for (i = 0; i < k; i++) {
    int u = std::max(0, std::min(uvec[i], t));  // u range: [0, totdim]
    RunStats_t st;
    auto [gamma, gamma0, objval] =
        EnvOpt::solveEnvelope(m_M, m_U, u, init0, m_maxiter, m_ftol, &st);
    // compute the log-likelihood value
    // yenv: loglik = -n * t / 2 * (log(2 * M_PI) + 1) - n / 2 * objfun
    // NOTE: the xenv loglik contains objval_offset
    double loglik =
        EnvOpt::calcLogLikelihood(objval + objval_offset, n, t, degree);
    double penalty = double(t) + t * (t + 1) / 2.0 + degree * double(u);
    double bic     = -2.0 * loglik + log(double(n)) * penalty;
    // NOTE: should we include AIC in the output?
    // double aic = -2.0 * loglik + 2 * penalty;
    bicv(i, 0) = bic;
  }
  // now compute the min of loglik
  double min_val = bicv(0, 0);
  int min_u      = uvec[0];
  for (int i = 0; i < k; i++) {
    if (bicv(i, 0) < min_val) {
      min_val = bicv(i, 0);
      min_u   = uvec[i];
    }
  }
  return {bicv, min_u};
}

std::pair<DMat, int> EnvOpt::sweepArr(const IArr& uarr, int n_samples,
                                      double objval_offset, int p)
{
  std::vector<int> uvec(uarr.data(), uarr.data() + uarr.size());
  return sweep(uvec, n_samples, objval_offset, p);
}

std::tuple<DMat, DMat, double> EnvOpt::solveEnvelope(DMatV M, DMatV U, int u,
                                                     DMatV init0, int maxiter,
                                                     double ftol,
                                                     RunStats_t* st)
{
  int r = M.rows();
  if (u > M.rows() || u < 0) {
    throw std::invalid_argument("u should be between 0 and r");
  }
  // based on chosen envelope subspace dimension
  if (u == 0) {
    // no material info
    DMat gammahat  = DMat();  // DMat::Zero(r, 1);
    DMat gamma0hat = DMat::Identity(r, r);
    DMat MU        = M + U;
    auto [ev, E]   = LAU::eig(MU);
    double objval  = ev.array().log().sum();
    return {std::move(gammahat), std::move(gamma0hat), objval};
  }
  else if (u == r) {
    // all material info
    DMat gammahat  = DMat::Identity(r, r);
    DMat gamma0hat = DMat();  // DMat::Zero(r, 1);
    auto [ev, E]   = LAU::eig(M);
    double objval  = ev.array().log().sum();
    return {std::move(gammahat), std::move(gamma0hat), objval};
  }
  else if (u == 1) {
    // material info in single dim u=1
    auto [gammahat, gamma0hat, objval] =
        EnvOpt::solve1(M, U, init0, maxiter, ftol, st);
    return {std::move(gammahat), std::move(gamma0hat), objval};
  }
  else if (u == r - 1 && u != 1) {
    // material info in all but 1 dim u=r-1
    auto [gammahat, gamma0hat, objval] =
        EnvOpt::solven(M, U, u, init0, true, maxiter, ftol, st);
    return {std::move(gammahat), std::move(gamma0hat), objval};
  }
  else {
    // material info somewhere between u=[2, r-2]
    auto [gammahat, gamma0hat, objval] =
        EnvOpt::solven(M, U, u, init0, false, maxiter, ftol, st);
    return {std::move(gammahat), std::move(gamma0hat), objval};
  }
}

void EnvOpt::setMaxIter(int maxiter)
{
  m_maxiter = maxiter;
}

DMat EnvOpt::getGammaHat() const
{
  return m_gammahat;
}

DMat EnvOpt::getGamma0Hat() const
{
  return m_gamma0hat;
}

std::pair<DMat, DMat> EnvOpt::getGammas() const
{
  return {m_gammahat, m_gamma0hat};
}

double EnvOpt::getObjValue() const
{
  return m_objval;
}

EnvOpt::RunStats_t EnvOpt::getStats() const
{
  return m_st;
}

double EnvOpt::calcObjective(DMatV x, DMatV sigma1, DMatV sigma2)
{
  DMat f1        = x.transpose() * sigma1 * x;
  DMat f2        = x.transpose() * sigma2 * x;
  auto [ev1, E1] = LAU::eig(f1);
  auto [ev2, E2] = LAU::eig(f2);
  double v1      = ev1.array().log().sum();
  double v2      = ev2.array().log().sum();
  return v1 + v2;
}

DMat EnvOpt::pickInitial(DMatV E, DMatV sigma1, unsigned ncols)
{
  int m = E.rows();
  int n = E.cols();
  std::vector<double> esvals;
  for (int j = 0; j < n; j++) {
    DMat vec     = E.col(j);
    double esval = (vec.transpose() * sigma1 * vec)(0);
    esvals.push_back(esval);
  }
  // descending sort index
  std::vector<size_t> idx = Vec::argsort(esvals, false);
  int nchoose             = std::min(ncols, unsigned(idx.size()));
  DMat E0                 = DMat::Zero(m, nchoose);
  for (int j = 0; j < nchoose; j++) {
    int ix    = idx[j];
    E0.col(j) = E.col(ix);
  }
  return E0;
}

std::tuple<DMat, DMat, DMat, double> EnvOpt::computeInitial(DMatV M, DMatV U,
                                                            int u, DMatV initog)
{
  DMat MU        = M + U;
  auto [ev, E]   = LAU::eig(MU);
  auto [mev, mE] = LAU::eig(M);
  int t          = M.rows();

  DMat evdiv = (1 / ev.array()).matrix().asDiagonal();
  DMat invMU = (E * evdiv) * E.transpose();

  DMat evsqrtdiv = (1 / ev.array().sqrt()).matrix().asDiagonal();
  DMat invMU2    = (E * evsqrtdiv) * E.transpose();

  DMat mevsqrtdiv = (1 / mev.array().sqrt()).matrix().asDiagonal();
  DMat invM2      = (mE * mevsqrtdiv) * mE.transpose();

  DMat init   = DMat::Zero(t, u);  // initialize mem with zeros
  double obj0 = 1e9;               // large number

  // note use of capture vars for update
  auto calc_min = [&init, &obj0, &M, &invMU, &u](const DMat& E,
                                                 const DMat& mid) {
    DMat init_tmp  = EnvOpt::pickInitial(E, mid, u);
    double obj_tmp = calcObjective(init_tmp, M, invMU);
    if (obj_tmp < obj0) {
      init = init_tmp;
      obj0 = obj_tmp;
    }
  };

  if (initog.cols() == u && initog.rows() == t) {
    init = initog;
    obj0 = calcObjective(init, M, invMU);
  }
  else {
    // 4 candidates for initial value
    calc_min(E, U);
    calc_min(E, invMU2 * U * invMU2.transpose());
    calc_min(mE, U);
    calc_min(mE, invM2 * U * invM2.transpose());
  }

  return {init, invMU, ev, obj0};
}

static DMat _remove_row(const DMat& m, int rm_row)
{
  DMat out(m.rows() - 1, m.cols());

  for (int i = 0, k = 0; i < m.rows(); i++) {
    if (i == rm_row) {
      continue;
    }
    out.row(k++) = m.row(i);
  }
  return out;
}

struct EvalData_t
{
  DMatV t2;
  DMatV t3;
  DMatV invt4;
  DMatV invC1;
  DMatV invC2;
  double Mj;
  double invMUj;
  int k;  // lbfgs iterations
};

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wunused-function"
static lbfgsfloatval_t evaluate(void* instance, const lbfgsfloatval_t* xptr,
                                lbfgsfloatval_t* gptr, const int n,
                                const lbfgsfloatval_t step)
{
  EvalData_t* d = reinterpret_cast<EvalData_t*>(instance);
  DMatV x(const_cast<double*>(xptr), 1, n);  // 1 x n
  DMat tmp2 = x + d->t2;                     // 1 x n
  DMat tmp3 = x + d->t3;                     // 1 x n
  DMat f1   = d->invt4 * x.transpose();      // n x 1 (invt4: n x n)
  DMat f2   = d->invC1 * tmp2.transpose();   // n x 1 (invC1: n x n)
  DMat f3   = d->invC2 * tmp3.transpose();   // n x 1 (invC2: n x n)
  // objective scalar
  double fval = -2 * log(1 + (x * f1)(0)) + log(1 + d->Mj * (tmp2 * f2)(0)) +
                log(1 + d->invMUj * (tmp3 * f3)(0));
  // gradient vector
  DMat gv = -4 * f1.array() / (1 + (x * f1)(0)) +
            2 * f2.array() / (1 / d->Mj + (tmp2 * f2)(0)) +
            2 * f3.array() / (1 / d->invMUj + (tmp3 * f3)(0));  // n x 1
  // update the gradient into gptr pointer
  for (int i = 0; i < gv.rows(); i++) {
    gptr[i] = gv(i, 0);
  }
  d->k++;
  return fval;
}

static int progress(void* instance, const lbfgsfloatval_t* x,
                    const lbfgsfloatval_t* g, const lbfgsfloatval_t fx,
                    const lbfgsfloatval_t xnorm, const lbfgsfloatval_t gnorm,
                    const lbfgsfloatval_t step, int n, int k, int ls)
{
  printf("lbfgs iter %d, fx = %f, xnorm = %f, gnorm = %f step = %f\n", k, fx,
         xnorm, gnorm, step);
  return 0;
}
#pragma GCC diagnostic pop

std::tuple<DMat, DMat, double> EnvOpt::solven(DMatV M, DMatV U, int u,
                                              DMatV init0, bool lastonly,
                                              int maxiter, double ftol,
                                              RunStats_t* stats)
{
  if (sizeof(lbfgsfloatval_t) != sizeof(double)) {
    throw std::runtime_error(
        "lbfgsfloatval_t should be the same size os double!");
  }
  auto start = std::chrono::high_resolution_clock::now();

  int r = M.rows();
  // compute the initial value
  auto [init, invMU, ev, obj0] = EnvOpt::computeInitial(M, U, u, init0);

  // get the top u equations, and create normalized initial value
  IArr geidx = LAU::geindex(init);
  DMat initu(u, init.cols());  // u x u
  for (int i = 0; i < u; i++) {
    initu.row(i) = init.row(geidx(i));
  }
  DMat I     = DMat::Identity(initu.rows(), initu.rows());
  DMat sol   = LAU::solve(initu, I);  // solve the subset u
  DMat Ginit = init * sol;            // (r x u) * (u x u) = r x u

  // create the matrices required for coordinate descent
  DMat GUG = Ginit.transpose() * M * Ginit;      // u x u
  DMat GVG = Ginit.transpose() * invMU * Ginit;  // u x u
  // initv "complement" of initu, created from Ginit
  DMat initv(r - u, init.cols());  // r-u x u
  for (int i = 0; i < r - u; i++) {
    initv.row(i) = Ginit.row(geidx(u + i));
  }
  DMat t4 = (initv.transpose() * initv) + DMat::Identity(u, u);  // u x u

  // now loop through and optimize
  DMat gammahat;
  DMat fullQ;
  double objval = obj0;
  // coordinate descent rows: default subset (r - u) rows
  IArr geidx_sub;
  if (lastonly) {
    geidx_sub    = IArr::Zero(1);
    geidx_sub(0) = geidx(r - 1);
  }
  else {
    geidx_sub = IArr::Zero(r - u);
    for (int i = 0; i < r - u; i++) {
      geidx_sub(i) = geidx(u + i);
    }
  }
  // n is same as u (dimension of envelope subspace, solve dim for lbfgs)
  int n = Ginit.cols();

  // L-BFGS related
  lbfgs_parameter_t param;
  lbfgs_parameter_init(&param);
  lbfgsfloatval_t fx;
  lbfgsfloatval_t* x = lbfgs_malloc(n);  // required for SSE2

  int i = 0, tot_i = 0, lbfgs_i = 0;
  while (i < maxiter) {
    // coordinate descent across equations (rows) in Ginit
    for (int ii = 0; ii < geidx_sub.size(); ii++) {
      int j         = geidx_sub(ii);
      double Mj     = M(j, j);
      double invMUj = invMU(j, j);
      // close to zero
      if (fabs(Mj) < 1e-8 || fabs(invMUj) < 1e-8) {
        std::cout << "EnvOpt::solven(): encountered zero for M(" << j << ","
                  << j << ")=" << Mj << " or invMU(" << j << "," << j
                  << ")=" << invMUj << " skipping due to bad input."
                  << std::endl;
        continue;
      }
      DMat g = Ginit.row(j);  // 1 x u

      // matrices with j-th row removed
      DMat Ginit_rest = _remove_row(Ginit, j);
      DMat M_rest     = _remove_row(M, j).col(j);
      DMat invMU_rest = _remove_row(invMU, j).col(j);

      // t2, t3 are 1 x u
      DMat t2 = ((Ginit_rest.transpose() * M_rest) / Mj).transpose();
      DMat t3 = ((Ginit_rest.transpose() * invMU_rest) / invMUj).transpose();
      // 1 x u
      DMat GUGt2 = g + t2;
      DMat GVGt2 = g + t3;
      // u x u
      GUG = GUG - (GUGt2.transpose() * GUGt2) * M(j, j);
      GVG = GVG - (GVGt2.transpose() * GVGt2) * invMU(j, j);
      t4  = t4 - (g.transpose() * g);  // expansion
      // u x u
      DMat invC1 = LAU::cholinv(GUG);
      DMat invC2 = LAU::cholinv(GVG);
      DMat invt4 = LAU::cholinv(t4);

      // same as x = Ginit[j,:]
      for (int colix = 0; colix < n; colix++) {
        x[colix] = Ginit(j, colix);  // copies out the j-th row for lbfgs
      }
      // input data for evaluate function
      EvalData_t data{t2, t3, invt4, invC1, invC2, Mj, invMUj, 0};

      // invoke L-BFGS optimization routine, progress = NULL to be faster
      int ret = lbfgs(n, x, &fx, evaluate, NULL, &data, &param);
      // NOTE: may not converge, and throwing here might halt often.
      //       we perform the L-BFGS optimization on best efforts basis
      if (ret != 0) {
        // lbfgs_free(x);
        // std::cerr << "LBFGS routine failed to converge for j-th row="
        //           << j " after evaluate iters=" << data.k << std::endl;
      }
      lbfgs_i += data.k;  // underlying lbfgs opt iterations

      // 1 x u
      DMatV xx(const_cast<double*>(x), 1, n);
      Ginit.row(j) = xx;

      g = Ginit.row(j);
      // 1 + u
      GUGt2 = g + t2;
      GVGt2 = g + t3;
      // u x u
      GUG = GUG + (GUGt2.transpose() * GUGt2) * M(j, j);
      GVG = GVG + (GVGt2.transpose() * GVGt2) * invMU(j, j);
      t4  = t4 + (g.transpose() * g);

      tot_i++;
    }

    auto [Q, R] = LAU::qr(Ginit, true);
    // save this out
    fullQ = std::move(Q);
    // first u columns are the material components
    gammahat.setZero(fullQ.rows(), u);
    for (int j = 0; j < u; j++) {
      gammahat.col(j) = fullQ.col(j);
    }
    objval = calcObjective(gammahat, M, invMU);
    if (fabs(obj0 - objval) < ftol * fabs(obj0)) {
      break;
    }
    obj0 = objval;
    i++;
  }
  // remaining (r - u) columns are the immaterial components
  DMat gamma0hat(fullQ.rows(), r - u);
  for (int j = 0; j < r - u; j++) {
    gamma0hat.col(j) = fullQ.col(u + j);
  }
  objval = objval + ev.array().log().sum();
  // clean up the x from lbfgs_malloc()
  lbfgs_free(x);

  std::chrono::duration<double> elapsed =
      std::chrono::high_resolution_clock::now() - start;
  // record stats if available
  if (stats) {
    stats->elapsed    = elapsed.count();
    stats->iter       = tot_i;
    stats->lbfgs_iter = lbfgs_i;
  }

  return std::tuple{gammahat, gamma0hat, objval};
}

struct EvalData1_t
{
  DMatV M;
  DMatV invMU;
  int k;  // lbfgs iterations
};

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
static lbfgsfloatval_t evaluate1(void* instance, const lbfgsfloatval_t* xptr,
                                 lbfgsfloatval_t* gptr, const int n,
                                 const lbfgsfloatval_t step)
{
  EvalData1_t* d = reinterpret_cast<EvalData1_t*>(instance);
  DMatV x(const_cast<double*>(xptr), n, 1);  // n x 1
  // function value
  double f1   = (x.transpose() * x)(0);
  double f2   = (x.transpose() * d->M * x)(0);
  double f3   = (x.transpose() * d->invMU * x)(0);
  double fval = -2 * log(f1) + log(f2) + log(f3);
  // gradient vector
  DMat d1 = x.array() / f1;               // n x 1
  DMat d2 = (d->M * x).array() / f2;      // n x 1
  DMat d3 = (d->invMU * x).array() / f3;  // n x 1
  DMat gv = -2 * d1 + d2 + d3;            // n x 1
  for (int i = 0; i < gv.rows(); i++) {
    gptr[i] = gv(i, 0);
  }
  d->k++;
  return fval;
}
#pragma GCC diagnostic pop

std::tuple<DMat, DMat, double> EnvOpt::solve1(DMatV M, DMatV U, DMatV initog,
                                              int maxiter, double ftol,
                                              RunStats_t* stats)
{
  if (sizeof(lbfgsfloatval_t) != sizeof(double)) {
    throw std::runtime_error(
        "lbfgsfloatval_t should be the same size os double!");
  }
  auto start = std::chrono::high_resolution_clock::now();
  int u      = 1;
  int r      = M.rows();
  // compute the initial value
  auto [init, invMU, ev, obj0] = EnvOpt::computeInitial(M, U, u, initog);
  // get the top first equation, and create normalized initial value
  IArr geidx = LAU::geindex(init);
  DMat initu(u, init.cols());
  for (int i = 0; i < u; i++) {
    initu.row(i) = init.row(geidx(i));
  }
  DMat I     = DMat::Identity(initu.rows(), initu.rows());
  DMat sol   = LAU::solve(initu, I);
  DMat Ginit = init * sol;  // r x u, since u=1, r x 1

  DMat gammahat;
  DMat fullQ;
  double objval = obj0;
  int n         = Ginit.rows();  // n is the number of data dim (r)
  // L-BFGS related
  lbfgs_parameter_t param;
  lbfgs_parameter_init(&param);
  lbfgsfloatval_t fx;
  lbfgsfloatval_t* x = lbfgs_malloc(n);  // required for SSE2

  int i = 0, tot_i = 0, lbfgs_i = 0;
  while (i < maxiter) {
    // prepare the initial value for lbfgs routine
    for (int rix = 0; rix < n; rix++) {
      x[rix] = Ginit(rix, 0);
    }
    // input data for evaluate function
    EvalData1_t data{M, invMU, 0};
    // invoke L-BFGS optimization routine, progress = NULL to be faster
    int ret = lbfgs(n, x, &fx, evaluate1, NULL, &data, &param);
    if (ret != 0) {
      // std::cerr << "LBFGS routine failed to converge for j-th row="
      //           << j " after evaluate iters=" << data.k << std::endl;
    }
    lbfgs_i += data.k;  // lbfgs iterations
    // 1 x r
    DMatV xx(const_cast<double*>(x), n, 1);
    Ginit.col(0) = xx;

    auto [Q, R] = LAU::qr(Ginit, true);
    // save this out
    fullQ = std::move(Q);
    // first u columns are the material components
    gammahat.setZero(fullQ.rows(), u);  // r x 1
    gammahat = fullQ.col(0);
    objval   = calcObjective(gammahat, M, invMU);
    if (fabs(obj0 - objval) < ftol * fabs(obj0)) {
      break;
    }
    obj0 = objval;
    i++;
    tot_i++;
  }
  // remaining immaterial projection matrix: r x (r-1)
  DMat gamma0hat(fullQ.rows(), r - u);
  for (int j = 0; j < r - u; j++) {
    gamma0hat.col(j) = fullQ.col(u + j);
  }
  objval = objval + ev.array().log().sum();
  // clean up the x from lbfgs_malloc()
  lbfgs_free(x);

  std::chrono::duration<double> elapsed =
      std::chrono::high_resolution_clock::now() - start;
  // record stats if available
  if (stats) {
    stats->elapsed    = elapsed.count();
    stats->iter       = tot_i;
    stats->lbfgs_iter = lbfgs_i;
  }
  return std::tuple{gammahat, gamma0hat, objval};
}
