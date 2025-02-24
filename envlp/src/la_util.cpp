#include "envelope/la_util.hpp"

#include <Eigen/Eigenvalues>
#include <iostream>

using namespace envelope;

DMat LAU::cov(DMatV a, DMatV b)
{
  double n = a.rows();
  DMat amu = a;  // copy
  amu.rowwise() -= amu.colwise().mean();

  if (!b.size()) {
    DMat cov = (amu.transpose() * amu) / (n - 1);
    return cov;
  }
  // compute the cov between a and b
  if (a.rows() != b.rows()) {
    throw std::invalid_argument("a and b does requires same number of rows");
  }
  DMat bmu = b;  // copy
  bmu.rowwise() -= bmu.colwise().mean();
  DMat cov = (amu.transpose() * bmu) / (n - 1);
  return cov;
}

std::pair<DMat, DMat> LAU::eig(DMatV Aview)
{
  using EigType = Eigen::EigenSolver<Eigen::MatrixXd>;
  EigType es(Aview);  // solve for eigen

  Eigen::ComputationInfo info = es.info();
  // check for failure and throw
  if (info != Eigen::ComputationInfo::Success) {
    std::string issue =
        (info == Eigen::ComputationInfo::NumericalIssue)  ? "NumericalIssue"
        : (info == Eigen::ComputationInfo::NoConvergence) ? "NoConvergence"
        : (info == Eigen::ComputationInfo::InvalidInput)  ? "InvalidInput"
                                                          : "UnKnown";
    throw std::runtime_error("LAU::eig(): failed with " + issue);
  }
  // discards the complex part
  DMat ev = es.pseudoEigenvalueMatrix().diagonal();
  DMat E  = es.pseudoEigenvectors();
  return std::pair{std::move(ev), std::move(E)};
}

std::pair<DMat, DMat> LAU::qr(DMatV Aview, bool reduced)
{
  if (reduced) {
    Eigen::HouseholderQR<Eigen::MatrixXd> qr(Aview);
    DMat Q = qr.householderQ();
    DMat R = Q.transpose() * Aview;
    return std::pair{std::move(Q), std::move(R)};
  }
  Eigen::FullPivHouseholderQR<Eigen::MatrixXd> fqr(Aview);
  // just issue warning?
  if (!fqr.isInvertible()) {
    // std::cerr << "LAU::qr(): WARN FullPvHouseholderQR not invertible"
    //           << std::endl;
  }
  DMat Q = fqr.matrixQ();
  DMat R = Q.transpose() * Aview;
  return std::pair{std::move(Q), std::move(R)};
}

DMat LAU::cholinv(DMatV Aview)
{
  Eigen::LLT<DMat> chol(Aview);
  Eigen::ComputationInfo info = chol.info();
  // check the status
  if (info != Eigen::ComputationInfo::Success) {
    std::string issue =
        (info == Eigen::ComputationInfo::NumericalIssue)  ? "NumericalIssue"
        : (info == Eigen::ComputationInfo::NoConvergence) ? "NoConvergence"
        : (info == Eigen::ComputationInfo::InvalidInput)  ? "InvalidInput"
                                                          : "UnKnown";
    // throw on these, but numerical issue prints warning
    if (info == Eigen::ComputationInfo::NoConvergence ||
        info == Eigen::ComputationInfo::InvalidInput) {
      throw std::runtime_error("LAU::cholinv(): failed with " + issue);
    }
    // else {
    //   std::cerr << "LAU::cholinv(): WARN NumericalIssue" << std::endl;
    // }
  }
  DMat L    = chol.matrixL();
  DMat Linv = L.inverse();  // robust?
  return Linv.transpose() * Linv;
}

DMat LAU::pinv(DMatV Aview)
{
  Eigen::PartialPivLU<DMat> piv(Aview);
  return piv.inverse();
}

DMat LAU::solve(DMatV A, DMatV b)
{
  // check for solve dimension here!
  if (A.rows() != b.rows()) {
    throw std::invalid_argument("LAU::solve(): A rows needs to match b rows");
  }
  Eigen::ColPivHouseholderQR<DMat> s(A);
  if (!s.isInvertible()) {
    throw std::invalid_argument(
        "LAU::solve(): ColPivHouseholderQR not invertible!");
  }
  return s.solve(b);
}

static IArr _range(int n)
{
  IArr a(n);
  for (int i = 0; i < n; i++) {
    a(i) = i;
  }
  return a;
}

static IArr _setdiff(const IArr& a, const IArr& b)
{
  IArr c(a.size());
  auto it = std::set_difference(a.data(), a.data() + a.size(), b.data(),
                                b.data() + b.size(), c.data());
  c.conservativeResize(std::distance(c.data(), it));
  return c;
}

static IArr _where_bycol_abs_eq(const DMat& A, int colix, double val)
{
  std::vector<int> idx;
  for (int i = 0; i < A.rows(); i++) {
    double aval = fabs(A(i, colix));
    if (fabs(aval - val) < 1e-9) {
      idx.push_back(i);
    }
  }
  IArr aidx(idx.size());
  for (int i = 0; i < int(idx.size()); i++) {
    aidx(i) = idx[i];
  }
  return aidx;
}

static DMat _slice(const DMat& A, const IArr& ivec, const IArr& jvec)
{
  DMat Asub(ivec.size(), jvec.size());
  for (int i = 0; i < ivec.size(); i++) {
    for (int j = 0; j < jvec.size(); j++) {
      Asub(i, j) = A(ivec(i), jvec(j));
    }
  }
  return Asub;
}

IArr LAU::geindex(DMatV Aview)
{
  DMat A = Aview;  // copy, since we will modify this matrix
  int n  = A.rows();
  int p  = A.cols();

  IArr idx     = IArr::Zero(p);
  IArr res_idx = _range(n);
  IArr ivec(1), rm_idx(1);
  int i = 0;
  while (i < p) {
    ivec(0)        = i;
    DMat Asub      = _slice(A, res_idx, ivec);
    double abs_max = Asub.cwiseAbs().maxCoeff();
    IArr max_idx   = _where_bycol_abs_eq(A, i, abs_max);
    IArr stmp      = _setdiff(max_idx, idx);
    if (stmp.size() > 0) {
      idx(i) = stmp(0);
    }
    rm_idx(0) = idx(i);
    res_idx   = _setdiff(res_idx, rm_idx);
    for (int j = 0; j < (n - i - 1); j++) {
      int rix      = res_idx(j);
      double invar = A(rix, i) / A(idx(i), i);
      // sweep across columns and update (eliminate)
      for (int cix = 0; cix < p; cix++) {
        A(rix, cix) = A(rix, cix) - invar * A(idx(i), cix);
      }
    }
    i++;
  }
  // concatenate
  IArr out(idx.size() + res_idx.size());
  int k = 0;
  for (int i = 0; i < idx.size(); i++, k++) {
    out(k) = idx(i);
  }
  for (int i = 0; i < res_idx.size(); i++, k++) {
    out(k) = res_idx(i);
  }
  return out;
}
