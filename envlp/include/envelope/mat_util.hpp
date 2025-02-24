#pragma once

#include <vector>

#include "envelope/eigen_types.hpp"

namespace envelope {

  class MatUtil
  {
  public:
    // simple eigen index producers to aid in prob construction
    static IArr range(int start, int end, int step = 1);
    static IArr toIArr(const std::vector<int> &vec);

    template <typename T>
    static Arr<T> toArr(const std::vector<T> &vec);

    // some convenient linarg wrappers
    template <typename T>
    static Mat<T> chol(MatV<T> mat, bool upper = true);
    template <typename T>
    static Mat<T> chol(const Mat<T> &mat, bool upper = true);

    template <typename T>
    static bool isInv(const Mat<T> &mat);

    // inv( E C E' + s )
    template <typename T>
    static Mat<T> fullInv(MatV<T> E, MatV<T> C, MatV<T> s,
                          bool do_pinv = false);
    template <typename T>
    static Mat<T> fullInv(const Mat<T> &E, const Mat<T> &C, const Mat<T> &s,
                          bool do_pinv = false);

    // E: (n x k), C: (k x k), s: (n x 1), where n >> k
    template <typename T>
    static Mat<T> woodburyInv(MatV<T> E, MatV<T> C, MatV<T> s);
    template <typename T>
    static Mat<T> woodburyInv(const Mat<T> &E, const Mat<T> &C,
                              const Mat<T> &s);

    // manipulation routines on double matrices
    static DMat vstack(const std::vector<DMatV> &mats);
    static DMat hstack(const std::vector<DMatV> &mats);
    static DMat subset(DMatV mat, const std::vector<int> idx,
                       bool dorow = true);
  };

  inline IArr MatUtil::range(int start, int end, int step)
  {
    if (start > end) {
      throw std::invalid_argument("getRange has invalid input");
    }
    int sz = (end - start) / step;
    return IArr::LinSpaced(sz, start, end);
  }

  inline IArr MatUtil::toIArr(const std::vector<int> &vec)
  {
    // copies
    return Eigen::Map<IArr, Eigen::Aligned>(const_cast<int *>(vec.data()),
                                            vec.size());
  }

  template <typename T>
  inline Arr<T> MatUtil::toArr(const std::vector<T> &vec)
  {
    // copies
    return Eigen::Map<Arr<T>, Eigen::Aligned>(const_cast<T *>(vec.data()),
                                              vec.size());
  }

  template <typename T>
  inline Mat<T> MatUtil::chol(MatV<T> mat, bool upper)
  {
    // standard cholesky: ldlt is the more robust version but does not give
    // recoverable lower, upper.
    // Eigen::LDLT<DMat> decomp(mat);
    Eigen::LLT<Mat<T>> decomp(mat);
    return upper ? Mat<T>(decomp.matrixU()) : Mat<T>(decomp.matrixL());
  }

  template <typename T>
  inline Mat<T> MatUtil::chol(const Mat<T> &mat, bool upper)
  {
    return MatUtil::chol(MatV<T>(mat), upper);
  }

  template <typename T>
  inline bool MatUtil::isInv(const Mat<T> &mat)
  {
    Eigen::FullPivLU<Mat<T>> lu(mat);
    return lu.isInvertible();
  }

  template <typename T>
  inline Mat<T> MatUtil::fullInv(MatV<T> E, MatV<T> C, MatV<T> s, bool do_pinv)
  {
    DMat full = E * C * E.transpose();
    full += s.asDiagonal();
    if (do_pinv) {
      Eigen::CompleteOrthogonalDecomposition<
          Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>
          cqr(full);
      return cqr.pseudoInverse();
    }
    return full.inverse();
  }

  template <typename T>
  inline Mat<T> MatUtil::fullInv(const Mat<T> &E, const Mat<T> &C,
                                 const Mat<T> &s, bool do_pinv)
  {
    return MatUtil::fullInv(MatV(E), MatV(C), MatV(s), do_pinv);
  }

  template <typename T>
  inline Mat<T> MatUtil::woodburyInv(MatV<T> E, MatV<T> C, MatV<T> s)
  {
    Mat<T> sinv       = 1.0 / s.array();
    Mat<T> fullcovinv = sinv.asDiagonal();
    fullcovinv -=
        sinv.asDiagonal() *
        (E * (C.inverse() + E.transpose() * sinv.asDiagonal() * E).inverse() *
         E.transpose()) *
        sinv.asDiagonal();
    return fullcovinv;
  }

  template <typename T>
  inline Mat<T> MatUtil::woodburyInv(const Mat<T> &E, const Mat<T> &C,
                                     const Mat<T> &s)
  {
    // surely, there's a better way to do this conversion w/ templates?
    return MatUtil::woodburyInv(MatV<T>(E), MatV<T>(C), MatV<T>(s));
  }

}  // namespace envelope
