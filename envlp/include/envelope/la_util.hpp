#pragma once

#include <rapidjson/document.h>

#include <Eigen/Dense>
#include <algorithm>
#include <numeric>
#include <vector>

#include "envelope/eigen_types.hpp"

namespace envelope {

  // Linear Algebra Util
  class LAU
  {
  public:
    // cov
    static DMat cov(DMatV a, DMatV b = DMatV());
    // eigen value decomp, returns tuple( eigen value, eigen vector )
    static std::pair<DMat, DMat> eig(DMatV A);
    // qr decomp
    static std::pair<DMat, DMat> qr(DMatV A, bool reduced = true);
    // cholesky based inverse
    static DMat cholinv(DMatV A);
    // partial inverse
    static DMat pinv(DMatV A);
    // equivalent to np.linalg.solve
    static DMat solve(DMatV A, DMatV b);
    // returns sorted index of equations from gaussian elimination
    static IArr geindex(DMatV A);
  };

  class Vec
  {
  public:
    template <typename T>
    static inline std::vector<size_t> argsort(const std::vector<T>& vec,
                                              bool ascending)
    {
      std::vector<size_t> idx(vec.size());
      std::iota(idx.begin(), idx.end(), 0);
      std::sort(idx.begin(), idx.end(),
                [&vec](int l, int r) -> bool { return vec[l] < vec[r]; });
      // reverse
      if (!ascending) {
        std::reverse(idx.begin(), idx.end());
      }
      return idx;
    }
  };

}  // namespace envelope
