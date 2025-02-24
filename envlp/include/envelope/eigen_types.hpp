#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>

namespace envelope {

  // naming convension: <TypeChar><Mat|Arr|Vec>[View]
  // DMat: D for dense or double, alias for Eigen::MatrixXd with storage
  // DMatV: D for dense or double, view on Eigen::MatrixXd without storage

  // convenience matrix adapter for dynamically resizable matrix
  // with column-major storage and alignment
  template <typename T>
  class MatV
    : public Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>,
                        Eigen::Unaligned, Eigen::OuterStride<>>
  {
  public:
    using BaseType =
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>,
                   Eigen::Unaligned, Eigen::OuterStride<>>;

  public:
    MatV()
      : BaseType(nullptr, 0, 0, Eigen::OuterStride<>(0))
    {
    }

    MatV(const MatV &other)
      : BaseType(other)
    {
    }

    MatV(T *data, size_t rows, size_t cols, size_t stride = 0)
      : BaseType(data, rows, cols,
                 Eigen::OuterStride<>(stride > 0 ? stride : rows))
    {
    }

    MatV(const Eigen::MatrixBase<
         Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> &m)
      : BaseType(const_cast<T *>(&m(0, 0)), m.rows(), m.cols(),
                 Eigen::OuterStride<>(m.outerStride()))
    {
    }

    // compatibility for Eigen::MatrixXd.col(j)
    // https://eigen.tuxfamily.org/dox/classEigen_1_1Block.html
    MatV(const Eigen::Block<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>,
                            Eigen::Dynamic, 1, true> &col)
      : BaseType(const_cast<T *>(&col(0)), col.rows(), 1,
                 Eigen::OuterStride<>(col.outerStride()))
    {
    }

    // compatibility for MatV<T>.col(j)
    MatV(const typename Eigen::DenseBase<
         Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>,
                    Eigen::Unaligned, Eigen::OuterStride<>>>::ConstColXpr &col)
      : BaseType(const_cast<T *>(col.data()), col.rows(), 1,
                 Eigen::OuterStride<>(col.rows()))
    {
    }

    // compatibility for Mat<T>.row(i)
    MatV(const typename Eigen::DenseBase<
         Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>::RowXpr &blk)
      : BaseType(
            const_cast<T *>(blk.data()), 1, blk.cols(),
            Eigen::OuterStride<>(blk.innerStride()))  // note the innerstride
    {
    }

    // compatibility for MatV<T>.row(i)
    MatV(const typename Eigen::DenseBase<
         Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>,
                    Eigen::Unaligned, Eigen::OuterStride<>>>::RowXpr &row)
      : BaseType(
            const_cast<T *>(row.data()), 1, row.cols(),
            Eigen::OuterStride<>(row.innerStride()))  // note the innerstride
    {
    }

    // compatibility for Mat<T>.block(p,q,psz,qsz)
    MatV(const typename Eigen::DenseBase<
         Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>::BlockXpr &blk)
      : BaseType(const_cast<T *>(blk.data()), blk.rows(), blk.cols(),
                 Eigen::OuterStride<>(blk.outerStride()))
    {
    }

    // compatibility for const MatV<T>.block(p,q,psz,qsz)
    MatV(const typename Eigen::DenseBase<Eigen::Map<
             Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>, Eigen::Unaligned,
             Eigen::OuterStride<>>>::ConstBlockXpr &blk)
      : BaseType(const_cast<T *>(blk.data()), blk.rows(), blk.cols(),
                 Eigen::OuterStride<>(blk.outerStride()))
    {
    }

    // compatibility for (non-const) MatV<T>.block(p,q,psz,qsz)
    MatV(const typename Eigen::DenseBase<
         Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>,
                    Eigen::Unaligned, Eigen::OuterStride<>>>::BlockXpr &blk)
      : BaseType(const_cast<T *>(blk.data()), blk.rows(), blk.cols(),
                 Eigen::OuterStride<>(blk.outerStride()))
    {
    }

    MatV(const Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>,
                          Eigen::Unaligned, Eigen::OuterStride<>> &m)
      : BaseType(m)
    {
    }

    // compatibility from array to matrix
    MatV(const Eigen::ArrayBase<Eigen::Array<T, Eigen::Dynamic, 1>> &a)
      : BaseType(const_cast<T *>(&a(0)), a.size(), 1,
                 Eigen::OuterStride<>(a.outerStride()))
    {
    }

    MatV(const Eigen::Map<Eigen::Array<T, Eigen::Dynamic, 1>, Eigen::Unaligned>
             &a)
      : BaseType(const_cast<T *>(a.data()), a.size(), 1,
                 Eigen::OuterStride<>(a.outerStride()))
    {
    }

    MatV &operator=(const MatV<T> &m)
    {
      // use the inplace new to construct and set into this pointer
      // if you do this naively: *this = MatV<T>(m);
      // assignment operator will be called recursively and cause segfault.
      new (this) MatV<T>(m);
      return *this;
    }

    // allows std::is_polymorphic check to pass
    virtual ~MatV() {};
  };

  // convenience array adapter for dynamically resizable 1D contiguous data
  template <typename T>
  class ArrV
    : public Eigen::Map<Eigen::Array<T, Eigen::Dynamic, 1>, Eigen::Unaligned>
  {
  public:
    using BaseType =
        Eigen::Map<Eigen::Array<T, Eigen::Dynamic, 1>, Eigen::Unaligned>;

  public:
    ArrV()
      : BaseType(nullptr, 0)
    {
    }

    ArrV(const ArrV &other)
      : BaseType(other)
    {
    }

    ArrV(T *data, size_t rows)
      : BaseType(data, rows)
    {
    }

    ArrV(const Eigen::ArrayBase<Eigen::Array<T, Eigen::Dynamic, 1>> &a)
      : BaseType(const_cast<T *>(&a(0)), a.rows())
    {
    }

    ArrV(const Eigen::Map<Eigen::Array<T, Eigen::Dynamic, 1>, Eigen::Unaligned>
             &a)
      : BaseType(a)
    {
    }

    // conversion from vector
    ArrV(std::vector<T> &vec)
      : BaseType(vec.data(), vec.size())
    {
    }

    // support (n x 1) 2D matrix conversion
    ArrV(const Eigen::MatrixBase<
         Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> &m)
      : BaseType(const_cast<T *>(&m(0, 0)), m.rows())
    {
      if (m.cols() > 1) {
        throw std::invalid_argument(
            "MatrixXd conversion only supported for ncol==1");
      }
    }

    // allows: DArrView colview = m.col(j); // m is MatrixXd
    ArrV(const Eigen::Block<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>,
                            Eigen::Dynamic, 1, true> &a)
      : BaseType(const_cast<T *>(&a(0)), a.size())
    {
    }

    // allows: DArrView colview = mview.col(j);
    ArrV(const Eigen::Block<
         Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>,
                    Eigen::Unaligned, Eigen::OuterStride<>>,
         Eigen::Dynamic, 1, true> &a)
      : BaseType(const_cast<T *>(a.data()), a.size())
    {
    }

    ArrV(const Eigen::Block<
         const Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>,
                          Eigen::Unaligned, Eigen::OuterStride<>>,
         Eigen::Dynamic, 1, true> &a)
      : BaseType(const_cast<T *>(a.data()), a.size())
    {
    }

    ArrV &operator=(const ArrV<T> &a)
    {
      // use the inplace new to construct and set into this pointer
      // if you do this naively: *this = ArrV<T>(m);
      // assignment operator will be called recursively and cause segfault.
      new (this) ArrV<T>(a);
      return *this;
    }

    virtual ~ArrV() {};
  };

  //
  // Convenient Aliases
  //

  // matrix types with storage
  template <typename T>
  using Mat  = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
  using FMat = Mat<float>;
  using DMat = Mat<double>;
  using BMat = Mat<int8_t>;
  using IMat = Mat<int>;
  using LMat = Mat<long>;

  // array types with storage
  template <typename T>
  using Arr  = Eigen::Array<T, Eigen::Dynamic, 1>;
  using FArr = Eigen::Array<float, Eigen::Dynamic, 1>;
  using DArr = Eigen::Array<double, Eigen::Dynamic, 1>;
  using BArr = Eigen::Array<int8_t, Eigen::Dynamic, 1>;
  using IArr = Eigen::Array<int, Eigen::Dynamic, 1>;
  using LArr = Eigen::Array<long, Eigen::Dynamic, 1>;

  // view types for matrix
  using FMatV = MatV<float>;
  using DMatV = MatV<double>;
  using BMatV = MatV<int8_t>;
  using IMatV = MatV<int>;
  using LMatV = MatV<long>;

  // view types for array
  using FArrV = ArrV<float>;
  using DArrV = ArrV<double>;
  using BArrV = ArrV<int8_t>;
  using IArrV = ArrV<int>;
  using LArrV = ArrV<long>;

  using SMat = Eigen::SparseMatrix<double>;
  using Tri  = Eigen::Triplet<double>;  // contains <i,j,val>

  // template deduction guideline
  // template <typename T>
  // MatV(const Mat<T>&) -> MatV<T>;

  // template <typename T>
  // MatV(Mat<T>&) -> MatV<T>;
}  // namespace envelope
