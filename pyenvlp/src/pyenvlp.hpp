#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "rapidjson/document.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"
#include <iostream>

#include "envelope/eigen_types.hpp"

using namespace envelope;

// forward decl
void init_envlp(pybind11::module &m);

namespace {

  namespace py = pybind11;

  template <typename T>
  class ArrVPy : public ArrV<T>
  {
  public:
    using ArrV<T>::ArrV;
    py::object obj;
  };

  template <typename T>
  class MatVPy : public MatV<T>
  {
  public:
    using MatV<T>::MatV;
    py::object obj;
  };

  // create templatized matrix view from numpy array acceps f_style
  template <typename T>
  static MatVPy<T> fromBuffer(
      py::array_t<T, py::array::f_style | py::array::forcecast> b)
  {
    typedef Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic> Stride;
    // by default we are always col major
    constexpr bool rowMajor =
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Flags &
        Eigen::RowMajorBit;
    // request a buffer descriptor from Python
    py::buffer_info info = b.request();
    // Some sanity checks
    if (info.format != py::format_descriptor<T>::format())
      throw std::runtime_error("MatVPy: incompatible format: expected " +
                               py::format_descriptor<T>::format());
    // single dimension vector gets converted to matrix with 1 column
    if (info.ndim == 1) {
      py::ssize_t itemsize = sizeof(T);
      auto strides         = Stride(1, info.strides[0] / itemsize);
      T *ptr               = static_cast<T *>(info.ptr);
      int rows             = info.shape[0];
      MatVPy<T> mview(ptr, rows, 1, strides.outer());
      // hold the underlying object for refcounting
      mview.obj = b;
      return mview;
    }
    if (info.ndim != 2)
      throw std::runtime_error("MatVPy: incompatible buffer dimension!");
    // NOTE: also detect the storage order (rowmajor vs colmajor)
    py::ssize_t itemsize = sizeof(T);

    auto strides = Stride(info.strides[rowMajor ? 0 : 1] / itemsize,
                          info.strides[rowMajor ? 1 : 0] / itemsize);

    T *ptr   = static_cast<T *>(info.ptr);
    int rows = info.shape[0];
    int cols = info.shape[1];
    MatVPy<T> mview(ptr, rows, cols, strides.outer());
    // hold the underlying object for refcounting
    mview.obj = b;
    return mview;
  }

  // return buffer def to create numpy data
  template <typename T>
  static py::buffer_info toBuffer(MatV<T> m)
  {
    constexpr bool rowMajor = MatV<T>::Flags & Eigen::RowMajorBit;
    return py::buffer_info(
        m.data(),                            // Pointer to buffer
        sizeof(T),                           // Size of one scalar
        py::format_descriptor<T>::format(),  // py struct-style format
                                             // descriptor
        2,                                   // Number of dimensions
        {m.rows(), m.cols()},                // Buffer dimensions
        {sizeof(T) * (rowMajor ? m.cols() : 1),
         sizeof(T) * (rowMajor ? 1 : m.outerStride())}
        // Stride (in bytes) for each index
    );
  }

  // create templatized array view from numpy buffer
  template <typename T>
  static ArrVPy<T> fromBufferArr(py::buffer b)
  {
    // request a buffer descriptor from Python
    py::buffer_info info = b.request();
    // NOTE: because we are creating views using the memory owned by
    //       the numpy array, we need to make sure the storage sticks
    //       around and not garbage collected. One way is to increase the
    //       ref count of the array passed in. But when ArrV<T> disappears
    //       does it decrease the ref count of the object whose refcount it
    //       initially incremented?
    // Some sanity checks
    if (info.format != py::format_descriptor<T>::format())
      throw std::runtime_error(
          "ArrVPy: incompatible format: expected a byte array!");
    if (info.ndim != 1)
      throw std::runtime_error(
          "ArrVPy: incompatible buffer dimension! Only single dimension "
          "supported");
    T *ptr   = static_cast<T *>(info.ptr);
    int rows = info.shape[0];
    ArrVPy<T> aview(ptr, rows);
    // hold the underlying object for refcounting
    aview.obj = b;
    return aview;
  }

  // return buffer def to create numpy data
  template <typename T>
  static py::buffer_info toBufferArr(ArrV<T> a)
  {
    return py::buffer_info(
        a.data(),                            // Pointer to buffer
        sizeof(T),                           // Size of one scalar
        py::format_descriptor<T>::format(),  // py struct-style format
                                             // descriptor
        1,                                   // Number of dimensions
        {a.rows()},                          // Buffer dimensions
        {sizeof(T)}                          // Stride (in bytes) for each index
    );
  }

  template <typename T>
  static py::tuple linearTypeToTuple(T &reg)
  {
    // get json as string and save as bytes
    namespace rj = rapidjson;
    rj::Document d;
    d.SetObject();
    rj::Document::AllocatorType &a = d.GetAllocator();
    rj::Value js                   = reg.toJson(a);
    js.Swap(d);
    rapidjson::StringBuffer buffer;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    d.Accept(writer);
    std::string jsstr = buffer.GetString();
    py::bytes bobj(jsstr);
    return py::make_tuple(bobj);
  }

  template <typename T>
  static T tupleToLinearType(py::tuple &t)
  {
    // load from json string
    if (t.size() != 1)
      throw std::runtime_error("Booster pickle invalid state!");
    py::bytes bobj = t[0];
    std::stringstream is(std::string(bobj), std::ios_base::in);
    namespace rj = rapidjson;
    rj::Document d;
    d.Parse(is.str().c_str());
    T reg;
    reg.fromJson(d);
    return reg;
  }

}  // namespace
