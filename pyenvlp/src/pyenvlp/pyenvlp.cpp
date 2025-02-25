#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wzero-as-null-pointer-constant"
#include "pyenvlp.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "envelope/eigen_types.hpp"
#include "envelope/envopt.hpp"
#include "envelope/la_util.hpp"
#include "envelope/regression.hpp"
#include "envelope/regression_env.hpp"

using namespace pybind11::literals;
using namespace envelope;
namespace py = pybind11;

void init_envlp(py::module& m)
{
  py::class_<EnvOpt>(m, "EnvOpt")
      .def_static(
          "solve_envelope",
          [](py::array M, py::array U, int u, std::optional<py::array> initarg,
             int maxiter, double ftol) {
            py::array init;
            if (initarg.has_value()) {
              init = initarg.value();
            }
            EnvOpt::RunStats_t st;
            auto [gamma, gamma0, objval] = EnvOpt::solveEnvelope(
                fromBuffer<double>(M), fromBuffer<double>(U), u,
                fromBuffer<double>(init), maxiter, ftol, &st);
            // return as dict
            return py::dict(
                "gammahat"_a  = py::array_t<double>(toBuffer<double>(gamma)),
                "gamma0hat"_a = py::array_t<double>(toBuffer<double>(gamma0)),
                "objfun"_a = objval, "niter"_a = st.iter,
                "opt_elapsed"_a = st.elapsed);
          },
          py::arg("M"), py::arg("U"), py::arg("u"),
          py::arg("init") = py::none(), py::arg("maxiter") = 100,
          py::arg("ftol") = 1e-3)
      .def(py::init<>())
      .def(py::init(
               [](py::array M, py::array U, std::optional<py::array> initarg) {
                 py::array init0;
                 if (initarg.has_value()) {
                   init0 = initarg.value();
                 }
                 EnvOpt::RunStats_t st;
                 return std::unique_ptr<EnvOpt>(
                     new EnvOpt(fromBuffer<double>(M), fromBuffer<double>(U),
                                fromBuffer<double>(init0)));
               }),
           py::arg("M"), py::arg("U"), py::arg("init") = py::none())
      .def(
          "set_data",
          [](EnvOpt& eo, py::array M, py::array U,
             std::optional<py::array> initarg) {
            py::array init0;
            if (initarg.has_value()) {
              init0 = initarg.value();
            }
            eo.setData(fromBuffer<double>(M), fromBuffer<double>(U),
                       fromBuffer<double>(init0));
          },
          py::arg("M"), py::arg("U"), py::arg("init") = py::none())
      .def("solve", &EnvOpt::solve)
      .def(
          "sweep",
          [](EnvOpt& eo, const std::vector<int>& uvec, int n_samples,
             double objval_offset, int p) {
            // std::pair becomes py::tuple automatically
            auto [bicv, min_u] = eo.sweep(uvec, n_samples, objval_offset, p);
            return py::make_tuple(py::array_t<double>(toBuffer<double>(bicv)),
                                  min_u);
          },
          "u_list"_a, "n_samples"_a, "objval_offset"_a = 0.0, "p"_a = 1)
      .def("get_objvalue", &EnvOpt::getObjValue)
      .def("get_gammas",
           [](EnvOpt& eo) {
             auto [g, g0] = eo.getGammas();
             return py::make_tuple(py::array_t<double>(toBuffer<double>(g)),
                                   py::array_t<double>(toBuffer<double>(g0)));
           })
      .def("get_stats", [](EnvOpt& eo) {
        EnvOpt::RunStats_t st = eo.getStats();
        return py::dict("elapsed"_a = st.elapsed, "iter"_a = st.iter,
                        "lbfgs_iter"_a = st.lbfgs_iter);
      });

  py::class_<Regression>(m, "Regression")
      .def(py::init<bool>(), py::arg("fit_intercept") = true)
      .def(
          "fit",
          [](Regression& reg, py::array x, py::array y, py::array w) {
            reg.fit(fromBuffer<double>(x), fromBuffer<double>(y),
                    fromBuffer<double>(w));
            return reg;
          },
          py::arg("x"), py::arg("y"), py::arg("w") = py::array_t<double>())
      .def(
          "fit_gram",
          [](Regression& reg, py::array xtx, py::array xty, py::array yty,
             int nrows) {
            reg.fitGram(fromBuffer<double>(xtx), fromBuffer<double>(xty),
                        fromBuffer<double>(yty), nrows);
            return reg;
          },
          py::arg("xtx"), py::arg("xty"),
          py::arg("yty") = py::array_t<double>(), py::arg("nrows") = 0)
      .def_property_readonly("intercept_",
                             [](Regression& reg) {
                               return py::array_t<double>(toBuffer<double>(
                                   reg.getInterceptOffset(true)));
                             })
      .def("coefs",
           [](Regression& reg) {
             return py::array_t<double>(toBuffer<double>(reg.coefs()));
           })
      .def_property_readonly(
          "coef_",
          [](Regression& reg) {
            return py::array_t<double>(toBuffer<double>(reg.coefs()));
          })
      .def(
          "predict",
          [](Regression& reg, py::array x, bool include_intercept) {
            Eigen::MatrixXd yh =
                reg.predict(fromBuffer<double>(x), include_intercept);
            return py::array_t<double>(toBuffer<double>(yh));
          },
          py::arg("x"), py::arg("include_intercept") = true)
      .def("get_nrows", &Regression::getNRows)
      .def("get_xsum",
           [](Regression& reg) {
             return py::array_t<double>(toBuffer<double>(reg.getXSum()));
           })
      .def("get_ysum",
           [](Regression& reg) {
             return py::array_t<double>(toBuffer<double>(reg.getYSum()));
           })
      .def("to_json_str", &Regression::toJsonString)
      .def("from_json_str", &Regression::fromJsonString)
      .def(py::pickle(
          // __getstate__
          [](const Regression& reg) { return linearTypeToTuple(reg); },
          // __setstate__
          [](py::tuple t) { return tupleToLinearType<Regression>(t); }));

  py::class_<Ridge, Regression>(m, "Ridge")
      .def(py::init<bool, double>(), py::arg("fit_intercept") = true,
           py::arg("l2_lambda") = 0.0)
      .def("set_l2_lambda", &Ridge::setL2Lambda)
      .def("get_l2_lambda", &Ridge::getL2Lambda)
      .def(
          "fit_gram",
          [](Ridge& reg, py::array xtx, py::array xty, py::array yty,
             int nrows) {
            reg.fitGram(fromBuffer<double>(xtx), fromBuffer<double>(xty),
                        fromBuffer<double>(yty), nrows);
            return reg;
          },
          py::arg("xtx"), py::arg("xty"),
          py::arg("yty") = py::array_t<double>(), py::arg("nrows") = 0)
      .def(py::pickle([](const Ridge& reg) { return linearTypeToTuple(reg); },
                      [](py::tuple t) { return tupleToLinearType<Ridge>(t); }));

  // fitter::linear compatible regression class
  py::class_<RidgeEnvlp, Ridge>(m, "RidgeEnvlp")
      .def(py::init<bool, double, bool, bool, bool, int>(),
           "fit_intercept"_a = true, "l2_lambda"_a = 0.0, "zscore_x"_a = false,
           "zscore_y"_a = false, "do_coef_scale"_a = false, "u_step"_a = 0)
      .def("set_l2_lambda", &Ridge::setL2Lambda)
      .def("get_l2_lambda", &Ridge::getL2Lambda)
      .def("set_x_zscoring", &RidgeEnvlp::setXZScoring)
      .def("set_y_zscoring", &RidgeEnvlp::setYZScoring)
      .def("get_x_zscoring", &RidgeEnvlp::getXZScoring)
      .def("get_y_zscoring", &RidgeEnvlp::getYZScoring)
      .def("set_u", &RidgeEnvlp::setEnvelopeDim)
      .def("get_u", &RidgeEnvlp::getEnvelopeDim)
      .def("get_u_list",
           [](RidgeEnvlp& reg) {
             return py::array_t<double>(toBuffer<double>(reg.getDimList()));
           })
      .def("get_bic_list",
           [](RidgeEnvlp& reg) {
             return py::array_t<double>(toBuffer<double>(reg.getBICList()));
           })
      .def("coefsbase",
           [](RidgeEnvlp& reg) {
             return py::array_t<double>(toBuffer<double>(reg.coefsBase()));
           })
      .def("get_objvalue", &RidgeEnvlp::getObjValue)
      .def("get_loglik", &RidgeEnvlp::getLogLikelihood)
      .def("gamma",
           [](RidgeEnvlp& reg) {
             return py::array_t<double>(toBuffer<double>(reg.getGamma()));
           })
      .def("gamma0",
           [](RidgeEnvlp& reg) {
             return py::array_t<double>(toBuffer<double>(reg.getGamma0()));
           })
      .def("omega",
           [](RidgeEnvlp& reg) {
             return py::array_t<double>(toBuffer<double>(reg.getOmega()));
           })
      .def("get_xsqsum",
           [](RidgeEnvlp& reg) {
             return py::array_t<double>(toBuffer<double>(reg.getXSqSum()));
           })
      .def("get_ysqsum",
           [](RidgeEnvlp& reg) {
             return py::array_t<double>(toBuffer<double>(reg.getYSqSum()));
           })
      .def(
          "fit_gram",
          [](RidgeEnvlp& reg, py::array xtx, py::array xty, py::array yty,
             int nrows) {
            reg.fitGram(fromBuffer<double>(xtx), fromBuffer<double>(xty),
                        fromBuffer<double>(yty), nrows);
            return reg;
          },
          py::arg("xtx"), py::arg("xty"),
          py::arg("yty") = py::array_t<double>(), py::arg("nrows") = 0)
      .def(py::pickle(
          [](const RidgeEnvlp& reg) { return linearTypeToTuple(reg); },
          [](py::tuple t) { return tupleToLinearType<RidgeEnvlp>(t); }));
}

#pragma GCC diagnostic pop
