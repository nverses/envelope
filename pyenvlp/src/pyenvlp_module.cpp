#include "pyenvlp.hpp"

#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(_pyenvlp, m)
{
  m.doc() = "pyenvlp pybind11 module";
  init_envlp(m);
}
