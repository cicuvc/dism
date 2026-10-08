#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "config_registry.h"

namespace py = pybind11;
void bind_frontend(py::module_&);
#define DECLARE(R,D,V) namespace dism_r##R##_d##D##_v##V { void bind_config(py::module_&); }
DISM_CONFIGS(DECLARE)
#undef DECLARE

PYBIND11_MODULE(PACKAGE_NAME, module) {
    py::dict configurations;
#define BIND(R,D,V) { \
    auto sub = module.def_submodule("r" #R "_d" #D "_v" #V); \
    dism_r##R##_d##D##_v##V::bind_config(sub); \
    configurations[py::make_tuple(R,D,V)] = sub; \
}
    DISM_CONFIGS(BIND)
#undef BIND
    module.attr("configs") = configurations;
    module.def("get_config", [configurations](int r, int d, int dv) {
        auto key = py::make_tuple(r,d,dv);
        if (!configurations.contains(key)) throw py::value_error("unsupported DISM R/D/DV configuration");
        return configurations[key];
    });
    module.def("fp32_enabled", [] { return bool(DISM_ENABLE_FP32); });
    bind_frontend(module);
    // Default-shape production aliases; diagnostic aliases exist only in probe builds.
    auto key = py::make_tuple(32,64,64);
    if (configurations.contains(key)) {
        py::dict defaults = configurations[key].attr("__dict__");
        for (auto item : defaults) {
            std::string name = py::str(item.first);
            if (name.rfind("__",0) != 0) module.attr(name.c_str()) = item.second;
        }
    }
}
