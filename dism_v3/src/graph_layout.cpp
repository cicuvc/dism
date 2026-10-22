#include <torch/csrc/utils/pybind.h>
#include <pybind11/stl.h>
#include "host/graph_layout.h"

void bind_graph_layout(pybind11::module_& m) {
    m.def("make_graph_layout",&dism_graph::create);
    m.def("update_graph_layout",&dism_graph::update);
    m.def("graph_layout_maximum",[](const at::Tensor& table) {
        auto* state=dism_graph::state(table);
        return state ? state->maximum : int64_t(0);
    });
}
