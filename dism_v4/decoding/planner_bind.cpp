#include "planner.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;
using namespace dism_decode;
PYBIND11_MODULE(_dism_decode_planner, m) {
    py::class_<Planner>(m, "Planner")
        .def(py::init<int, int, int, int, double>())
        .def("needs_rebuild", &Planner::needs_rebuild)
        .def("rebuild", &Planner::rebuild)
        .def(
            "append",
            [](Planner &p, int k, int q, bool reset) {
                py::list result;
                for (auto t : p.append(k, q, reset))
                    result.append(py::make_tuple(t.kind, t.index, t.coefficient));
                return result;
            },
            py::arg("key"), py::arg("query"), py::arg("reset") = false)
        .def_readonly("fallback", &Planner::fallback)
        .def_readonly("max_length", &Planner::max_length)
        .def_readonly("snapshot_size", &Planner::snapshot_size)
        .def("topology", [](const Planner &p) {
            std::vector<int> links, lengths, positions;
            for (auto &n : p.nodes) {
                links.push_back(n.link);
                lengths.push_back(n.length);
                positions.push_back(n.position);
            }
            return py::make_tuple(links, lengths, positions, p.order, p.materialized, p.samples);
        });
}
