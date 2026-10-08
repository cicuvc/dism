#include "prefill.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
namespace py=pybind11;
using namespace dism_prefill;

template<class T> py::array_t<T> copy_array(const std::vector<T> &v) {
    py::array_t<T> a(v.size());
    std::copy(v.begin(),v.end(),a.mutable_data());
    return a;
}

template<class T>
py::array_t<T> mock(const Program &p,
    py::array_t<T,py::array::c_style> sq,
    py::array_t<T,py::array::c_style> sk,
    py::array_t<T,py::array::c_style> value) {
    if (sq.ndim()!=2 || sk.ndim()!=2 || value.ndim()!=2 || sq.shape(0)!=p.n
        || sk.shape(0)!=p.n || value.shape(0)!=p.n || sq.shape(1)!=sk.shape(1)
        || sq.shape(1)<=0 || value.shape(1)<=0)
        throw std::invalid_argument("expected sq/sk [N,R], value [N,DV]");
    py::array_t<T> out({py::ssize_t(p.n),value.shape(1)});
    {
        py::gil_scoped_release release;
        execute(p,sq.data(),sk.data(),value.data(),int(sq.shape(1)),int(value.shape(1)),out.mutable_data());
    }
    return out;
}

PYBIND11_MODULE(_dism_prefill,m) {
    py::class_<Program>(m,"Program")
        .def_readonly("n",&Program::n)
        .def("statistics",[](const Program &p) {
            py::dict d;
            d["n"]=p.n; d["states"]=p.states; d["streams"]=p.offsets.size()-1;
            d["events"]=p.rows.size();
            d["program_bytes"]=p.offsets.size()*8+p.rows.size()*20+p.logden.size()*8;
            return d;
        })
        .def("arrays",[](const Program &p) {
            py::dict d;
            d["offsets"]=copy_array(p.offsets); d["rows"]=copy_array(p.rows);
            d["decay"]=copy_array(p.decay); d["weight"]=copy_array(p.weight);
            d["logden"]=copy_array(p.logden);
            return d; // Owned copies, never writable aliases to the plan.
        })
        .def("chunks",[](const Program &p,int width) {
            ChunkProgram c;
            { py::gil_scoped_release release; c=chunk_program(p,width); }
            py::dict d;
            d["offsets"]=copy_array(c.offsets); d["rows"]=copy_array(c.rows);
            d["prefixes"]=copy_array(c.prefixes); d["weights"]=copy_array(c.weights);
            d["lengths"]=copy_array(c.lengths); d["resets"]=copy_array(c.resets);
            return d;
        })
        .def("execute_fp64",&mock<double>,py::arg("sq").noconvert(),py::arg("sk").noconvert(),py::arg("value").noconvert())
        .def("execute_fp32",&mock<float>,py::arg("sq").noconvert(),py::arg("sk").noconvert(),py::arg("value").noconvert());
    m.def("plan",[](const Ids &q,const Ids &k,const Ids &reset,double tau) {
        py::gil_scoped_release release;
        return Builder(q,k,reset,tau).finish();
    });
}
