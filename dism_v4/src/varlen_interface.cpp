#include <ATen/Tensor.h>
#include <tuple>
#include <vector>
#include <pybind11/stl.h>
#include <pybind11/pybind11.h>
#include <torch/csrc/utils/pybind.h>

#include "variant.cuh"
namespace DISM_VARIANT {

at::Tensor varlen_pack_cuda(at::Tensor,at::Tensor,bool);
at::Tensor varlen_tma_probe_cuda(at::Tensor,at::Tensor,int64_t);
std::tuple<at::Tensor,at::Tensor> varlen_summary_cuda(std::vector<at::Tensor>,at::Tensor);
at::Tensor varlen_chunk_cuda(at::Tensor,at::Tensor,at::Tensor,int64_t);
at::Tensor varlen_backward_chunk_cuda(at::Tensor,at::Tensor,at::Tensor,int64_t);
std::tuple<at::Tensor,at::Tensor,at::Tensor> varlen_forward_cuda(std::vector<at::Tensor>,at::Tensor,bool,bool);
at::Tensor varlen_unpack_cuda(at::Tensor,at::Tensor,int64_t,int64_t);
at::Tensor varlen_delta_cuda(at::Tensor,at::Tensor,at::Tensor);
std::vector<at::Tensor> varlen_backward_summary_cuda(std::vector<at::Tensor>,at::Tensor,
    at::Tensor,at::Tensor,at::Tensor,at::Tensor,bool);
std::vector<at::Tensor> varlen_backward_qk_cuda(std::vector<at::Tensor>,at::Tensor,
    at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,bool);

void bind_varlen(pybind11::module_& module) {
#if DISM_BUILD_PROBES
    module.def("varlen_pack",&varlen_pack_cuda);
#endif
#if DISM_BUILD_PROBES
    module.def("varlen_tma_probe",&varlen_tma_probe_cuda);
#endif
    module.def("varlen_summary",&varlen_summary_cuda);
    module.def("varlen_chunk",&varlen_chunk_cuda);
    module.def("varlen_backward_chunk",&varlen_backward_chunk_cuda);
    module.def("varlen_forward",&varlen_forward_cuda,pybind11::arg("operands"),
               pybind11::arg("table"),pybind11::arg("save_boundaries"),pybind11::arg("fp32_output")=false);
#if DISM_BUILD_PROBES
    module.def("varlen_unpack",&varlen_unpack_cuda);
#endif
    module.def("varlen_delta",&varlen_delta_cuda);
    module.def("varlen_backward_summary",&varlen_backward_summary_cuda);
    module.def("varlen_backward_qk",&varlen_backward_qk_cuda);
}

} // namespace DISM_VARIANT
