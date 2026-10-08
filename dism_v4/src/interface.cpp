#include <cstdio>
#include <tuple>
#include <optional>
#include <pybind11/stl.h>

#include <ATen/Tensor.h>
#include <pybind11/pybind11.h>
#include <torch/csrc/utils/pybind.h>

#include "variant.cuh"
namespace DISM_VARIANT {

std::tuple<at::Tensor, at::Tensor> summarization_cuda(at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
                              at::Tensor, at::Tensor, at::Tensor, at::Tensor, int64_t,
                              int64_t, std::optional<at::Tensor>);
at::Tensor scan_probe_cuda(at::Tensor);
std::tuple<at::Tensor,at::Tensor> backward_scan_probe_cuda(at::Tensor,at::Tensor,at::Tensor);
at::Tensor backward_recompute_probe_cuda(std::vector<at::Tensor>,at::Tensor,int64_t);
std::vector<at::Tensor> backward_summary_cuda(std::vector<at::Tensor>,at::Tensor,at::Tensor,
                                            at::Tensor,at::Tensor,int64_t,int64_t,bool);
at::Tensor backward_chunk_cuda(at::Tensor,at::Tensor);
at::Tensor backward_delta_cuda(at::Tensor,at::Tensor);
int backward_tma_mask();
int backward_summary_k();
bool backward_debug_enabled();
bool backward_key_shared();
int backward_shared_bytes();
int backward_input_stages();
int forward_readout_dim_cuda();
std::vector<at::Tensor> backward_summary_debug_cuda(std::vector<at::Tensor>,at::Tensor,at::Tensor,
                                                  at::Tensor,at::Tensor,int64_t);
std::vector<at::Tensor> backward_qk_debug_cuda(std::vector<at::Tensor>,at::Tensor,at::Tensor,
                                             at::Tensor,at::Tensor,at::Tensor,int64_t);
std::vector<at::Tensor> backward_qk_cuda(std::vector<at::Tensor>,at::Tensor,at::Tensor,
                                       at::Tensor,at::Tensor,at::Tensor,int64_t,int64_t,bool);
std::tuple<at::Tensor, at::Tensor> forward_cuda(
    at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
    at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, int64_t, int64_t,
    std::optional<at::Tensor>, bool, std::optional<at::Tensor>);
std::tuple<at::Tensor, at::Tensor> forward_scan_probe_cuda(at::Tensor, at::Tensor, bool);
at::Tensor chunk_scan_cuda(at::Tensor, at::Tensor, int64_t);
at::Tensor pipe_probe_cuda(at::Tensor, int64_t);
at::Tensor metadata_pipe_probe_cuda(at::Tensor, bool, bool);
at::Tensor rv_load_probe_cuda(at::Tensor, bool);
int summary_variant_cuda();
int summary_key_dim_cuda();
int summary_k_stages_cuda();
int summary_allocator_mode_cuda();
int summary_shared_bytes_cuda();
at::Tensor allocator_probe_cuda(at::Tensor, int64_t);
at::Tensor output_store_probe_cuda(at::Tensor, int64_t, bool);
bool forward_output_shared_cuda();
int forward_lse_mode_cuda();
bool forward_output_bf16_cuda();
int forward_head_dim_cuda();
int forward_shared_bytes_cuda(bool);
int forward_stages_cuda();
int forward_warp_k_size_cuda();
int summary_lse_mode_cuda();
int summary_metadata_mode_cuda();
at::Tensor lse_probe_cuda(at::Tensor);
std::tuple<at::Tensor, at::Tensor> key_metadata_probe_cuda(at::Tensor, at::Tensor, bool);

void bind_varlen(pybind11::module_&);

void bind_config(pybind11::module_& m) {
    bind_varlen(m);
    m.def("summarization", &summarization_cuda,
        pybind11::arg("q"),pybind11::arg("k"),pybind11::arg("lq"),pybind11::arg("lk"),
        pybind11::arg("iq"),pybind11::arg("ik"),pybind11::arg("hard"),pybind11::arg("direction"),
        pybind11::arg("tau"),pybind11::arg("n"),pybind11::arg("ctas"),pybind11::arg("gate_delta")=pybind11::none());
    m.def("forward_output", &forward_cuda,
          pybind11::arg("q"), pybind11::arg("k"), pybind11::arg("sq"), pybind11::arg("sk"),
          pybind11::arg("v"), pybind11::arg("q_lse"), pybind11::arg("k_lse"),
          pybind11::arg("q_label"), pybind11::arg("k_label"), pybind11::arg("hard"),
          pybind11::arg("direction"), pybind11::arg("tau"), pybind11::arg("boundary"),
          pybind11::arg("n"), pybind11::arg("ctas"), pybind11::arg("vertical") = pybind11::none(),
          pybind11::arg("fp32_output") = false, pybind11::arg("gate_delta") = pybind11::none());
    m.def("forward_head_dim", &forward_head_dim_cuda);
    m.def("chunk_scan", &chunk_scan_cuda, pybind11::arg("summary_a"),
          pybind11::arg("summary_b"), pybind11::arg("seqlen"));
    m.def("summary_key_dim", &summary_key_dim_cuda);
    m.def("backward_summary", &backward_summary_cuda);
    m.def("backward_chunk", &backward_chunk_cuda,pybind11::arg("a"),pybind11::arg("b"));
    m.def("backward_delta", &backward_delta_cuda);
    m.def("forward_readout_dim", &forward_readout_dim_cuda);
    m.def("backward_qk", &backward_qk_cuda);
#if DISM_BUILD_PROBES
    m.def("output_store_probe", &output_store_probe_cuda, pybind11::arg("anchor"),
          pybind11::arg("n"), pybind11::arg("fp32_output") = false);
    m.def("forward_output_shared", &forward_output_shared_cuda);
    m.def("forward_lse_mode", &forward_lse_mode_cuda);
    m.def("forward_output_bf16", &forward_output_bf16_cuda);
    m.def("forward_shared_bytes", &forward_shared_bytes_cuda, pybind11::arg("fp32_output") = false);
    m.def("forward_stages", &forward_stages_cuda);
    m.def("forward_warp_k_size", &forward_warp_k_size_cuda);
    m.def("summary_variant", &summary_variant_cuda);
    m.def("summary_k_stages", &summary_k_stages_cuda);
    m.def("summary_allocator_mode", &summary_allocator_mode_cuda);
    m.def("summary_shared_bytes", &summary_shared_bytes_cuda);
    m.def("allocator_probe", &allocator_probe_cuda);
    m.def("summary_lse_mode", &summary_lse_mode_cuda);
    m.def("summary_metadata_mode", &summary_metadata_mode_cuda);
    m.def("lse_probe", &lse_probe_cuda);
    m.def("key_metadata_probe", &key_metadata_probe_cuda);
    m.def("scan_probe", &scan_probe_cuda);
    m.def("backward_scan_probe", &backward_scan_probe_cuda);
    m.def("backward_recompute_probe", &backward_recompute_probe_cuda);
    m.def("backward_tma_mask", &backward_tma_mask);
    m.def("backward_summary_k", &backward_summary_k);
    m.def("backward_key_shared", &backward_key_shared);
    m.def("backward_shared_bytes", &backward_shared_bytes);
    m.def("backward_input_stages", &backward_input_stages);
    m.def("forward_scan_probe", &forward_scan_probe_cuda, pybind11::arg("score"),
          pybind11::arg("top"), pybind11::arg("exact") = false);
    m.def("pipe_probe", &pipe_probe_cuda);
    m.def("metadata_pipe_probe", &metadata_pipe_probe_cuda);
    m.def("rv_load_probe", &rv_load_probe_cuda);
#endif
#if DISM_BUILD_PROBES || DISM_BACKWARD_DEBUG
    m.def("backward_debug_enabled", &backward_debug_enabled);
    m.def("backward_summary_debug", &backward_summary_debug_cuda);
    m.def("backward_qk_debug", &backward_qk_debug_cuda);
#endif
}

} // namespace DISM_VARIANT
