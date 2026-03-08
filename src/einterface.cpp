#include <ATen/Tensor.h>
#include <pybind11/pybind11.h>
#include <torch/csrc/utils/pybind.h>

namespace{

struct BaselineNoPEAttnStateImpl {};

struct BaselineNoPEAttnState {
    at::Tensor FwdQ, FwdK, FwdV;
    at::Tensor FwdQCache, FwdKCache;
    at::Tensor FwdVBuffer;
    at::Tensor FwdHBuffer;
    at::Tensor FwdOutput;
    at::Tensor FwdMax;

    BaselineNoPEAttnState(at::Tensor q, at::Tensor k, at::Tensor v)
        : FwdQ(std::move(q)), FwdK(std::move(k)), FwdV(std::move(v)) {}

    static void registerType(py::class_<BaselineNoPEAttnState> &clazz) {
        clazz.def(py::init<at::Tensor, at::Tensor, at::Tensor>());
        clazz.def_readwrite("fwd_q", &BaselineNoPEAttnState::FwdQ);
        clazz.def_readwrite("fwd_k", &BaselineNoPEAttnState::FwdK);
        clazz.def_readwrite("fwd_v", &BaselineNoPEAttnState::FwdV);
        clazz.def_readwrite("fwd_q_cache", &BaselineNoPEAttnState::FwdQCache);
        clazz.def_readwrite("fwd_k_cache", &BaselineNoPEAttnState::FwdKCache);
        clazz.def_readwrite("fwd_v_buffer", &BaselineNoPEAttnState::FwdVBuffer);
        clazz.def_readwrite("fwd_h_buffer", &BaselineNoPEAttnState::FwdHBuffer);
        clazz.def_readwrite("fwd_output", &BaselineNoPEAttnState::FwdOutput);
        clazz.def_readwrite("fwd_max", &BaselineNoPEAttnState::FwdMax);
    }
};

PYBIND11_MODULE(dism_C, m) {
    m.doc() = "Suffix matching discrete attention acceleration kernels";

    auto clz_baseline_nope_state = py::class_<BaselineNoPEAttnState>(m, "BaselineNoPEAttnState");
    BaselineNoPEAttnState::registerType(clz_baseline_nope_state);
}

} // namespace