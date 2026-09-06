#include <cassert>
#include <fmt/core.h>
#include <ATen/Tensor.h>
#include <ATen/Functions.h>
#include <pybind11/pybind11.h>
#include <torch/csrc/utils/pybind.h>

#include <dism_baseline_nope.hpp>

extern void invokeTMA(void *buffer, size_t batch, size_t seq, size_t head, int b, int s, int n);

namespace{


struct BaselineNoPEAttnState {
    at::Tensor FwdQ, FwdK, FwdV, RcpTau;
    at::Tensor FwdVBuffer;
    at::Tensor FwdHBuffer;
    at::Tensor FwdOutput;
    at::Tensor FwdMax;

    size_t Batch, Seqlen, Head, QkDim, HeadDim;

    BaselineNoPEAttnState(at::Tensor q, at::Tensor k, at::Tensor v, at::Tensor rtau, at::Tensor betas)
        : FwdQ(std::move(q)), FwdK(std::move(k)), FwdV(std::move(v)), RcpTau(std::move(rtau)) {
            const auto& q_shape = FwdQ.sizes();
            const auto& k_shape = FwdK.sizes();
            
            assert(q_shape.size() == 4);
            assert(k_shape.size() == 4);
            
            Batch = q_shape.at(0), Seqlen = q_shape.at(1), Head = q_shape.at(2), QkDim = q_shape.at(3), HeadDim = FwdV.size(3);
            assert(k_shape.at(0) == Batch);
            assert(k_shape.at(1) == Seqlen);
            assert(k_shape.at(2) == Head);
            assert(k_shape.at(3) == QkDim);
            assert(RcpTau.at(0) == Head);
            assert(FwdMax.at(0) == Batch);
            assert(FwdMax.at(1) == Head);
            assert(FwdMax.at(2) == Seqlen);

            if((QkDim != 64) || (HeadDim != 64)){
                throw std::runtime_error(fmt::format("Currently only headdim = 64 are supported. Got qkdim = {}, headdim = {}", QkDim, HeadDim));
            }

            auto vh_buffer_shape = BaselineNoPEAttnStateImpl<64, 64>::getVHBufferShape(Batch, Head, Seqlen);

            auto opt = at::TensorOptions { at::ScalarType::Float }.device(FwdQ.device());
            auto opt_bf16 = at::TensorOptions { at::ScalarType::BFloat16 }.device(FwdQ.device());
            FwdVBuffer = at::empty({int64_t(vh_buffer_shape[0]), int64_t(vh_buffer_shape[1]), int64_t(vh_buffer_shape[2]), int64_t(vh_buffer_shape[3])}, opt);
            FwdHBuffer = at::empty({int64_t(vh_buffer_shape[0]), int64_t(vh_buffer_shape[1]), int64_t(vh_buffer_shape[2]), int64_t(vh_buffer_shape[3])}, opt);
            FwdOutput = at::zeros({int64_t(Batch), int64_t(Seqlen), int64_t(Head), int64_t(HeadDim)}, opt_bf16);
            FwdMax = at::empty({int64_t(Batch), int64_t(Head), int64_t(Seqlen)}, opt).copy_(betas);
        }

    void invokeFwdPreprocess(){
        BaselineNoPEAttnStateImpl<64, 64>::invokeFwdPreprocess({
            Batch, Seqlen, Head,
            (half*)FwdQ.data_ptr(), (half*)FwdK.data_ptr(),
            (nv_bfloat16*)FwdV.data_ptr(),
            (nv_bfloat16*)FwdOutput.data_ptr(),
            (float*)FwdVBuffer.data_ptr(), (float*)FwdHBuffer.data_ptr(), (float*)RcpTau.data_ptr(), (float*)FwdMax.data_ptr()
        });
    }

    void invokeFwd(){
        BaselineNoPEAttnStateImpl<64, 64>::invokeFwd({
            Batch, Seqlen, Head,
            (half*)FwdQ.data_ptr(), (half*)FwdK.data_ptr(),
            (nv_bfloat16*)FwdV.data_ptr(),
            (nv_bfloat16*)FwdOutput.data_ptr(),
            (float*)FwdVBuffer.data_ptr(), (float*)FwdHBuffer.data_ptr(), (float*)RcpTau.data_ptr(), (float*)FwdMax.data_ptr()
        });
    }

    void invokeBwdPreprocess(at::Tensor dO, at::Tensor dV){
        BaselineNoPEAttnStateImpl<64, 64>::invokeBwdPreprocess({
            Batch, Seqlen, Head,
            (half*)FwdQ.data_ptr(), (half*)FwdK.data_ptr(),
            (nv_bfloat16*)FwdV.data_ptr(),
            (nv_bfloat16*)dO.data_ptr(),
            (nv_bfloat16*)dV.data_ptr(),
            (float*)FwdVBuffer.data_ptr(), (float*)FwdHBuffer.data_ptr(), (float*)RcpTau.data_ptr(), (float*)FwdMax.data_ptr()
        });
    }

    static void registerType(py::class_<BaselineNoPEAttnState> &clazz) {
        clazz.def(py::init<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>());
        clazz.def_readwrite("fwd_q", &BaselineNoPEAttnState::FwdQ);
        clazz.def_readwrite("fwd_k", &BaselineNoPEAttnState::FwdK);
        clazz.def_readwrite("fwd_v", &BaselineNoPEAttnState::FwdV);
        clazz.def_readwrite("fwd_v_buffer", &BaselineNoPEAttnState::FwdVBuffer);
        clazz.def_readwrite("fwd_h_buffer", &BaselineNoPEAttnState::FwdHBuffer);
        clazz.def_readwrite("fwd_output", &BaselineNoPEAttnState::FwdOutput);
        clazz.def_readwrite("fwd_max", &BaselineNoPEAttnState::FwdMax);

        clazz.def("invoke_fwd_preprocess", &BaselineNoPEAttnState::invokeFwdPreprocess);
        clazz.def("invoke_fwd", &BaselineNoPEAttnState::invokeFwd);
        clazz.def("invoke_bwd_preprocess", &BaselineNoPEAttnState::invokeBwdPreprocess);
    }
};

PYBIND11_MODULE(dism_C, m) {
    m.doc() = "Suffix matching discrete attention acceleration kernels";

    auto clz_baseline_nope_state = py::class_<BaselineNoPEAttnState>(m, "BaselineNoPEAttnState");
    BaselineNoPEAttnState::registerType(clz_baseline_nope_state);

    m.def("testTMA", [](at::Tensor data, int b, int s, int h){
        invokeTMA(data.data_ptr(), data.size(0), data.size(1), data.size(2), b, s, h);
    });
}

} // namespace