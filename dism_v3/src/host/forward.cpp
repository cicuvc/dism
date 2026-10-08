#define DISM_HOST_API 1
#include "host/torch.cuh"
#include "forward/producer.cuh"
#include "forward/readout.cuh"

#include "variant.cuh"

namespace DISM_VARIANT {
using namespace dism_forward;
template<bool FP32> const void* forward_address();
std::tuple<at::Tensor, at::Tensor> forward_cuda(
        at::Tensor q, at::Tensor k, at::Tensor sq, at::Tensor sk, at::Tensor v,
        at::Tensor q_lse, at::Tensor k_lse, at::Tensor q_label, at::Tensor k_label,
        at::Tensor hard, at::Tensor direction, at::Tensor tau, at::Tensor boundary,
        int64_t n, int64_t requested_ctas, std::optional<at::Tensor> vertical, bool fp32_output) {
    auto run = [&]<bool FP32Output>() -> std::tuple<at::Tensor,at::Tensor> {
        using Args = ArgsT<FP32Output>;
        using DType = OutputDType<FP32Output>;
        using OutputMap = OutputGlobal<FP32Output>;
        TORCH_CHECK(q.is_cuda() && q.scalar_type() == at::kBFloat16 && q.dim() == 4,
                    "q must be CUDA BF16 [B,padded_N,H,D]");
        c10::cuda::CUDAGuard guard(q.device());
        int batch = q.size(0), padded = q.size(1), heads = q.size(2);
        TORCH_CHECK(batch > 0 && heads > 0 && n > 0 && n == padded && n % 256 == 0,
                    "sequence length must be256-token aligned; padding is not accepted");
        auto check = [&](const at::Tensor& x, at::ScalarType dtype, at::IntArrayRef shape,
                         const char* name) {
            TORCH_CHECK(x.device() == q.device() && x.scalar_type() == dtype &&
                            x.is_contiguous() && x.sizes() == shape,
                        name, ": invalid device, dtype, shape or stride");
        };
        check(q, at::kBFloat16, {batch,padded,heads,CONFIG.KcKeyDim}, "q");
        check(k, at::kBFloat16, q.sizes(), "k");
        check(sq, at::kBFloat16, {batch,padded,heads,READOUT_DIM}, "sq");
        check(sk, at::kBFloat16, sq.sizes(), "sk");
        check(v, at::kBFloat16, {batch,padded,heads,DIM}, "v");
        check(q_lse, at::kFloat, {batch,heads,padded}, "q_lse");
        check(k_lse, at::kFloat, q_lse.sizes(), "k_lse");
        check(q_label, at::kInt, q_lse.sizes(), "q_label");
        check(k_label, at::kInt, q_lse.sizes(), "k_label");
        check(hard, at::kByte, q_lse.sizes(), "hard");
        check(direction, at::kByte, {batch,heads}, "direction");
        check(tau, at::kFloat, {heads}, "tau");
        int checkpoints = (n - 1) / 32;
        check(boundary, at::kFloat, {batch,heads,checkpoints,padded}, "boundary");
        if (vertical)
            check(*vertical, at::kFloat, {batch,heads,(n-1)/16,padded}, "vertical");
        auto output = at::empty({batch,n,heads,DIM},
                               q.options().dtype(FP32Output ? at::kFloat : at::kBFloat16));
        auto lse2 = at::empty({batch,heads,n}, q.options().dtype(at::kFloat));
        auto q_stride = kt::gl_strides{size_t(padded * heads * CONFIG.KcKeyDim),
                                       size_t(CONFIG.KcKeyDim), size_t(heads * CONFIG.KcKeyDim)};
        auto sq_stride = kt::gl_strides{size_t(padded * heads * READOUT_DIM),
                                        size_t(READOUT_DIM), size_t(heads * READOUT_DIM)};
        using QG = decltype(Args::q); using SQG = decltype(Args::sq);
        using KG = decltype(Args::k); using SKG = decltype(Args::sk); using VG = decltype(Args::v);
        using FG = decltype(Args::q_lse); using IG = decltype(Args::q_label);
        Args args{batch,heads,int(n),padded,checkpoints,
            QG(reinterpret_cast<kt::bf16*>(q.data_ptr()),batch,heads,padded,CONFIG.KcKeyDim,q_stride),
            SQG(reinterpret_cast<kt::bf16*>(sq.data_ptr()),batch,heads,padded,READOUT_DIM,sq_stride),
            KG(reinterpret_cast<kt::bf16*>(k.data_ptr()),batch,padded,heads,CONFIG.KcKeyDim),
            SKG(reinterpret_cast<kt::bf16*>(sk.data_ptr()),batch,padded,heads,READOUT_DIM),
            VG(reinterpret_cast<kt::bf16*>(v.data_ptr()),batch,padded,heads,DIM),
            FG(q_lse.data_ptr<float>(),batch,heads,1,padded),
            FG(k_lse.data_ptr<float>(),batch,heads,1,padded),
            IG(q_label.data_ptr<int>(),batch,heads,1,padded),
            IG(k_label.data_ptr<int>(),batch,heads,1,padded),
            hard.data_ptr<uint8_t>(),direction.data_ptr<uint8_t>(),tau.data_ptr<float>(),
            boundary.data_ptr<float>(),reinterpret_cast<DType*>(output.data_ptr()),lse2.data_ptr<float>(),
            vertical ? vertical->data_ptr<float>() : nullptr
            , OutputMap(reinterpret_cast<DType*>(output.data_ptr()),batch,heads,n,DIM,
                kt::gl_strides{size_t(n*heads*DIM),DIM,size_t(heads*DIM)})
        };
        int tasks = batch * heads * ((n + QROWS - 1) / QROWS);
        int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
        int ctas = std::min(tasks, requested_ctas > 0 ? int(requested_ctas) : sms);
        C10_CUDA_CHECK(cudaFuncSetAttribute(forward_address<FP32Output>(),
            cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes<FP32Output>));
        launch_kernel(forward_address<FP32Output>(),ctas,384,shared_bytes<FP32Output>,at::cuda::getCurrentCUDAStream(),args);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return {output,lse2};
    };
    #if DISM_ENABLE_FP32
    return fp32_output ? run.template operator()<true>() : run.template operator()<false>();
#else
    TORCH_CHECK(!fp32_output,"FP32 output instances are disabled in this build");
    return run.template operator()<false>();
#endif
}

bool forward_output_shared_cuda() { return 1 != 0; }
int forward_lse_mode_cuda() { return 0; }
bool forward_output_bf16_cuda() { return true; } // Default output precision.
int forward_head_dim_cuda() { return DIM; }
int forward_readout_dim_cuda() { return READOUT_DIM; }
int forward_shared_bytes_cuda(bool fp32_output) { return fp32_output ? shared_bytes<true> : shared_bytes<false>; }
int forward_stages_cuda() { return STAGES; }
int forward_warp_k_size_cuda() { return WarpKSize; }

} // namespace DISM_VARIANT
