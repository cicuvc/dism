#define DISM_HOST_API 1
#include "host/torch.cuh"
#include "forward/types.cuh"
namespace DISM_VARIANT {
constexpr int DIM=dism_forward::DIM;
template<bool FP32> const void* output_probe_address();
at::Tensor output_store_probe_cuda(at::Tensor anchor, int64_t n, bool fp32_output) {
    auto run = [&]<bool FP32Output>() {
        using DType = dism_forward::OutputDType<FP32Output>;
        using Global = dism_forward::OutputGlobal<FP32Output>;
        TORCH_CHECK(anchor.is_cuda() && n > 0 && n <= 1024,"invalid probe input");
        c10::cuda::CUDAGuard guard(anchor.device());
        constexpr int B = 2, H = 3;
        auto output = at::empty({B,n,H,DIM},
                               anchor.options().dtype(FP32Output ? at::kFloat : at::kBFloat16));
        Global global(reinterpret_cast<DType*>(output.data_ptr()),B,H,n,DIM,
                      kt::gl_strides{size_t(n*H*DIM),DIM,H*DIM});
        launch_kernel(output_probe_address<FP32Output>(),2,256,0,at::cuda::getCurrentCUDAStream(),global);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return output;
    };
    #if DISM_ENABLE_FP32
    return fp32_output ? run.template operator()<true>() : run.template operator()<false>();
#else
    TORCH_CHECK(!fp32_output,"FP32 output instances are disabled in this build");
    return run.template operator()<false>();
#endif
}

} // namespace DISM_VARIANT
