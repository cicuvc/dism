#define DISM_HOST_API 1
#include "host/torch.cuh"
#include "forward/scan.cuh"

#include "variant.cuh"

namespace DISM_VARIANT {
const void* forward_scan_probe_kernel_address0();
const void* forward_scan_probe_kernel_address1();
std::tuple<at::Tensor, at::Tensor> forward_scan_probe_cuda(at::Tensor score, at::Tensor top, bool exact) {
    // This diagnostic file also contains ordinary EX2 for exact LSE. Do not
    // apply the blanket summary SASS patch to it; reject the unpatched path.
    TORCH_CHECK(score.is_cuda() && score.scalar_type() == at::kFloat && score.dim() == 2 &&
                    score.is_contiguous() && score.size(0) == 16 && score.size(1) > 0 &&
                    score.size(1) % dism_forward::WarpKSize == 0, "expected contiguous CUDA FP32 score [16,K], K%WarpKSize=0");
    TORCH_CHECK(top.device() == score.device() && top.scalar_type() == at::kFloat &&
                    top.dim() == 1 && top.size(0) == score.size(1) && top.is_contiguous(),
                "expected same-device contiguous FP32 top [K]");
    c10::cuda::CUDAGuard guard(score.device());
    auto output = at::empty_like(score);
    auto bottom = at::empty_like(top);
    auto kernel = exact ? forward_scan_probe_kernel_address0() : forward_scan_probe_kernel_address1();
    launch_kernel(kernel,1, 32, 0, at::cuda::getCurrentCUDAStream(),
        score.data_ptr<float>(), top.data_ptr<float>(), output.data_ptr<float>(),
        bottom.data_ptr<float>(), score.size(1));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {output, bottom};
}


} // namespace DISM_VARIANT
