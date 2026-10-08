#define DISM_HOST_API 1
#include "host/torch.cuh"
#include "summary/primitives.cuh"

#include "variant.cuh"

namespace DISM_VARIANT {
using Reverse = pscore::AltLayoutSplitScanBuffer<16,32,pscore::BinaryElement,
                                                pscore::AffineComposeOp>;
using Scalar = pscore::AltLayoutSplitScanBuffer<16,32,pscore::UnaryElement>;

const void* reverse_probe_address0();
std::tuple<at::Tensor,at::Tensor> backward_scan_probe_cuda(
        at::Tensor pairs, at::Tensor bottom, at::Tensor right) {
    TORCH_CHECK(pairs.is_cuda() && pairs.scalar_type()==at::kFloat && pairs.is_contiguous() &&
                pairs.sizes()==at::IntArrayRef({16,32,2}), "pairs must be CUDA FP32 [16,32,2]");
    TORCH_CHECK(bottom.device()==pairs.device() && right.device()==pairs.device() &&
                bottom.scalar_type()==at::kFloat && right.scalar_type()==at::kFloat &&
                bottom.is_contiguous() && right.is_contiguous() &&
                bottom.sizes()==at::IntArrayRef({32}) && right.sizes()==at::IntArrayRef({16}),
                "bottom/right must be contiguous FP32 [32]/[16]");
    c10::cuda::CUDAGuard guard(pairs.device());
    auto result = at::empty_like(pairs);
    auto edge = at::empty({32,2},pairs.options());
    launch_kernel(reverse_probe_address0(),1,32,0,at::cuda::getCurrentCUDAStream(),
        reinterpret_cast<float2*>(pairs.data_ptr<float>()),bottom.data_ptr<float>(),
        right.data_ptr<float>(),reinterpret_cast<float2*>(result.data_ptr<float>()),
        reinterpret_cast<float2*>(edge.data_ptr<float>()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {result,edge};
}


} // namespace DISM_VARIANT
