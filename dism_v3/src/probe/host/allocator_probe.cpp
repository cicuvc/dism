#define DISM_HOST_API 1
#include "host/torch.cuh"
#include "summary/primitives.cuh"

#include "variant.cuh"

namespace DISM_VARIANT {
const void* allocator_probe_kernel_address0();
at::Tensor allocator_probe_cuda(at::Tensor anchor, int64_t offset) {
    TORCH_CHECK(anchor.is_cuda() && offset >= 0 && offset < 1024, "invalid probe input");
    c10::cuda::CUDAGuard guard(anchor.device());
    auto output = at::empty({45}, anchor.options().dtype(at::kInt));
    launch_kernel(allocator_probe_kernel_address0(),1, 32, 0, at::cuda::getCurrentCUDAStream(),
        output.data_ptr<int>(), int(offset));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}


} // namespace DISM_VARIANT
