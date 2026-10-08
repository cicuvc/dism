#define DISM_HOST_API 1
#include "host/torch.cuh"
#include "summary/primitives.cuh"
#include <c10/cuda/CUDAException.h>

#include "variant.cuh"

namespace DISM_VARIANT {
const void* metadata_pipe_probe_kernel_address0();
at::Tensor metadata_pipe_probe_cuda(at::Tensor source, bool local_wait, bool unit_arrivals) {
    TORCH_CHECK(source.is_cuda() && source.scalar_type() == at::kInt &&
                    source.is_contiguous() && source.dim() == 2 &&
                    source.size(0) > 0 && source.size(0) <= 10000 && source.size(1) == 256,
                "expected contiguous CUDA int32 [tasks,256]");
    c10::cuda::CUDAGuard guard(source.device());
    auto output = at::empty_like(source);
    launch_kernel(metadata_pipe_probe_kernel_address0(),1, 384, 2048, at::cuda::getCurrentCUDAStream(),
        source.data_ptr<int>(), output.data_ptr<int>(), source.size(0), local_wait, unit_arrivals);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}


} // namespace DISM_VARIANT
