#define DISM_HOST_API 1
#include "host/torch.cuh"
#include "summary/primitives.cuh"
#include "backward/config.cuh"

#include "variant.cuh"

namespace DISM_VARIANT {
namespace {
constexpr int Step=dism_backward::SummaryK;
using Global=kt::gl<float,-1,-1,-1,-1>;
}
const void* backward_chunk_address_0();
const void* backward_chunk_address_1();
at::Tensor backward_chunk_cuda(at::Tensor a,at::Tensor b) {
    TORCH_CHECK(a.is_cuda() && a.scalar_type()==at::kFloat && a.dim()==4 && a.is_contiguous() &&
                b.device()==a.device() && b.scalar_type()==at::kFloat && b.is_contiguous() &&
                b.sizes()==a.sizes() && a.size(0)>0 && a.size(1)>0 && a.size(2)>0 && a.size(3)>0,
                "expected FP32 contiguous affine SoA [B,H,chunks,padded]");
    c10::cuda::CUDAGuard guard(a.device());
    auto output=at::empty_like(a);
    int chunks=a.size(2),padded=a.size(3);
    if (padded%16==0) {
        int diagonals=padded+((chunks-1)*Step+31)/32*32;
        Global ga(a.data_ptr<float>(),a.size(0)*a.size(1),1,chunks,padded);
        Global gb(b.data_ptr<float>(),b.size(0)*b.size(1),1,chunks,padded);
        launch_kernel(backward_chunk_address_0(),dim3((diagonals+127)/128,a.size(0)*a.size(1)),128,0,
                  at::cuda::getCurrentCUDAStream(),ga,gb,output.data_ptr<float>(),chunks,padded);
    } else {
        int diagonals=padded+(chunks-1)*Step;
        launch_kernel(backward_chunk_address_1(),dim3((diagonals+127)/128,a.size(0)*a.size(1)),128,0,
                    at::cuda::getCurrentCUDAStream(),a.data_ptr<float>(),b.data_ptr<float>(),
                                                       output.data_ptr<float>(),chunks,padded);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

} // namespace DISM_VARIANT
