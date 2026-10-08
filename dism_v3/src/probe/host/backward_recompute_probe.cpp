#define DISM_HOST_API 1
#include "host/torch.cuh"
#include "backward/recompute.cuh"

#include "variant.cuh"

namespace DISM_VARIANT {
using namespace dism_backward;
const void* recompute_probe_address0();
at::Tensor backward_recompute_probe_cuda(std::vector<at::Tensor> operands,
                                         at::Tensor vertical, int64_t n) {
    TORCH_CHECK(!operands.empty() && operands[0].is_cuda(),"expected saved CUDA operands");
    c10::cuda::CUDAGuard guard(operands[0].device());
    auto args = dism_backward::make_recompute_args(operands,vertical,n);
    auto output = at::empty({args.batch,args.heads,n,n},vertical.options());
    launch_kernel(recompute_probe_address0(),dim3((n+31)/32,(n+15)/16,args.batch*args.heads),32,0,
                      at::cuda::getCurrentCUDAStream(),args,output.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}


} // namespace DISM_VARIANT
