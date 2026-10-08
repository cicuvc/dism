#define DISM_HOST_API 1
#include "host/torch.cuh"
#include "summary/primitives.cuh"
#include "dimensions.cuh"

#include "variant.cuh"

namespace DISM_VARIANT {
constexpr int Channels=ActiveConfig::DV;
const void* backward_delta_address_0();
const void* backward_delta_address_1();
at::Tensor backward_delta_cuda(at::Tensor output,at::Tensor dout) {
    TORCH_CHECK(output.is_cuda() && output.dim()==4 && output.size(3)==Channels &&
                output.is_contiguous() && (output.scalar_type()==at::kFloat || output.scalar_type()==at::kBFloat16) &&
                dout.device()==output.device() && dout.is_contiguous() && dout.sizes()==output.sizes() &&
                dout.scalar_type()==at::kBFloat16,"expected contiguous O/dO matching compiled DV");
    c10::cuda::CUDAGuard guard(output.device());
    int b=output.size(0),n=output.size(1),h=output.size(2),rows=b*n*h;
    TORCH_CHECK(n%256==0,"sequence length must be256-token aligned");
    TORCH_CHECK(rows>0,"empty backward input");
    auto result=at::empty({b,h,n},output.options().dtype(at::kFloat));
    auto stream=at::cuda::getCurrentCUDAStream();
    auto* derivative=reinterpret_cast<const __nv_bfloat16*>(dout.data_ptr());
    if (output.scalar_type()==at::kFloat)
        launch_kernel(backward_delta_address_0(),(rows+7)/8,256,0,stream,output.data_ptr<float>(),derivative,result.data_ptr<float>(),n,h,rows);
    else launch_kernel(backward_delta_address_1(),(rows+7)/8,256,0,stream,reinterpret_cast<const __nv_bfloat16*>(output.data_ptr()),
                                              derivative,result.data_ptr<float>(),n,h,rows);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return result;
}

} // namespace DISM_VARIANT
