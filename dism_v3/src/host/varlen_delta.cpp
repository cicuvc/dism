#define DISM_HOST_API 1
#include "host/torch.cuh"
#include "summary/primitives.cuh"
#include "varlen/layout.cuh"

#include "variant.cuh"

namespace DISM_VARIANT {

const void* varlen_delta_address_0();
const void* varlen_delta_address_1();
at::Tensor varlen_delta_cuda(at::Tensor output,at::Tensor dout,at::Tensor table) {
    using namespace dism_varlen;
    Layout layout(table);
    TORCH_CHECK(output.is_cuda() && output.dim()==4 && output.size(0)==1 &&
                output.size(1)==layout.tokens && output.size(2)>0 && output.size(3)==ActiveConfig::DV &&
                output.is_contiguous() && (output.scalar_type()==at::kFloat || output.scalar_type()==at::kBFloat16) &&
                dout.device()==output.device() && dout.is_contiguous() && dout.sizes()==output.sizes() &&
                dout.scalar_type()==at::kBFloat16,"expected packed O/dO [1,T,H,64]");
    c10::cuda::CUDAGuard guard(output.device());
    int h=output.size(2);
    auto result=at::empty({layout.padded*h},output.options().dtype(at::kFloat));
    if (!layout.tokens) return result;
    auto device_table=dism_metadata::documents(table,output.device());
    dim3 grid((layout.max_padded*h+7)/8,std::min<int64_t>(layout.sequences,65535));
    auto* derivative=reinterpret_cast<const kt::bf16*>(dout.data_ptr());
    auto stream=at::cuda::getCurrentCUDAStream();
    if (output.scalar_type()==at::kFloat)
        launch_kernel(varlen_delta_address_0(),grid,256,0,stream,output.data_ptr<float>(),derivative,
            result.data_ptr<float>(),device_table.data_ptr<int64_t>(),layout.sequences,h,int(layout.tokens));
    else
        launch_kernel(varlen_delta_address_1(),grid,256,0,stream,reinterpret_cast<const kt::bf16*>(output.data_ptr()),
            derivative,result.data_ptr<float>(),device_table.data_ptr<int64_t>(),layout.sequences,h,int(layout.tokens));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return result;
}

} // namespace DISM_VARIANT
