#define DISM_HOST_API 1
#include "host/torch.cuh"
#include "backward/gradients.cuh"
#include "backward/launch.cuh"
#include "varlen/backward.cuh"

#include "variant.cuh"

namespace DISM_VARIANT {
using namespace dism_backward;
template<bool FP32> const void* varlen_backward_summary_address();
std::vector<at::Tensor> varlen_backward_summary_cuda(std::vector<at::Tensor> operands,
        at::Tensor vertical,at::Tensor dout,at::Tensor normalizer,at::Tensor delta,
        at::Tensor table,bool fp32_output) {
    using namespace dism_varlen;
    BackwardInputs input(operands,vertical,dout,normalizer,delta,table);
    const auto& layout=input.layout;
    int h=input.heads;
    c10::cuda::CUDAGuard guard(operands[0].device());
    auto options=normalizer.options();
    auto private_options=options.dtype(fp32_output?at::kFloat:at::kBFloat16);
    auto dv=at::empty({1,layout.tokens,h,DV},private_options);
    auto dsk=at::empty({1,layout.tokens,h,R},private_options);
    auto dsq=at::zeros({1,layout.tokens,h,R},options);
    auto a=at::zeros({layout.backward*h},options),b=at::zeros_like(a);
    if (!layout.tokens) return {dv,dsq,dsk,a,b};
    auto args=input.common();
    args.dv=dv.data_ptr(); args.dsk=dsk.data_ptr(); args.dsq=dsq.data_ptr<float>();
    set_key_output(args,0,dv);
    set_key_output(args,1,dsk);
    set_query_output(args,dsq);
    args.summary_a=a.data_ptr<float>(); args.summary_b=b.data_ptr<float>();
#if DISM_ENABLE_FP32
    if (fp32_output)
        launch_backward(varlen_backward_summary_address<true>(),args,layout,operands[0].device());
    else
#else
    TORCH_CHECK(!fp32_output,"FP32 output instances are disabled in this build");
#endif
        launch_backward(varlen_backward_summary_address<false>(),args,layout,operands[0].device());
    return {dv,dsq,dsk,a,b};
}

} // namespace DISM_VARIANT
