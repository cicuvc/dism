#define DISM_HOST_API 1
#include "host/torch.cuh"
#include "backward/gradients.cuh"
#include "backward/launch.cuh"
#include "varlen/backward.cuh"

#include "variant.cuh"

namespace DISM_VARIANT {
using namespace dism_backward;
template<bool FP32> const void* varlen_backward_qk_address();
std::vector<at::Tensor> varlen_backward_qk_cuda(std::vector<at::Tensor> operands,
        at::Tensor vertical,at::Tensor dout,at::Tensor normalizer,at::Tensor delta,
        at::Tensor boundary,at::Tensor table,bool fp32_output) {
    using namespace dism_varlen;
    BackwardInputs input(operands,vertical,dout,normalizer,delta,table);
    const auto& layout=input.layout;
    int h=input.heads;
    TORCH_CHECK(boundary.device()==vertical.device() && boundary.scalar_type()==at::kFloat &&
                boundary.is_contiguous() && boundary.dim()==1 && boundary.numel()==layout.backward*h,
                "invalid ragged backward boundary");
    c10::cuda::CUDAGuard guard(operands[0].device());
    auto options=normalizer.options();
    auto private_options=options.dtype(fp32_output?at::kFloat:at::kBFloat16);
    auto dq=at::zeros({1,layout.tokens,h,D},options);
    auto dk=at::empty({1,layout.tokens,h,D},private_options);
    // Global [H,T]; both T and document starts are256-token aligned.
    auto dlq=at::zeros({layout.padded*h},options);
    auto dlk=at::empty({layout.padded*h},private_options);
    auto dtau=at::zeros({h},options);
    if (!layout.tokens) return {dq,dk,dlq,dlk,dtau};
    auto args=input.common();
    args.dq=dq.data_ptr<float>(); args.dk=dk.data_ptr();
    args.dlq=dlq.data_ptr<float>(); args.dlk=dlk.data_ptr(); args.dtau=dtau.data_ptr<float>();
    args.g_boundary=boundary.data_ptr<float>();
    set_key_output(args,2,dk);
    set_query_output(args,dq);
    // Flat view of global [H,T], addressed at head*T+document_begin+query.
    kt::gl<float,-1,-1,-1,-1,GradientVector> lse_map(
        dlq.data_ptr<float>(),1,1,1,layout.tokens*h);
    copy_output_map(args.lse_output,lse_map);
#if DISM_ENABLE_FP32
    if (fp32_output)
        launch_backward(varlen_backward_qk_address<true>(),args,layout,operands[0].device());
    else
#else
    TORCH_CHECK(!fp32_output,"FP32 output instances are disabled in this build");
#endif
        launch_backward(varlen_backward_qk_address<false>(),args,layout,operands[0].device());
    return {dq,dk,dlq,dlk,dtau};
}

} // namespace DISM_VARIANT
