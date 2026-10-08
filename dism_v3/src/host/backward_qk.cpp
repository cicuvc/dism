#define DISM_HOST_API 1
#include "host/torch.cuh"
#include "backward/gradients.cuh"
#include "backward/launch.cuh"

#include "variant.cuh"

namespace DISM_VARIANT {
using namespace dism_backward;
template<bool FP32> const void* backward_qk_address();
static std::vector<at::Tensor> backward_qk_impl(std::vector<at::Tensor> operands,
        at::Tensor vertical,at::Tensor dout,at::Tensor normalizer,at::Tensor delta,
        at::Tensor boundary,int64_t n,int64_t ctas,bool fp32_output,bool diagnostic) {
    TORCH_CHECK(!operands.empty() && operands[0].is_cuda(),"expected saved CUDA operands");
    TORCH_CHECK(n>0 && n%256==0 && operands[0].dim()==4 && operands[0].size(1)==n,
                "sequence length must be256-token aligned; padding is not accepted");
    TORCH_CHECK(!diagnostic || DISM_BACKWARD_DEBUG,
                "dense diagnostics require DISM_BACKWARD_DEBUG=1 at build time");
    c10::cuda::CUDAGuard guard(operands[0].device());
    auto args=make_args(operands,vertical,dout,normalizer,delta,n);
    int b=args.score.batch,h=args.score.heads,np=args.score.padded;
    TORCH_CHECK(boundary.device()==vertical.device() && boundary.scalar_type()==at::kFloat &&
                boundary.is_contiguous() && boundary.sizes()==at::IntArrayRef({b,h,(n+SummaryK-1)/SummaryK,np}),
                "invalid reverse chunk boundary");
    auto options=vertical.options();
    auto dq=at::zeros({b,n,h,D},options);
    auto dk=at::empty({b,n,h,D},options.dtype(fp32_output?at::kFloat:at::kBFloat16));
    auto dlq=at::zeros({b,h,n},options);
    auto dlk=at::empty({b,h,n},options.dtype(fp32_output?at::kFloat:at::kBFloat16));
    auto dtau=at::zeros({h},options);
    args.dq=dq.data_ptr<float>(); args.dk=dk.data_ptr();
    set_key_output(args,2,dk);
    args.dlq=dlq.data_ptr<float>(); args.dlk=dlk.data_ptr(); args.dtau=dtau.data_ptr<float>();
    set_query_output(args,dq);
    set_lse_output(args,dlq);
    args.g_boundary=boundary.data_ptr<float>();
    at::Tensor g;
#if DISM_BACKWARD_DEBUG
    if (diagnostic) {
        g=at::zeros({b,h,n,n},options);
        args.debug_g=g.data_ptr<float>();
    }
#endif
#if DISM_ENABLE_FP32
    if (fp32_output) launch(backward_qk_address<true>(),args,ctas);
    else
#else
    TORCH_CHECK(!fp32_output,"FP32 output instances are disabled in this build");
#endif
        launch(backward_qk_address<false>(),args,ctas);
    if (diagnostic) return {dq,dk,dlq,dlk,dtau,g};
    return {dq,dk,dlq,dlk,dtau};
}

std::vector<at::Tensor> backward_qk_cuda(std::vector<at::Tensor> operands,
        at::Tensor vertical,at::Tensor dout,at::Tensor normalizer,at::Tensor delta,
        at::Tensor boundary,int64_t n,int64_t ctas,bool fp32_output) {
    return backward_qk_impl(operands,vertical,dout,normalizer,delta,boundary,n,ctas,fp32_output,false);
}
std::vector<at::Tensor> backward_qk_debug_cuda(std::vector<at::Tensor> operands,
        at::Tensor vertical,at::Tensor dout,at::Tensor normalizer,at::Tensor delta,
        at::Tensor boundary,int64_t n) {
    return backward_qk_impl(operands,vertical,dout,normalizer,delta,boundary,n,1,true,true);
}

} // namespace DISM_VARIANT
