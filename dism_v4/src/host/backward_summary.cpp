#define DISM_HOST_API 1
#include "host/torch.cuh"
#include "backward/gradients.cuh"
#include "backward/launch.cuh"

#include "variant.cuh"

namespace DISM_VARIANT {
using namespace dism_backward;
template<bool FP32> const void* backward_summary_address();
static std::vector<at::Tensor> backward_summary_impl(std::vector<at::Tensor> operands,
        at::Tensor vertical,at::Tensor dout,at::Tensor normalizer,at::Tensor delta,
        int64_t n,int64_t ctas,bool fp32_output,bool diagnostic) {
    TORCH_CHECK(!operands.empty() && operands[0].is_cuda(),"expected saved CUDA operands");
    TORCH_CHECK(n>0 && n%256==0 && operands[0].dim()==4 && operands[0].size(1)==n,
                "sequence length must be256-token aligned; padding is not accepted");
    TORCH_CHECK(!diagnostic || DISM_BACKWARD_DEBUG,
                "dense diagnostics require DISM_BACKWARD_DEBUG=1 at build time");
    c10::cuda::CUDAGuard guard(operands[0].device());
    auto args=make_args(operands,vertical,dout,normalizer,delta,n);
    int b=args.score.batch,h=args.score.heads,np=args.score.padded;
    auto options=vertical.options();
    auto key_options=options.dtype(fp32_output?at::kFloat:at::kBFloat16);
    auto dv=at::empty({b,n,h,DV},key_options),dsk=at::empty({b,n,h,R},key_options);
    auto dsq=at::zeros({b,n,h,R},options);
    auto a=at::zeros({b,h,(n+SummaryK-1)/SummaryK,np},options),s=at::zeros_like(a);
    args.dv=dv.data_ptr(); args.dsk=dsk.data_ptr(); args.dsq=dsq.data_ptr<float>();
    set_key_output(args,0,dv);
    set_key_output(args,1,dsk);
    set_query_output(args,dsq);
    args.summary_a=a.data_ptr<float>(); args.summary_b=s.data_ptr<float>();
    at::Tensor ca,cb;
#if DISM_BACKWARD_DEBUG
    if (diagnostic) {
        ca=at::zeros({b,h,n,n},options); cb=at::zeros_like(ca);
        args.debug_ca=ca.data_ptr<float>(); args.debug_cb=cb.data_ptr<float>();
    }
#endif
#if DISM_ENABLE_FP32
    if (fp32_output) launch(backward_summary_address<true>(),args,ctas);
    else
#else
    TORCH_CHECK(!fp32_output,"FP32 output instances are disabled in this build");
#endif
        launch(backward_summary_address<false>(),args,ctas);
    if (diagnostic) return {dv,dsq,dsk,a,s,ca,cb};
    return {dv,dsq,dsk,a,s};
}

int backward_tma_mask() { return 7; }
int backward_summary_k() { return SummaryK; }
bool backward_debug_enabled() { return DISM_BACKWARD_DEBUG; }
bool backward_key_shared() { return true; }
int backward_shared_bytes() { return SharedBytes; }
int backward_input_stages() { return InputStages; }

std::vector<at::Tensor> backward_summary_cuda(std::vector<at::Tensor> operands,
        at::Tensor vertical,at::Tensor dout,at::Tensor normalizer,at::Tensor delta,
        int64_t n,int64_t ctas,bool fp32_output) {
    return backward_summary_impl(operands,vertical,dout,normalizer,delta,n,ctas,fp32_output,false);
}
std::vector<at::Tensor> backward_summary_debug_cuda(std::vector<at::Tensor> operands,
        at::Tensor vertical,at::Tensor dout,at::Tensor normalizer,at::Tensor delta,int64_t n) {
    return backward_summary_impl(operands,vertical,dout,normalizer,delta,n,1,true,true);
}

} // namespace DISM_VARIANT
