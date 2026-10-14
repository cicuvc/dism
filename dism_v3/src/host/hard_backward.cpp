#define DISM_HOST_API 1
#include "host/torch.cuh"
#include "backward/hard.cuh"
#include "backward/launch.cuh"

namespace DISM_VARIANT {
const void* hard_backward_address();
const void* hard_dlog_weights_address();
const void* hard_tau_reduce_address();

std::vector<at::Tensor> hard_backward_cuda(std::vector<at::Tensor> x,at::Tensor boundary,
        at::Tensor logw,at::Tensor normalizer,int64_t requested_ctas) {
    namespace hb=dism_hard_backward;
    using namespace dism_backward;
    TORCH_CHECK(x.size()==7,"expected SQ,SK,V,dO,query labels,key labels,hard");
    const auto& sq=x[0];
    TORCH_CHECK(sq.is_cuda() && sq.dim()==4,"SQ must be CUDA BNHR");
    c10::cuda::CUDAGuard guard(sq.device());
    int64_t b=sq.size(0),n=sq.size(1),h=sq.size(2);
    TORCH_CHECK(b>0 && h>0 && b*h<=INT_MAX && n>0 && n<=INT_MAX-256 && n%256==0,
                "positive dimensions and 256-aligned length required");
    TORCH_CHECK(b*h*(n/128)<=INT_MAX && b*n<=INT_MAX,"hard backward grid is too large");
    TORCH_CHECK(requested_ctas>=0 && requested_ctas<=INT_MAX,"invalid CTA count");
    auto check=[&](const at::Tensor& t,at::ScalarType dtype,at::IntArrayRef shape,const char* name) {
        TORCH_CHECK(t.device()==sq.device() && t.scalar_type()==dtype && t.is_contiguous() && t.sizes()==shape,
                    name,": invalid device, dtype, shape or stride");
    };
    check(sq,at::kBFloat16,{b,n,h,R},"SQ");
    check(x[1],at::kBFloat16,{b,n,h,R},"SK");
    check(x[2],at::kBFloat16,{b,n,h,DV},"V");
    check(x[3],at::kBFloat16,{b,n,h,DV},"dO");
    check(x[4],at::kInt,{b,h,n},"query labels");
    check(x[5],at::kInt,{b,h,n},"key labels");
    check(x[6],at::kBool,{b,h},"sequence hard flags");
    check(boundary,at::kInt,{b,h,(n-1)/32,n},"integer boundary");
    check(logw,at::kFloat,{h,n+1},"log weight LUT");
    check(normalizer,at::kFloat,{b,h,n},"log2 normalizer");
    auto options=sq.options().dtype(at::kFloat);
    auto dsq=at::zeros({b,n,h,R},options),dsk=at::zeros_like(x[1]),dv=at::zeros_like(x[2]);
    auto moments=at::zeros({4,b,h,n},options),dlogw=at::empty_like(logw),dtau=at::empty({h},options);
    hb::Args args{{int(b),int(h),int(n)},
        SoftQueryGlobal(reinterpret_cast<kt::bf16*>(sq.data_ptr()),b,n,h,R),
        DerivativeGlobal(reinterpret_cast<kt::bf16*>(x[3].data_ptr()),b,n,h,DV),
        HeldSoftGlobal(reinterpret_cast<kt::bf16*>(x[1].data_ptr()),b,h,n,R,
            kt::gl_strides{size_t(n*h*R),R,size_t(h*R)}),
        HeldValueGlobal(reinterpret_cast<kt::bf16*>(x[2].data_ptr()),b,h,n,DV,
            kt::gl_strides{size_t(n*h*DV),DV,size_t(h*DV)}),
        x[4].data_ptr<int>(),x[5].data_ptr<int>(),boundary.data_ptr<int>(),x[6].data_ptr<bool>(),
        normalizer.data_ptr<float>(),logw.data_ptr<float>(),dlogw.data_ptr<float>(),
        moments.data_ptr<float>(),dv.data_ptr(),dsk.data_ptr()};
    args.key_output[0]=KeyOutputGlobal(reinterpret_cast<kt::bf16*>(dv.data_ptr()),b,h,n,DV,
        kt::gl_strides{size_t(n*h*DV),DV,size_t(h*DV)});
    args.key_output[1]=KeyOutputGlobal(reinterpret_cast<kt::bf16*>(dsk.data_ptr()),b,h,n,R,
        kt::gl_strides{size_t(n*h*R),R,size_t(h*R)});
    kt::gl<float,-1,-1,-1,-1,GradientTile> query_map(dsq.data_ptr<float>(),b,h,n,R,
        kt::gl_strides{size_t(n*h*R),R,size_t(h*R)});
    copy_output_map(args.query_output,query_map);
    auto stream=at::cuda::getCurrentCUDAStream();
    launch_kernel(hard_dlog_weights_address(),dim3((n+256)/256,h),256,0,stream,
                  logw.data_ptr<float>(),dlogw.data_ptr<float>(),int(n+1),int(h));
    int sms=at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    int ctas=std::min(int(b*h*(n/128)),requested_ctas?int(requested_ctas):sms);
    C10_CUDA_CHECK(cudaFuncSetAttribute(hard_backward_address(),cudaFuncAttributeMaxDynamicSharedMemorySize,hb::SharedBytes));
    launch_kernel(hard_backward_address(),ctas,384,hb::SharedBytes,stream,args);
    launch_kernel(hard_tau_reduce_address(),h,256,0,stream,moments.data_ptr<float>(),normalizer.data_ptr<float>(),
                  dlogw.data_ptr<float>(),dtau.data_ptr<float>(),int(b),int(h),int(n));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {dsq,dsk,dv,dtau};
}
} // namespace DISM_VARIANT
