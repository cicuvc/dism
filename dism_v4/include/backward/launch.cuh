#pragma once
#include "backward/types.cuh"

#include "variant.cuh"
namespace DISM_VARIANT {

namespace dism_backward {
inline void set_key_output(Args& args, int index, const at::Tensor& output) {
    if (output.scalar_type()==at::kBFloat16) {
        int b=output.size(0),n=output.size(1),h=output.size(2),c=output.size(3);
        args.key_output[index]=KeyOutputGlobal(reinterpret_cast<kt::bf16*>(output.data_ptr()),
            b,h,n,c,kt::gl_strides{size_t(n*h*c),size_t(c),size_t(h*c)});
    }
}
template<class Global>
inline void copy_output_map(OutputMap& output, Global& global) {
    std::vector<CUtensorMap*> maps;
    global.collect_tmaps(maps);
    TORCH_CHECK(maps.size()==1,"expected one output tensor map");
    output.descriptor = *maps[0];
}
inline void set_query_output(Args& args, const at::Tensor& output) {
    int b=output.size(0),n=output.size(1),h=output.size(2),c=output.size(3);
    kt::gl<float,-1,-1,-1,-1,GradientTile> global(output.data_ptr<float>(),b,h,n,c,
        kt::gl_strides{size_t(n*h*c),size_t(c),size_t(h*c)});
    copy_output_map(args.query_output,global);
}
inline void set_lse_output(Args& args, const at::Tensor& output) {
    // Tensor-map strides must be 16-byte aligned; other lengths use scalar adds.
    if (args.score.n%4==0) {
        kt::gl<float,-1,-1,-1,-1,GradientVector> global(output.data_ptr<float>(),
            args.score.batch,args.score.heads,1,args.score.n);
        copy_output_map(args.lse_output,global);
    }
}
inline Args make_args(const std::vector<at::Tensor>& operands,const at::Tensor& vertical,
        const at::Tensor& dout,const at::Tensor& normalizer,const at::Tensor& delta,int n) {
    auto score=make_recompute_args(operands,vertical,n);
    int b=score.batch,h=score.heads,np=score.padded;
    auto check=[&](const at::Tensor& x,at::ScalarType type,at::IntArrayRef shape) {
        TORCH_CHECK(x.device()==operands[0].device() && x.scalar_type()==type && x.is_contiguous() &&
                    x.sizes()==shape,"invalid backward operand");
    };
    check(operands[2],at::kBFloat16,{b,np,h,R});
    check(operands[3],at::kBFloat16,{b,np,h,R});
    check(operands[4],at::kBFloat16,{b,np,h,DV});
    check(dout,at::kBFloat16,{b,np,h,DV});
    check(normalizer,at::kFloat,{b,h,n});
    check(delta,at::kFloat,{b,h,n});
    return {score,
        SoftQueryGlobal(reinterpret_cast<kt::bf16*>(operands[2].data_ptr()),b,np,h,R),
        DerivativeGlobal(reinterpret_cast<kt::bf16*>(dout.data_ptr()),b,np,h,DV),
        HeldKeyGlobal(reinterpret_cast<kt::bf16*>(operands[1].data_ptr()),b,h,np,D,
                      kt::gl_strides{size_t(np*h*D),D,size_t(h*D)}),
        HeldSoftGlobal(reinterpret_cast<kt::bf16*>(operands[3].data_ptr()),b,h,np,R,
                       kt::gl_strides{size_t(np*h*R),R,size_t(h*R)}),
        HeldValueGlobal(reinterpret_cast<kt::bf16*>(operands[4].data_ptr()),b,h,np,DV,
                        kt::gl_strides{size_t(np*h*DV),DV,size_t(h*DV)}),
        normalizer.data_ptr<float>(),delta.data_ptr<float>(),nullptr,
        nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr};
}
template<class Kernel>
inline void launch(Kernel kernel,const Args& args,int requested_ctas) {
    int tasks=args.score.batch*args.score.heads*((args.score.n+127)/128);
    int sms=at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    int ctas=std::min(tasks,requested_ctas>0?requested_ctas:sms);
    C10_CUDA_CHECK(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,SharedBytes));
    launch_kernel(kernel,ctas,384,SharedBytes,at::cuda::getCurrentCUDAStream(),args);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
} // namespace dism_backward

} // namespace DISM_VARIANT
