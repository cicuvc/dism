#pragma once
#include "summary/primitives.cuh"
#include "varlen/layout.cuh"

#include "variant.cuh"
namespace DISM_VARIANT {

namespace dism_varlen {
#if DISM_HOST_API
inline int validate_operands(const std::vector<at::Tensor>& operands,const Layout& layout) {
    TORCH_CHECK(operands.size()>=12,"expected twelve staged varlen operands");
    const auto& q=operands[0];
    TORCH_CHECK(q.is_cuda() && q.dim()==3 && q.size(1)>0,"expected staged CUDA Q");
    int h=q.size(1);
    auto check=[&](int i,at::ScalarType dtype,at::IntArrayRef shape) {
        const auto& x=operands[i];
        TORCH_CHECK(x.device()==q.device() && x.scalar_type()==dtype &&
                    x.is_contiguous() && x.sizes()==shape,"invalid varlen operand ",i);
    };
    check(0,at::kBFloat16,{layout.padded,h,ActiveConfig::D});
    check(1,at::kBFloat16,{layout.padded,h,ActiveConfig::D});
    check(2,at::kBFloat16,{layout.padded,h,ActiveConfig::R});
    check(3,at::kBFloat16,{layout.padded,h,ActiveConfig::R});
    check(4,at::kBFloat16,{layout.padded,h,ActiveConfig::DV});
    check(5,at::kFloat,{layout.padded*h});
    check(6,at::kFloat,{layout.padded*h});
    check(7,at::kInt,{layout.padded*h});
    check(8,at::kInt,{layout.padded*h});
    check(9,at::kByte,{layout.padded*h});
    check(10,at::kByte,{1,h});
    check(11,at::kFloat,{h});
    if(operands.size()==14) check(13,at::kFloat,{layout.padded*h});
    return h;
}

inline void append_tasks(std::vector<int2>& tasks,int record,int count) {
    TORCH_CHECK(int64_t(tasks.size())+count<=INT_MAX,"too many varlen tasks");
    for (int i=0;i<count;++i) tasks.push_back(make_int2(record,i));
}

#endif

template<class Common> struct PackedArgs {
    Common common;
    const int64_t* documents;
    const int2* tasks;
};

#if DISM_HOST_API
template<class Kernel,class Common>
inline void launch_common(Kernel kernel,const Common& common,const Layout& layout,
        const std::vector<int2>& tasks,size_t shared_bytes,at::Device device,int threads=384) {
    if (tasks.empty()) return;
    auto documents=dism_metadata::documents(layout.cpu,device);
    auto work=upload_records(tasks,device);
    PackedArgs<Common> args{common,documents.data_ptr<int64_t>(),
                           reinterpret_cast<const int2*>(work.data_ptr())};
    if (shared_bytes)
        C10_CUDA_CHECK(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,shared_bytes));
    launch_kernel(kernel,tasks.size(),threads,shared_bytes,at::cuda::getCurrentCUDAStream(),args);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
#endif
} // namespace dism_varlen

} // namespace DISM_VARIANT
