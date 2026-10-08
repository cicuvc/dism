#define DISM_HOST_API 1
#include "host/torch.cuh"
#include "summary/kernel_common.cuh"
#include "varlen/summary.cuh"

#include "variant.cuh"

namespace DISM_VARIANT {

const void* varlen_summary_address();
std::tuple<at::Tensor,at::Tensor> varlen_summary_cuda(
        std::vector<at::Tensor> operands,at::Tensor table) {
    using namespace dism_varlen;
    Layout layout(table);
    int h=validate_operands(operands,layout);
    c10::cuda::CUDAGuard guard(operands[0].device());
    auto options=operands[0].options().dtype(at::kFloat);
    auto a=at::empty({layout.forward*h},options);
    auto b=at::empty_like(a);
    if (!layout.tokens) return {a,b};
    TORCH_CHECK(layout.tokens*h<=INT_MAX,"packed metadata coordinate exceeds int32");
    int t=layout.tokens;
    const auto& x=operands;
    using QG=decltype(TmaSummarizationKernelArgs::QVec);
    using KG=decltype(TmaSummarizationKernelArgs::KVec);
    using FG=decltype(TmaSummarizationKernelArgs::QLseVec);
    using IG=decltype(TmaSummarizationKernelArgs::IdxQ);
    TmaSummarizationKernelArgs args{1,t,h,t,0,
        QG(reinterpret_cast<kt::bf16*>(x[0].data_ptr()),1,h,t,CONFIG.KcKeyDim,
           kt::gl_strides{size_t(t)*h*CONFIG.KcKeyDim,CONFIG.KcKeyDim,size_t(h*CONFIG.KcKeyDim)}),
        KG(reinterpret_cast<kt::bf16*>(x[1].data_ptr()),1,t,h,CONFIG.KcKeyDim),
        FG(x[5].data_ptr<float>(),1,1,1,t*h),FG(x[6].data_ptr<float>(),1,1,1,t*h),
        IG(x[7].data_ptr<int>(),1,1,1,t*h),IG(x[8].data_ptr<int>(),1,1,1,t*h),
        x[9].data_ptr<uint8_t>(),x[10].data_ptr<uint8_t>(),x[11].data_ptr<float>(),
        a.data_ptr<float>(),b.data_ptr<float>()};
    std::vector<int2> tasks;
    for (int s=0;s<layout.sequences;++s)
        append_tasks(tasks,s,h*(layout.get(s,Length)/CONFIG.getQBlockSize()));
    launch_common(varlen_summary_address(),args,layout,tasks,SUMMARY_SHARED_BYTES,operands[0].device());
    return {a,b};
}

} // namespace DISM_VARIANT
