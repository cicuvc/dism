#define DISM_HOST_API 1
#include "host/torch.cuh"
#include "varlen/probe.cuh"
namespace DISM_VARIANT {
namespace dism_varlen {
template<int Rows> const void* descriptor_address();
template<int Rows>
at::Tensor run_descriptor_probe(at::Tensor input,const Layout& layout) {
    int heads=input.size(1);
    int64_t total=layout.padded;
    for (int64_t s=0;s<layout.sequences;++s) if (layout.get(s,Length)) total+=Rows;
    auto output=at::empty({total,heads,64},input.options());
    std::vector<ProbeRecord<Rows>> records;
    std::vector<int3> tasks;
    int64_t out_start=0;
    for (int64_t s=0;s<layout.sequences;++s) {
        int p=layout.get(s,PaddedLength);
        if (!p) continue;
        auto* src=reinterpret_cast<kt::bf16*>(input.data_ptr())+layout.get(s,PaddedBegin)*heads*64;
        auto* dst=reinterpret_cast<kt::bf16*>(output.data_ptr())+out_start*heads*64;
        int index=records.size();
        records.push_back({typename ProbeRecord<Rows>::Global(src,1,p,heads,64),dst,heads});
        for (int h=0;h<heads;++h)
            for (int start=0;start<=p;start+=Rows) tasks.push_back(make_int3(index,h,start));
        out_start+=p+Rows;
    }
    if (tasks.empty()) return output;
    auto device_records=upload_records(records,input.device());
    auto device_tasks=upload_records(tasks,input.device());
    launch_kernel(descriptor_address<Rows>(),tasks.size(),32,0,at::cuda::getCurrentCUDAStream(),
        reinterpret_cast<const ProbeRecord<Rows>*>(device_records.data_ptr()),
        reinterpret_cast<const int3*>(device_tasks.data_ptr()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
} // namespace dism_varlen

at::Tensor varlen_tma_probe_cuda(at::Tensor input,at::Tensor table,int64_t rows) {
    using namespace dism_varlen;
    Layout layout(table);
    TORCH_CHECK(input.is_cuda() && input.scalar_type()==at::kBFloat16 &&
                input.is_contiguous() && input.dim()==3 && input.size(0)==layout.padded &&
                input.size(1)>0 && input.size(2)==64,"expected packed BF16 [padded,H,64]");
    TORCH_CHECK(rows==32 || rows==64,"probe tile rows must be32 or64");
    c10::cuda::CUDAGuard guard(input.device());
    return rows==32 ? run_descriptor_probe<32>(input,layout) : run_descriptor_probe<64>(input,layout);
}

} // namespace DISM_VARIANT
