#define DISM_HOST_API 1
#include "host/torch.cuh"
#include "varlen/chunk.cuh"
namespace DISM_VARIANT {
namespace dism_varlen {
template<bool Reverse> const void* varlen_chunk_address();
template<bool Reverse>
at::Tensor launch_chunk(at::Tensor a,at::Tensor b,at::Tensor table,int64_t heads) {
    Layout layout(table);
    int64_t size=(Reverse?layout.backward:layout.forward)*heads;
    TORCH_CHECK(heads>0 && heads<=65535 && a.is_cuda() && a.scalar_type()==at::kFloat &&
                a.dim()==1 && a.is_contiguous() && a.numel()==size &&
                b.device()==a.device() && b.scalar_type()==at::kFloat &&
                b.is_contiguous() && b.sizes()==a.sizes(),"invalid varlen affine buffers");
    c10::cuda::CUDAGuard guard(a.device());
    auto output=at::empty_like(a);
    if (!size) return output;
    TORCH_CHECK(size/32<=INT_MAX,"packed checkpoint row coordinate exceeds int32");
    ChunkArgs args{ChunkGlobal(a.data_ptr<float>(),1,1,size/32,32),
                   ChunkGlobal(b.data_ptr<float>(),1,1,size/32,32),
                   output.data_ptr<float>(),int(heads)};
    std::vector<int2> tasks;
    for (int64_t s=0;s<layout.sequences;++s) {
        int n=layout.get(s,Length),p=layout.get(s,PaddedLength);
        int count=Reverse?(n+31)/32:(n?((n-1)/32):0);
        if (!count) continue;
        int blocks=(p+(count-1)*32+127)/128;
        append_tasks(tasks,s,heads*blocks);
    }
    launch_common(varlen_chunk_address<Reverse>(),args,layout,tasks,0,a.device(),128);
    return output;
}
} // namespace dism_varlen

at::Tensor varlen_chunk_cuda(at::Tensor a,at::Tensor b,at::Tensor table,int64_t heads) {
    return dism_varlen::launch_chunk<false>(a,b,table,heads);
}
at::Tensor varlen_backward_chunk_cuda(at::Tensor a,at::Tensor b,at::Tensor table,int64_t heads) {
    return dism_varlen::launch_chunk<true>(a,b,table,heads);
}

} // namespace DISM_VARIANT
