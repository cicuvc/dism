#define DISM_HOST_API 1
#include "host/torch.cuh"
#include "varlen/layout.cuh"
namespace DISM_VARIANT {
namespace dism_varlen {
template<class E,bool Pack> const void* pack_address();
template<class Element,bool Pack>
void launch_copy(const at::Tensor& source,const at::Tensor& output,const Layout& layout,
                 int heads,int channels,int mode) {
    auto device_table=layout.cpu.to(source.device());
    int64_t extent=layout.max_padded*heads*channels;
    dim3 grid(std::min<int64_t>((extent+255)/256,65535),
              std::min<int64_t>(layout.sequences,65535));
    auto* src=static_cast<const Element*>(source.data_ptr());
    auto* dst=static_cast<Element*>(output.data_ptr());
    auto stream=at::cuda::getCurrentCUDAStream();
    if constexpr (Pack)
        launch_kernel(pack_address<Element,true>(),grid,256,0,stream,src,dst,device_table.data_ptr<int64_t>(),
            layout.sequences,layout.tokens,heads);
    else
        launch_kernel(pack_address<Element,false>(),grid,256,0,stream,src,dst,device_table.data_ptr<int64_t>(),
            layout.sequences,layout.tokens,heads,mode);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<bool Pack>
void dispatch_copy(const at::Tensor& source,const at::Tensor& output,const Layout& layout,
                   int64_t heads,int64_t channels,int mode) {
    TORCH_CHECK(heads>0 && channels>0 && layout.max_padded*heads*channels<=INT_MAX,
                "per-document staged element count exceeds32-bit packing limit");
    if (!output.numel()) return;
    switch (source.element_size()) {
        case 1: launch_copy<uint8_t,Pack>(source,output,layout,heads,channels,mode); break;
        case 2: launch_copy<uint16_t,Pack>(source,output,layout,heads,channels,mode); break;
        case 4: launch_copy<uint32_t,Pack>(source,output,layout,heads,channels,mode); break;
        case 8: launch_copy<uint64_t,Pack>(source,output,layout,heads,channels,mode); break;
        default: TORCH_CHECK(false,"unsupported packing element size");
    }
}
} // namespace dism_varlen

at::Tensor varlen_pack_cuda(at::Tensor input,at::Tensor table,bool vectors) {
    using namespace dism_varlen;
    Layout layout(table);
    TORCH_CHECK(input.is_cuda() && input.is_contiguous() &&
                input.dim()==(vectors?4:3) && input.size(0)==1,
                "expected contiguous CUDA vectors [1,T,H,C] or metadata [1,H,T]");
    TORCH_CHECK(input.size(vectors?1:2)==layout.tokens,"packed token count mismatch");
    int64_t heads=input.size(vectors?2:1),channels=vectors?input.size(3):1;
    c10::cuda::CUDAGuard guard(input.device());
    // Vector inputs already satisfy the alignment contract. Never allocate or
    // launch a copy for them, including low-level callers.
    if (vectors) return input.squeeze(0);
    auto output=at::empty({layout.tokens*heads},input.options());
    dispatch_copy<true>(input,output,layout,heads,channels,vectors);
    return output;
}

at::Tensor varlen_unpack_cuda(at::Tensor input,at::Tensor table,int64_t heads,int64_t mode) {
    using namespace dism_varlen;
    Layout layout(table);
    TORCH_CHECK(mode>=0 && mode<=2,"scalar layout mode must be compact0, padded1 or aligned-compact2");
    TORCH_CHECK(heads>0 && heads<=INT_MAX && input.is_cuda() && input.dim()==1 &&
                input.is_contiguous() && input.numel()==(mode?layout.padded:layout.tokens)*heads,
                "invalid sequence-major scalar buffer");
    c10::cuda::CUDAGuard guard(input.device());
    auto output=at::empty({1,heads,layout.tokens},input.options());
    dispatch_copy<false>(input,output,layout,heads,1,mode);
    return output;
}

} // namespace DISM_VARIANT
