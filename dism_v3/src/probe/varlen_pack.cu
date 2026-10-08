#include "varlen/layout.cuh"

#include "variant.cuh"
namespace DISM_VARIANT {

namespace dism_varlen {
// Copy raw elements, not floating-point conversions. Local element indices
// are32-bit (validated below); global offsets remain64-bit. This avoids
// compiler-emitted out-of-line64-bit division helpers in the packing kernels.
template<class Element>
__global__ void pack_kernel(const Element* source,Element* output,
        const int64_t* table,int64_t sequences,int64_t tokens,int heads) {
    for (int64_t s=blockIdx.y;s<sequences;s+=gridDim.y) {
        const int64_t* row=table+s*Fields;
        int n=row[Length];
        int64_t total=int64_t(n)*heads;
        for (int64_t linear=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
             linear<total;linear+=int64_t(gridDim.x)*blockDim.x) {
            int element=int(linear),token=element%n,head=element/n;
            int64_t src=int64_t(head)*tokens+row[Begin]+token;
            output[row[Begin]*heads+linear]=source[src];
        }
    }
}

template<class Element>
__global__ void unpack_metadata_kernel(const Element* source,Element* output,
        const int64_t* table,int64_t sequences,int64_t tokens,int heads,int mode) {
    for (int64_t s=blockIdx.y;s<sequences;s+=gridDim.y) {
        const int64_t* row=table+s*Fields;
        int n=row[Length],pitch=mode==1?row[PaddedLength]:n;
        int64_t base=(mode?row[PaddedBegin]:row[Begin])*heads;
        for (int64_t linear=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
             linear<int64_t(n)*heads;linear+=int64_t(gridDim.x)*blockDim.x) {
            int element=int(linear),head=element/n,token=element%n;
            output[int64_t(head)*tokens+row[Begin]+token]=
                source[base+int64_t(head)*pitch+token];
        }
    }
}

template<class E,bool Pack> const void* pack_address() { if constexpr (Pack) return reinterpret_cast<const void*>(pack_kernel<E>); else return reinterpret_cast<const void*>(unpack_metadata_kernel<E>); }
template const void* pack_address<uint8_t,false>();
template const void* pack_address<uint8_t,true>();
template const void* pack_address<uint16_t,false>();
template const void* pack_address<uint16_t,true>();
template const void* pack_address<uint32_t,false>();
template const void* pack_address<uint32_t,true>();
template const void* pack_address<uint64_t,false>();
template const void* pack_address<uint64_t,true>();
} // namespace dism_varlen
} // namespace DISM_VARIANT
