#pragma once
#include "kittens.cuh"

namespace fp8gemm {
// Complete row-layout tiles. One STMATRIX x4 stores a 16-row x 32-byte
// fragment, for either BF16 or FP8. Keep the aligned base outside the XOR.
template<int GW,int R,class ST> struct FactoredStoreAddress {
    static constexpr int SB=ST::swizzle_bytes;
    static constexpr int PanelTiles=SB/32;
    static_assert(sizeof(typename ST::dtype)==1 || sizeof(typename ST::dtype)==2);
    static_assert(ST::rows==GW*R && ST::underlying_rows==ST::rows &&
                  ST::underlying_cols==ST::cols);
    static_assert(R%16==0 && alignof(ST)>=8*SB && ST::required_alignment>=8*SB);
    uint32_t base,offset;
    __device__ FactoredStoreAddress()=default;
    __device__ __forceinline__ explicit FactoredStoreAddress(ST &dst) {
        namespace kt=kittens;
        int lane=kt::laneid(),warp=kt::group<GW>::warpid();
        if constexpr(GW%4==0)warp=warp/4+(warp%4)*(GW/4);
        base=uint32_t(__cvta_generic_to_shared(dst.data));
        offset=ST::idx(0u,{lane%16,(lane/16)*16/int(sizeof(typename ST::dtype))})
               ^ (warp*R*SB);
        // Keep the derived column addresses local to this store operation.
        // Without this compiler barrier, ptxas can keep several addresses
        // alive across the entire GEMM loop and FP8 quantization.
        asm volatile("" : "+r"(offset));
    }
    __device__ __forceinline__ uint32_t get(int i,int j) const {
        return base+(offset^((j%PanelTiles)*32))+i*16*SB+(j/PanelTiles)*ST::rows*SB;
    }
};

template<int GW,class ST,class RT>
__device__ __forceinline__ void store_factored(ST &dst,const RT &src) {
    namespace kt=kittens;
    static_assert(std::is_same_v<typename RT::layout,kt::ducks::rt_layout::row>);
    static_assert(ST::cols==RT::cols && ST::width==RT::width);
    using U2=typename kt::base_types::packing<typename ST::dtype>::packed_type;
    FactoredStoreAddress<GW,RT::rows,ST> address(dst);
#pragma unroll
    for(int i=0;i<RT::height;i++) {
#pragma unroll
        for(int j=0;j<RT::width;j++) {
            U2 tmp[4];
#pragma unroll
            for(int p=0;p<4;p++)
                tmp[p]=kt::base_types::convertor<U2,typename RT::dtype>::convert(src.tiles[i][j].data[p]);
            kt::move<U2>::stsm4(address.get(i,j),tmp[0],tmp[1],tmp[2],tmp[3]);
        }
    }
}
}
