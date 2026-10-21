#pragma once
#include "kittens.cuh"
#include "shared_store.cuh"

namespace fp8gemm {
// Each word is a column-layout m16n8.b8 fragment. This is deliberately not
// the standard FP8 MMA register layout.
template<int R,int C> struct MXOutputTransposed {
    uint32_t data[R/16][C/32][4];
};

template<int R,int C>
__device__ __forceinline__ void quantize_output_movmatrix(
    MXOutputTransposed<R,C> &dst,kt::mxfp8_scale<R,C> &scales,
    const kt::rt_bf<R,C> &src) {
    static_assert(C%32==0);
    int lane=kt::laneid();
#pragma unroll
    for(int i=0;i<R/16;i++) {
#pragma unroll
        for(int k=0;k<C/32;k++) {
            uint32_t rows[2];
#pragma unroll
            for(int r=0;r<2;r++) {
                uint32_t v=0;
#pragma unroll
                for(int j=0;j<2;j++) {
#pragma unroll
                    for(int p=r;p<4;p+=2)
                        v=kt::detail::online_max_u16x2(v,
                            kt::detail::online_bf16_bits(src.tiles[i][2*k+j].data[p])&0x7f807f80u);
                }
                rows[r]=max(v&0xffffu,v>>16);
            }
            uint32_t e=rows[0]|(rows[1]<<16);
            e=kt::detail::online_max_u16x2(e,__shfl_xor_sync(0xffffffff,e,1));
            e=kt::detail::online_max_u16x2(e,__shfl_xor_sync(0xffffffff,e,2));
            e=kt::detail::online_shift_e8m0_u16x2(e);
            scales.data[i][k]=e>>(7+16*(lane&1));
            uint32_t rb=(0x7f007f00u-e)&0x7f807f80u;
#pragma unroll
            for(int r=0;r<2;r++) {
                uint32_t h=(rb>>(16*r))&0xffffu,rcp=h|(h<<16);
#pragma unroll
                for(int j=0;j<2;j++) {
                    uint16_t q[2];
#pragma unroll
                    for(int p=0;p<2;p++) {
                        uint32_t x=kt::detail::online_bf16_bits(src.tiles[i][2*k+j].data[r+2*p]);
                        uint32_t scaled,transposed;
                        // Normalize in the original row layout. After MOVM,
                        // the two BF16 elements belong to different rows.
                        asm("mul.bf16x2 %0, %1, %2;":"=r"(scaled):"r"(x),"r"(rcp));
                        asm("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;"
                            :"=r"(transposed):"r"(scaled));
                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;"
                            :"=h"(q[p]):"r"(transposed));
                    }
                    dst.data[i][k][r+2*j]=uint32_t(q[0])|(uint32_t(q[1])<<16);
                }
            }
        }
    }
}

template<int GW,bool Factored=false,int R,int C,class ST>
__device__ __forceinline__ void store_output_movmatrix(
    ST &dst,const MXOutputTransposed<R,C> &src) {
    static_assert(ST::rows==GW*R && ST::cols==C);
    int warp=(threadIdx.x/32)%GW,lane=kt::laneid();
    uint32_t base=uint32_t(__cvta_generic_to_shared(dst.data));
    FactoredStoreAddress<GW,R,ST> address;
    if constexpr(Factored)address=FactoredStoreAddress<GW,R,ST>(dst);
#pragma unroll
    for(int i=0;i<R/16;i++) {
#pragma unroll
        for(int j=0;j<C/32;j++) {
            uint32_t addr;
            if constexpr(Factored)addr=address.get(i,j);
            else addr=ST::idx(base,{warp*R+i*16+lane%16,j*32+(lane/16)*16});
            auto &d=src.data[i][j];
            asm volatile("stmatrix.sync.aligned.m16n8.x4.trans.shared::cta.b8 [%0], {%1,%2,%3,%4};"
                         ::"r"(addr),"r"(d[0]),"r"(d[1]),"r"(d[2]),"r"(d[3]):"memory");
        }
    }
}

// Next GEMM's A-scale layout: uint32 [ceil(N/128), roundup(M,16)].
// Complete groups use word stores; C32/C64 tiles own disjoint scale bytes.
template<int R,int C>
__device__ __forceinline__ void store_output_scales(
    uint32_t *dst,const kt::mxfp8_scale<R,C> &scales,int mr,int nc,int m,int n) {
    static_assert(C%32==0);
    int lane=kt::laneid(),mp=(m+15)/16*16;
    if((lane&3)<2) {
#pragma unroll
        for(int i=0;i<R/16;i++) {
            int row=mr+i*16+lane/4+8*(lane&1);
            if(row<mp) {
                if constexpr(C%128==0) {
#pragma unroll
                    for(int j=0;j<C/128;j++) {
                        uint32_t word=0;
#pragma unroll
                        for(int k=0;k<4;k++)word|=(scales.data[i][4*j+k]&255u)<<(8*k);
                        if(nc/128+j<(n+127)/128)
                            dst[size_t(nc/128+j)*mp+row]=word;
                    }
                } else {
                    auto *bytes=reinterpret_cast<uint8_t*>(dst);
#pragma unroll
                    for(int j=0;j<C/32;j++) {
                        int block=nc/32+j;
                        if(block<(n+31)/32)
                            bytes[(size_t(block/4)*mp+row)*4+block%4]=scales.data[i][j];
                    }
                    // Deterministic zero padding, including reused graph outputs.
                    if(nc+C>=n) {
                        for(int block=(n+31)/32;block<(n+127)/128*4;block++)
                            bytes[(size_t(block/4)*mp+row)*4+block%4]=0;
                    }
                }
            }
        }
    }
}
} // namespace fp8gemm
