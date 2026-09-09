#pragma once
#if DISM_BWD_OPT >= 3
namespace dism_v2::bwd_input {
// Same direct RHS mapping validated in summary_persistent.cuh. Absorb the
// transpose in LDSM addressing/register ownership, not a post-load MOV pass.
template<int ROWS,int COLS>
__device__ __forceinline__ void load_rhs(
        kittens::rt_bf<COLS,ROWS,kittens::ducks::rt_layout::col>& dst,
        const kittens::st_bf<ROWS,COLS>& src) {
    int lane=kittens::laneid();
    uint32_t addr=smaddr(src.data);
    #pragma unroll
    for(int c=0;c<dst.height;++c) {
        #pragma unroll
        for(int r=0;r<dst.width;++r) {
            kittens::bf16_2 tmp[4];
            int row=16*r+(lane/16)*8+(lane%8),col=16*c+((lane/8)%2)*8;
            kittens::move<kittens::bf16_2>::ldsm4(tmp[0],tmp[2],tmp[1],tmp[3],src.idx(addr,{row,col}));
            dst.tiles[c][r].data[0]=tmp[0]; dst.tiles[c][r].data[1]=tmp[1];
            dst.tiles[c][r].data[2]=tmp[2]; dst.tiles[c][r].data[3]=tmp[3];
        }
    }
}
template<int D> CUtensorMap held_map(const Args& p,const void* source) {
    constexpr int S=D==32?32:64;
    const cuuint64_t dims[]{S,cuuint64_t(p.n),cuuint64_t(p.batch_heads),D/S,1};
    const cuuint64_t strides[]{D*2,cuuint64_t(p.n)*D*2,S*2,S*2};
    const cuuint32_t box[]{S,128,1,D/S,1},steps[]{1,1,1,1,1};
    CUtensorMap map{};
    auto e=cuTensorMapEncodeTiled(&map,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,5,
        const_cast<void*>(source),dims,strides,box,steps,CU_TENSOR_MAP_INTERLEAVE_NONE,
        D==32?CU_TENSOR_MAP_SWIZZLE_64B:CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if(e!=CUDA_SUCCESS) throw std::runtime_error("backward held B/V tensor map failed");
    return map;
}
__device__ __forceinline__ void load(const CUtensorMap* map,void* dst,uint64_t* bar,int4 coord) {
    kittens::tma::atoms::load_async_atom<kittens::cache_policy::NORMAL>(
        smaddr(dst),reinterpret_cast<uint64_t>(map),coord,*reinterpret_cast<kittens::semaphore*>(bar));
}
}
#endif
