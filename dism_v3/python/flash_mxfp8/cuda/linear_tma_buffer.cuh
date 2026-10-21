#pragma once
#include "kittens.cuh"
#include <stdexcept>

namespace fp8gemm {
// Row-major shared storage, with no MMA layout or swizzle.
template<class T,int Rows,int Cols> struct alignas(128) LinearTile {
    using dtype=T;
    static constexpr int rows=Rows, cols=Cols, required_alignment=128;
    T data[Rows*Cols];
};
}

namespace kittens::tma {
template<class T,int Rows,int Cols>
struct buffer_traits<fp8gemm::LinearTile<T,Rows,Cols>> {
    using buffer_type=fp8gemm::LinearTile<T,Rows,Cols>;
    using dtype=T;
    using coord_type=kittens::coord<>;
    static constexpr bool default_swizzle=false;
    // TK's load/store atoms issue tensor.5d; the fifth dimension is a singleton.
    static constexpr int rank=5;
    static constexpr uint32_t transfer_bytes=Rows*Cols*sizeof(T);
    template<bool EnableSwizzle,class GlobalT>
    __host__ static void encode(CUtensorMap *map,const global_tensor_view<GlobalT>& view) {
        static_assert(!EnableSwizzle && std::is_same_v<T,GlobalT>);
        static_assert(std::is_same_v<T,float> || std::is_same_v<T,kittens::fp8e4m3>);
        if(view.stride[3]!=1) throw std::runtime_error("Linear TMA requires unit column stride");
        uint64_t shape[5]={view.shape[3],view.shape[2],view.shape[1],view.shape[0],1};
        uint64_t stride[4]={view.stride[2]*sizeof(T),view.stride[1]*sizeof(T),
                            view.stride[0]*sizeof(T),view.stride[0]*view.shape[0]*sizeof(T)};
        uint32_t box[5]={Cols,Rows,1,1,1}, step[5]={1,1,1,1,1};
        auto format=std::is_same_v<T,float> ? CU_TENSOR_MAP_DATA_TYPE_FLOAT32 : CU_TENSOR_MAP_DATA_TYPE_UINT8;
        auto result=cuTensorMapEncodeTiled(map,format,rank,view.data,shape,stride,box,step,
            CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_NONE,
            CU_TENSOR_MAP_L2_PROMOTION_NONE,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
        if(result!=CUDA_SUCCESS) throw std::runtime_error("Invalid unswizzled TMA descriptor");
    }
    __device__ static void *shared_address(buffer_type& tile) {return tile.data;}
    __device__ static const void *shared_address(const buffer_type& tile) {return tile.data;}
    template<class Coord,class GL>
    __device__ static int5 coordinates(const Coord& idx,const GL&) {
        return {idx.c,idx.r,idx.d,idx.b,0};
    }
};
}
