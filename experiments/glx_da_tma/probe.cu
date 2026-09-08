#include <cuda.h>
#include <kittens.cuh>
#include <stdexcept>
namespace kt=kittens;
namespace da_probe {
template<int D> struct Scratch {
    kt::st_bf<16,64> grad;
    kt::st_bf<16,D> key;
    alignas(128) float out[2][16][32];
};
__device__ __forceinline__ void wait_read() {
    asm volatile("cp.async.bulk.wait_group.read 1;" ::: "memory");
}
template<int D,int WARPS>
__global__ void run(const float* g,const __nv_bfloat16* b,int n,int groups,float scale,
                   __grid_constant__ const CUtensorMap output) {
    extern __shared__ __align__(128) unsigned char memory[];
    int warp=threadIdx.x/32,lane=threadIdx.x%32;
    int group=blockIdx.x*WARPS+warp;
    if(group>=groups) return; // Warp-private scratch; no CTA-wide barriers.
    auto& s=reinterpret_cast<Scratch<D>*>(memory)[warp];
    for(int i=lane;i<16*D;i+=32) s.key[int2{i/D,i%D}]=b[group*16*D+i];
    __syncwarp();
    int issued=0;
    for(int qb=0;qb<n;qb+=64) {
        // Same (lane,register,element) coordinates as unrolled GLX 16x64 G.
        // Scatter logical query columns here, removing the score-MMA permutation.
        #pragma unroll
        for(int r=0;r<2;++r) {
            #pragma unroll
            for(int c=0;c<8;++c) {
                int k=r*8+lane/4,q=c+8*(lane%4);
                float x=qb+q<n?g[(group*16+k)*n+qb+q]:0;
                float y=qb+q+32<n?g[(group*16+k)*n+qb+q+32]:0;
                s.grad[int2{k,q}]=__float2bfloat16_rn(x);
                s.grad[int2{k,q+32}]=__float2bfloat16_rn(y);
            }
        }
        __syncwarp();
        #pragma unroll
        for(int qt=0;qt<4;++qt) {
            kt::rt_bf<16,16,kt::ducks::rt_layout::col> gt;
            auto gs=s.grad.template subtile<16,16>({0,qt});
            kt::warp::load(gt,gs);
            #pragma unroll
            for(int f=0;f<D/32;++f) {
                kt::rt_bf<16,32,kt::ducks::rt_layout::col> key;
                auto bs=s.key.template subtile<16,32>({0,f});
                kt::warp::load(key,bs);
                kt::rt_fl<16,32> da{0.f};
                kt::warp::wmma::mma_AtB(da,gt,key,da);
                int slot=issued%2;
                if(lane==0 && issued>=2) wait_read();
                __syncwarp(); // Old TMA has finished reading this slot.
                #pragma unroll
                for(int c=0;c<2;++c) {
                    #pragma unroll
                    for(int r=0;r<4;++r) {
                        auto x=da.tiles[0][c].data[r];
                        int row=(r%2)*8+lane/4,col=c*16+(r/2)*8+(lane%4)*2;
                        s.out[slot][row][col]=x.x*scale;
                        s.out[slot][row][col+1]=x.y*scale;
                    }
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                __syncwarp(); // Every writer publishes its stores to async proxy.
                if(lane==0) {
                    unsigned addr=static_cast<unsigned>(__cvta_generic_to_shared(s.out[slot]));
                    asm volatile("cp.reduce.async.bulk.tensor.2d.global.shared::cta.add.tile.bulk_group "
                                 "[%0, {%2, %3}], [%1];" ::
                                 "l"(&output),"r"(addr),"r"(f*32),"r"(qb+qt*16):"memory");
                    asm volatile("cp.async.bulk.commit_group;" ::: "memory");
                }
                ++issued;
            }
        }
        __syncwarp();
    }
    if(lane==0) asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
    __syncwarp();
}
template<int D,int W> void launch(const float* g,const void* b,float* out,int n,int groups,float scale,
                                  const CUtensorMap& map,cudaStream_t stream) {
    constexpr int bytes=W*sizeof(Scratch<D>);
    auto err=cudaFuncSetAttribute(run<D,W>,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes);
    if(err!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    run<D,W><<<(groups+W-1)/W,32*W,bytes,stream>>>(g,static_cast<const __nv_bfloat16*>(b),n,groups,scale,map);
}
template<int D> void dispatch(const float* g,const void* b,float* out,int n,int groups,int warps,float scale,
                             const CUtensorMap& map,cudaStream_t stream) {
    if(warps==1) launch<D,1>(g,b,out,n,groups,scale,map,stream);
    else launch<D,8>(g,b,out,n,groups,scale,map,stream);
}
}
void launch_da_probe(const float* g,const void* b,float* out,int n,int d,int groups,int warps,float scale,cudaStream_t stream) {
    CUtensorMap map;
    cuuint64_t dims[2]={static_cast<cuuint64_t>(d),static_cast<cuuint64_t>(n)};
    cuuint64_t strides[1]={static_cast<cuuint64_t>(d)*4};
    cuuint32_t box[2]={32,16},steps[2]={1,1};
    auto status=cuTensorMapEncodeTiled(&map,CU_TENSOR_MAP_DATA_TYPE_FLOAT32,2,out,dims,strides,box,steps,
        CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_NONE,CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if(status!=CUDA_SUCCESS) throw std::runtime_error("output tensor map encoding failed");
    switch(d) {
        case 32: da_probe::dispatch<32>(g,b,out,n,groups,warps,scale,map,stream); break;
        case 64: da_probe::dispatch<64>(g,b,out,n,groups,warps,scale,map,stream); break;
        case 128: da_probe::dispatch<128>(g,b,out,n,groups,warps,scale,map,stream); break;
    }
}

namespace da_probe {
template<int D> __global__ void db(const float* g,const __nv_bfloat16* a,float* out,int n,float scale) {
    __shared__ kt::st_bf<16,64> gs;
    __shared__ kt::st_bf<64,D> as;
    int lane=threadIdx.x,group=blockIdx.x;
    kt::rt_fl<16,D> acc{0.f};
    for(int qb=0;qb<n;qb+=64) {
        for(int x=lane;x<16*64;x+=32) gs[int2{x/64,x%64}]=qb+x%64<n?
            __float2bfloat16_rn(g[(group*16+x/64)*n+qb+x%64]):__float2bfloat16(0);
        for(int x=lane;x<64*D;x+=32) as[int2{x/D,x%D}]=qb+x/D<n?a[(qb+x/D)*D+x%D]:__float2bfloat16(0);
        __syncwarp();
        kt::rt_bf<16,64> grad;
        kt::rt_bf<64,D,kt::ducks::rt_layout::col> query;
        kt::warp::load(grad,gs);kt::warp::load(query,as);
        kt::warp::wmma::mma_AB(acc,grad,query,acc);
        __syncwarp();
    }
    #pragma unroll
    for(int c=0;c<D/16;++c) {
        #pragma unroll
        for(int r=0;r<4;++r) {
            auto x=acc.tiles[0][c].data[r];
            int k=(r%2)*8+lane/4,j=c*16+(r/2)*8+2*(lane%4);
            out[(group*16+k)*D+j]=x.x*scale;
            out[(group*16+k)*D+j+1]=x.y*scale;
        }
    }
}
}
void launch_db_probe(const float* g,const void* a,float* out,int n,int d,int groups,float scale,cudaStream_t stream) {
    auto input=static_cast<const __nv_bfloat16*>(a);
    switch(d) {
        case 32:da_probe::db<32><<<groups,32,0,stream>>>(g,input,out,n,scale);break;
        case 64:da_probe::db<64><<<groups,32,0,stream>>>(g,input,out,n,scale);break;
        case 128:da_probe::db<128><<<groups,32,0,stream>>>(g,input,out,n,scale);break;
    }
}
