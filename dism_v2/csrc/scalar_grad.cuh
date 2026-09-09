#pragma once
// Included inside a kernel namespace after log_affine.cuh.
// G is dL/d(natural logM), before the BF16 conversion used by gradient MMA.
template<bool TAU_ONLY=false>
__device__ __forceinline__ void scalar_gradients(
        const Args& p,const Scalar& value,int bh,int kb,int qb,
        uint32_t hard0,uint32_t hard1,float* dlse,float& tau,float (&key_lse)[2]) {
    int lane=threadIdx.x&31,g=lane&3,l=lane/4;
    #pragma unroll
    for(int c=0;c<8;++c) {
        int q=qb+c+8*g;
        float a=0,b=0;
        #pragma unroll
        for(int r=0;r<2;++r) {
            int k=kb+8*r+l;
            auto x=value.data[r][c].value;
            float u=k<p.n && q<p.n && k<=q?x.u0:0.f;
            float v=k<p.n && q+32<p.n && k<=q+32?x.u1:0.f;
            tau+=u+v; // Hard matches participate; hard breaks have exact G=0.
            if constexpr(!TAU_ONLY) {
            u=((hard0>>(c+8*g))&1)?0.f:u;
            v=((hard1>>(c+8*g))&1)?0.f:v;
            if(p.column_lse) key_lse[r]-=u+v;
            else {a-=u;b-=v;}
            }
        }
        if constexpr(!TAU_ONLY) {
        if(!p.column_lse) {
            #pragma unroll
            for(int shift=4;shift<=16;shift*=2) {
                a+=__shfl_xor_sync(0xffffffff,a,shift);
                b+=__shfl_xor_sync(0xffffffff,b,shift);
            }
            if(l==0 && kb<p.n) {
                if(q<p.n) atomicAdd(dlse+int64_t(bh)*p.n+q,a);
                if(q+32<p.n) atomicAdd(dlse+int64_t(bh)*p.n+q+32,b);
            }
        }
        }
    }
}
__device__ __forceinline__ void store_key_lse(
        const Args& p,int bh,int kb,float* dlse,float (&key_lse)[2]) {
    if(p.column_lse) {
        int lane=threadIdx.x&31;
        #pragma unroll
        for(int r=0;r<2;++r) {
            float x=key_lse[r];
            x+=__shfl_xor_sync(0xffffffff,x,1);
            x+=__shfl_xor_sync(0xffffffff,x,2);
            int k=kb+8*r+lane/4;
            if((lane&3)==0 && k<p.n) dlse[int64_t(bh)*p.n+k]=x;
        }
    }
}
__device__ __forceinline__ float warp_sum(float x) {
    #pragma unroll
    for(int shift=16;shift>0;shift/=2) x+=__shfl_down_sync(0xffffffff,x,shift);
    return x;
}
