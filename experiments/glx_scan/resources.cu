// Scan-only resource/liveness probe, NOT a complete attention kernel.
#include <glx/diagonal_scan.cuh>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
using namespace glx;

__host__ __device__ float lse(float x, float y) {
    if (x == -INFINITY) return y;
    if (y == -INFINITY) return x;
    return fmaxf(x,y) + log1pf(exp2f(-fabsf(x-y))) * 1.4426950408889634f;
}
struct LogAffine {
    template<class T> __host__ __device__ static BinaryElement<T> identity() {
        return {T{0.f}, T{-INFINITY}};
    }
    __host__ __device__ static BinaryElement<float> apply(
            BinaryElement<float> l, BinaryElement<float> r) {
        return {l.first+r.first, lse(l.second+r.first,r.second)};
    }
    __device__ static BinaryElement<F32x1> apply(
            BinaryElement<F32x1> l, BinaryElement<F32x1> r) {
        return {{l.first.u0+r.first.u0},{lse(l.second.u0+r.first.u0,r.second.u0)}};
    }
    __device__ static BinaryElement<F32x2> apply(
            BinaryElement<F32x2> l, BinaryElement<F32x2> r) {
        return {l.first+r.first,
            {lse(l.second.u0+r.first.u0,r.second.u0),
             lse(l.second.u1+r.first.u1,r.second.u1)}};
    }
};

template<int C, bool REVERSE, bool REDUCE, bool KEEP_SCORE, bool SCALAR_ROLL=true>
__global__ void probe(const float* input, float* output) {
    using Op = std::conditional_t<REVERSE, AffineComposeOp, LogAffine>;
    using Buffer = MMABuffer<16,C,BinaryElement,Op,F32x2>;
    MMABuffer<16,C,UnaryElement,AddOp,F32x2> scalar;
    Buffer tile;
    float original[2][C/8][2];
    const int lane = threadIdx.x & 31;
    #pragma unroll
    for(int r=0;r<2;++r) {
        #pragma unroll
        for(int c=0;c<C/8;++c) {
            auto p=Buffer::layout(r,c,0), q=Buffer::layout(r,c,1);
            F32x2 x{input[p.first*C+p.second],input[q.first*C+q.second]};
            // Opaque values forbid reloading/rematerializing the kept score.
            asm volatile("" : "+f"(x.u0), "+f"(x.u1));
            if constexpr(KEEP_SCORE) {
                original[r][c][0]=x.u0; original[r][c][1]=x.u1;
            }
            scalar.data[r][c].value=x;
        }
    }
    if constexpr(!REVERSE && SCALAR_ROLL) scalar.roll();
    #pragma unroll
    for(int r=0;r<2;++r) {
        #pragma unroll
        for(int c=0;c<C/8;++c) {
            auto x=scalar.data[r][c].value;
            if constexpr(REVERSE) tile.data[r][c] = {F32x2{0.5f},x};
            else tile.data[r][c] = {x,x};
        }
    }
    if constexpr(REVERSE) tile.reverse_roll();
    else if constexpr(!SCALAR_ROLL) tile.roll();
    typename Buffer::StatePair state;
    if constexpr(REDUCE) {
        if constexpr(REVERSE) state=tile.reduce_backward({},{});
        else state=tile.reduce_forward({},{});
    } else {
        if constexpr(REVERSE) {
            state=tile.reverse_inclusive_scan({},{});
            tile.template reverse_roll<false>();
        } else {
            state=tile.inclusive_scan({},{});
            tile.template roll<false>();
        }
        #pragma unroll
        for(int r=0;r<2;++r) {
            #pragma unroll
            for(int c=0;c<C/8;++c) {
                auto p=Buffer::layout(r,c,0), q=Buffer::layout(r,c,1);
                output[p.first*C+p.second]=tile.data[r][c].second.u0;
                output[q.first*C+q.second]=tile.data[r][c].second.u1;
                if constexpr(KEEP_SCORE) {
                    output[16*C+p.first*C+p.second]=original[r][c][0];
                    output[16*C+q.first*C+q.second]=original[r][c][1];
                }
            }
        }
    }
    // Keep both outgoing summaries observable, including for reduce-only.
    int off=32*C+lane*12;
    #pragma unroll
    for(int r=0;r<2;++r) {
        output[off+2*r]=state.first.init[r].first.u0;
        output[off+2*r+1]=state.first.init[r].second.u0;
    }
    output[off+4]=state.second.init[0].first.u0;
    output[off+5]=state.second.init[0].first.u1;
    output[off+6]=state.second.init[0].second.u0;
    output[off+7]=state.second.init[0].second.u1;
}

void check(cudaError_t err) {
    if(err!=cudaSuccess) { fprintf(stderr,"%s\n",cudaGetErrorString(err)); exit(1); }
}
template<int C,bool R,bool D,bool K=false,bool S=true> void report(float* in,float* out) {
    cudaFuncAttributes a;
    check(cudaFuncGetAttributes(&a,probe<C,R,D,K,S>));
    printf("16x%d %-7s %-6s keep=%d scalar_roll=%d regs=%d local=%zu shared=%zu\n",
           C,R?"affine":"log-lse",D?"reduce":"scan",K,!R && S,a.numRegs,a.localSizeBytes,a.sharedSizeBytes);
    probe<C,R,D,K,S><<<1,32>>>(in,out);
    check(cudaGetLastError()); check(cudaDeviceSynchronize());
}
template<int C> void shape(float* in,float* out) {
    report<C,false,false>(in,out);
    float saved[16*C];
    for(int i=0;i<16*C;++i) saved[i]=out[i];
    report<C,false,false,false,false>(in,out);
    for(int i=0;i<16*C;++i) if(std::memcmp(&saved[i],&out[i],sizeof(float))!=0) {
        fprintf(stderr,"scalar/tuple roll mismatch at %d\n",i); exit(1);
    }
    printf("16x%d scalar/tuple roll scan outputs bit-equivalent for probe input\n",C);
    report<C,false,false,true>(in,out);
    report<C,false,true>(in,out); report<C,true,false>(in,out);
    report<C,true,true>(in,out);
}
int main() {
    float *in,*out;
    check(cudaMallocManaged(&in,16*64*sizeof(float)));
    check(cudaMallocManaged(&out,(32*64+32*12)*sizeof(float)));
    for(int i=0;i<16*64;++i) in[i]=(i%13==0)?-INFINITY:-0.125f*(i%7);
    shape<32>(in,out); shape<64>(in,out);
    check(cudaFree(out)); check(cudaFree(in));
    return 0;
}
