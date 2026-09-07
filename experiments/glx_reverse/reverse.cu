// Standalone diagnostic: full W/G buffers exist only for oracle comparisons.
#include "../../dism_v2/csrc/log_affine.cuh"
#include "../../dism_v2/csrc/pipeline.cuh"
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
using namespace dism_v2;
using Reverse=glx::MMABuffer<16,64,glx::BinaryElement,glx::AffineComposeOp,glx::F32x2>;

void check(cudaError_t e) { if(e!=cudaSuccess) { std::fprintf(stderr,"%s\n",cudaGetErrorString(e)); std::exit(2); } }
__host__ __device__ float dp(int q,int k) { return float((q*13+k*7)%29-14)*.125f; }
__device__ __forceinline__ float reciprocal(float x) {
    float y; asm("rcp.approx.ftz.f32 %0,%1;":"=f"(y):"f"(x));
    return fmaf(y,fmaf(-x,y,1.f),y);
}
__device__ __forceinline__ float2 coefficient(const float* w,const float* norm,int q,int k,int n,int np) {
    if(q>=n || k>=n) return {1,0};
    float x=w[q*np+k];
    if(x==-INFINITY) return {0,0};
    float z=exp2f(-fabsf(x));
    float a=reciprocal(1+z); if(x<0) a*=z;
    return {a,exp2f(x-norm[q])*dp(q,k)};
}

template<bool SCAN> __global__ void reverse_tiles(const float* w,const float* norm,
        float2* summary,const float* boundary,float* gradient,int n,int np,int cp) {
    __shared__ Reverse::HState::SharedStorage mail[4][2];
    __shared__ uint64_t ready[4][2],free[4][2];
    int warp=threadIdx.x/32,lane=threadIdx.x&31,g=lane&3,l=lane/4;
    int chunk=blockIdx.x*4+(warp&3),kb=chunk*32+(warp/4)*16;
    if(threadIdx.x==0) {
        for(int p=0;p<4;++p) for(int s=0;s<2;++s) {
            init_bar(&ready[p][s],32); init_bar(&free[p][s],32);
        }
    }
    __syncthreads();
    Reverse::VState right;
    for(int t=0;t*64<np;++t) {
        int qb=np-64-t*64,s=t%2,phase=(t/2)&1;
        Reverse data;
        #pragma unroll
        for(int r=0;r<2;++r) {
            #pragma unroll
            for(int c=0;c<8;++c) {
                auto pos=Reverse::layout(r,c,0);
                auto a=coefficient(w,norm,qb+pos.second,kb+pos.first,n,np);
                auto b=coefficient(w,norm,qb+pos.second+32,kb+pos.first,n,np);
                data.data[r][c]={{a.x,b.x},{a.y,b.y}};
            }
        }
        // Components differ: both must undergo reverse roll.
        data.reverse_roll();
        Reverse::HState bottom;
        if(warp<4) {
            wait(&ready[warp][s],phase);
            bottom=Reverse::HState::load_shared(mail[warp][s]);
            arrive(&free[warp][s]);
        } else if constexpr(SCAN) {
            if(chunk+1<cp) {
                int q=qb+8*g+8-l;
                bottom.init[0].first={1,1};
                bottom.init[0].second={q<np?boundary[(chunk+1)*np+q]:0,
                    q+32<np?boundary[(chunk+1)*np+q+32]:0};
            }
        }
        Reverse::StatePair result;
        if constexpr(SCAN) result=data.reverse_inclusive_scan(right,bottom);
        else result=data.reduce_backward(right,bottom);
        right=result.first;
        if(warp>=4) {
            if(t>=2) wait(&free[warp-4][s],phase^1);
            result.second.store_shared(mail[warp-4][s]);
            arrive(&ready[warp-4][s]);
        } else if constexpr(!SCAN) {
            if(chunk<cp) {
                int q=qb+8*g+8-l;
                auto x=result.second.init[0];
                // Reverse HState encodes local columns 1..64; column0 is VState.
                if(q<qb+64) summary[chunk*np+q]={x.first.u0,x.second.u0};
                if(q+32<qb+64) summary[chunk*np+q+32]={x.first.u1,x.second.u1};
                if(lane==0) summary[chunk*np+qb]={right.init[0].first.u0,right.init[0].second.u0};
            }
        }
        if constexpr(SCAN) {
            Scalar scalar;
            #pragma unroll
            for(int r=0;r<2;++r) {
                #pragma unroll
                for(int c=0;c<8;++c) scalar.data[r][c].value=data.data[r][c].second;
            }
            scalar.template reverse_roll<false>();
            #pragma unroll
            for(int r=0;r<2;++r) {
                #pragma unroll
                for(int c=0;c<8;++c) {
                    auto pos=Reverse::layout(r,c,0); auto x=scalar.data[r][c].value;
                    int k=kb+pos.first,q=qb+pos.second;
                    if(k<np) { gradient[q*np+k]=x.u0; gradient[(q+32)*np+k]=x.u1; }
                }
            }
        }
    }
    __syncthreads();
}

__global__ void reverse_passing(const float2* summary,float* boundary,int np,int cp) {
    int d=int(blockIdx.x*blockDim.x+threadIdx.x)-(cp-1)*32;
    if(d>=np) return;
    float x=0;
    for(int s=cp-1;s>=0;--s) {
        int q=d+s*32;
        if(q>=0 && q<np) {
            auto pair=summary[s*np+q]; x=fmaf(pair.x,x,pair.y); boundary[s*np+q]=x;
        }
    }
}

double ladd(double a,double b) {
    if(a==-INFINITY) return b; if(b==-INFINITY) return a;
    return std::max(a,b)+std::log1p(std::exp(-std::abs(a-b)));
}
bool run(int n,int mode) {
    int np=(n+63)/64*64,cp=np/32;
    std::vector<float> w(np*np,-INFINITY),norm(np,0);
    for(int q=0;q<n;++q) {
        double l=0;
        for(int k=0;k<=q;++k) {
            double m=double((q*17+k*11)%23-16)/8;
            if(mode==1 && q%3==0) m=q%7==k%7?.2:-INFINITY;
            if(mode==2) m=-INFINITY;
            if(mode==3) m=std::log(64.);
            if(mode==4) m=-16.;
            double previous=q>0 && k>0?w[(q-1)*np+k-1]*std::log(2.):-INFINITY;
            w[q*np+k]=float((m+ladd(0,previous))/std::log(2.));
            l=ladd(l,w[q*np+k]*std::log(2.));
        }
        norm[q]=float(l/std::log(2.));
    }
    std::vector<double> a(np*np,1),e(np*np),expected(np*np);
    for(int q=0;q<n;++q) for(int k=0;k<n;++k) {
        double x=w[q*np+k],z=std::exp2(-std::abs(x));
        a[q*np+k]=(x>=0?1:z)/(1+z);
        e[q*np+k]=std::exp2(x-norm[q])*dp(q,k);
    }
    for(int q=np-1;q>=0;--q) for(int k=np-1;k>=0;--k)
        expected[q*np+k]=e[q*np+k]+a[q*np+k]*(q+1<np && k+1<np?expected[(q+1)*np+k+1]:0);
    float *dw,*dn,*db,*dg; float2* ds;
    check(cudaMalloc(&dw,np*np*sizeof(float))); check(cudaMalloc(&dn,np*sizeof(float)));
    check(cudaMalloc(&db,cp*np*sizeof(float))); check(cudaMalloc(&dg,np*np*sizeof(float)));
    check(cudaMalloc(&ds,cp*np*sizeof(float2)));
    check(cudaMemcpy(dw,w.data(),w.size()*sizeof(float),cudaMemcpyHostToDevice));
    check(cudaMemcpy(dn,norm.data(),norm.size()*sizeof(float),cudaMemcpyHostToDevice));
    reverse_tiles<false><<<(cp+3)/4,256>>>(dw,dn,ds,db,dg,n,np,cp);
    reverse_passing<<<(np+(cp-1)*32+127)/128,128>>>(ds,db,np,cp);
    reverse_tiles<true><<<(cp+3)/4,256>>>(dw,dn,ds,db,dg,n,np,cp);
    check(cudaGetLastError()); check(cudaDeviceSynchronize());
    std::vector<float> got(np*np),bound(cp*np); std::vector<float2> sum(cp*np);
    check(cudaMemcpy(got.data(),dg,got.size()*sizeof(float),cudaMemcpyDeviceToHost));
    check(cudaMemcpy(bound.data(),db,bound.size()*sizeof(float),cudaMemcpyDeviceToHost));
    check(cudaMemcpy(sum.data(),ds,sum.size()*sizeof(float2),cudaMemcpyDeviceToHost));
    double error=0; bool ok=true;
    auto compare=[&](double x,double y) { error=std::max(error,std::abs(x-y)); if(!std::isfinite(x) || std::abs(x-y)>3e-5+3e-5*std::abs(y)) ok=false; };
    for(int q=0;q<np;++q) for(int k=0;k<np;++k) compare(got[q*np+k],expected[q*np+k]);
    for(int c=0;c<cp;++c) for(int q=0;q<np;++q) {
        double ca=1,cb=0;
        for(int k=c*32+31;k>=c*32;--k) {
            int qi=q+k-c*32;
            if(qi<np) { cb=e[qi*np+k]+a[qi*np+k]*cb; ca=a[qi*np+k]*ca; }
        }
        compare(sum[c*np+q].x,ca); compare(sum[c*np+q].y,cb);
        compare(bound[c*np+q],expected[q*np+c*32]);
    }
    void* allocations[]{dw,dn,db,dg,ds};
    for(void* p:allocations) check(cudaFree(p));
    std::printf("%s n=%d mode=%d max_abs=%.9g\n",ok?"PASS":"FAIL",n,mode,error);
    return ok;
}
int main() {
    bool ok=true;
    for(int n:{1,17,31,64,65,129,139,257,513}) for(int mode=0;mode<5;++mode) ok=run(n,mode)&&ok;
    return ok?0:1;
}
