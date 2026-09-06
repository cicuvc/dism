// Summary -> diagonal checkpoint propagation -> parallel recomputation probe.
// Output W and row decisions are test diagnostics only, never production inputs.
#define main resource_probe_main
#include "../glx_scan/resources.cu"
#undef main
#include <vector>
#include <algorithm>

using B64=glx::MMABuffer<16,64,glx::BinaryElement,LogAffine,glx::F32x2>;
using Pair=glx::BinaryElement<float>;
constexpr int WIDTH=192;
__host__ __device__ unsigned counter(unsigned row, unsigned seed) {
    unsigned x=row^seed; x^=x>>16; x*=0x7feb352du; x^=x>>15;
    x*=0x846ca68bu; return x^(x>>16);
}
__host__ __device__ float score(int i,int j,int mode) {
    bool hard=mode==1 || mode==3 || (mode==2 && (counter(i,123)&1));
    if(j>i) return -INFINITY;
    if(hard) return mode!=3 && i%7==j%7 ? 0.5f : -INFINITY;
    return 0.25f-((i*3+j*5)%11)*0.0625f;
}
__device__ B64 make_tile(int stripe,int tile,int n,int mode) {
    glx::MMABuffer<16,64,glx::UnaryElement,glx::AddOp,glx::F32x2> scalar;
    #pragma unroll
    for(int r=0;r<2;++r) {
        #pragma unroll
        for(int c=0;c<8;++c) {
            auto p=B64::layout(r,c,0),q=B64::layout(r,c,1);
            scalar.data[r][c].value={score(stripe*16+p.first,tile*64+p.second,mode),
                                    score(stripe*16+q.first,tile*64+q.second,mode)};
        }
    }
    scalar.roll(); B64 result;
    #pragma unroll
    for(int r=0;r<2;++r) {
        #pragma unroll
        for(int c=0;c<8;++c) {
            auto x=scalar.data[r][c].value;
            result.data[r][c]={x,x};
            // Identity is not (logM,logM). Padding carries state unchanged;
            // a valid hard mismatch is (-inf,-inf), which resets the chain.
            int i=stripe*16+((r*8+(int(threadIdx.x)&31)/4-(7-c)+16)%16);
            int j=tile*64+c+(threadIdx.x&3)*8;
            if(i>=n || j>=n) { result.data[r][c].first.u0=0; result.data[r][c].second.u0=-INFINITY; }
            if(i>=n || j+32>=n) { result.data[r][c].first.u1=0; result.data[r][c].second.u1=-INFINITY; }
        }
    }
    return result;
}

__global__ void summarize(Pair* summaries,unsigned* rng,int n,int mode) {
    int stripe=blockIdx.x, lane=threadIdx.x;
    if(lane<16) rng[stripe*16+lane]=counter(stripe*16+lane,123);
    B64::VState left;
    for(int t=0;t<3;++t) {
        auto data=make_tile(stripe,t,n,mode);
        auto state=data.reduce_forward(left,{}); left=state.first;
        // For 16x64 HState, lane=4*l+g, element=e encodes bottom-row
        // column 8*g+32*e+6-l, including -1 and excluding 63.
        int j=8*(lane&3)+6-lane/4;
        auto h=state.second.init[0];
        if(j>=0) summaries[stripe*WIDTH+t*64+j]={h.first.u0,h.second.u0};
        summaries[stripe*WIDTH+t*64+j+32]={h.first.u1,h.second.u1};
        if(lane==31) summaries[stripe*WIDTH+t*64+63]={left.init[1].first.u0,left.init[1].second.u0};
    }
}

__global__ void propagate(const Pair* summary,float* boundary,int stripes) {
    // One thread per checkpoint diagonal. No scores, RNG or rescans here.
    int d=int(blockIdx.x*blockDim.x+threadIdx.x)-(stripes-1)*16;
    if(d>=WIDTH) return;
    float x=-INFINITY;
    for(int s=0;s<stripes;++s) {
        int j=d+s*16;
        if(j>=0 && j<WIDTH) {
            auto a=summary[s*WIDTH+j]; x=lse(x+a.first,a.second);
            boundary[s*WIDTH+j]=x;
        }
    }
}

__global__ void recompute(const float* boundary,float* output,unsigned* rng,int n,int mode,int stripes) {
    int stripe=blockIdx.x*(blockDim.x/32)+threadIdx.x/32, lane=threadIdx.x&31;
    if(stripe>=stripes) return;
    if(lane<16) rng[stripe*16+lane]=counter(stripe*16+lane,123);
    B64::VState left;
    for(int t=0;t<3;++t) {
        B64::HState top;
        int j=t*64+8*(lane&3)+6-lane/4;
        if(stripe>0) {
            top.init[0].first={0,0};
            top.init[0].second={j>=0?boundary[(stripe-1)*WIDTH+j]:-INFINITY,
                               boundary[(stripe-1)*WIDTH+j+32]};
        }
        auto data=make_tile(stripe,t,n,mode);
        auto state=data.inclusive_scan(left,top); left=state.first;
        data.template roll<false>();
        #pragma unroll
        for(int r=0;r<2;++r) {
            #pragma unroll
            for(int c=0;c<8;++c) {
                auto p=B64::layout(r,c,0),q=B64::layout(r,c,1);
                output[(stripe*16+p.first)*WIDTH+t*64+p.second]=data.data[r][c].second.u0;
                output[(stripe*16+q.first)*WIDTH+t*64+q.second]=data.data[r][c].second.u1;
            }
        }
    }
}

double dlse(double x,double y) {
    if(x==-INFINITY) return y; if(y==-INFINITY) return x;
    return std::max(x,y)+std::log1p(std::exp2(-std::abs(x-y)))/std::log(2.0);
}
bool close(double a,double b,double& worst) {
    if(a==b) return true;
    if(!std::isfinite(a) || !std::isfinite(b)) return false;
    double err=std::abs(a-b); worst=std::max(worst,err); return err<3e-5;
}
bool run(int n,int mode) {
    int stripes=(n+15)/16, rows=stripes*16;
    Pair* summary; float *boundary,*out; unsigned *rng0,*rng1;
    check(cudaMallocManaged(&summary,stripes*WIDTH*sizeof(Pair)));
    check(cudaMallocManaged(&boundary,stripes*WIDTH*4)); check(cudaMallocManaged(&out,rows*WIDTH*4));
    check(cudaMallocManaged(&rng0,rows*4)); check(cudaMallocManaged(&rng1,rows*4));
    summarize<<<stripes,32>>>(summary,rng0,n,mode); check(cudaGetLastError());
    propagate<<<(WIDTH+(stripes-1)*16+127)/128,128>>>(summary,boundary,stripes); check(cudaGetLastError());
    recompute<<<(stripes+3)/4,128>>>(boundary,out,rng1,n,mode,stripes); check(cudaGetLastError());
    check(cudaDeviceSynchronize());
    bool ok=true; double worst=0;
    std::vector<double> ref(rows*WIDTH,-INFINITY);
    for(int i=0;i<rows;++i) {
        ok &= rng0[i]==rng1[i] && rng0[i]==counter(i,123);
        for(int j=0;j<WIDTH;++j) {
            double prev=i&&j?ref[(i-1)*WIDTH+j-1]:-INFINITY;
            bool padding=i>=n || j>=n;
            ref[i*WIDTH+j]=padding?prev:double(score(i,j,mode))+dlse(0,prev);
            ok &= close(out[i*WIDTH+j],ref[i*WIDTH+j],worst);
            if(i%16==15) ok &= close(boundary[(i/16)*WIDTH+j],ref[i*WIDTH+j],worst);
        }
    }
    // Independently verify both components of each local stripe summary.
    for(int s=0;s<stripes;++s) for(int j=0;j<WIDTH;++j) {
        double a=0,b=-INFINITY;
        for(int r=0;r<16;++r) {
            int i=s*16+r,c=j-15+r;
            if(c<0) continue;
            if(i>=n || c>=n) continue; // affine identity
            double m=score(i,c,mode); a+=m; b=m+dlse(0,b);
        }
        auto actual=summary[s*WIDTH+j];
        ok &= close(actual.first,a,worst) && close(actual.second,b,worst);
    }
    std::fprintf(stdout,"checkpoint N=%d mode=%d stripes=%d max_log2_abs=%.8g scratch=%zu %s\n",n,mode,stripes,worst,
                 size_t(stripes*WIDTH*12),ok?"PASS":"FAIL");
    check(cudaFree(summary));check(cudaFree(boundary));check(cudaFree(out));check(cudaFree(rng0));check(cudaFree(rng1));
    return ok;
}
int main() {
    bool ok=true; for(int n:{1,17,65,139,192}) for(int mode=0;mode<4;++mode) ok &= run(n,mode);
    return ok?0:1;
}
