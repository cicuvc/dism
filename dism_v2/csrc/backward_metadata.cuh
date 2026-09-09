#pragma once
#ifndef DISM_BWD_OPT
#define DISM_BWD_OPT 0
#endif
#ifndef DISM_BWD_STAGES
#define DISM_BWD_STAGES 2
#endif

namespace dism_v2::bwd_metadata {
__device__ __forceinline__ float select(bool pred,float yes,float no) {
    float out;
    asm("{ .reg .pred p; setp.ne.u32 p,%3,0; selp.f32 %0,%1,%2,p; }"
        :"=f"(out):"f"(yes),"f"(no),"r"(int(pred)));
    return out;
}
struct Key {
    float tau2,tau,lse[2];
    long long label[2];
    __device__ __forceinline__ Key(const Args& p,int bh,int kb) {
        tau=p.tau[bh%p.heads];
        tau2=tau*LOG2E;
        #pragma unroll
        for(int r=0;r<2;++r) {
            int k=kb+8*r+(threadIdx.x%32)/4;
            lse[r]=0; label[r]=0;
            if(p.column_lse && p.hard_prob!=1.f && k<p.n)
                lse[r]=p.lse[int64_t(bh)*p.n+k];
            if(p.hard_prob!=0.f && k<p.n) label[r]=p.key_label(int64_t(bh)*p.n+k);
        }
    }
};
struct Query {
    float lse[2],normalizer[2],delta[2];
    long long label[2];
    __device__ __forceinline__ Query(const Args& p,const float* d,int bh,int qb) {
        // Two distributed logical query columns per lane; no shared staging.
        #pragma unroll
        for(int e=0;e<2;++e) {
            int q=qb+(threadIdx.x%32)+32*e;
            int64_t off=int64_t(bh)*p.n+q;
            lse[e]=0; label[e]=0; normalizer[e]=0; delta[e]=0;
            if(q<p.n) {
                normalizer[e]=p.normalizer[off]; delta[e]=d[off];
                if(!p.column_lse && p.hard_prob!=1.f)
                    lse[e]=p.lse[off];
                if(p.hard_prob!=0.f) label[e]=p.query_label(off);
            }
        }
    }
};
template<int E>
__device__ __forceinline__ float score(const Args& p,const Key& key,const Query& query,
        float dot,int r,int col,int q,int k,bool hard) {
    float lse=__shfl_sync(0xffffffff,query.lse[E],col);
    if(p.column_lse) lse=key.lse[r];
    long long label=__shfl_sync(0xffffffff,query.label[E],col);
    // Keep the historical backward association for now: the single-FFMA
    // variant crossed BF16 G rounding boundaries in two strict GEMM probes.
    float soft=(dot*p.scale-lse+key.tau)*LOG2E;
    float hs=select(key.label[r]==label,key.tau2,LOG_ZERO);
    return select(q<p.n && k<p.n && k<=q,select(hard,hs,soft),LOG_ZERO);
}
template<int E>
__device__ __forceinline__ float probability(const Args& p,const Query& query,float w,
        int col,int q,int k) {
    float norm=__shfl_sync(0xffffffff,query.normalizer[E],col);
    return select(k<p.n && q<p.n,exp2_ftz(w-norm),0.f);
}
template<int E>
__device__ __forceinline__ float2 coefficient(const Args& p,const Query& query,float w,float dot,
        int col,int q,int k) {
    float norm=__shfl_sync(0xffffffff,query.normalizer[E],col);
    float delta=__shfl_sync(0xffffffff,query.delta[E],col);
    float t;
    asm("tanh.approx.f32 %0,%1;":"=f"(t):"f"(w*0.3465735902799726547f));
    float a=fmaf(.5f,t,.5f),b=exp2_ftz(w-norm)*(dot-delta);
    bool valid=k<p.n && q<p.n;
    // Missing coordinates are affine identity; a hard break is a zero map.
    return {select(valid,select(w==LOG_ZERO,0.f,a),1.f),
            select(valid && w!=LOG_ZERO,b,0.f)};
}
} // namespace dism_v2::bwd_metadata
