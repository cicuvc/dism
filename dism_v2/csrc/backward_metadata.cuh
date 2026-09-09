#pragma once
#ifndef DISM_BWD_OPT
#define DISM_BWD_OPT 0
#endif
#ifndef DISM_BWD_STAGES
#define DISM_BWD_STAGES 2
#endif

namespace dism_v2::bwd_metadata {
#if DISM_BWD_OPT >= 13 && defined(DISM_FINITE_SENTINEL) && DISM_FINITE_SENTINEL
inline constexpr bool finite_zero = true;
#else
inline constexpr bool finite_zero = false;
#endif
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
// Ten variants: two LSE directions x (soft,hard32,hard64,mixed32,mixed64).
template<int S> struct Policy {
    static constexpr bool column=S>=5;
    static constexpr int variant=S%5;
    static constexpr int mode=variant==0?0:(variant<=2?1:2);
    using Label=std::conditional_t<variant==0 || variant==1 || variant==3,int,long long>;
};
template<int S> struct SpecialKey {
    using P=Policy<S>; using Label=typename P::Label;
    float tau2,tau;
#if DISM_BWD_OPT >= 12
    float scale2,bias2[2];
#else
    float lse[2];
#endif
    Label label[2];
    __device__ __forceinline__ SpecialKey(const Args& p,int bh,int kb) {
        tau=p.tau[bh%p.heads]; tau2=tau*LOG2E;
#if DISM_BWD_OPT >= 12
        scale2=p.scale*LOG2E;
#endif
        #pragma unroll
        for(int r=0;r<2;++r) {
            int k=kb+8*r+(threadIdx.x%32)/4;
            label[r]=0;
#if DISM_BWD_OPT >= 12
            bias2[r]=0;
            if constexpr(P::column && P::mode!=1)
                if(k<p.n) bias2[r]=(tau-p.lse[int64_t(bh)*p.n+k])*LOG2E;
#else
            lse[r]=0;
            if constexpr(P::column && P::mode!=1)
                if(k<p.n) lse[r]=p.lse[int64_t(bh)*p.n+k];
#endif
            if constexpr(P::mode!=0)
                if(k<p.n) label[r]=reinterpret_cast<const Label*>(p.k_label)[int64_t(bh)*p.n+k];
        }
    }
};
template<int S> struct SpecialQuery {
    using P=Policy<S>; using Label=typename P::Label;
#if DISM_BWD_OPT >= 12
    float bias2[2];
#else
    float lse[2];
#endif
    float normalizer[2],delta[2]; Label label[2];
    __device__ __forceinline__ SpecialQuery(const Args& p,const float* d,int bh,int qb,float tau=0.f) {
        #pragma unroll
        for(int e=0;e<2;++e) {
            int q=qb+(threadIdx.x%32)+32*e;
            int64_t off=int64_t(bh)*p.n+q;
            // Invalid query probabilities become zero without a per-element mask.
            // +INF is a normalization mask, not an affine sentinel/identity.
            label[e]=0; normalizer[e]=finite_zero?INFINITY:0.f; delta[e]=0;
#if DISM_BWD_OPT >= 12
            bias2[e]=0;
#else
            lse[e]=0;
#endif
            if(q<p.n) {
                normalizer[e]=p.normalizer[off]; delta[e]=d[off];
#if DISM_BWD_OPT >= 12
                if constexpr(!P::column && P::mode!=1) bias2[e]=(tau-p.lse[off])*LOG2E;
#else
                if constexpr(!P::column && P::mode!=1) lse[e]=p.lse[off];
#endif
                if constexpr(P::mode!=0) label[e]=reinterpret_cast<const Label*>(p.q_label)[off];
            }
        }
    }
};
template<int S> using KeyFor=std::conditional_t<(S<0),Key,SpecialKey<S>>;
template<int S> using QueryFor=std::conditional_t<(S<0),Query,SpecialQuery<S>>;
template<int E,int S>
__device__ __forceinline__ float score(const Args& p,const SpecialKey<S>& key,
        const SpecialQuery<S>& query,float dot,int r,int col,int q,int k,bool hard) {
    using P=Policy<S>;
    float result;
    if constexpr(P::mode!=1) {
#if DISM_BWD_OPT >= 12
        float bias;
        if constexpr(P::column) bias=key.bias2[r];
        else bias=__shfl_sync(0xffffffff,query.bias2[E],col);
        result=fmaf(dot,key.scale2,bias);
#else
        float lse;
        if constexpr(P::column) lse=key.lse[r];
        else lse=__shfl_sync(0xffffffff,query.lse[E],col);
        result=(dot*p.scale-lse+key.tau)*LOG2E;
#endif
    }
    if constexpr(P::mode!=0) {
        auto label=__shfl_sync(0xffffffff,query.label[E],col);
        float hs=select(key.label[r]==label,key.tau2,LOG_ZERO);
        if constexpr(P::mode==1) result=hs;
        else result=select(hard,hs,result);
    }
    if constexpr(finite_zero) return select(q<p.n && k<=q,result,LOG_ZERO);
    else return select(q<p.n && k<p.n && k<=q,result,LOG_ZERO);
}
template<typename F> void dispatch(const Args& p,F&& f) {
    auto direction=[&]<int BASE>() {
        if(p.hard_prob==0.f) f.template operator()<BASE>();
        else if(p.hard_prob==1.f) {
            if(p.label32) f.template operator()<BASE+1>();
            else f.template operator()<BASE+2>();
        } else {
            if(p.label32) f.template operator()<BASE+3>();
            else f.template operator()<BASE+4>();
        }
    };
    if(p.column_lse) direction.template operator()<5>();
    else direction.template operator()<0>();
}
template<int E,typename Q>
__device__ __forceinline__ float probability(const Args& p,const Q& query,float w,
        int col,int q,int k) {
    float norm=__shfl_sync(0xffffffff,query.normalizer[E],col);
    // Invalid q: norm=+INF. Invalid k with valid q: k>q, hence W is
    // unreachable on the entire causal diagonal and EX2 underflows to zero.
    if constexpr(finite_zero) return exp2_ftz(w-norm);
    else return select(k<p.n && q<p.n,exp2_ftz(w-norm),0.f);
}
template<int E,typename Q>
__device__ __forceinline__ float2 coefficient(const Args& p,const Q& query,float w,float dot,
        int col,int q,int k) {
    float norm=__shfl_sync(0xffffffff,query.normalizer[E],col);
    float delta=__shfl_sync(0xffffffff,query.delta[E],col);
    float t;
    asm("tanh.approx.f32 %0,%1;":"=f"(t):"f"(w*0.3465735902799726547f));
    float a=fmaf(.5f,t,.5f),b=exp2_ftz(w-norm)*(dot-delta);
    bool valid=k<p.n && q<p.n;
    // Missing coordinates are affine identity; a hard break is a zero map.
    if constexpr(finite_zero) {
        // At finite LOG_ZERO TANH saturates to -1 and EX2 flushes to zero.
        // Padding beta is zero by the same normalization/causality invariant
        // as probability(), but alpha MUST remain1 to preserve scan identity.
        return {select(valid,a,1.f),b};
    } else {
        return {select(valid,select(w==LOG_ZERO,0.f,a),1.f),
                select(valid && w!=LOG_ZERO,b,0.f)};
    }
}
} // namespace dism_v2::bwd_metadata
