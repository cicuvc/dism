#include "decoupled_readout_plan.hpp"
#ifdef DISM_PRODUCTION_TOPK
#include "../python/flash_dism/inference/csrc/decoupled_readout_direct.hpp"
#else
#include "decoupled_readout_direct.hpp"
#endif
#include <cassert>
#include <cstring>
#include <iostream>
#include <random>
using Matrix=std::vector<double>;
template<class T>void identical(const std::vector<T>&a,const T*b) {
    if(!a.empty())assert(std::memcmp(a.data(),b,a.size()*sizeof(T))==0);
}
void direct_check(const topk_prefill::Program&p,int bk,int bq,int width,int threshold,bool compact) {
    auto chunks=topk_prefill::chunk_program(p,width);
    topk_prefill::TrainingJob job{1,2,256,256+p.n,0.}; // nonzero B/H/document offsets
    int physical_n=p.n+512;
    decoupled::Plan ref(width,bk,bq,p.topk);
    for(size_t s=1;s<p.offsets.size();++s) {
        if(threshold>=0&&chunks.offsets[s]-chunks.offsets[s-1]>threshold)ref.route(p,job,physical_n,3,1,p.offsets[s-1],p.offsets[s]);
        else ref.legacy(p,job,physical_n,3,1,p.offsets[s-1],p.offsets[s],compact);
    }
    auto size=decoupled::direct_count(p,width,bk,bq,threshold,compact);decoupled::DirectLayout layout;
    layout.layout(size,width,bk,bq);auto allocation=std::make_unique<unsigned char[]>(layout.bytes);
    std::memset(allocation.get(),0xa5,layout.bytes); // catch missing initialization, including padding slots
    layout.data=allocation.get();layout.bytes=0;auto view=layout.layout(size,width,bk,bq);
    view.offsets[0]=view.qoffsets[0]=0;for(auto&a:view.old)a.offsets[0]=0;
    decoupled::direct_pack(p,job,physical_n,3,width,bk,bq,threshold,compact,{},view);
    assert(size.blocks==int64_t(ref.kl.size())&&size.tiles==int64_t(ref.ql.size())&&size.streams+1==int64_t(ref.offsets.size()));
#define SAME(name) identical(ref.name,view.name)
    SAME(kr);SAME(kp);SAME(kw);SAME(kd);SAME(kl);SAME(kreset);SAME(ktau);
    SAME(qr);SAME(qp);SAME(qw);SAME(qd);SAME(qcount);SAME(ql);SAME(qblock);SAME(offsets);SAME(qoffsets);
#undef SAME
    for(int i=0;i<4;++i) {
        auto&a=i==3?ref.multi:ref.buckets[i];auto&v=view.old[i];
        assert(size.old[i].blocks==int64_t(a.lengths.size())&&size.old[i].streams==int64_t(a.heads.size()));
#define SAME(name) identical(a.name,v.name)
        SAME(rows);SAME(prefixes);SAME(weights);SAME(dlog);SAME(lengths);SAME(heads);SAME(resets);SAME(offsets);
#undef SAME
    }
}
std::pair<Matrix,Matrix> reference(const topk_prefill::Program&p) {
    Matrix out(size_t(p.n)*p.topk*p.n,0),der(out.size(),0);
    for(size_t s=1;s<p.offsets.size();++s) {
        Matrix state(p.n,0),tangent(p.n,0);
        for(auto e=p.offsets[s-1];e<p.offsets[s];++e) {
            int row=p.rows[e];double w=p.weight[e],dw=p.dlogweight[e];
            if(row>=0){for(int j=0;j<p.n;++j){state[j]*=p.decay[e];tangent[j]*=p.decay[e];}state[row]+=w;tangent[row]+=w*dw;}
            else{int i=-row-1;for(int j=0;j<p.n;++j){out[size_t(i)*p.n+j]+=w*state[j];der[size_t(i)*p.n+j]+=w*(dw*state[j]+tangent[j]);}}
        }
    }return {out,der};
}
std::pair<Matrix,Matrix> evaluate(const decoupled::Plan&p,int n) {
    Matrix out(size_t(n)*p.topk*n,0),der(out.size(),0);
    for(size_t s=1;s<p.offsets.size();++s) {
        Matrix state(n,0),tangent(n,0);
        for(auto block=p.offsets[s-1];block<p.offsets[s];++block) {
            for(auto tile=p.qoffsets[block];tile<p.qoffsets[block+1];++tile)for(int q=0;q<p.ql[tile];++q) {
                auto at=tile*p.bq+q;int i=p.qr[at];double scale=p.kreset[block]?0:p.qw[at]*std::exp(p.qp[at]);
                for(int j=0;j<n;++j){out[size_t(i)*n+j]+=scale*state[j];der[size_t(i)*n+j]+=scale*(p.qd[at]*state[j]+tangent[j]);}
                for(int k=0;k<p.qcount[at];++k){auto ka=block*p.bk+k;int j=p.kr[ka];double weight=double(p.qw[at])*p.kw[ka]*std::exp(double(p.qp[at])-p.kp[ka]);out[size_t(i)*n+j]+=weight;der[size_t(i)*n+j]+=weight*(double(p.qd[at])+p.kd[ka]);}
            }
            double last=p.kp[block*p.bk+p.kl[block]-1],alpha=p.kreset[block]?0:std::exp(last);
            for(int j=0;j<n;++j){state[j]*=alpha;tangent[j]*=alpha;}
            for(int k=0;k<p.kl[block];++k){auto ka=block*p.bk+k;double w=p.kw[ka]*std::exp(last-p.kp[ka]);state[p.kr[ka]]+=w;tangent[p.kr[ka]]+=w*p.kd[ka];}
        }
    }return {out,der};
}
int main() {
    for(int which=0;which<3;++which) {
        decoupled::DirectSize huge;
        if(which==0)huge.blocks=INT_MAX/64+1;
        if(which==1)huge.tiles=INT_MAX/64+1;
        if(which==2)huge.old[3].blocks=INT_MAX/64+1;
        bool rejected=false;try{decoupled::DirectLayout layout;layout.layout(huge,64,64,64);}catch(const std::overflow_error&){rejected=true;}
        assert(rejected);
    }
    std::mt19937 rng(821);int count=0;double max_weight=0,max_derivative=0;
    topk_prefill::Builder builder;builder.cache_forward_weights=true;builder.lazy_gradient_weights=true;
    for(int n:{1,17,65,129})for(int K:{1,2,4})for(int pattern=0;pattern<3;++pattern)for(double tau:{-.2,0.,.5}) {
        std::vector<int>q(n),k(n),reset(n,0),cand(n*K);
        for(int i=0;i<n;++i){q[i]=rng()%7;k[i]=rng()%7;if(pattern==1)q[i]=k[i]=0;if(pattern==2){q[i]=0;k[i]=i+1;}for(int c=0;c<K;++c)cand[i*K+c]=(q[i]+c)%7;}
        if(pattern==2)k[0]=0;if(pattern==0&&n>17)reset[n/2]=1;
        builder.build(q.data(),k.data(),reset.data(),n,tau,false,true,cand.data(),K);auto p=builder.finish();auto ref=reference(p);
        for(int bk:{16,32,64})for(int bq:{16,32,64}) {
            for(int threshold:{-1,0,1,64}) {
                direct_check(p,bk,bq,16,threshold,false);
                direct_check(p,bk,bq,64,threshold,true);
            }
            decoupled::Plan plan(64,bk,bq,K);topk_prefill::TrainingJob job{0,0,0,n,tau};
            for(size_t s=1;s<p.offsets.size();++s)plan.route(p,job,n,1,1,p.offsets[s-1],p.offsets[s]);
            auto actual=evaluate(plan,n);
            for(size_t i=0;i<ref.first.size();++i) {
                double e=std::abs(ref.first[i]-actual.first[i]),d=std::abs(ref.second[i]-actual.second[i]);
                max_weight=std::max(max_weight,e);max_derivative=std::max(max_derivative,d);
                // This checks the intentionally FP32 GPU metadata against FP64
                // event coefficients, not a bitwise-equivalent decomposition.
                assert(e<=2e-6*(1+std::abs(ref.first[i])));
                assert(d<=2e-5*(1+std::abs(ref.second[i])));
            }++count;
        }
    }
    std::cout<<"PASS "<<count<<" coefficient/derivative cases; max errors "<<max_weight<<" "<<max_derivative<<"\n";
}
