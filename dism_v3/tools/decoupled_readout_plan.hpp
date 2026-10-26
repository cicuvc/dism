#pragma once
// Isolated execution-format experiment. No normalization or matching changes.
#include "topk_prefill_training.hpp"
#include <array>

namespace decoupled {
struct Arena {
    int width;
    std::vector<int> rows,lengths,heads,resets;
    std::vector<int64_t> offsets{0};
    std::vector<float> prefixes,weights,dlog;
    explicit Arena(int c):width(c){}
    template<class T>static void extend(std::vector<T>&a,const std::vector<T>&b){a.insert(a.end(),b.begin(),b.end());}
    void append(Arena&b) {
        int64_t base=lengths.size();for(size_t i=1;i<b.offsets.size();++i)offsets.push_back(base+b.offsets[i]);
        extend(rows,b.rows);extend(lengths,b.lengths);extend(heads,b.heads);extend(resets,b.resets);
        extend(prefixes,b.prefixes);extend(weights,b.weights);extend(dlog,b.dlog);
    }
};
struct Plan {
    std::array<Arena,3> buckets{Arena(16),Arena(32),Arena(64)};
    Arena multi;
    int bk,bq,topk;
    std::vector<int> kr,kl,kreset,ktau,qr,ql,qblock,qcount;
    std::vector<float> kp,kw,kd,qp,qw,qd;
    std::vector<int64_t> offsets{0},qoffsets{0};
    int64_t routed_events=0,legacy_events=0;
    Plan(int c,int k,int q,int topk):multi(c),bk(k),bq(q),topk(topk){}
    void append(Plan&b) {
        int64_t block_base=kl.size(),tile_base=ql.size();
        for(size_t i=1;i<b.offsets.size();++i)offsets.push_back(block_base+b.offsets[i]);
        for(size_t i=1;i<b.qoffsets.size();++i)qoffsets.push_back(tile_base+b.qoffsets[i]);
        for(auto &id:b.qblock)id+=int(block_base);
#define EXTEND(x) Arena::extend(x,b.x)
        EXTEND(kr);EXTEND(kl);EXTEND(kreset);EXTEND(ktau);EXTEND(kp);EXTEND(kw);EXTEND(kd);
        EXTEND(qr);EXTEND(ql);EXTEND(qblock);EXTEND(qcount);EXTEND(qp);EXTEND(qw);EXTEND(qd);
#undef EXTEND
        multi.append(b.multi);for(int i=0;i<3;++i)buckets[i].append(b.buckets[i]);
        legacy_events+=b.legacy_events;routed_events+=b.routed_events;
    }
    void reserve(const std::vector<std::unique_ptr<Plan>>&parts) {
#define RESERVE(x) {size_t total=0;for(auto&p:parts)total+=p->x.size();x.reserve(total);}
        RESERVE(kr);RESERVE(kl);RESERVE(kreset);RESERVE(ktau);RESERVE(kp);RESERVE(kw);RESERVE(kd);
        RESERVE(qr);RESERVE(ql);RESERVE(qblock);RESERVE(qcount);RESERVE(qp);RESERVE(qw);RESERVE(qd);RESERVE(offsets);RESERVE(qoffsets);
        RESERVE(multi.rows);RESERVE(multi.lengths);RESERVE(multi.heads);RESERVE(multi.resets);RESERVE(multi.offsets);RESERVE(multi.prefixes);RESERVE(multi.weights);RESERVE(multi.dlog);
        for(int i=0;i<3;++i){RESERVE(buckets[i].rows);RESERVE(buckets[i].lengths);RESERVE(buckets[i].heads);RESERVE(buckets[i].resets);RESERVE(buckets[i].offsets);RESERVE(buckets[i].prefixes);RESERVE(buckets[i].weights);RESERVE(buckets[i].dlog);}
#undef RESERVE
    }

    int mapped(int row,const topk_prefill::TrainingJob&j,int n,int h) const {
        int64_t base=int64_t(j.batch)*n+j.start;
        if(row>=0)return int((base+row)*h+j.head);
        int id=-row-1;
        return -int(((base+id/topk)*h+j.head)*topk+id%topk)-1;
    }
    void legacy(const topk_prefill::Program&p,const topk_prefill::TrainingJob&j,
                int n,int h,int tau_rows,int64_t begin,int64_t end,bool compact) {
        std::vector<std::pair<int64_t,int64_t>> ranges;
        for(auto lo=begin;lo<end;) {
            auto hi=std::min(lo+multi.width,end);
            for(auto e=lo+1;e<hi;++e)if(p.rows[e]>=0&&p.decay[e]==0.){hi=e;break;}
            ranges.push_back({lo,hi});lo=hi;
        }
        bool single=compact&&ranges.size()==1;
        Arena&a=single?buckets[topk_prefill::compact_bucket(int(end-begin))]:multi;
        int head=(tau_rows==1?0:j.batch)*h+j.head;
        a.heads.push_back(head);
        for(auto [lo,hi]:ranges) {
            int length=int(hi-lo);bool reset=p.rows[lo]>=0&&p.decay[lo]==0.;
            a.lengths.push_back(length);a.resets.push_back(reset);
            double prefix=0.;
            for(int t=0;t<a.width;++t) {
                if(t<length) {
                    auto e=lo+t;if(p.rows[e]>=0&&!(reset&&t==0))prefix+=std::log(p.decay[e]);
                    a.rows.push_back(mapped(p.rows[e],j,n,h));a.prefixes.push_back(float(prefix));
                    a.weights.push_back(float(p.weight[e]));a.dlog.push_back(p.dlogweight.empty()?0.f:float(p.dlogweight[e]));
                } else {a.rows.push_back(mapped(0,j,n,h));a.prefixes.push_back(0.f);a.weights.push_back(0.f);a.dlog.push_back(0.f);}
            }
        }
        a.offsets.push_back(a.lengths.size());legacy_events+=end-begin;
    }

    // Each key block owns ONE matrix prefix. Queries reference it and retain
    // the number of causal keys inside the block. Many query tiles may share it.
    void segment(const topk_prefill::Program&p,const topk_prefill::TrainingJob&j,
                 int n,int h,int tau_rows,int64_t begin,int64_t end) {
        std::vector<int64_t> keys;
        std::vector<std::vector<std::pair<int64_t,int>>> queries;
        for(auto e=begin;e<end;++e) {
            if(p.rows[e]>=0){keys.push_back(e);if((keys.size()-1)%bk==0)queries.emplace_back();}
            else if(!keys.empty())queries.back().push_back({e,int((keys.size()-1)%bk)+1});
        }
        if(keys.empty())return;
        int head=(tau_rows==1?0:j.batch)*h+j.head;
        for(size_t first=0;first<keys.size();first+=bk) {
            int block=int(kl.size()),length=int(std::min(size_t(bk),keys.size()-first));
            kl.push_back(length);kreset.push_back(first==0);ktau.push_back(head);
            double prefix=0.;std::vector<float> local_prefix(length);
            for(int t=0;t<bk;++t) {
                if(t<length) {
                    auto e=keys[first+t];if(first+t>0)prefix+=std::log(p.decay[e]);
                    local_prefix[t]=float(prefix);kr.push_back(mapped(p.rows[e],j,n,h));
                    kp.push_back(float(prefix));kw.push_back(float(p.weight[e]));kd.push_back(p.dlogweight.empty()?0.f:float(p.dlogweight[e]));
                }else{kr.push_back(0);kp.push_back(0.f);kw.push_back(0.f);kd.push_back(0.f);}
            }
            const auto&reads=queries[first/bk];
            for(size_t qfirst=0;qfirst<reads.size();qfirst+=bq) {
                int count=int(std::min(size_t(bq),reads.size()-qfirst));ql.push_back(count);qblock.push_back(block);
                for(int t=0;t<bq;++t) {
                    if(t<count) {
                        auto [e,seen]=reads[qfirst+t];qr.push_back(-mapped(p.rows[e],j,n,h)-1);
                        qp.push_back(local_prefix[seen-1]);qw.push_back(float(p.weight[e]));
                        qd.push_back(p.dlogweight.empty()?0.f:float(p.dlogweight[e]));qcount.push_back(seen);
                    }else{qr.push_back(0);qp.push_back(0.f);qw.push_back(0.f);qd.push_back(0.f);qcount.push_back(0);}
                }
            }
            qoffsets.push_back(ql.size());
        }
        offsets.push_back(kl.size());
    }
    void route(const topk_prefill::Program&p,const topk_prefill::TrainingJob&j,
               int n,int h,int tau_rows,int64_t begin,int64_t end) {
        routed_events+=end-begin;
        for(auto lo=begin;lo<end;) {
            auto hi=lo+1;
            while(hi<end&&!(p.rows[hi]>=0&&p.decay[hi]==0.))++hi;
            segment(p,j,n,h,tau_rows,lo,hi);lo=hi;
        }
    }
};
} // namespace decoupled
