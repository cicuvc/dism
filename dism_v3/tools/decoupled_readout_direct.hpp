#pragma once
// Direct count/write packing. No old boundary arrays, per-block containers or
// per-job packed outputs. Workers own disjoint ranges of the final allocation.
#include "decoupled_readout_plan.hpp"
#include <climits>

namespace decoupled {
struct ArenaSize { int64_t blocks=0,streams=0; };
struct DirectSize {
    std::array<ArenaSize,4> old{};
    int64_t blocks=0,tiles=0,streams=0,routed=0,legacy=0;
    int64_t short_streams=0,long_streams=0,long_chunks=0;
    DirectSize& operator+=(const DirectSize&b) {
        for(int i=0;i<4;++i){old[i].blocks+=b.old[i].blocks;old[i].streams+=b.old[i].streams;}
        blocks+=b.blocks;tiles+=b.tiles;streams+=b.streams;routed+=b.routed;legacy+=b.legacy;
        short_streams+=b.short_streams;long_streams+=b.long_streams;long_chunks+=b.long_chunks;return *this;
    }
};
struct ArenaView {
    int width;
    int *rows,*lengths,*heads,*resets;
    float *prefixes,*weights,*dlog;
    int64_t *offsets;
};
struct DirectView {
    std::array<ArenaView,4> old;
    int *kr,*kl,*kreset,*ktau,*qr,*ql,*qblock,*qcount;
    float *kp,*kw,*kd,*qp,*qw,*qd;
    int64_t *offsets,*qoffsets;
    int64_t *stream_ids,*chunk_ids,*local_offsets,*short_stream_ids,*long_stream_ids,*long_chunk_ids,*long_local_offsets;
};
inline int64_t chunk_end(const topk_prefill::Program&p,int64_t begin,int64_t end,int width) {
    auto stop=std::min(begin+width,end);
    for(auto e=begin+1;e<stop;++e)if(p.rows[e]>=0&&p.decay[e]==0.)return e;
    return stop;
}
// For long streams, event count alone proves that the threshold is exceeded.
// Only short streams need a reset-aware count; no boundary list is generated.
inline int64_t old_chunks(const topk_prefill::Program&p,int64_t begin,int64_t end,int width,int threshold) {
    if(threshold>=0 && (end-begin+width-1)/width>threshold)return int64_t(threshold)+1;
    int64_t count=0;
    for(auto e=begin;e<end;){e=chunk_end(p,e,end,width);++count;if(threshold>=0&&count>threshold)break;}
    return count;
}
inline DirectSize direct_count(const topk_prefill::Program&p,int width,int bk,int bq,int threshold,bool compact) {
    DirectSize size;
    for(size_t s=1;s<p.offsets.size();++s) {
        auto begin=p.offsets[s-1],end=p.offsets[s];auto chunks=old_chunks(p,begin,end,width,threshold);
        if(threshold<0||chunks<=threshold) {
            int bucket=compact&&chunks==1?topk_prefill::compact_bucket(int(end-begin)):3;
            size.old[bucket].blocks+=chunks;++size.old[bucket].streams;size.legacy+=end-begin;
            if(!compact){if(chunks>64){++size.long_streams;size.long_chunks+=chunks;}else ++size.short_streams;}
            continue;
        }
        size.routed+=end-begin;bool active=false;int keys=0,queries=0;
        for(auto e=begin;e<end;++e) {
            if(p.rows[e]>=0) {
                if(p.decay[e]==0.)active=false;
                if(!active){++size.streams;keys=0;active=true;}
                if(keys==0||keys==bk){++size.blocks;keys=0;queries=0;}
                ++keys;
            }else if(active){if(queries%bq==0)++size.tiles;++queries;}
        }
    }
    return size;
}
inline int direct_row(int row,const topk_prefill::TrainingJob&job,int n,int h,int topk) {
    int64_t base=int64_t(job.batch)*n+job.start;
    if(row>=0)return int((base+row)*h+job.head);
    int id=-row-1;return -int(((base+id/topk)*h+job.head)*topk+id%topk)-1;
}
inline void direct_pack(const topk_prefill::Program&p,const topk_prefill::TrainingJob&job,
                        int n,int h,int width,int bk,int bq,int threshold,bool compact,
                        const DirectSize&base,const DirectView&out) {
    auto cursor=base;bool gradients=!p.dlogweight.empty();
    auto mapped=[&](int row){return direct_row(row,job,n,h,p.topk);};
    for(size_t s=1;s<p.offsets.size();++s) {
        auto begin=p.offsets[s-1],end=p.offsets[s];auto chunks=old_chunks(p,begin,end,width,threshold);
        if(threshold<0||chunks<=threshold) {
            int bucket=compact&&chunks==1?topk_prefill::compact_bucket(int(end-begin)):3;
            auto&a=out.old[bucket];auto&at=cursor.old[bucket];
            a.heads[at.streams]=job.head;
            if(!compact) {
                out.stream_ids[at.streams]=at.streams;out.local_offsets[at.streams]=at.blocks;
                if(chunks>64){out.long_stream_ids[cursor.long_streams]=at.streams;out.long_local_offsets[cursor.long_streams++]=cursor.long_chunks;}
                else out.short_stream_ids[cursor.short_streams++]=at.streams;
            }
            for(auto lo=begin;lo<end;) {
                auto hi=chunk_end(p,lo,end,width);int length=int(hi-lo);
                bool reset=p.rows[lo]>=0&&p.decay[lo]==0.;double prefix=0.;
                int64_t slot=at.blocks*a.width;a.lengths[at.blocks]=length;a.resets[at.blocks]=reset;
                if(!compact){out.chunk_ids[at.blocks]=at.blocks;if(chunks>64)out.long_chunk_ids[cursor.long_chunks++]=at.blocks;}
                for(int t=0;t<length;++t) {
                    auto e=lo+t;if(p.rows[e]>=0&&!(reset&&t==0)&&p.decay[e]!=1.)prefix+=std::log(p.decay[e]);
                    a.rows[slot+t]=mapped(p.rows[e]);a.prefixes[slot+t]=float(prefix);
                    a.weights[slot+t]=float(p.weight[e]);a.dlog[slot+t]=gradients?float(p.dlogweight[e]):0.f;
                }
                for(int t=length;t<a.width;++t){a.rows[slot+t]=mapped(0);a.prefixes[slot+t]=0.f;a.weights[slot+t]=0.f;a.dlog[slot+t]=0.f;}
                ++at.blocks;lo=hi;
            }
            a.offsets[++at.streams]=at.blocks;
            continue;
        }
        bool active=false,block_open=false;int keys=0,queries=0;double prefix=0.;
        auto close_block=[&]() {
            if(!block_open)return;
            int64_t block=cursor.blocks,slot=block*bk;
            out.kl[block]=keys;
            for(int t=keys;t<bk;++t){out.kr[slot+t]=0;out.kp[slot+t]=0.f;out.kw[slot+t]=0.f;out.kd[slot+t]=0.f;}
            if(queries) {
                int used=(queries-1)%bq+1;out.ql[cursor.tiles-1]=used;
                auto qslot=(cursor.tiles-1)*bq;
                for(int t=used;t<bq;++t){out.qr[qslot+t]=0;out.qcount[qslot+t]=0;out.qp[qslot+t]=0.f;out.qw[qslot+t]=0.f;out.qd[qslot+t]=0.f;}
            }
            out.qoffsets[++cursor.blocks]=cursor.tiles;block_open=false;
        };
        for(auto e=begin;e<end;++e) {
            if(p.rows[e]>=0) {
                if(p.decay[e]==0.&&active){close_block();out.offsets[++cursor.streams]=cursor.blocks;active=false;}
                bool first=!active;
                if(!block_open||keys==bk) {
                    close_block();block_open=true;keys=queries=0;prefix=0.;
                    out.kreset[cursor.blocks]=first;out.ktau[cursor.blocks]=job.head;
                }
                if(!first&&p.decay[e]!=1.)prefix+=std::log(p.decay[e]);
                auto slot=cursor.blocks*bk+keys++;
                out.kr[slot]=mapped(p.rows[e]);out.kp[slot]=float(prefix);
                out.kw[slot]=float(p.weight[e]);out.kd[slot]=gradients?float(p.dlogweight[e]):0.f;active=true;
            }else if(active) {
                if(queries%bq==0) {
                    if(queries)out.ql[cursor.tiles-1]=bq;
                    out.qblock[cursor.tiles]=int(cursor.blocks);++cursor.tiles;
                }
                auto slot=(cursor.tiles-1)*bq+queries%bq;++queries;
                out.qr[slot]=-mapped(p.rows[e])-1;out.qcount[slot]=keys;out.qp[slot]=float(prefix);
                out.qw[slot]=float(p.weight[e]);out.qd[slot]=gradients?float(p.dlogweight[e]):0.f;
            }
        }
        if(active){close_block();out.offsets[++cursor.streams]=cursor.blocks;}
    }
}

// One aligned, uninitialized arena. Allocation and Python views are handled by
// the binding. The same layout function sizes and populates the typed pointers.
struct DirectLayout {
    size_t bytes=0;unsigned char*data=nullptr;
    template<class T>T*take(int64_t count) {
        if(count<0||uint64_t(count)>SIZE_MAX/sizeof(T))throw std::overflow_error("metadata size overflow");
        if(bytes>SIZE_MAX-7)throw std::overflow_error("arena alignment overflow");
        bytes=(bytes+7)/8*8;size_t start=bytes,extra=size_t(count)*sizeof(T);
        if(bytes>SIZE_MAX-extra)throw std::overflow_error("arena size overflow");
        bytes+=extra;return data?reinterpret_cast<T*>(data+start):nullptr;
    }
    DirectView layout(const DirectSize&s,int width,int bk,int bq) {
        if(s.blocks>INT_MAX/bk||s.tiles>INT_MAX/bq)throw std::overflow_error("tiled metadata exceeds int32 indexing");
        DirectView v;
        for(int i=0;i<4;++i) {
            auto&a=v.old[i];auto&z=s.old[i];a.width=i==3?width:(16<<i);
            if(z.blocks>INT_MAX/a.width)throw std::overflow_error("legacy metadata exceeds int32 indexing");
            a.rows=take<int>(z.blocks*a.width);a.prefixes=take<float>(z.blocks*a.width);
            a.weights=take<float>(z.blocks*a.width);a.dlog=take<float>(z.blocks*a.width);
            a.lengths=take<int>(z.blocks);a.heads=take<int>(z.streams);a.resets=take<int>(z.blocks);a.offsets=take<int64_t>(z.streams+1);
        }
        v.kr=take<int>(s.blocks*bk);v.kp=take<float>(s.blocks*bk);v.kw=take<float>(s.blocks*bk);v.kd=take<float>(s.blocks*bk);
        v.kl=take<int>(s.blocks);v.kreset=take<int>(s.blocks);v.ktau=take<int>(s.blocks);
        v.qr=take<int>(s.tiles*bq);v.qp=take<float>(s.tiles*bq);v.qw=take<float>(s.tiles*bq);v.qd=take<float>(s.tiles*bq);v.qcount=take<int>(s.tiles*bq);
        v.ql=take<int>(s.tiles);v.qblock=take<int>(s.tiles);v.offsets=take<int64_t>(s.streams+1);v.qoffsets=take<int64_t>(s.blocks+1);
        auto routed_streams=s.short_streams+s.long_streams;
        v.stream_ids=take<int64_t>(routed_streams);v.local_offsets=take<int64_t>(routed_streams);
        v.chunk_ids=take<int64_t>(routed_streams?s.old[3].blocks:0);
        v.short_stream_ids=take<int64_t>(s.short_streams);v.long_stream_ids=take<int64_t>(s.long_streams);
        v.long_chunk_ids=take<int64_t>(s.long_chunks);v.long_local_offsets=take<int64_t>(s.long_streams);
        return v;
    }
};
} // namespace decoupled
