#pragma once
// Experimental program rewrite, applied AFTER per-query normalization.
// Replay a bounded key prefix in independent query tiles. This duplicates K
// events but eliminates carried matrix states for sparse long streams.
#include <utility>
#include <vector>
#include <stdexcept>

template<class Program>
Program split_sparse_reads(Program p, int width, int max_keys) {
    if (max_keys == 0) return p;
    if (max_keys < 0 || max_keys > 4 || width <= max_keys)
        throw std::invalid_argument("sparse readout requires 1..4 keys and width > keys");
    std::vector<unsigned char> eligible(p.offsets.size()-1,0);
    bool any=false;
    for (size_t s=1;s<p.offsets.size();++s) {
        auto begin=p.offsets[s-1],end=p.offsets[s];
        if (end-begin<=width) continue;
        int keys=0; bool reset=false;
        for (auto e=begin;e<end;++e) if (p.rows[e]>=0) {
            if (++keys>max_keys) break;
            reset |= e>begin && p.decay[e]==0.;
        }
        eligible[s-1]=keys>0 && keys<=max_keys && !reset
            && p.rows[begin]>=0 && p.decay[begin]==0.;
        any |= eligible[s-1];
    }
    if (!any) return p;
    // Keep scalar normalizers/derivatives and all non-event fields untouched.
    auto offsets=std::move(p.offsets); auto rows=std::move(p.rows);
    auto decay=std::move(p.decay); auto weight=std::move(p.weight);
    auto dlog=std::move(p.dlogweight);
    p.offsets={0};p.rows.clear();p.decay.clear();p.weight.clear();p.dlogweight.clear();
    size_t reserve=rows.size()+rows.size()/width*max_keys;
    p.rows.reserve(reserve);p.decay.reserve(reserve);p.weight.reserve(reserve);
    if (!dlog.empty()) p.dlogweight.reserve(reserve);
    p.offsets.reserve(offsets.size()+rows.size()/width);
    auto append=[&](int64_t e) {
        p.rows.push_back(rows[e]);p.decay.push_back(decay[e]);p.weight.push_back(weight[e]);
        if (!dlog.empty()) p.dlogweight.push_back(dlog[e]);
    };
    for (size_t s=1;s<offsets.size();++s) {
        auto begin=offsets[s-1],end=offsets[s];
        if (!eligible[s-1]) {
            for(auto e=begin;e<end;++e) append(e);
            p.offsets.push_back(p.rows.size());continue;
        }
        std::vector<int64_t> prefix;prefix.reserve(max_keys);
        auto cursor=begin;
        while(cursor<end) {
            int count=0;
            for(auto e:prefix) {append(e);++count;}
            while(cursor<end && count<width) {
                if(rows[cursor]>=0) prefix.push_back(cursor);
                append(cursor++);++count;
            }
            p.offsets.push_back(p.rows.size());
        }
    }
    return p;
}
