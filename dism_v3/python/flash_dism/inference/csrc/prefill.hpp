#pragma once

// Offline control plane. No Torch/CUDA dependencies and no vector payloads.
#include "planner.hpp"
#include "prefill_matcher.hpp"
#include <deque>
#include <functional>
#include <limits>
#include <numeric>
#include <queue>

namespace dism_prefill {
using Ids = std::vector<int>;
constexpr double neg_inf = -std::numeric_limits<double>::infinity();

inline double logadd(double a, double b) {
    if (a == neg_inf) return b;
    if (b == neg_inf) return a;
    return std::max(a, b) + std::log1p(std::exp(-std::abs(a-b)));
}

// GPU-facing program: each stream starts with M=0.
// row>=0: M = decay*M + weight*outer(sk[row], v[row]).
// row<0:  O[-row-1] += weight * sq[-row-1]^T M.
// Independent streams overlap output rows; GPU execution needs FP32 reduction.
struct Program {
    int n = 0, states = 0;
    std::vector<int64_t> offsets{0};
    Ids rows;
    std::vector<double> decay, weight, logden;
    // Populated only by the explicit training planner; inference is unchanged.
    std::vector<double> dlogweight, dlogden;
};

struct ChunkProgram {
    std::vector<int64_t> offsets{0};
    Ids rows, lengths, resets;
    std::vector<float> prefixes, weights;
    std::vector<float> dlog;
};

inline ChunkProgram chunk_program(const Program &p, int width) {
    if (width!=16 && width!=32 && width!=64)
        throw std::invalid_argument("chunk width must be 16, 32 or 64");
    ChunkProgram out;
    out.offsets.reserve(p.offsets.size());
    // Discover boundaries first; allocate the four padded slot arrays exactly
    // once instead of repeatedly growing/copying them for every chunk.
    size_t estimate=p.rows.size()/width+p.offsets.size();
    out.lengths.reserve(estimate); out.resets.reserve(estimate);
    for (size_t stream=1;stream<p.offsets.size();++stream) {
        int64_t start=p.offsets[stream-1], end=p.offsets[stream];
        while (start<end) {
            int64_t stop=std::min(start+width,end);
            for (int64_t e=start+1;e<stop;++e)
                if (p.rows[e]>=0 && p.decay[e]==0.) { stop=e; break; }
            out.lengths.push_back(int(stop-start));
            out.resets.push_back(int(p.rows[start]>=0 && p.decay[start]==0.));
            start=stop;
        }
        out.offsets.push_back(out.lengths.size());
    }
    size_t slots=out.lengths.size()*width;
    out.rows.resize(slots,0); out.prefixes.resize(slots,0.f);
    out.weights.resize(slots,0.f);
    if (!p.dlogweight.empty()) out.dlog.resize(slots,0.f);
    int64_t e=0;
    for (size_t chunk=0;chunk<out.lengths.size();++chunk) {
        double prefix=0.;
        for (int i=0;i<out.lengths[chunk];++i,++e) {
            if (p.rows[e]>=0 && !(out.resets[chunk] && i==0)) prefix+=std::log(p.decay[e]);
            size_t slot=chunk*width+i;
            out.rows[slot]=p.rows[e];
            out.prefixes[slot]=float(prefix);
            out.weights[slot]=float(p.weight[e]);
            if (!p.dlogweight.empty()) out.dlog[slot]=float(p.dlogweight[e]);
        }
    }
    return out;
}

class Builder {
    // Index ranges survive flat vector growth; no per-stream heap objects.
    struct Stream { size_t qb, qe, kb, ke; };
    struct Range { int b, e; };
    // Per-recursion-level reusable scratch. Removing a centroid leaves
    // components of at most half the size, so the recursion depth is O(log n).
    // Per-component and per-merge lists live in flat index arenas instead of a
    // heap vector per group; capacity is retained across calls and jobs.
    struct Scratch {
        Ids order;                    // component node ids, traversal order
        Ids node_arena;               // component node lists
        std::vector<Range> comps;     // component ranges into node_arena
        Ids qs, ks;                   // component query/key lists (+ merges)
        std::vector<Range> gq, gk;    // initial component ranges into qs/ks
        std::vector<Range> wq, wk;    // merge-tree ranges into qs/ks
        std::vector<int> count, cursor;
        std::vector<std::pair<int,int>> heap;
        Ids qp, qd, kp, kd;           // cross() partition scratch
    };
    std::deque<Scratch> scratch;
    Scratch &scratch_at(int depth) {
        if (size_t(depth)>=scratch.size()) scratch.resize(depth+1);
        return scratch[depth];
    }
    PrefillMatcher sam;
    int n=0;
    double tau=0.;
    double cached_tau=std::numeric_limits<double>::quiet_NaN();
    int cached_length=0;
    bool reference_lca=false;
    bool gradients=false;
    std::vector<double> logweights, derivatives;
    Ids endpoint, qnode, matched, depth, owner, parent, size, lca_node, lca_chain, lca_stack;
    std::vector<Ids> up;
    Ids adj_offsets, adj_nodes, adj_cursor;
    struct Neighbors {
        const int *b, *e;
        const int *begin() const { return b; }
        const int *end() const { return e; }
    };
    Neighbors neighbors(int v) const {
        return {adj_nodes.data()+adj_offsets[v],adj_nodes.data()+adj_offsets[v+1]};
    }
    std::vector<char> on_path;
    std::vector<bool> removed;
    std::vector<Stream> streams;
    Ids stream_q, stream_k;
    Ids forest_qhead, forest_qtail, forest_qnext, forest_queries, forest_keys;
    std::vector<double> stream_a, stream_b, stream_da, stream_db;

    double logweight(int length) const {
        if (!length) return neg_inf;
        if (!tau) return std::log(double(length));
        double t = std::abs(tau);
        return (tau > 0 ? length*tau : tau)
            + std::log(-std::expm1(-length*t)) - std::log(-std::expm1(-t));
    }

    int lca(int a, int b) const {
        if (depth[a] < depth[b]) std::swap(a,b);
        int gap = depth[a]-depth[b];
        for (int bit=0; gap; ++bit, gap>>=1)
            if (gap&1) a=up[bit][a];
        if (a==b) return a;
        for (int bit=int(up.size())-1; bit>=0; --bit)
            if (up[bit][a]!=up[bit][b]) { a=up[bit][a]; b=up[bit][b]; }
        return sam.nodes[a].link;
    }

    // Production path reads lca_node[]; reference_lca keeps the original
    // binary-lifting lookup as the unit-test oracle.
    int lca_len(int v, int c) const {
        return reference_lca ? sam.nodes[lca(v,c)].length : sam.nodes[lca_node[v]].length;
    }

    // Only the parent-side component ever asks for lca(v,c). Walk the real link
    // chain from c up to the component's topmost node p, then sweep downward
    // from p; removed[c] stops the sweep before the child components, so this
    // touches exactly the parent-side nodes. O(parent component).
    void fill_lca(int c) {
        lca_chain.clear();
        for (int v=c; ; ) {
            lca_chain.push_back(v);
            int up=sam.nodes[v].link;
            if (up<0 || removed[up]) break;
            v=up;
        }
        for (int v: lca_chain) on_path[v]=1;
        int p=lca_chain.back();
        lca_node[p]=p;
        lca_stack.clear();
        lca_stack.push_back(p);
        while (!lca_stack.empty()) {
            int v=lca_stack.back(); lca_stack.pop_back();
            for (int w: neighbors(v)) {
                if (w==sam.nodes[v].link || removed[w]) continue;
                lca_node[w] = on_path[w] ? w : lca_node[v];
                lca_stack.push_back(w);
            }
        }
        for (int v: lca_chain) on_path[v]=0;
    }

    // Append the ascending merge of two ranges of `arena` as a new tail range.
    // Tail append never moves an existing range's contents, so callers may keep
    // older ranges live while new merged lists are added after them.
    static void merge_into(Ids &arena, Range a, Range b, Range &out) {
        // Arena ranges are immutable: an empty merge can reuse the other range.
        if (a.b==a.e) { out=b; return; }
        if (b.b==b.e) { out=a; return; }
        out.b = int(arena.size());
        arena.resize(arena.size()+(a.e-a.b)+(b.e-b.b));
        int i=a.b, j=b.b, k=out.b;
        while (i<a.e && j<b.e) {
            int va=arena[i], vb=arena[j];
            if (vb<va) { arena[k++]=vb; ++j; } else { arena[k++]=va; ++i; }
        }
        while (i<a.e) arena[k++]=arena[i++];
        while (j<b.e) arena[k++]=arena[j++];
        out.e=k;
    }

    // mode 0: downward/downward; 1: parent/downward; 2: downward/parent;
    // 3: centroid singleton. Prune zero weights before building event streams.
    void emit(const Ids &qs, Range qr, const Ids &ks, Range kr, int c, int mode) {
        if (qr.b==qr.e || kr.b==kr.e || ks[kr.b]>qs[qr.e-1]) return;
        Stream s{stream_q.size(),0,stream_k.size(),0};
        for (int t=qr.b;t<qr.e;++t) {
            int i=qs[t];
            int length = mode==3 ? matched[i] : std::min(matched[i],sam.nodes[c].length);
            if (mode==1) length=std::min(matched[i],lca_len(qnode[i],c));
            double a = mode==2 ? 0. : (gradients ? logweights[length] : logweight(length));
            if (a!=neg_inf) {
                stream_q.push_back(i); stream_a.push_back(a);
                if (gradients) stream_da.push_back(mode==2 ? 0. : derivatives[length]);
            }
        }
        s.qe=stream_q.size();
        if (s.qb==s.qe) return;
        for (int t=kr.b;t<kr.e;++t) {
            int j=ks[t];
            if (j>stream_q.back()) break;
            int length = mode==2 ? lca_len(endpoint[j],c) : 0;
            double b = mode==2 ? (gradients ? logweights[length] : logweight(length)) : 0.;
            if (b!=neg_inf) {
                stream_k.push_back(j); stream_b.push_back(b);
                if (gradients) stream_db.push_back(mode==2 ? derivatives[length] : 0.);
            }
        }
        s.ke=stream_k.size();
        if (s.kb!=s.ke) streams.push_back(s);
        else {
            stream_q.resize(s.qb); stream_a.resize(s.qb);
            if (gradients) stream_da.resize(s.qb);
        }
    }

    // `aq` are the queries of the first component and `bk` the keys of the
    // second; cross() never recurses, so the partition scratch stays in S.
    // At most one live merge group contains the original parent component.
    // Only that side needs partitioning; the other side is entirely downward.
    void cross(const Ids &aq, Range aqr, const Ids &bk, Range bkr, int c, int pg,
               bool parent_a, bool parent_b, Scratch &S) {
        if (aqr.b==aqr.e || bkr.b==bkr.e || bk[bkr.b]>aq[aqr.e-1]) return;
        if (!parent_a && !parent_b) { emit(aq,aqr,bk,bkr,c,0); return; }
        if (parent_a && parent_b) throw std::logic_error("split parent component");
        if (parent_a) {
            auto &qp=S.qp, &qd=S.qd; qp.clear(); qd.clear();
            for (int t=aqr.b;t<aqr.e;++t) {
                int i=aq[t]; (owner[qnode[i]]==pg ? qp:qd).push_back(i);
            }
            emit(qd,{0,int(qd.size())},bk,bkr,c,0);
            emit(qp,{0,int(qp.size())},bk,bkr,c,1);
        } else {
            auto &kp=S.kp, &kd=S.kd; kp.clear(); kd.clear();
            for (int t=bkr.b;t<bkr.e;++t) {
                int j=bk[t]; (owner[endpoint[j]]==pg ? kp:kd).push_back(j);
            }
            emit(aq,aqr,kd,{0,int(kd.size())},c,0);
            emit(aq,aqr,kp,{0,int(kp.size())},c,2);
        }
    }

    // Per-branch query buckets are populated during matching. Key buckets
    // are populated by SAM extension, with clone positions excluded.
    void build_root_forest() {
        // F(0)=0: distinct root components have no cross contribution.
        // Removing only the root isolates branches without changing suffix links.
        removed[0]=true;
        for (size_t g=0;g<sam.branch_roots.size();++g) {
            int first_key=sam.key_head[g],last_query=forest_qtail[g];
            if (last_query<first_key || last_query<0) continue;
            forest_queries.clear();forest_keys.clear();
            for (int i=forest_qhead[g];i>=0;i=forest_qnext[i])
                if (i>=first_key) forest_queries.push_back(i);
            for (int j=first_key;j>=0 && j<=last_query;j=sam.key_next[j])
                forest_keys.push_back(j);
            decompose(sam.branch_roots[g],forest_queries,{0,int(forest_queries.size())},
                      forest_keys,{0,int(forest_keys.size())},0);
        }
    }

    void decompose(int root, const Ids &qs, Range qr, const Ids &ks, Range kr, int depth,
                   const Ids *known_nodes=nullptr, Range known_range={0,0}) {
        if (qr.b==qr.e || kr.b==kr.e || ks[kr.b]>qs[qr.e-1]) return;
        Scratch &S=scratch_at(depth);
        // Parent partitioning already produced this component's BFS order,
        // orientation and subtree sizes. Each forest root needs one initial BFS.
        parent[root]=-1;
        if (!known_nodes) {
            auto &initial=S.order;initial.clear();initial.push_back(root);
            for (size_t i=0;i<initial.size();++i) {
                int v=initial[i];size[v]=1;
                for (int w:neighbors(v)) if (!removed[w] && w!=parent[v]) {
                    parent[w]=v;initial.push_back(w);
                }
            }
            for (size_t i=initial.size();i-->1;) size[parent[initial[i]]]+=size[initial[i]];
        }
        // known_nodes belongs to the parent recursion level and stays immutable
        // until this call returns. scratch is a deque, so deeper growth is safe.
        const Ids &order=known_nodes ? *known_nodes : S.order;
        const int begin=known_nodes ? known_range.b : 0;
        const int end=known_nodes ? known_range.e : int(order.size());
        const int total=end-begin;
        int c=root,best=total;
        for (int t=begin;t<end;++t) {
            int v=order[t],largest=total-size[v];
            for (int w:neighbors(v)) if (!removed[w] && parent[w]==v) largest=std::max(largest,size[w]);
            if (largest<best) {best=largest;c=v;}
        }
        // Child-side sizes are unchanged by removing c. In the parent-side
        // component only the old root-to-c path reverses. Repair those sizes
        // before partitioning overwrites parent[] with the new orientations.
        int cut=size[c];
        for (int v=parent[c];v>=0;v=parent[v]) {
            int previous=size[v];size[v]=total-cut;cut=previous;
        }
        removed[c]=true;
        S.node_arena.clear(); S.comps.clear();
        S.node_arena.push_back(c); S.comps.push_back(Range{0,1});
        int pg=-1; owner[c]=0;
        for (int neighbor:neighbors(c)) if (!removed[neighbor]) {
            int g=int(S.comps.size());
            if (neighbor==sam.nodes[c].link) pg=g;
            Range r{int(S.node_arena.size()),0};
            S.node_arena.push_back(neighbor); parent[neighbor]=c;
            for (int t=r.b;t<int(S.node_arena.size());++t) {
                int v=S.node_arena[t]; owner[v]=g;
                for (int w:neighbors(v)) if (!removed[w] && w!=parent[v]) {
                    parent[w]=v; S.node_arena.push_back(w);
                }
            }
            r.e=int(S.node_arena.size());
            S.comps.push_back(r);
        }
        // Group queries and keys by owning component through counting sort,
        // preserving each component's original input order without per-group
        // vectors. Initial lists sit at the front of qs/ks; merge-tree lists are
        // appended after them, so the component lists stay valid for recursion.
        const int nc=int(S.comps.size());
        S.gq.assign(nc,Range{0,0}); S.gk.assign(nc,Range{0,0});
        S.qs.clear(); S.ks.clear();
        S.count.assign(nc+1,0);
        for (int t=qr.b;t<qr.e;++t) ++S.count[owner[qnode[qs[t]]]+1];
        for (int g=0;g<nc;++g) S.count[g+1]+=S.count[g];
        S.qs.resize(S.count[nc]);
        S.cursor.assign(S.count.begin(),S.count.end());
        for (int g=0;g<nc;++g) S.gq[g]=Range{S.count[g],S.count[g+1]};
        for (int t=qr.b;t<qr.e;++t) { int i=qs[t]; S.qs[S.cursor[owner[qnode[i]]]++]=i; }
        S.count.assign(nc+1,0);
        for (int t=kr.b;t<kr.e;++t) ++S.count[owner[endpoint[ks[t]]]+1];
        for (int g=0;g<nc;++g) S.count[g+1]+=S.count[g];
        S.ks.resize(S.count[nc]);
        S.cursor.assign(S.count.begin(),S.count.end());
        for (int g=0;g<nc;++g) S.gk[g]=Range{S.count[g],S.count[g+1]};
        for (int t=kr.b;t<kr.e;++t) { int j=ks[t]; S.ks[S.cursor[owner[endpoint[j]]]++]=j; }
        if (!reference_lca && pg>=0 && (S.gq[pg].b!=S.gq[pg].e || S.gk[pg].b!=S.gk[pg].e))
            fill_lca(c);
        // The SAM root has length zero: every cross-component match has zero
        // weight. Skip the merge tree there but still recurse below.
        if (sam.nodes[c].length>0) {
            S.wq.assign(S.gq.begin(),S.gq.end());
            S.wk.assign(S.gk.begin(),S.gk.end());
            // Track the parent component through unions without scanning the
            // event lists to rediscover whether a group contains it.
            int parent_group=pg;
            auto &heap=S.heap; heap.clear(); heap.reserve(nc);
            for (int g=0;g<nc;++g) heap.push_back({int(S.comps[g].e-S.comps[g].b),g});
            std::make_heap(heap.begin(),heap.end(),std::greater<std::pair<int,int>>());
            while (heap.size()>1) {
                std::pop_heap(heap.begin(),heap.end(),std::greater<std::pair<int,int>>());
                auto [wa,a]=heap.back(); heap.pop_back();
                std::pop_heap(heap.begin(),heap.end(),std::greater<std::pair<int,int>>());
                auto [wb,b]=heap.back(); heap.pop_back();
                bool parent_a=a==parent_group, parent_b=b==parent_group;
                cross(S.qs,S.wq[a],S.ks,S.wk[b],c,pg,parent_a,parent_b,S);
                cross(S.qs,S.wq[b],S.ks,S.wk[a],c,pg,parent_b,parent_a,S);
                // The final merged lists have no consumer; recursion uses gq/gk.
                if (heap.empty()) break;
                Range qj,kj;
                merge_into(S.qs,S.wq[a],S.wq[b],qj);
                merge_into(S.ks,S.wk[a],S.wk[b],kj);
                S.wq[a]=Range{0,0}; S.wk[a]=Range{0,0};
                S.wq[b]=Range{0,0}; S.wk[b]=Range{0,0};
                int joined=int(S.wq.size());
                if (parent_a || parent_b) parent_group=joined;
                S.wq.push_back(qj); S.wk.push_back(kj);
                heap.push_back({wa+wb,joined});
                std::push_heap(heap.begin(),heap.end(),std::greater<std::pair<int,int>>());
            }
        }
        emit(S.qs,S.gq[0],S.ks,S.gk[0],c,3);
        for (int g=1;g<nc;++g)
            decompose(S.node_arena[S.comps[g].b],S.qs,S.gq[g],S.ks,S.gk[g],depth+1,
                      &S.node_arena,S.comps[g]);
    }

  public:
    Builder() = default;
    Builder(const Ids &q, const Ids &k, const Ids &reset, double tau_, bool reference_lca_=false, bool gradients_=false) {
        if (q.size()!=k.size() || reset.size()!=k.size())
            throw std::invalid_argument("equal label/reset lengths required");
        build(q.data(),k.data(),reset.data(),k.size(),tau_,reference_lca_,gradients_);
    }
    // Inputs are borrowed only until build returns. A null reset means no
    // interior resets (the training job already represents one document).
    void build(const int *q, const int *k, const int *reset, size_t count_,
               double tau_, bool reference_lca_=false, bool gradients_=false) {
        if (!q || !k || !count_ || count_>size_t(std::numeric_limits<int>::max()/8) || !std::isfinite(tau_))
            throw std::invalid_argument("nonempty bounded label lengths and finite tau required");
        n=int(count_); tau=tau_; reference_lca=reference_lca_; gradients=gradients_;
        endpoint.clear(); qnode.clear(); matched.clear(); up.clear();
        streams.clear(); stream_q.clear(); stream_k.clear();
        stream_a.clear(); stream_b.clear(); stream_da.clear(); stream_db.clear();
        if (std::abs(tau)>std::numeric_limits<double>::max()/n)
            throw std::invalid_argument("tau*N exceeds FP64 log-domain range");
        if (gradients) {
            if (cached_tau!=tau) {
                cached_tau=std::numeric_limits<double>::quiet_NaN(); cached_length=0;
                logweights.assign(1,neg_inf); derivatives.assign(1,0.);
            }
            int first=cached_length+1;
            if (first<=n) { logweights.resize(n+1); derivatives.resize(n+1); }
            for (int l=first;l<=n;++l) {
                logweights[l]=logweight(l);
                derivatives[l]=std::exp(logweights[l-1]-logweights[l])*derivatives[l-1]
                    + std::exp(l*tau-logweights[l])*l;
            }
            cached_tau=tau; cached_length=std::max(cached_length,n);
        }
        // Build the offline SAM with the same node/link order as online decode.
        int last=0;
        sam.reset(n); endpoint.reserve(n);
        for (int j=0;j<n;++j) {
            if (reset && reset[j]!=0 && reset[j]!=1) throw std::invalid_argument("reset must be 0 or 1");
            sam.extend(k[j],j,last); endpoint.push_back(last);
        }
        int count=int(sam.nodes.size());
        owner.resize(count);
        parent.resize(count); size.resize(count); removed.assign(count,false);
        lca_node.resize(count);
        on_path.assign(count,0);
        adj_offsets.assign(count+1,0);
        for (int v=1;v<count;++v) {
            ++adj_offsets[v+1]; ++adj_offsets[sam.nodes[v].link+1];
        }
        for (int v=1;v<=count;++v) adj_offsets[v]+=adj_offsets[v-1];
        adj_nodes.resize(2*(count-1)); adj_cursor=adj_offsets;
        for (int v=1;v<count;++v) {
            int p=sam.nodes[v].link;
            adj_nodes[adj_cursor[v]++]=p;
            adj_nodes[adj_cursor[p]++]=v;
        }
        if (reference_lca) {
            depth.resize(count); depth[0]=0;
            Ids order{0};
            for (size_t i=0;i<order.size();++i) for (int w:neighbors(order[i]))
                if (w!=sam.nodes[order[i]].link) { depth[w]=depth[order[i]]+1; order.push_back(w); }
            up.emplace_back(count);
            for (int v=0;v<count;++v) up[0][v]=std::max(0,sam.nodes[v].link);
            for (int64_t span=2;span<=count;span*=2) {
                Ids table(count);
                for (int v=0;v<count;++v) table[v]=up.back()[up.back()[v]];
                up.push_back(std::move(table));
            }
        }
        int state=0, length=0;
        qnode.reserve(n); matched.reserve(n);
        forest_qhead.assign(sam.branch_roots.size(),-1);
        forest_qtail.assign(sam.branch_roots.size(),-1);forest_qnext.resize(n);
        for (int i=0;i<n;++i) {
            if (reset && reset[i]) state=length=0;
            int edge=sam.find(state,q[i]);
            while (state && edge<0) {
                state=sam.nodes[state].link; length=std::min(length,sam.nodes[state].length);
                edge=sam.find(state,q[i]);
            }
            if (edge<0) state=length=0;
            else { state=sam.edges[edge].target; ++length; }
            qnode.push_back(state); matched.push_back(length);
            if (length) {
                int group=sam.branch[state];forest_qnext[i]=-1;
                if (forest_qtail[group]>=0) forest_qnext[forest_qtail[group]]=i;
                else forest_qhead[group]=i;
                forest_qtail[group]=i;
            }
        }
        build_root_forest();
    }

    Program finish() const {
        if (!n) throw std::logic_error("build must precede finish");
        Program out; out.n=n; out.states=int(sam.nodes.size()); out.logden.assign(n,0.);
        if (gradients) out.dlogden.assign(n,0.);
        // Per-query scaled denominator: den = exp(P)*Z, dlogden = D/Z.
        // A stream contributes exp(a+pivot)*mass and its unnormalized tau
        // derivative exp(a+pivot)*(da*mass+dmass). Treat pivots as numerical
        // scales, not differentiable model parameters. Rescale both sums when
        // P increases; this needs one exp per contribution and one final log
        // per query, instead of logadd and derivative exps at every event.
        // P=0, Z=1, D=0 preserves the constant unit denominator contribution.
        static thread_local std::vector<double> norm_pivot, norm_mass, norm_derivative;
        norm_pivot.assign(n,0.); norm_mass.assign(n,1.);
        if (gradients) norm_derivative.assign(n,0.);
        // Scalar pass. Only causal keys participate in each stream's pivot.
        // The exponential pair (decay, weight) for a key depends only on the
        // running pivot, so the two passes walk the identical key sequence and
        // recompute the identical values. Evaluate each pair once here and hand
        // it to the coefficient pass; exp is not CSE-able across calls because
        // it may set errno, so the compiler cannot do this on its own.
        static thread_local std::vector<double> key_decay, key_weight;
        key_decay.clear(); key_weight.clear();
        for (const auto &s:streams) {
            size_t cursor=s.kb; double pivot=neg_inf, mass=0., dmass=0.;
            for (size_t qi=s.qb;qi<s.qe;++qi) {
                int i=stream_q[qi];
                while (cursor<s.ke && stream_k[cursor]<=i) {
                    double b=stream_b[cursor], next=std::max(pivot,b);
                    double decay=std::exp(pivot-next), weight=std::exp(b-next);
                    if (gradients) dmass=dmass*decay+weight*stream_db[cursor];
                    ++cursor;
                    mass=mass*decay+weight; pivot=next;
                    key_decay.push_back(decay); key_weight.push_back(weight);
                }
                if (mass) {
                    double t=stream_a[qi]+pivot;
                    double d=gradients ? stream_da[qi]*mass+dmass : 0.;
                    if (t>norm_pivot[i]) {
                        double scale=std::exp(norm_pivot[i]-t);
                        norm_mass[i]=norm_mass[i]*scale+mass;
                        if (gradients) norm_derivative[i]=norm_derivative[i]*scale+d;
                        norm_pivot[i]=t;
                    } else {
                        double scale=std::exp(t-norm_pivot[i]);
                        norm_mass[i]+=mass*scale;
                        if (gradients) norm_derivative[i]+=d*scale;
                    }
                }
            }
        }
        for (int i=0;i<n;++i) {
            out.logden[i]=norm_pivot[i]+std::log(norm_mass[i]);
            if (gradients) out.dlogden[i]=norm_derivative[i]/norm_mass[i];
        }
        // Compile all floating point transcendental work into coefficients.
        size_t capacity=stream_q.size()+stream_k.size();
        out.rows.reserve(capacity); out.decay.reserve(capacity); out.weight.reserve(capacity);
        if (gradients) out.dlogweight.reserve(capacity);
        out.offsets.reserve(streams.size()+1);
        size_t key_cursor=0;
        for (const auto &s:streams) {
            size_t cursor=s.kb; double pivot=neg_inf;
            for (size_t qi=s.qb;qi<s.qe;++qi) {
                int i=stream_q[qi];
                while (cursor<s.ke && stream_k[cursor]<=i) {
                    double b=stream_b[cursor];
                    if (gradients) out.dlogweight.push_back(stream_db[cursor]);
                    out.rows.push_back(stream_k[cursor]);
                    out.decay.push_back(key_decay[key_cursor]);
                    out.weight.push_back(key_weight[key_cursor]);
                    ++key_cursor; ++cursor;
                    pivot=std::max(pivot,b);
                }
                if (pivot!=neg_inf) {
                    out.rows.push_back(-i-1); out.decay.push_back(1.);
                    out.weight.push_back(std::exp(stream_a[qi]+pivot-out.logden[i]));
                    if (gradients) out.dlogweight.push_back(stream_da[qi]-out.dlogden[i]);
                }
            }
            if (int64_t(out.rows.size())!=out.offsets.back()) out.offsets.push_back(out.rows.size());
        }
        return out;
    }
};

// CPU stand-in for the future device backend: deliberately uses only Program.
template<class T>
void execute(const Program &p, const T *sq, const T *sk, const T *v, int r, int dv, T *out) {
    std::fill(out,out+int64_t(p.n)*dv,T(0));
    std::vector<T> matrix(int64_t(r)*dv);
    for (size_t s=1;s<p.offsets.size();++s) {
        std::fill(matrix.begin(),matrix.end(),T(0));
        for (int64_t e=p.offsets[s-1];e<p.offsets[s];++e) {
            int row=p.rows[e]; T w=T(p.weight[e]);
            if (row>=0) {
                T decay=T(p.decay[e]);
                for (int a=0;a<r;++a) for (int b=0;b<dv;++b)
                    matrix[a*dv+b]=decay*matrix[a*dv+b]+w*sk[int64_t(row)*r+a]*v[int64_t(row)*dv+b];
            } else {
                row=-row-1;
                for (int b=0;b<dv;++b) {
                    T sum=0;
                    for (int a=0;a<r;++a) sum+=sq[int64_t(row)*r+a]*matrix[a*dv+b];
                    out[int64_t(row)*dv+b]+=w*sum;
                }
            }
        }
    }
}
} // namespace dism_prefill
