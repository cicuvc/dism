#pragma once

// Offline control plane. No Torch/CUDA dependencies and no vector payloads.
#include "planner.hpp"
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
};

struct ChunkProgram {
    std::vector<int64_t> offsets{0};
    Ids rows, lengths, resets;
    std::vector<float> prefixes, weights;
};

inline ChunkProgram chunk_program(const Program &p, int width) {
    if (width!=16 && width!=32 && width!=64)
        throw std::invalid_argument("chunk width must be 16, 32 or 64");
    ChunkProgram out;
    for (size_t stream=1;stream<p.offsets.size();++stream) {
        int64_t start=p.offsets[stream-1], end=p.offsets[stream];
        while (start<end) {
            int64_t stop=std::min(start+width,end);
            for (int64_t e=start+1;e<stop;++e)
                if (p.rows[e]>=0 && p.decay[e]==0.) { stop=e; break; }
            bool reset=p.rows[start]>=0 && p.decay[start]==0.;
            double prefix=0.;
            size_t base=out.rows.size();
            out.rows.resize(base+width,0);
            out.prefixes.resize(base+width,0.f);
            out.weights.resize(base+width,0.f);
            for (int64_t e=start;e<stop;++e) {
                if (p.rows[e]>=0 && !(reset && e==start)) prefix+=std::log(p.decay[e]);
                size_t slot=base+e-start;
                out.rows[slot]=p.rows[e];
                out.prefixes[slot]=float(prefix);
                out.weights[slot]=float(p.weight[e]);
            }
            out.lengths.push_back(int(stop-start)); out.resets.push_back(int(reset));
            start=stop;
        }
        out.offsets.push_back(out.lengths.size());
    }
    return out;
}

class Builder {
    struct Stream { Ids q, k; std::vector<double> a, b; };
    struct Group { Ids q, k; };
    dism_decode::Planner sam;
    int n;
    double tau;
    bool reference_lca;
    Ids endpoint, qnode, matched, depth, owner, parent, size, lca_node, lca_chain, lca_stack;
    std::vector<Ids> adj, up;
    std::vector<char> on_path;
    std::vector<bool> removed;
    std::vector<Stream> streams;

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
            for (int w: adj[v]) {
                if (w==sam.nodes[v].link || removed[w]) continue;
                lca_node[w] = on_path[w] ? w : lca_node[v];
                lca_stack.push_back(w);
            }
        }
        for (int v: lca_chain) on_path[v]=0;
    }

    static Ids merge(const Ids &a, const Ids &b) {
        Ids out; out.reserve(a.size()+b.size());
        std::merge(a.begin(),a.end(),b.begin(),b.end(),std::back_inserter(out));
        return out;
    }

    // mode 0: downward/downward; 1: parent/downward; 2: downward/parent;
    // 3: centroid singleton. Prune zero weights before building event streams.
    void emit(const Ids &queries, const Ids &keys, int c, int mode) {
        Stream s;
        for (int i: queries) {
            int length = mode==3 ? matched[i] : std::min(matched[i],sam.nodes[c].length);
            if (mode==1) length=std::min(matched[i],lca_len(qnode[i],c));
            double a = mode==2 ? 0. : logweight(length);
            if (a!=neg_inf) { s.q.push_back(i); s.a.push_back(a); }
        }
        if (s.q.empty()) return;
        for (int j: keys) {
            if (j>s.q.back()) break;
            double b = mode==2 ? logweight(lca_len(endpoint[j],c)) : 0.;
            if (b!=neg_inf) { s.k.push_back(j); s.b.push_back(b); }
        }
        if (!s.k.empty()) streams.push_back(std::move(s));
    }

    void cross(const Group &a, const Group &b, int c, int pg) {
        Ids qp,qd,kp,kd;
        for (int i:a.q) (owner[qnode[i]]==pg ? qp:qd).push_back(i);
        for (int j:b.k) (owner[endpoint[j]]==pg ? kp:kd).push_back(j);
        if (!qp.empty() && !kp.empty()) throw std::logic_error("split parent component");
        emit(qd,kd,c,0); emit(qp,kd,c,1); emit(qd,kp,c,2);
    }

    void decompose(int root, const Ids &queries, const Ids &keys) {
        if (queries.empty() || keys.empty()) return;
        Ids order{root}; parent[root]=-1;
        for (size_t i=0;i<order.size();++i) {
            int v=order[i]; size[v]=1;
            for (int w:adj[v]) if (!removed[w] && w!=parent[v]) {
                parent[w]=v; order.push_back(w);
            }
        }
        for (size_t i=order.size();i-->1;) size[parent[order[i]]]+=size[order[i]];
        int c=root, best=int(order.size());
        for (int v:order) {
            int largest=int(order.size())-size[v];
            for (int w:adj[v]) if (!removed[w] && parent[w]==v) largest=std::max(largest,size[w]);
            if (largest<best) { best=largest; c=v; }
        }
        removed[c]=true;
        std::vector<Ids> components(1,Ids{c});
        int pg=-1; owner[c]=0;
        for (int neighbor:adj[c]) if (!removed[neighbor]) {
            int g=int(components.size());
            if (neighbor==sam.nodes[c].link) pg=g;
            Ids nodes{neighbor}; parent[neighbor]=c;
            for (size_t i=0;i<nodes.size();++i) {
                int v=nodes[i]; owner[v]=g;
                for (int w:adj[v]) if (!removed[w] && w!=parent[v]) {
                    parent[w]=v; nodes.push_back(w);
                }
            }
            components.push_back(std::move(nodes));
        }
        std::vector<Group> groups(components.size());
        for (int i:queries) groups[owner[qnode[i]]].q.push_back(i);
        for (int j:keys) groups[owner[endpoint[j]]].k.push_back(j);
        if (!reference_lca && pg>=0 && (!groups[pg].q.empty() || !groups[pg].k.empty()))
            fill_lca(c);
        // Keep original component lists for recursion; merged lists die here.
        {
            std::vector<Group> work=groups;
            using Item=std::pair<int,int>;
            std::priority_queue<Item,std::vector<Item>,std::greater<Item>> heap;
            for (int g=0;g<int(groups.size());++g) heap.emplace(int(components[g].size()),g);
            while (heap.size()>1) {
                auto [wa,a]=heap.top(); heap.pop();
                auto [wb,b]=heap.top(); heap.pop();
                cross(work[a],work[b],c,pg); cross(work[b],work[a],c,pg);
                Group joined{merge(work[a].q,work[b].q),merge(work[a].k,work[b].k)};
                work[a]=Group{}; work[b]=Group{};
                heap.emplace(wa+wb,int(work.size())); work.push_back(std::move(joined));
            }
        }
        emit(groups[0].q,groups[0].k,c,3);
        for (size_t g=1;g<groups.size();++g)
            decompose(components[g][0],groups[g].q,groups[g].k);
    }

  public:
    Builder(const Ids &q, const Ids &k, const Ids &reset, double tau_, bool reference_lca_=false)
        : sam(1,1,1,2,0.), n(int(k.size())), tau(tau_), reference_lca(reference_lca_) {
        if (!n || q.size()!=k.size() || reset.size()!=k.size() || !std::isfinite(tau))
            throw std::invalid_argument("nonempty equal label/reset lengths and finite tau required");
        if (std::abs(tau)>std::numeric_limits<double>::max()/n)
            throw std::invalid_argument("tau*N exceeds FP64 log-domain range");
        // Reuse the existing SAM extension primitive, not its online snapshots.
        int last=0;
        for (int j=0;j<n;++j) {
            if (reset[j]!=0 && reset[j]!=1) throw std::invalid_argument("reset must be 0 or 1");
            sam.extend(k[j],j,last); endpoint.push_back(last);
        }
        int count=int(sam.nodes.size());
        adj.resize(count); owner.resize(count);
        parent.resize(count); size.resize(count); removed.resize(count,false);
        lca_node.resize(count);
        on_path.assign(count,0);
        for (int v=1;v<count;++v) {
            int p=sam.nodes[v].link; adj[v].push_back(p); adj[p].push_back(v);
        }
        if (reference_lca) {
            depth.resize(count);
            Ids order{0};
            for (size_t i=0;i<order.size();++i) for (int w:adj[order[i]])
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
        for (int i=0;i<n;++i) {
            if (reset[i]) state=length=0;
            while (state && !sam.nodes[state].next.count(q[i])) {
                state=sam.nodes[state].link; length=std::min(length,sam.nodes[state].length);
            }
            auto it=sam.nodes[state].next.find(q[i]);
            if (it==sam.nodes[state].next.end()) state=length=0;
            else { state=it->second; ++length; }
            qnode.push_back(state); matched.push_back(length);
        }
        Ids ids(n); std::iota(ids.begin(),ids.end(),0);
        decompose(0,ids,ids);
    }

    Program finish() const {
        Program out; out.n=n; out.states=int(sam.nodes.size()); out.logden.assign(n,0.);
        // Scalar pass. Only causal keys participate in each stream's pivot.
        for (const auto &s:streams) {
            size_t cursor=0; double pivot=neg_inf, mass=0.;
            for (size_t qi=0;qi<s.q.size();++qi) {
                int i=s.q[qi];
                while (cursor<s.k.size() && s.k[cursor]<=i) {
                    double b=s.b[cursor++], next=std::max(pivot,b);
                    mass=mass*std::exp(pivot-next)+std::exp(b-next); pivot=next;
                }
                if (mass) out.logden[i]=logadd(out.logden[i],s.a[qi]+pivot+std::log(mass));
            }
        }
        // Compile all floating point transcendental work into coefficients.
        for (const auto &s:streams) {
            size_t cursor=0; double pivot=neg_inf;
            for (size_t qi=0;qi<s.q.size();++qi) {
                int i=s.q[qi];
                while (cursor<s.k.size() && s.k[cursor]<=i) {
                    double b=s.b[cursor], next=std::max(pivot,b);
                    out.rows.push_back(s.k[cursor++]);
                    out.decay.push_back(std::exp(pivot-next)); out.weight.push_back(std::exp(b-next));
                    pivot=next;
                }
                if (pivot!=neg_inf) {
                    out.rows.push_back(-i-1); out.decay.push_back(1.);
                    out.weight.push_back(std::exp(s.a[qi]+pivot-out.logden[i]));
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
