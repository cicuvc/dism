// Experimental Builder members, injected only into isolated build headers.
// HEAVY_WEIGHTED selects weight-median ternary splits; otherwise midpoint binary.
// Leaves are SAM nodes. Internal nodes combine either light children at one
// attachment vertex (anchor >= 0), or ordered intervals of a heavy chain.
    struct HNode { int parent=-1, left=-1, right=-1, anchor=-1, weight=1; };
    std::vector<HNode> h_nodes;
    Ids h_order,h_heavy,h_roots,h_rakes,h_chain,h_prefix;
    Ids h_qcount,h_kcount,h_lastq,h_qcursor,h_kcursor;
    std::vector<std::pair<int,int>> h_heap;
    int h_leaf_count=0;

    int h_join(int a,int b,int anchor) {
        if (a<0) return b;
        if (b<0) return a;
        int id=int(h_nodes.size());
        h_nodes.push_back({-1,a,b,anchor,h_nodes[a].weight+h_nodes[b].weight});
        h_nodes[a].parent=id; h_nodes[b].parent=id;
        return id;
    }
    int h_split(int lo,int hi) {
        if (lo==hi) return -1;
        if (hi-lo==1) return h_rakes[h_chain[lo]];
#if HEAVY_WEIGHTED
        int target=h_prefix[lo]+(h_prefix[hi]-h_prefix[lo]+1)/2;
        int pivot=int(std::lower_bound(h_prefix.begin()+lo+1,h_prefix.begin()+hi+1,target)-h_prefix.begin())-1;
        int left=h_split(lo,pivot), right=h_split(pivot+1,hi);
        return h_join(left,h_join(h_rakes[h_chain[pivot]],right,-1),-1);
#else
        int mid=lo+(hi-lo)/2;
        int left=h_split(lo,mid),right=h_split(mid,hi);
        return h_join(left,right,-1);
#endif
    }
    // Each leaf-to-root walk visits exactly the streams containing this event.
    // A flat (rake) join changes its attachment to anchor; a chain join retains
    // the attachment, which determines the LCA when this event is on the left.
    // length == -1 denotes the constant factor 1, not F(1).
    template<class Fn> void h_visit(int v,bool query,int match,Fn &&fn) const {
        fn(v,query ? match : -1);
        int attachment=v;
        for (int child=v,p=h_nodes[v].parent;p>=0;child=p,p=h_nodes[p].parent) {
            const auto &node=h_nodes[p]; bool left=node.left==child;
            int sid=h_leaf_count+2*(p-h_leaf_count)+(query ? !left : left);
            int length=-1;
            if (node.anchor>=0) {
                attachment=node.anchor;
                if (query) length=std::min(match,sam.nodes[attachment].length);
            } else if (left) {
                length=sam.nodes[attachment].length;
                if (query) length=std::min(match,length);
            }
            fn(sid,length);
        }
    }
    void build_heavy() {
        const int m=int(sam.nodes.size()); h_leaf_count=m;
        h_order.clear(); h_order.push_back(0);
        for (size_t i=0;i<h_order.size();++i) {
            int v=h_order[i];
            for (int w:neighbors(v)) if (sam.nodes[w].link==v) h_order.push_back(w);
        }
        size.assign(m,1); h_heavy.assign(m,-1);
        for (size_t i=h_order.size();i-->0;) {
            int v=h_order[i];
            for (int w:neighbors(v)) if (sam.nodes[w].link==v) {
                size[v]+=size[w];
                if (h_heavy[v]<0 || size[w]>size[h_heavy[v]]) h_heavy[v]=w;
            }
        }
        h_nodes.clear(); h_nodes.reserve(2*size_t(m)); h_nodes.resize(m);
        h_roots.assign(m,-1); h_rakes.resize(m);
        // Light-child chain roots are built before their parent chain.
        for (size_t i=h_order.size();i-->0;) {
            int head=h_order[i],par=sam.nodes[head].link;
            if (par>=0 && h_heavy[par]==head) continue;
            h_chain.clear(); h_prefix.clear(); h_prefix.push_back(0);
            for (int v=head;v>=0;v=h_heavy[v]) {
                h_chain.push_back(v); h_heap.clear(); h_heap.push_back({1,v});
                // LCA at the SAM root has F(0)=0. Leave its light-child
                // chain trees disconnected: no cross-stream can contribute.
                // This mirrors production's root merge-tree pruning.
                for (int w:neighbors(v)) if (v!=0 && sam.nodes[w].link==v && w!=h_heavy[v])
                    h_heap.push_back({size[w],h_roots[w]});
                std::make_heap(h_heap.begin(),h_heap.end(),std::greater<std::pair<int,int>>());
                while (h_heap.size()>1) {
                    std::pop_heap(h_heap.begin(),h_heap.end(),std::greater<std::pair<int,int>>());
                    auto [wa,a]=h_heap.back();h_heap.pop_back();
                    std::pop_heap(h_heap.begin(),h_heap.end(),std::greater<std::pair<int,int>>());
                    auto [wb,b]=h_heap.back();h_heap.pop_back();
                    int joined=h_join(a,b,v);
                    h_heap.push_back({wa+wb,joined});
                    std::push_heap(h_heap.begin(),h_heap.end(),std::greater<std::pair<int,int>>());
                }
                h_rakes[v]=h_heap.front().second;
                h_prefix.push_back(h_prefix.back()+h_nodes[h_rakes[v]].weight);
            }
            h_roots[head]=h_split(0,int(h_chain.size()));
        }
        const int slots=m+2*(int(h_nodes.size())-m);
        h_qcount.assign(slots,0);h_kcount.assign(slots,0);h_lastq.assign(slots,-1);
        // Count before allocating; key events later than the final query are
        // dropped exactly as emit() does in the production planner.
        for (int i=0;i<n;++i) if (matched[i]) h_visit(qnode[i],true,matched[i],[&](int s,int len) {
            if (len) {++h_qcount[s];h_lastq[s]=i;}
        });
        for (int j=0;j<n;++j) h_visit(endpoint[j],false,0,[&](int s,int len) {
            if (len && j<=h_lastq[s]) ++h_kcount[s];
        });
        h_qcursor.assign(slots,-1);h_kcursor.assign(slots,-1);
        size_t nq=0,nk=0;
        for (int s=0;s<slots;++s) if (h_qcount[s] && h_kcount[s]) {
            h_qcursor[s]=int(nq); h_kcursor[s]=int(nk);
            streams.push_back({nq,nq+h_qcount[s],nk,nk+h_kcount[s]});
            nq+=h_qcount[s];nk+=h_kcount[s];
        }
        stream_q.resize(nq);stream_a.resize(nq);stream_k.resize(nk);stream_b.resize(nk);
        if (gradients) {stream_da.resize(nq);stream_db.resize(nk);}
        // Global token order gives sorted per-stream lists without list merges.
        for (int i=0;i<n;++i) if (matched[i]) h_visit(qnode[i],true,matched[i],[&](int s,int len) {
            if (!len || !h_kcount[s]) return;
            int slot=h_qcursor[s]++;
            stream_q[slot]=i;
            stream_a[slot]=len<0 ? 0. : (gradients ? logweights[len] : logweight(len));
            if (gradients) stream_da[slot]=len<0 ? 0. : derivatives[len];
        });
        for (int j=0;j<n;++j) h_visit(endpoint[j],false,0,[&](int s,int len) {
            if (!len || j>h_lastq[s] || !h_kcount[s]) return;
            int slot=h_kcursor[s]++;
            stream_k[slot]=j;
            stream_b[slot]=len<0 ? 0. : (gradients ? logweights[len] : logweight(len));
            if (gradients) stream_db[slot]=len<0 ? 0. : derivatives[len];
        });
    }
    void decompose(int, const Ids &, Range, const Ids &, Range, int) {
        build_heavy();
    }
