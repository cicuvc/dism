#pragma once

// Offline SAM with contiguous transition storage. Online decode keeps its
// separate Planner representation; node creation/link semantics are identical.
#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace dism_prefill {
class PrefillMatcher {
  public:
    struct Node { int link=-1, length=0, position=-1, first=-1; };
    struct Edge { int node, symbol, target, next; };
    std::vector<Node> nodes;
    std::vector<Edge> edges;
    // Stable root-component ids; cloning never changes existing membership.
    std::vector<int> branch, branch_roots, key_head, key_tail, key_next;
  private:
    std::vector<int> table;
    size_t mask=0;
    size_t hash(int node,int symbol) const {
        uint64_t x=(uint64_t(uint32_t(node))<<32)|uint32_t(symbol);
        x^=x>>33; x*=UINT64_C(0xff51afd7ed558ccd); x^=x>>33;
        return size_t(x)&mask;
    }
    void insert(int node,int symbol,int target) {
        // A length-n SAM has fewer than 3n transitions. The table reserves at
        // least 8n entries, including copies created by cloning states.
        if (edges.size()+1>=table.size()/2)
            throw std::length_error("offline SAM transition bound exceeded");
        size_t slot=hash(node,symbol);
        while(table[slot]>=0)slot=(slot+1)&mask;
        table[slot]=int(edges.size());
        edges.push_back({node,symbol,target,nodes[node].first});
        nodes[node].first=int(edges.size())-1;
    }
  public:
    void reset(size_t n) {
        if (!n || n>size_t(std::numeric_limits<int>::max()/8))
            throw std::invalid_argument("offline SAM length exceeds index range");
        nodes.clear(); edges.clear();
        branch.clear();branch.reserve(2*n);branch.push_back(-1);
        branch_roots.clear();key_head.clear();key_tail.clear();
        key_next.clear();key_next.reserve(n);
        nodes.reserve(2*n); edges.reserve(3*n);
        size_t capacity=16;
        while(capacity<8*n)capacity*=2;
        table.assign(capacity,-1); mask=capacity-1;
        nodes.push_back({});
    }
    int find(int node,int symbol) const {
        int first=nodes[node].first;
        if(first<0)return -1;
        if(edges[first].symbol==symbol)return first;
        if(edges[first].next<0)return -1;
        size_t slot=hash(node,symbol);
        for(;;) {
            int index=table[slot];
            if(index<0)return -1;
            const auto& e=edges[index];
            if(e.node==node && e.symbol==symbol)return index;
            slot=(slot+1)&mask;
        }
    }
    void extend(int symbol,int position,int& last) {
        int current=int(nodes.size());nodes.push_back({});
        branch.push_back(-1);key_next.push_back(-1);
        nodes[current].length=nodes[last].length+1;
        nodes[current].position=position;
        int p=last, edge=-1;
        while(p>=0) {
            edge=find(p,symbol);
            if(edge>=0)break;
            insert(p,symbol,current);p=nodes[p].link;
        }
        if(p<0)nodes[current].link=0;
        else {
            int q=edges[edge].target;
            if(nodes[p].length+1==nodes[q].length)nodes[current].link=q;
            else {
                Node copy=nodes[q];copy.length=nodes[p].length+1;copy.position=-1;copy.first=-1;
                int clone=int(nodes.size());nodes.push_back(copy);
                int group=branch[q];branch.push_back(group);
                // A root-child clone becomes the new root of the same branch.
                // Existing nodes retain their ids; no subtree relabeling is needed.
                if (copy.link==0) branch_roots[group]=clone;
                for(int e=nodes[q].first;e>=0;) {
                    // insert can reallocate edges; retain values, never references.
                    Edge original=edges[e];insert(clone,original.symbol,original.target);e=original.next;
                }
                while(p>=0) {
                    int e=find(p,symbol);
                    if(e<0 || edges[e].target!=q)break;
                    edges[e].target=clone;p=nodes[p].link;
                }
                nodes[current].link=nodes[q].link=clone;
            }
        }
        int target=nodes[current].link;
        int group=target==0 ? int(branch_roots.size()) : branch[target];
        if (target==0) {
            branch_roots.push_back(current);
            key_head.push_back(-1);key_tail.push_back(-1);
        }
        branch[current]=group;
        if (key_tail[group]>=0) key_next[key_tail[group]]=position;
        else key_head[group]=position;
        key_tail[group]=position;
        last=current;
    }
};
} // namespace dism_prefill
