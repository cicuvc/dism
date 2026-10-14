#pragma once
#include "backward/gradients.cuh"
#include "summary/hard.cuh"

namespace DISM_VARIANT::dism_hard_backward {
using namespace dism_backward;
struct Shape { int batch, heads, n; };
struct Args {
    Shape score;
    SoftQueryGlobal sq;
    DerivativeGlobal dout;
    HeldSoftGlobal sk;
    HeldValueGlobal v;
    const int *q_label, *k_label, *boundary;
    const bool *hard;
    const float *normalizer, *log_weights, *dlog_weights;
    float *moments;
    void *dv, *dsk;
    KeyOutputGlobal key_output[3]={{nullptr,0,0,0,0},{nullptr,0,0,0,0},{nullptr,0,0,0,0}};
    OutputMap query_output, lse_output;
};
struct Shared {
    struct Held { HeldSoftTile sk; HeldValueTile v; } held[1];
    struct Input { SoftQueryTile sq; DerivativeTile dout; } input[2];
    struct Labels {
        struct alignas(16) Held { int value[160]; } held[1];
        struct alignas(16) Input { int value[32]; } input[2];
    } labels;
    kt::st_bf<K,Q> transpose[8];
    float moments[8][4][Q]; // necessary inter-warp reduction, no shared atomics
    GradientTile query_output[8];
    GradientVector lse_output; // query_gradient's optional LSE branch is disabled.
};
constexpr size_t SharedBytes=4096+sizeof(Shared);
struct Scheduler {
    struct Task { int batch,head,bh,k0; };
    int index,blocks,heads,total;
    __device__ Scheduler(const Args& a):index(blockIdx.x),blocks(a.score.n/128),
        heads(a.score.heads),total(blocks*heads*a.score.batch) {}
    __device__ bool next(Task& t) {
        if(index>=total) return false;
        int bh=index/blocks;
        t={bh/heads,bh%heads,bh,(index%blocks)*128};
        index+=gridDim.x;
        return true;
    }
};
template<class HeldPipe,class InputPipe>
__device__ __forceinline__ void produce(const Args& a,Shared& s,HeldPipe& held,
        InputPipe& input,Scheduler& scheduler) {
    kt::warpgroup::decrease_registers<40>();
    if(kt::warpgroup::warpid()!=0) return;
    bool leader=kt::warp::elect_leader();
    int lane=kt::warp::laneid();
    Scheduler::Task t;
    while(scheduler.next(t)) {
        if(!a.hard[t.bh]) continue;
        auto first=held.waitBuffer(0,s.held,s.labels.held);
        if(leader) {
            kt::tma::expect_bytes(first.getBarrier(),sizeof(Shared::Held));
            kt::tma::load_async(first.template get<0>().sk,a.sk,{t.batch,t.head,t.k0,0},first.getBarrier());
            kt::tma::load_async(first.template get<0>().v,a.v,{t.batch,t.head,t.k0,0},first.getBarrier());
        }
        // Halo needed to reconstruct up to32 diagonal steps from a horizontal
        // checkpoint. Labels are loaded once per held key tile, never per pair.
        for(int j=lane;j<160;j+=32) {
            int key=t.k0+j-32;
            first.template get<1>().value[j]=key>=0?a.k_label[int64_t(t.bh)*a.score.n+key]:0;
        }
        __syncwarp();
        first.submitToNextAndTrigger(); held.moveNext();
        for(int q0=a.score.n-Q;q0>=t.k0;q0-=Q) {
            auto packet=input.waitBuffer(0,s.input,s.labels.input);
            if(leader) {
                kt::tma::expect_bytes(packet.getBarrier(),sizeof(Shared::Input));
                kt::tma::load_async(packet.template get<0>().sq,a.sq,{t.batch,q0,t.head,0},packet.getBarrier());
                kt::tma::load_async(packet.template get<0>().dout,a.dout,{t.batch,q0,t.head,0},packet.getBarrier());
            }
            packet.template get<1>().value[lane]=a.q_label[int64_t(t.bh)*a.score.n+q0+lane];
            __syncwarp();
            packet.submitToNextAndTrigger(); input.moveNext();
        }
    }
}

// Correctness-first bounded halo reconstruction. O(Q) with Q fixed at32;
// checkpoint stores the last mismatching KEY coordinate, not a length.
__device__ __forceinline__ int length(const Args& a,const int* q,const int* k,
        int bh,int tile_k,int key,int q0,int query) {
    if(key>query) return 0;
    int count=0,local=query-q0;
    while(count<=local && count<=key && q[local-count]==k[key-tile_k+32-count]) ++count;
    if(count==local+1 && key>=count && q0>0) {
        int previous_key=key-count;
        int64_t offset=(int64_t(bh)*((a.score.n-1)/32)+q0/32-1)*a.score.n+previous_key;
        count+=previous_key-a.boundary[offset];
    }
    return count;
}

using IntScan=pscore::AltLayoutSplitScanBuffer<K,Q,pscore::UnaryElement,pscore::HardMaxOp,pscore::I32x2>;
__device__ __forceinline__ IntScan recompute(const Args& a,const int* q,const int* k,
        int bh,int tile_k,int k0,int q0) {
    IntScan scan;
    int lane=kt::warp::laneid();
#pragma unroll
    for(int r=0;r<2;++r) {
#pragma unroll
        for(int c=0;c<4;++c) {
            int key=k0+IntScan::physical_row(r,c,lane>>2);
            int col=c+8*(lane&3);
            int label=k[key-tile_k+32];
            scan.data[r][c].value={label==q[col]?-1:key,label==q[col+4]?-1:key};
        }
    }
    IntScan::HState top;
    IntScan::VState left;
    // B1's top edge is a vertical key16 checkpoint. Reconstruct just this
    // edge from the saved horizontal query32 checkpoint and shared halo.
    if(k0>0 && lane/4<4) {
        int query=q0+8*(lane&3)+3-lane/4;
        top.init[0].value={
            k0-1-length(a,q,k,bh,tile_k,k0-1,q0,query),
            k0-1-length(a,q,k,bh,tile_k,k0-1,q0,query+4)};
    }
    if(q0>0 && (lane&3)==3) {
#pragma unroll
        for(int r=0;r<2;++r) {
            int key=k0+8*r+lane/4-1;
            if(key>=0 && key<q0)
                left.init[r].value.u0=a.boundary[(int64_t(bh)*((a.score.n-1)/32)+q0/32-1)*a.score.n+key];
        }
    }
    auto state=scan.inclusive_prescan(left,top);
    scan.inclusive_postscan(top,state.intermediate);
    scan.roll<false>();
    return scan;
}
} // namespace DISM_VARIANT::dism_hard_backward
