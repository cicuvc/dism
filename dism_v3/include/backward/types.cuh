#pragma once
#include "backward/recompute.cuh"
#include "backward/config.cuh"

#include "variant.cuh"
namespace DISM_VARIANT {

namespace dism_backward {
#ifndef DISM_BACKWARD_DEBUG
#define DISM_BACKWARD_DEBUG 0
#endif
using KeyOutputGlobal = kt::gl<kt::bf16,-1,-1,-1,-1>;
constexpr int InputStages = D==128 ? 1 : 2;
using GradientTile = kt::st_fl<16,16>;
using GradientVector = kt::sv_fl<32>;
// Descriptor-only view: TK owns map encoding and device store primitives.
struct OutputMap {
    using identifier = kt::ducks::gl::identifier;
    CUtensorMap descriptor;
    template<class Tile>
    __device__ const CUtensorMap* get_tma() const { return &descriptor; }
};
using SoftQueryTile = KPermutationSharedBuffer<kt::bf16,Q,R>;
using DerivativeTile = KPermutationSharedBuffer<kt::bf16,Q,DV>;
using HeldKeyTile = kt::st_bf<128,D>;
using HeldSoftTile = kt::st_bf<128,R>;
using HeldValueTile = kt::st_bf<128,DV>;
using SoftQueryGlobal = kt::gl<kt::bf16,-1,-1,-1,R,SoftQueryTile>;
using DerivativeGlobal = kt::gl<kt::bf16,-1,-1,-1,DV,DerivativeTile>;
using HeldKeyGlobal = kt::gl<kt::bf16,-1,-1,-1,D,HeldKeyTile>;
using HeldSoftGlobal = kt::gl<kt::bf16,-1,-1,-1,R,HeldSoftTile>;
using HeldValueGlobal = kt::gl<kt::bf16,-1,-1,-1,DV,HeldValueTile>;

struct Args {
    RecomputeArgs score;
    SoftQueryGlobal sq;
    DerivativeGlobal dout;
    HeldKeyGlobal k;
    HeldSoftGlobal sk;
    HeldValueGlobal v;
    const float *normalizer, *delta, *g_boundary;
    void *dv;
    float *dsq;
    void *dsk;
    float *dq;
    void *dk;
    float *dlq;
    void *dlk;
    float *dtau;
    float *summary_a, *summary_b;
    // Dense diagnostics are absent from production device code and arguments.
#if DISM_BACKWARD_DEBUG
    float *debug_ca=nullptr, *debug_cb=nullptr, *debug_g=nullptr;
#endif
    KeyOutputGlobal key_output[3] = {
        {nullptr,0,0,0,0}, {nullptr,0,0,0,0}, {nullptr,0,0,0,0}};
    OutputMap query_output;
    OutputMap lse_output;
};

struct Shared {
    struct Held {
        HeldKeyTile k;
        HeldSoftTile sk;
        HeldValueTile v;
    } held[1];
    struct Input {
        QueryTile q;
        SoftQueryTile sq;
        DerivativeTile dout;
    } input[InputStages];
    // Reverse affine boundary exchange between the two compute warpgroups.
    struct Mail { float4 pairs[4][32]; } mail[2];
    // Warp-private GEMM transpose staging, reused for each query gradient.
    kt::st_bf<K,Q> transpose[8];
    // Optional B3 row-LSE partials, reduced by one owning warp after the
    // existing consumer rendezvous. No shared-memory atomic operations.
    float row_lse[8][Q];
    // Output-layout staging: warp-private query slots and warp0's LSE slot.
    // Read-wait before reuse, full-wait before kernel exit.
    GradientTile query_output[8];
    GradientVector lse_output;
};
constexpr size_t SharedBytes = 4*1024 + sizeof(Shared);

struct Scheduler {
    struct Task { int batch, head, bh, k0; };
    int index, blocks, heads, total;
    __device__ Scheduler(const Args& a)
        : index(blockIdx.x),blocks((a.score.n+127)/128),heads(a.score.heads),
          total(blocks*heads*a.score.batch) {}
    __device__ __forceinline__ bool next(Task& task) {
        if (index >= total) return false;
        task.bh = index/blocks;
        task.head = task.bh%heads;
        task.batch = task.bh/heads;
        task.k0 = (index%blocks)*128;
        index += gridDim.x;
        return true;
    }
};

template<class HeldPipe,class InputPipe>
__device__ __forceinline__ void produce(const Args& a,Shared& shared,
        HeldPipe& held,InputPipe& input,Scheduler& scheduler) {
    kt::warpgroup::decrease_registers<40>();
    if (kt::warpgroup::warpid()!=0) return;
    bool leader = kt::warp::elect_leader();
    Scheduler::Task task;
#pragma unroll 1
    while (scheduler.next(task)) {
        auto first = held.waitBuffer(0,shared.held);
        if (leader) {
            auto& data = first.template get<0>();
            kt::tma::expect_bytes(first.getBarrier(),sizeof(Shared::Held));
            kt::tma::load_async(data.k,a.k,{task.batch,task.head,task.k0,0},first.getBarrier());
            kt::tma::load_async(data.sk,a.sk,{task.batch,task.head,task.k0,0},first.getBarrier());
            kt::tma::load_async(data.v,a.v,{task.batch,task.head,task.k0,0},first.getBarrier());
        }
        first.submitToNextAndTrigger();
        held.moveNext();
        // The independent held slot may prefetch the next workload while the
        // current consumers finish their final reductions and output stores.
#pragma unroll 1
        for (int q0=((a.score.n+Q-1)/Q-1)*Q;q0>=task.k0;q0-=Q) {
            auto packet = input.waitBuffer(0,shared.input);
            if (leader) {
                auto& data = packet.template get<0>();
                kt::tma::expect_bytes(packet.getBarrier(),sizeof(Shared::Input));
                kt::tma::load_async(data.q,a.score.q,{task.batch,q0,task.head,0},packet.getBarrier());
                kt::tma::load_async(data.sq,a.sq,{task.batch,q0,task.head,0},packet.getBarrier());
                kt::tma::load_async(data.dout,a.dout,{task.batch,q0,task.head,0},packet.getBarrier());
            }
            packet.submitToNextAndTrigger();
            input.moveNext();
        }
    }
}
} // namespace dism_backward

} // namespace DISM_VARIANT
