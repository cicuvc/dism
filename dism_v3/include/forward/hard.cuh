#pragma once
#include "summary/hard.cuh"
#include "forward/readout.cuh"

namespace DISM_VARIANT::dism_hard_forward {
using namespace dism_forward;
using IntScan = pscore::AltLayoutSplitScanBuffer<16,64,
    pscore::UnaryElement,pscore::HardMaxOp,pscore::I32x2>;

struct Shared {
    union Tiles {
        struct Prefetch { SQTile sq; } prefetch[1];
        struct KeyValue { SKTile sk; VTile v; } kv[STAGES];
    } tiles;
    struct Metadata {
        // Q labels live across all K tiles and must not alias the K ring.
        struct alignas(16) Prefetch { int labels[QROWS]; } prefetch[1];
        struct alignas(16) KeyValue { int labels[WarpKSize]; } kv[STAGES];
    } metadata;
    struct Mail { int2 value[4][32]; } mail[2];
    OutputTile<false> output[8];
};
inline constexpr size_t SharedBytes = 4*1024 + sizeof(Shared);

struct Task {
    // upload_records caches byte identities: make alignment padding explicit
    // and initialize it, otherwise repeated equal tasks can miss the cache.
    int batch, head, begin, n, q_start, reserved;
    int64_t metadata, boundary, hard_index;
};
static_assert(sizeof(Task)==48 && offsetof(Task,metadata)==24);
struct Args {
    int batch, heads, n, task_count, lut_stride;
    kt::gl<kt::bf16,-1,-1,-1,READOUT_DIM,SQTile> sq;
    kt::gl<kt::bf16,-1,-1,-1,READOUT_DIM,SKTile> sk;
    kt::gl<kt::bf16,-1,-1,-1,DIM,VTile> v;
    const int *query_label, *key_label, *boundary;
    const float *log_weights;
    const bool *hard;
    const Task *tasks;
    float *lse2;
    OutputGlobal<false> output_map;
};

struct Scheduler {
    const Args& args;
    int current;
    __device__ Scheduler(const Args& a) : args(a), current(blockIdx.x) {}
    __device__ __forceinline__ bool next(Task& task) {
        if (current >= args.task_count) return false;
        if (args.tasks) task = args.tasks[current];
        else {
            int blocks = args.n / QROWS;
            int bh = current / blocks;
            task = {bh / args.heads, bh % args.heads, 0, args.n,
                    (current % blocks)*QROWS, 0, int64_t(bh)*args.n,
                    int64_t(bh)*((args.n-1)/32)*args.n, bh};
        }
        current += gridDim.x;
        return true;
    }
};

template<class Input, class Prefetch>
__device__ __forceinline__ void produce(const Args& args, Shared& shared,
        Input& input, Prefetch& prefetch, Scheduler& scheduler) {
    kt::warpgroup::decrease_registers<40>();
    if (kt::warpgroup::warpid() != 0) return;
    bool leader = kt::warp::elect_leader();
    Task task;
    while (scheduler.next(task)) {
        if (!args.hard[task.hard_index]) continue;
        prefetch.setup();
        auto first = prefetch.waitBuffer(0,shared.tiles.prefetch,shared.metadata.prefetch);
        if (leader) {
            kt::tma::expect_bytes(first.getBarrier(),sizeof(SQTile));
            kt::tma::load_async(first.template get<0>().sq,args.sq,
                {task.batch,task.head,task.begin+task.q_start,0},first.getBarrier());
        }
        load_async_any(first.template get<1>(),reinterpret_cast<uint8_t*>(
            const_cast<int*>(args.query_label+task.metadata+task.q_start)));
        // Same correctness-first label prefetch completion as hard summary.
        // SQ's TMA transfer remains tracked by the prefetch mbarrier.
        kt::warp::load_async_commit_group();
        kt::warp::load_async_wait<0>();
        first.submitToNextAndTrigger();
        prefetch.moveNext();
        prefetch.waitSlot(0);
        input.setup();
        int last = 0;
        for (int tile=0; tile<(task.q_start+QROWS)/WarpKSize; ++tile) {
            auto packet = input.waitBuffer(0,shared.tiles.kv,shared.metadata.kv);
            auto& kv = packet.template get<0>();
            if (leader) {
                kt::tma::expect_bytes(packet.getBarrier(),sizeof(SKTile)+sizeof(VTile));
                kt::tma::load_async(kv.sk,args.sk,
                    {task.batch,task.begin+tile*WarpKSize,task.head,0},packet.getBarrier());
                kt::tma::load_async(kv.v,args.v,
                    {task.batch,task.begin+tile*WarpKSize,task.head,0},packet.getBarrier());
            }
            load_async_any(packet.template get<1>(),reinterpret_cast<uint8_t*>(
                const_cast<int*>(args.key_label+task.metadata+tile*WarpKSize)));
            kt::warp::load_async_commit_group(packet.getBarrier());
            last = packet.SlotIdx;
            packet.submitToNextAndTrigger();
            input.moveNext();
        }
        // Also protects the persistent shared Q labels from next-task writes.
        input.waitSlot(last);
    }
}

__device__ __forceinline__ IntScan make_scan(const int* q,const int* k,int key_start) {
    IntScan scan;
    int lane = kt::warp::laneid();
#pragma unroll
    for (int r=0;r<IntScan::ROW_BLOCKS;++r) {
#pragma unroll
        for (int c=0;c<IntScan::COL_BLOCKS;++c) {
            int row = IntScan::physical_row(r,c,lane>>2);
            int col = c+2*(lane&3)*IntScan::COL_BLOCKS;
            int label = q[row];
            scan.data[r][c].value = {label==k[col] ? -1 : key_start+col,
                label==k[col+IntScan::COL_BLOCKS] ? -1 : key_start+col+IntScan::COL_BLOCKS};
        }
    }
    return scan;
}

__device__ __forceinline__ Scalar finish_scan(IntScan& scan,
        const IntScan::HState& incoming,const IntScan::ImmState& intermediate,
        const float* lut,int q_start,int key_start) {
    scan.inclusive_postscan(incoming,intermediate);
    // Unroll only the scalar integer state using the existing permutation.
    scan.roll<false>();
    Scalar values;
#pragma unroll
    for (int r=0;r<IntScan::ROW_BLOCKS;++r) {
#pragma unroll
        for (int c=0;c<IntScan::COL_BLOCKS;++c) {
            auto pos = Scalar::layout(r,c,0);
            auto state = scan.data[r][c].value;
            int j = key_start+pos.second;
            // Noncausal incoming checkpoints are intentionally identities.
            // Never use their spurious chain lengths to index the LUT.
            values.data[r][c].value = {
                j<=q_start+pos.first ? lut[j-state.u0] : LOG_ZERO,
                j+IntScan::COL_BLOCKS<=q_start+pos.first ?
                    lut[j+IntScan::COL_BLOCKS-state.u1] : LOG_ZERO};
        }
    }
    return values;
}
} // namespace DISM_VARIANT::dism_hard_forward
