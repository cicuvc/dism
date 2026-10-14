#pragma once

#include "summary/primitives.cuh"

// Integer payloads adapt the existing scan without changing its placement or
// dependency graph. Never reinterpret the soft affine checkpoint buffers.
namespace pscore {
struct I32x1 {
    int u0;
    __host__ __device__ I32x1() = default;
    __host__ __device__ I32x1(int x) : u0(x) {}
    __device__ __forceinline__ I32x1 shuffle(int lane) const {
        return {__shfl_sync(0xffffffff, u0, lane)};
    }
    __device__ __forceinline__ I32x1 extract(bool) const { return *this; }
    __device__ __forceinline__ void insert(bool, I32x1 x) { u0 = x.u0; }
    __device__ __forceinline__ static I32x1 select(unsigned p, I32x1 a, I32x1 b) {
        return p ? b : a;
    }
    __host__ __device__ static I32x1 maximum(I32x1 a, I32x1 b) {
        return {a.u0 > b.u0 ? a.u0 : b.u0};
    }
};
struct I32x2 {
    int u0, u1;
    __host__ __device__ I32x2() = default;
    __host__ __device__ I32x2(int x) : u0(x), u1(x) {}
    __host__ __device__ I32x2(int x, int y) : u0(x), u1(y) {}
    __device__ __forceinline__ I32x2 shuffle(int lane) const {
        return {__shfl_sync(0xffffffff, u0, lane), __shfl_sync(0xffffffff, u1, lane)};
    }
    __device__ __forceinline__ I32x1 extract(bool half) const { return {half ? u1 : u0}; }
    __device__ __forceinline__ void insert(bool half, I32x1 x) {
        if (half) u1 = x.u0; else u0 = x.u0;
    }
    __device__ __forceinline__ static I32x2 select(unsigned p, I32x2 a, I32x2 b) {
        return p ? b : a;
    }
    __host__ __device__ static I32x2 maximum(I32x2 a, I32x2 b) {
        return {a.u0 > b.u0 ? a.u0 : b.u0, a.u1 > b.u1 ? a.u1 : b.u1};
    }
};
template<> struct PackTraits<I32x2> {
    using unpacked_type = I32x1;
    using scalar_type = int;
    static constexpr int lanes = 2;
};
struct HardMaxOp {
    template<class T>
    __host__ __device__ static UnaryElement<T> identity() { return {T{-1}}; }
    template<class T>
    __host__ __device__ __forceinline__ static UnaryElement<T>
    apply(const UnaryElement<T>& a, const UnaryElement<T>& b) {
        return {T::maximum(a.value, b.value)};
    }
};
} // namespace pscore

namespace DISM_VARIANT {
using HardSummaryScan = pscore::AltLayoutSplitScanBuffer<32, 64,
    pscore::UnaryElement, pscore::HardMaxOp, pscore::I32x2>;

struct HardSummaryTask {
    int64_t metadata, output, hard_index;
    int n, q_start;
};
struct HardSummaryArgs {
    const int *query, *key;
    const bool *hard;
    int *summary;
    const HardSummaryTask *tasks; // packed only; fixed reuses FixedLengthScheduler
    int batch, heads, seqlen, task_count;
};
struct HardSummaryShared {
    struct alignas(16) Key { int labels[64]; } keys[3];
    struct alignas(16) Query { int labels[256]; } prefetch[1];
    // Q remains shared across K tiles. Unlike soft Q, it is not copied to an
    // MMA register tile, so it must NOT alias the K ring. The producer's final
    // slot drain protects both buffers before the next task overwrites Q.
};
inline constexpr size_t HARD_SUMMARY_SHARED_BYTES = 4 * 1024 + sizeof(HardSummaryShared);

struct HardSummaryScheduler {
    const HardSummaryArgs& args;
    FixedLengthScheduler fixed;
    int current;
    __device__ HardSummaryScheduler(const HardSummaryArgs& a)
        : args(a), fixed(a.batch, a.seqlen, a.heads), current(blockIdx.x) {}
    __device__ __forceinline__ bool next(HardSummaryTask& out) {
        if (args.tasks) {
            if (current >= args.task_count) return false;
            out = args.tasks[current];
            current += gridDim.x;
            return true;
        }
        FixedLengthScheduler::TaskInfo task;
        if (!fixed.getNextTask(task)) return false;
        int64_t bh = int64_t(task.Batch) * args.heads + task.Head;
        out = {bh * args.seqlen, bh * ((args.seqlen - 1) / 32) * args.seqlen,
               bh, args.seqlen, task.QStart};
        return true;
    }
};

// Construct the already-rolled tile directly from shared labels. Both packed
// halves have the same physical row; only their key columns differ. No roll,
// MMA, floating point conversion, or per-position global label access.
__device__ __forceinline__ HardSummaryScan make_hard_summary_scan(
        const int* query, const int* key, int key_start) {
    HardSummaryScan scan;
    int lane = kt::warp::laneid();
#pragma unroll
    for (int r = 0; r < scan.ROW_BLOCKS; ++r) {
#pragma unroll
        for (int c = 0; c < scan.COL_BLOCKS; ++c) {
            int row = scan.physical_row(r, c, lane >> 2);
            int col = c + 2 * (lane & 3) * scan.COL_BLOCKS;
            int q = query[row];
            scan.data[r][c].value = {
                q == key[col] ? -1 : key_start + col,
                q == key[col + scan.COL_BLOCKS] ? -1 : key_start + col + scan.COL_BLOCKS};
        }
    }
    return scan;
}
} // namespace DISM_VARIANT
