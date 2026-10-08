#define DISM_HOST_API 1
#include "host/torch.cuh"
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <climits>
#include <kittens.cuh>

#include "variant.cuh"

namespace DISM_VARIANT {
namespace {
constexpr int WORKLOAD_ROWS=256,CHECKPOINT_ROWS=32;
using Global=kittens::gl<float,-1,-1,-1,-1>;
}
const void* chunk_scan_address_0();
at::Tensor chunk_scan_cuda(at::Tensor summary_a, at::Tensor summary_b, int64_t seqlen) {
    TORCH_CHECK(seqlen>0 && seqlen%256==0,"sequence length must be256-token aligned");
    TORCH_CHECK(summary_a.is_cuda() && summary_a.scalar_type() == at::kFloat &&
                    summary_a.dim() == 4 && summary_a.is_contiguous(),
                "summary_a must be contiguous CUDA FP32 [B,H,S,padded_N]");
    TORCH_CHECK(summary_b.device() == summary_a.device() &&
                    summary_b.scalar_type() == at::kFloat && summary_b.is_contiguous() &&
                    summary_b.sizes() == summary_a.sizes(),
                "summary_b must have the same device, dtype, shape and contiguous layout");
    int64_t batch = summary_a.size(0), heads = summary_a.size(1);
    int64_t checkpoints = summary_a.size(2), padded_n = summary_a.size(3);
    TORCH_CHECK(batch > 0 && heads > 0 && batch * heads <= 65535,
                "invalid batch/head grid size");
    TORCH_CHECK(seqlen > 0 && seqlen == padded_n && padded_n % WORKLOAD_ROWS == 0 &&
                    checkpoints == (seqlen - 1) / CHECKPOINT_ROWS,
                "invalid sequence length or summary checkpoint/padding shape");
    TORCH_CHECK(padded_n + checkpoints * CHECKPOINT_ROWS <= INT_MAX,
                "sequence dimensions exceed kernel indexing range");
    c10::cuda::CUDAGuard guard(summary_a.device());
    auto boundary = at::empty_like(summary_b);
    if (checkpoints == 0) return boundary;

    constexpr int THREADS = 128;
    int64_t diagonals = padded_n + (checkpoints - 1) * CHECKPOINT_ROWS;
    dim3 grid((diagonals + THREADS - 1) / THREADS, batch * heads);
    Global a(summary_a.data_ptr<float>(), batch * heads, 1, checkpoints, padded_n);
    Global b(summary_b.data_ptr<float>(), batch * heads, 1, checkpoints, padded_n);
    launch_kernel(chunk_scan_address_0(),grid, THREADS, 0, at::cuda::getCurrentCUDAStream(),
        a, b, boundary.data_ptr<float>(), int(checkpoints), int(padded_n));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return boundary;
}

} // namespace DISM_VARIANT
