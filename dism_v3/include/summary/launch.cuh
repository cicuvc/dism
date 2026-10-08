#pragma once

#include <tuple>

#include "summary/primitives.cuh"

#include "variant.cuh"
namespace DISM_VARIANT {

// Low-level padded interface; shape/device checks live here, packing in Python.
// Retain tensors at caller scope through launch, on the current PyTorch stream.
template <class Kernel>
std::tuple<at::Tensor, at::Tensor> launch_summary(Kernel kernel, at::Tensor query, at::Tensor key, at::Tensor q_lse,
                          at::Tensor k_lse, at::Tensor q_labels, at::Tensor k_labels,
                          at::Tensor hard, at::Tensor direction, at::Tensor tau, int64_t seqlen,
                          int64_t requested_ctas) {
    TORCH_CHECK(query.is_cuda() && query.scalar_type() == at::kBFloat16, "query must be CUDA BF16");
    c10::cuda::CUDAGuard guard(query.device());
    TORCH_CHECK(query.dim() == 4 && query.size(3) == CONFIG.KcKeyDim,
                "expected [B,padded_N,H,", CONFIG.KcKeyDim, "] for this build");
    int batch = query.size(0), padded = query.size(1), heads = query.size(2);
    TORCH_CHECK(batch > 0 && heads > 0 && seqlen > 0 && seqlen <= padded &&
                    padded % CONFIG.getQBlockSize() == 0,
                "invalid padded length");
    auto check = [&](const at::Tensor &tensor, at::ScalarType type, at::IntArrayRef shape,
                     const char *name) {
        TORCH_CHECK(tensor.device() == query.device() && tensor.scalar_type() == type &&
                        tensor.is_contiguous() && tensor.sizes() == shape,
                    name, ": invalid device, dtype, shape or stride");
    };
    check(query, at::kBFloat16, {batch, padded, heads, CONFIG.KcKeyDim}, "query");
    check(key, at::kBFloat16, query.sizes(), "key");
    check(q_lse, at::kFloat, {batch, heads, padded}, "q_lse");
    check(k_lse, at::kFloat, q_lse.sizes(), "k_lse");
    check(q_labels, at::kInt, q_lse.sizes(), "q_labels");
    check(k_labels, at::kInt, q_lse.sizes(), "k_labels");
    check(hard, at::kByte, q_lse.sizes(), "hard");
    check(direction, at::kByte, {batch, heads}, "direction");
    check(tau, at::kFloat, {heads}, "tau");
    // Omit only the32-row warp block containing the final token, not its CTA.
    int checkpoints = (seqlen - 1) / CONFIG.WarpQSize;
    constexpr int warps_per_workload = CONFIG.getQBlockSize() / CONFIG.WarpQSize;
    int summary_workloads = (checkpoints + warps_per_workload - 1) / warps_per_workload;
    auto summary_a =
        at::empty({batch, heads, checkpoints, padded}, query.options().dtype(at::kFloat));
    auto summary_b = at::empty_like(summary_a);
    if (summary_workloads == 0) return {summary_a, summary_b};
    using QGlobal = decltype(TmaSummarizationKernelArgs::QVec);
    using KGlobal = decltype(TmaSummarizationKernelArgs::KVec);
    using LGlobal = decltype(TmaSummarizationKernelArgs::QLseVec);
    using IGlobal = decltype(TmaSummarizationKernelArgs::IdxQ);
    // Q descriptor sees B,H,N,D with physical B,N,H,D strides.
    QGlobal q_global(reinterpret_cast<kt::bf16 *>(query.data_ptr()), batch, heads, padded, CONFIG.KcKeyDim,
                     kt::gl_strides{size_t(padded * heads * CONFIG.KcKeyDim),
                                    size_t(CONFIG.KcKeyDim), size_t(heads * CONFIG.KcKeyDim)});
    KGlobal k_global(reinterpret_cast<kt::bf16 *>(key.data_ptr()), batch, padded, heads, CONFIG.KcKeyDim);
    TmaSummarizationKernelArgs args{batch,
                                    int(seqlen),
                                    heads,
                                    padded,
                                    checkpoints,
                                    q_global,
                                    k_global,
                                    LGlobal(q_lse.data_ptr<float>(), batch, heads, 1, padded),
                                    LGlobal(k_lse.data_ptr<float>(), batch, heads, 1, padded),
                                    IGlobal(q_labels.data_ptr<int>(), batch, heads, 1, padded),
                                    IGlobal(k_labels.data_ptr<int>(), batch, heads, 1, padded),
                                    hard.data_ptr<uint8_t>(),
                                    direction.data_ptr<uint8_t>(),
                                    tau.data_ptr<float>(),
                                    summary_a.data_ptr<float>(),
                                    summary_b.data_ptr<float>()};
    // Two pipe allocations, each rounded to1024, then the1024-aligned union.
    constexpr size_t shared_bytes = SUMMARY_SHARED_BYTES;
    int tasks = batch * heads * summary_workloads;
    int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    int ctas = std::min(tasks, requested_ctas > 0 ? int(requested_ctas) : sms);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    C10_CUDA_CHECK(
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes));
    launch_kernel(kernel,ctas, 384, shared_bytes, stream,args);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {summary_a, summary_b};
}

} // namespace DISM_VARIANT
