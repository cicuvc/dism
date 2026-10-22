#define DISM_HOST_API 1
#include "host/torch.cuh"
#include "summary/hard_chunk.cuh"
#include "varlen/layout.cuh"

namespace DISM_VARIANT {
const void* hard_chunk_address();

at::Tensor hard_chunk_cuda(at::Tensor summary, std::optional<at::Tensor> table, int64_t heads) {
    TORCH_CHECK(summary.is_cuda() && summary.scalar_type() == at::kInt && summary.is_contiguous(),
                "hard summary must be contiguous CUDA int32");
    c10::cuda::CUDAGuard guard(summary.device());
    at::Tensor task_storage;
    dim3 grid;
    int n = 0;
    if (table) {
        using namespace dism_varlen;
        Layout layout(*table);
        TORCH_CHECK(heads > 0 && heads <= 65535 && summary.dim() == 1 &&
                    summary.numel() == layout.forward * heads, "invalid packed hard summary/heads");
        auto build=[heads](const at::Tensor& table) {
        Layout layout(table);
        std::vector<HardChunkTask> records;
        for (int s = 0; s < layout.sequences; ++s) {
            int64_t length = layout.get(s, Length);
            if (!length) continue;
            int64_t checkpoints = (length - 1) / 32;
            TORCH_CHECK(length + checkpoints * 32 <= INT_MAX, "sequence dimensions exceed int32 indexing");
            int blocks = (length + (checkpoints - 1) * 32 + 127) / 128;
            for (int h = 0; h < heads; ++h)
                for (int block = 0; block < blocks; ++block)
                    records.push_back({layout.get(s, ForwardOffset) * heads + h * checkpoints * length,
                                       int(length), block});
        }
        return records;
        };
        auto records=build(*table);
        TORCH_CHECK(records.size() <= INT_MAX, "too many packed chunk tasks");
        if (records.empty()) return at::empty_like(summary);
        task_storage = dism_graph::state(*table) ? dism_graph::records(*table,
            "hard_chunk:"+std::to_string(heads),summary.device(),
            [build](const at::Tensor& table) { return dism_graph::bytes(build(table)); })
            : upload_records(records, summary.device());
        grid = dim3(records.size());
    } else {
        TORCH_CHECK(summary.dim() == 4, "fixed hard summary must be [B,H,C,N]");
        int64_t batch_heads = summary.size(0) * summary.size(1), length = summary.size(3);
        TORCH_CHECK(batch_heads > 0 && batch_heads <= 65535 &&
                    (heads == 0 || heads == summary.size(1)), "invalid fixed batch/head count");
        TORCH_CHECK(length > 0 && length <= INT_MAX - 255 && length % 256 == 0 &&
                    summary.size(2) == (length - 1) / 32 &&
                    length + summary.size(2) * 32 <= INT_MAX, "invalid hard checkpoint shape/alignment");
        n = length;
        grid = dim3((n + (summary.size(2) - 1) * 32 + 127) / 128, batch_heads);
    }
    auto output = at::empty_like(summary);
    HardChunkArgs args{summary.data_ptr<int>(), output.data_ptr<int>(),
        table ? reinterpret_cast<const HardChunkTask*>(task_storage.data_ptr()) : nullptr, n};
    launch_kernel(hard_chunk_address(), grid, 128, 0, at::cuda::getCurrentCUDAStream(), args);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
} // namespace DISM_VARIANT
