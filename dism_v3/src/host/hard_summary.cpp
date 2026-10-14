#define DISM_HOST_API 1
#include "host/torch.cuh"
#include "summary/hard.cuh"
#include "varlen/layout.cuh"

namespace DISM_VARIANT {
const void* hard_summary_address();

// Explicit standalone prototype. No production dispatch, tau or readout.
at::Tensor hard_summary_cuda(at::Tensor query, at::Tensor key, at::Tensor hard,
                            std::optional<at::Tensor> table, int64_t requested_ctas) {
    TORCH_CHECK(query.is_cuda() && query.scalar_type() == at::kInt && query.dim() == 3 &&
                query.is_contiguous(), "query labels must be contiguous CUDA int32 [B,H,N]");
    c10::cuda::CUDAGuard guard(query.device());
    int64_t b = query.size(0), h = query.size(1), n = query.size(2);
    TORCH_CHECK(b > 0 && h > 0 && b * h <= INT_MAX && n > 0 && n <= INT_MAX - 255 && n % 256 == 0,
                "positive dimensions and 256-aligned sequence length required");
    TORCH_CHECK(key.device() == query.device() && key.scalar_type() == at::kInt &&
                key.sizes() == query.sizes() && key.is_contiguous(), "invalid key labels");
    TORCH_CHECK(hard.device() == query.device() && hard.scalar_type() == at::kBool &&
                hard.dim() == 2 && hard.size(0) > 0 && hard.size(1) == h && hard.is_contiguous(),
                "hard must be contiguous bool [training_rows,H]; token-level masks are unsupported");
    TORCH_CHECK(requested_ctas >= 0 && requested_ctas <= INT_MAX, "invalid CTA count");
    auto options = query.options();
    at::Tensor output, task_storage;
    int64_t tasks = b * h * (n / 256);
    if (table) {
        using namespace dism_varlen;
        Layout layout(*table);
        TORCH_CHECK(b == 1 && layout.tokens == n, "packed labels/layout length mismatch");
        TORCH_CHECK(n % hard.size(0) == 0 && (n / hard.size(0)) % 256 == 0,
                    "packed hard rows must evenly divide tokens into 256-aligned training rows");
        int64_t row_tokens = n / hard.size(0);
        std::vector<HardSummaryTask> records;
        for (int s = 0; s < layout.sequences; ++s) {
            int64_t begin = layout.get(s, Begin), length = layout.get(s, Length);
            if (!length) continue;
            TORCH_CHECK(begin / row_tokens == (begin + length - 1) / row_tokens,
                        "document cannot cross a hard training-row boundary");
            for (int head = 0; head < h; ++head)
                for (int q = 0; q < length; q += 256)
                    records.push_back({head * n + begin,
                        layout.get(s, ForwardOffset) * h + head * ((length - 1) / 32) * length,
                        (begin / row_tokens) * h + head, int(length), q});
        }
        tasks = records.size();
        task_storage = upload_records(records, query.device());
        output = at::full({layout.forward * h}, -1, options);
    } else {
        TORCH_CHECK(hard.size(0) == b, "fixed hard must be [B,H]");
        output = at::full({b, h, (n - 1) / 32, n}, -1, options);
    }
    TORCH_CHECK(tasks > 0 && tasks <= INT_MAX, "invalid summary task count");
    HardSummaryArgs args{query.data_ptr<int>(), key.data_ptr<int>(), hard.data_ptr<bool>(),
        output.data_ptr<int>(), table ? reinterpret_cast<const HardSummaryTask*>(task_storage.data_ptr()) : nullptr,
        int(b), int(h), int(n), int(tasks)};
    int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    int ctas = std::min(int(tasks), requested_ctas ? int(requested_ctas) : sms);
    launch_kernel(hard_summary_address(), ctas, 384, HARD_SUMMARY_SHARED_BYTES,
                  at::cuda::getCurrentCUDAStream(), args);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
} // namespace DISM_VARIANT
