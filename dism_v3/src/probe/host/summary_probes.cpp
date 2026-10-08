#define DISM_HOST_API 1
#include "host/torch.cuh"
#include "summary/primitives.cuh"
#include "summary/key_metadata.cuh"

#include "variant.cuh"

namespace DISM_VARIANT {
const void* key_metadata_probe_kernel_address0();
const void* lse_probe_kernel_address0();
const void* rv_load_probe_kernel_address0();
const void* rv_load_probe_kernel_address1();
const void* scan_probe_kernel_address0();
const void* pipe_probe_kernel_address0();
int summary_allocator_mode_cuda() { return 0; }

std::tuple<at::Tensor, at::Tensor> key_metadata_probe_cuda(
    at::Tensor labels, at::Tensor bias, bool query_lse) {
    TORCH_CHECK(labels.is_cuda() && labels.scalar_type() == at::kInt &&
                labels.is_contiguous() && labels.numel() == 64,
                "expected CUDA int32 [64] labels");
    TORCH_CHECK(bias.device() == labels.device() && bias.scalar_type() == at::kFloat &&
                bias.is_contiguous() && bias.numel() == 64,
                "expected same-device FP32 [64] bias");
    c10::cuda::CUDAGuard guard(labels.device());
    auto output_labels = at::empty({32, 4, 2, 2}, labels.options());
    auto output_bias = at::empty({32, 4, 2, 2}, bias.options());
    using IGlobal = decltype(TmaSummarizationKernelArgs::IdxK);
    using LGlobal = decltype(TmaSummarizationKernelArgs::KLseVec);
    launch_kernel(key_metadata_probe_kernel_address0(),1, 32, 0, at::cuda::getCurrentCUDAStream(),
        IGlobal(labels.data_ptr<int>(), 1, 1, 1, 64),
        LGlobal(bias.data_ptr<float>(), 1, 1, 1, 64),
        output_labels.data_ptr<int>(), output_bias.data_ptr<float>(), query_lse);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {output_labels, output_bias};
}

int summary_lse_mode_cuda() { return 0; }

int summary_metadata_mode_cuda() { return 0; }

int summary_k_stages_cuda() { return CONFIG.KStages; }

int summary_shared_bytes_cuda() { return SUMMARY_SHARED_BYTES; }

at::Tensor lse_probe_cuda(at::Tensor inputs) {
    TORCH_CHECK(inputs.is_cuda() && inputs.scalar_type() == at::kFloat &&
                inputs.is_contiguous() && inputs.dim() == 2 && inputs.size(1) == 2 &&
                inputs.size(0) > 0, "expected nonempty CUDA FP32 [N,2]");
    c10::cuda::CUDAGuard guard(inputs.device());
    auto output = at::empty({inputs.size(0)}, inputs.options());
    launch_kernel(lse_probe_kernel_address0(),(inputs.size(0) + 127) / 128, 128, 0,
                       at::cuda::getCurrentCUDAStream(),
        reinterpret_cast<const float2*>(inputs.data_ptr<float>()),
        output.data_ptr<float>(), inputs.size(0));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

at::Tensor rv_load_probe_cuda(at::Tensor source, bool direct) {
    TORCH_CHECK(source.is_cuda() && source.scalar_type() == at::kFloat &&
                source.is_contiguous() && source.numel() == 32, "expected CUDA FP32 [32]");
    c10::cuda::CUDAGuard guard(source.device());
    auto output = at::empty_like(source);
    auto stream = at::cuda::getCurrentCUDAStream();
    if (direct) launch_kernel(rv_load_probe_kernel_address0(),1, 32, 0, stream,source.data_ptr<float>(), output.data_ptr<float>());
    else launch_kernel(rv_load_probe_kernel_address1(),1, 32, 0, stream,source.data_ptr<float>(), output.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

at::Tensor scan_probe_cuda(at::Tensor values) {
    TORCH_CHECK(values.is_cuda() && values.scalar_type() == at::kFloat && values.is_contiguous() &&
                    values.dim() == 2 && values.size(0) == 32 && values.size(1) > 0 &&
                    values.size(1) % 64 == 0,
                "expected FP32 [32,K], K%64=0");
    c10::cuda::CUDAGuard guard(values.device());
    auto result = at::empty({values.size(1), 2}, values.options());
    auto stream = at::cuda::getCurrentCUDAStream();
    launch_kernel(scan_probe_kernel_address0(),1, 32, 0, stream,
            values.data_ptr<float>(), reinterpret_cast<float2 *>(result.data_ptr<float>()),
            values.size(1));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return result;
}

at::Tensor pipe_probe_cuda(at::Tensor anchor, int64_t tasks) {
    TORCH_CHECK(anchor.is_cuda() && tasks > 0 && tasks <= 10000, "invalid probe arguments");
    c10::cuda::CUDAGuard guard(anchor.device());
    auto output = at::full({tasks, 8, 7}, -2, anchor.options().dtype(at::kInt));
    auto source = at::arange(tasks * 7 * 4, anchor.options().dtype(at::kInt));
    launch_kernel(pipe_probe_kernel_address0(),1, 384, 1024, at::cuda::getCurrentCUDAStream(),
        source.data_ptr<int>(), output.data_ptr<int>(), tasks);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}


} // namespace DISM_VARIANT
