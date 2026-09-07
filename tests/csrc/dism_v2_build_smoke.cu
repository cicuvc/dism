// Build/ABI/current-stream probe only; not part of the attention implementation.
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cstdint>

__global__ void copy_bf16_bits(const uint16_t* input, uint16_t* output, int64_t n) {
    int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) output[i] = input[i];
}

torch::Tensor copy_probe(const torch::Tensor& input) {
    TORCH_CHECK(input.is_cuda() && input.is_contiguous(), "expected contiguous CUDA tensor");
    TORCH_CHECK(input.scalar_type() == at::kBFloat16, "expected BF16 tensor");
    const c10::cuda::CUDAGuard guard(input.device());
    auto output = torch::empty_like(input);
    int64_t n = input.numel();
    if (n) {
        copy_bf16_bits<<<(n + 127) / 128, 128, 0, c10::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<const uint16_t*>(input.data_ptr()),
            reinterpret_cast<uint16_t*>(output.data_ptr()), n);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("copy_probe", &copy_probe);
}
