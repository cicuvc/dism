#pragma once
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <tuple>
#include <optional>

// Host-only parameter packing; no Torch dependency in kernel translation units.
template<class... Args>
inline void launch_kernel(const void* kernel, dim3 grid, dim3 block,
                          size_t shared_bytes, cudaStream_t stream, Args... args) {
    void* parameters[] = {const_cast<void*>(static_cast<const void*>(&args))...};
    C10_CUDA_CHECK(cudaLaunchKernel(kernel, grid, block, parameters, shared_bytes, stream));
}
