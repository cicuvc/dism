#pragma once

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cstring>
#include <list>
#include <string>

namespace dism_metadata {
// Only immutable integer document/task records belong here. Content identity
// is essential: callers reuse cu_seqlens storage with different contents.
// Per-thread, device+stream-specific entries avoid cross-stream readiness and
// allocator-lifetime hazards. Both host keys and device allocations are bounded.
struct Entry {
        std::string bytes;
        int device;
        int64_t stream;
        at::Tensor tensor;
    };
inline std::list<Entry>& cache_entries() {
    static thread_local std::list<Entry> cache;
    return cache;
}

// Release CUDA tensors while the runtime is alive, before main-thread TLS teardown.
inline void clear_cache() { cache_entries().clear(); }

inline at::Tensor upload(const void* data, size_t bytes, at::Device device) {
    auto& cache = cache_entries();
    constexpr size_t MaxEntries = 32, MaxBytes = 8 * 1024 * 1024;
    auto stream = at::cuda::getCurrentCUDAStream(device.index()).id();
    for (auto it = cache.begin(); it != cache.end(); ++it) {
        if (it->device == device.index() && it->stream == stream &&
            it->bytes.size() == bytes && (!bytes || !std::memcmp(it->bytes.data(), data, bytes))) {
            auto result = it->tensor;
            cache.splice(cache.begin(), cache, it);
            return result;
        }
    }
    auto host = at::empty({int64_t(bytes)}, at::TensorOptions()
        .dtype(at::kByte).device(at::kCPU).pinned_memory(true));
    if (bytes) std::memcpy(host.data_ptr(), data, bytes);
    auto result = host.to(device, /*non_blocking=*/true);
    if (bytes <= MaxBytes) {
        size_t used = bytes;
        for (const auto& entry : cache) used += entry.bytes.size();
        while (!cache.empty() && (cache.size() >= MaxEntries || used > MaxBytes)) {
            used -= cache.back().bytes.size();
            cache.pop_back();
        }
        cache.push_front({std::string(static_cast<const char*>(host.data_ptr()), bytes),
                          device.index(), stream, result});
    }
    return result;
}

inline at::Tensor documents(const at::Tensor& table, at::Device device) {
    return upload(table.data_ptr(), table.nbytes(), device)
        .view(at::kLong).view(table.sizes());
}
} // namespace dism_metadata
