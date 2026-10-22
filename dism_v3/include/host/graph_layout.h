#pragma once

// Runtime-only packed metadata for CUDA Graph replay. No Python registry and no
// checkpoint fields: the CPU table's Storage owns the staging state. Device
// translation units never include this header.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <functional>
#include <algorithm>
#include <map>
#include <memory>
#include <vector>
#include <cstring>
#include <climits>

namespace dism_graph {
using Bytes = std::vector<uint8_t>;
using Builder = std::function<Bytes(const at::Tensor&)>;
struct Slot { Builder build; at::Tensor device; };
struct State {
    int64_t tokens, maximum, row_tokens, documents;
    std::vector<int64_t> rows;
    std::map<std::string, Slot> slots;
    int device_index = -1;
    int64_t stream_id = -1;
};
inline void destroy(void* pointer) { delete static_cast<State*>(pointer); }
inline State* state(const at::Tensor& table) {
    return table.storage().data_ptr().get_deleter() == destroy
        ? static_cast<State*>(table.storage().data_ptr().get_context()) : nullptr;
}
inline void outside_capture(at::Device device) {
    c10::cuda::CUDAGuard guard(device);
    cudaStreamCaptureStatus status;
    C10_CUDA_CHECK(cudaStreamIsCapturing(at::cuda::getCurrentCUDAStream(), &status));
    TORCH_CHECK(status == cudaStreamCaptureStatusNone,
                "prepare/update graph metadata outside CUDA Graph capture");
}
inline void check_stream(State& s, at::Device device) {
    auto stream = at::cuda::getCurrentCUDAStream(device.index());
    if (s.device_index < 0) { s.device_index = device.index(); s.stream_id = stream.id(); }
    TORCH_CHECK(s.device_index == device.index() && s.stream_id == stream.id(),
                "graph metadata must be staged, captured and replayed on its owning stream");
}
template<class T> Bytes bytes(const std::vector<T>& data) {
    Bytes result(data.size() * sizeof(T));
    if (!result.empty()) std::memcpy(result.data(), data.data(), result.size());
    return result;
}
inline at::Tensor host_bytes(const Bytes& data) {
    auto host = at::empty({int64_t(data.size())}, at::TensorOptions()
        .dtype(at::kByte).device(at::kCPU).pinned_memory(true));
    if (!data.empty()) std::memcpy(host.data_ptr(), data.data(), data.size());
    return host;
}
inline at::Tensor records(const at::Tensor& table, const std::string& key,
                          at::Device device, Builder builder) {
    auto* s = state(table);
    TORCH_CHECK(s, "expected an owned graph layout");
    check_stream(*s, device);
    auto found = s->slots.find(key);
    if (found != s->slots.end()) return found->second.device;
    outside_capture(device);
    auto host = host_bytes(builder(table));
    auto gpu = host.to(device, true);
    s->slots.emplace(key, Slot{std::move(builder), gpu});
    return gpu;
}
inline void update(const at::Tensor& table, const at::Tensor& boundaries) {
    auto* s = state(table);
    TORCH_CHECK(s, "expected an owned graph layout");
    TORCH_CHECK(boundaries.device().is_cpu() && boundaries.scalar_type() == at::kInt &&
                boundaries.dim() == 1 && boundaries.is_contiguous() && boundaries.numel() >= 2,
                "graph boundaries must be contiguous CPU int32 [documents+1]");
    const int* cu = boundaries.data_ptr<int>();
    TORCH_CHECK(cu[0] == 0 && cu[boundaries.numel()-1] == s->tokens,
                "invalid graph boundary endpoints");
    std::vector<int64_t> rows(s->documents * 7);
    int64_t f=0, v=0, b=0, count=0;
    for (int64_t i=0; i+1<boundaries.numel(); ++i) {
        int64_t begin=cu[i], n=int64_t(cu[i+1])-begin;
        TORCH_CHECK(begin>=0 && begin%256==0 && n>=0 && n%256==0 && n<=s->maximum,
                    "graph documents must be monotonic, 256-aligned and within max_seqlen");
        if (!n) continue; // Empty documents have no DISM work or loss contribution.
        TORCH_CHECK(begin/s->row_tokens == (begin+n-1)/s->row_tokens,
                    "document cannot cross a hard training-row boundary");
        TORCH_CHECK(count<s->documents, "too many graph documents");
        int64_t row[] = {begin,n,begin,n,f,v,b};
        std::copy(row,row+7,rows.data()+7*count++);
        f+=(n-1)/32*n; v+=(n-1)/16*n; b+=n/32*n;
    }
    for (; count<s->documents; ++count) {
        int64_t row[] = {s->tokens,0,s->tokens,0,f,v,b};
        std::copy(row,row+7,rows.data()+7*count);
    }
    if (s->device_index>=0) {
        auto device=at::Device(at::kCUDA,s->device_index);
        outside_capture(device);
        check_stream(*s,device);
    }
    // Validate all input before mutating the live layout. Copies are ordered
    // after the previous replay and before the next replay on the same stream.
    std::copy(rows.begin(),rows.end(),s->rows.begin());
    for (auto& [key,slot] : s->slots) {
        auto data=slot.build(table);
        TORCH_CHECK(int64_t(data.size())==slot.device.numel(),
                    "graph metadata capacity changed for ",key);
        slot.device.copy_(host_bytes(data),true);
    }
}
inline at::Tensor create(const at::Tensor& boundaries,int64_t tokens,
                         int64_t maximum,int64_t row_tokens) {
    TORCH_CHECK(tokens>0 && tokens<=INT_MAX-256 && tokens%256==0 &&
                maximum>0 && maximum<=tokens && maximum%256==0 &&
                row_tokens>0 && row_tokens%256==0 && tokens%row_tokens==0 && maximum<=row_tokens,
                "invalid graph token/max_seqlen/training-row capacities");
    auto owner=std::make_unique<State>();
    owner->tokens=tokens; owner->maximum=maximum; owner->row_tokens=row_tokens;
    owner->documents=tokens/256;
    owner->rows.resize(owner->documents*7);
    auto* s=owner.release();
    auto table=at::for_blob(s->rows.data(),{s->documents,7}).context(s,destroy)
        .options(at::TensorOptions().dtype(at::kLong).device(at::kCPU)).make_tensor();
    update(table,boundaries);
    return table;
}
} // namespace dism_graph
