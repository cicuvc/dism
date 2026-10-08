#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#ifndef DEVICE_LOG_HOST_THROW
#define DEVICE_LOG_HOST_THROW(msg) throw std::runtime_error(msg)
#endif

namespace device_log {

// --- fixed_string NTTP ---
template <std::size_t N>
struct fixed_string {
  char v[N]{};
  constexpr fixed_string(const char (&s)[N]) {
    for (std::size_t i = 0; i < N; ++i) v[i] = s[i];
  }
  constexpr std::size_t strlen() const { return N ? (N - 1) : 0; }
};
template <std::size_t N>
fixed_string(const char (&)[N]) -> fixed_string<N>;

// --- FNV-1a 64 ---
constexpr std::uint64_t fnv1a64_bytes(const char* data, std::size_t len) {
  std::uint64_t h = 14695981039346656037ull;
  for (std::size_t i = 0; i < len; ++i) {
    h ^= static_cast<std::uint64_t>(static_cast<unsigned char>(data[i]));
    h *= 1099511628211ull;
  }
  return h;
}
template <fixed_string Name>
struct tag_type_id {
  static constexpr std::uint64_t value = fnv1a64_bytes(Name.v, Name.strlen());
};

// --- record ---
struct alignas(16) Record {
  std::uint64_t type;
  std::uint64_t value;
};

// --- sublog view ---
struct SubLogView {
  Record* records = nullptr;          // [total_rings * capacity]
  std::uint32_t* write_idx = nullptr; // [total_rings]
  std::uint32_t rings_per_block = 0;  // n_wg4 or m_sw
};

// --- device view ---
struct DeviceLogView {
  SubLogView wg4{}; // n_wg4 rings/CTA
  SubLogView sw{};  // m_sw rings/CTA
  std::uint32_t capacity = 0;      // power of 2
  std::uint32_t capacity_mask = 0; // capacity-1
  std::uint32_t n_wg4 = 0;
  std::uint32_t m_sw = 0;
};
static_assert(std::is_trivially_copyable_v<DeviceLogView>);
static_assert(std::is_trivially_copyable_v<SubLogView>);

// --- device helpers ---
__device__ __forceinline__ std::uint32_t lane_id() {
  return static_cast<std::uint32_t>(threadIdx.x) & 31u;
}
__device__ __forceinline__ std::uint32_t warp_id_in_block() {
  return static_cast<std::uint32_t>(threadIdx.x) >> 5;
}
__device__ __forceinline__ std::uint32_t wg4_region_warp_count(DeviceLogView log) {
  return log.n_wg4 << 2;
}

enum class AutoDomain : std::uint32_t { None = 0, WarpGroup4 = 1, SingleWarp = 2 };

__device__ __forceinline__ AutoDomain auto_domain(DeviceLogView log) {
  const std::uint32_t warp = warp_id_in_block();
  const std::uint32_t wg_warps = wg4_region_warp_count(log);
  if (warp < wg_warps) return AutoDomain::WarpGroup4;
  if (warp < wg_warps + log.m_sw) return AutoDomain::SingleWarp;
  return AutoDomain::None;
}

__device__ __forceinline__ bool is_writer_auto(AutoDomain d) {
  if (d == AutoDomain::WarpGroup4) {
    return (static_cast<std::uint32_t>(threadIdx.x) & 127u) == 0; // threadIdx.x % 128 == 0
  }
  if (d == AutoDomain::SingleWarp) {
    return lane_id() == 0;
  }
  return false;
}

__device__ __forceinline__ std::uint32_t group_id_in_block_auto(DeviceLogView log, AutoDomain d) {
  const std::uint32_t warp = warp_id_in_block();
  const std::uint32_t wg_warps = wg4_region_warp_count(log);
  if (d == AutoDomain::WarpGroup4) return warp >> 2; // warp/4
  return warp - wg_warps;                            // single warp index
}

__device__ __forceinline__ void append_auto_lane0(DeviceLogView log,
                                                  std::uint64_t type_id,
                                                  std::uint64_t value) {
  if (log.capacity == 0) return;

  const AutoDomain d = auto_domain(log);
  if (d == AutoDomain::None) return;
  if (!is_writer_auto(d)) return;

  const std::uint32_t C = log.capacity;
  const std::uint32_t mask = log.capacity_mask;

  if (d == AutoDomain::WarpGroup4) {
    const std::uint32_t ring =
        static_cast<std::uint32_t>(blockIdx.x) * log.wg4.rings_per_block + group_id_in_block_auto(log, d);
    const std::uint32_t idx = log.wg4.write_idx[ring];
    log.wg4.write_idx[ring] = idx + 1u;
    const std::uint32_t slot = idx & mask;
    log.wg4.records[static_cast<std::size_t>(ring) * C + slot] = Record{type_id, value};
    return;
  }

  // SingleWarp
  const std::uint32_t ring =
      static_cast<std::uint32_t>(blockIdx.x) * log.sw.rings_per_block + group_id_in_block_auto(log, d);
  const std::uint32_t idx = log.sw.write_idx[ring];
  log.sw.write_idx[ring] = idx + 1u;
  const std::uint32_t slot = idx & mask;
  log.sw.records[static_cast<std::size_t>(ring) * C + slot] = Record{type_id, value};
}

// --- tag wrapper ---
template <fixed_string Name>
struct DeviceLogTag {
  static constexpr std::uint64_t type_id = tag_type_id<Name>::value;
  __device__ __forceinline__ static void append(DeviceLogView log, std::uint64_t value) {
    append_auto_lane0(log, type_id, value);
  }
};

// --- query ---
enum class QueryDomain : std::uint32_t { WarpGroup4 = 0, SingleWarp = 1 };

__global__ void query_ring_type_kernel(DeviceLogView log,
                                       QueryDomain domain,
                                       std::uint32_t target_block,
                                       std::uint32_t target_group,
                                       std::uint64_t type_id,
                                       std::uint64_t* out_values,
                                       std::uint32_t* out_count) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;

  const std::uint32_t C = log.capacity;
  if (C == 0) {
    *out_count = 0;
    return;
  }

  const SubLogView& s = (domain == QueryDomain::WarpGroup4) ? log.wg4 : log.sw;
  const std::uint32_t rings_per_block = s.rings_per_block;
  if (target_group >= rings_per_block) {
    *out_count = 0;
    return;
  }

  const std::uint32_t ring = target_block * rings_per_block + target_group;

  const std::uint32_t head = s.write_idx[ring];
  const std::uint32_t valid = (head < C) ? head : C;
  const std::uint32_t start = (head - valid) & log.capacity_mask;

  const Record* base = s.records + static_cast<std::size_t>(ring) * C;

  std::uint32_t n = 0;
  for (std::uint32_t i = 0; i < valid; ++i) {
    const std::uint32_t slot = (start + i) & log.capacity_mask;
    const Record r = base[slot];
    if (r.type == type_id) out_values[n++] = r.value;
  }
  *out_count = n;
}

// --- host RAII ---
class DeviceLog {
public:
  DeviceLog() = default;

  DeviceLog(std::size_t grid_size,
            std::uint32_t n_wg4,
            std::uint32_t m_sw,
            std::uint32_t capacity = 1024)
      : grid_size_(grid_size), n_wg4_(n_wg4), m_sw_(m_sw), capacity_(capacity) {
    if (grid_size_ == 0) DEVICE_LOG_HOST_THROW("grid_size must be > 0.");

    block_threads_ = 128u * n_wg4_ + 32u * m_sw_;
    if (block_threads_ == 0) DEVICE_LOG_HOST_THROW("CTA must have > 0 threads.");

    if (capacity_ == 0) DEVICE_LOG_HOST_THROW("capacity must be > 0.");
    if (!isPowerOf2(capacity_)) DEVICE_LOG_HOST_THROW("capacity must be a power of 2.");

    allocateAll();
  }

  ~DeviceLog() { reset(); }

  DeviceLog(const DeviceLog&) = delete;
  DeviceLog& operator=(const DeviceLog&) = delete;

  DeviceLog(DeviceLog&& other) noexcept { moveFrom(std::move(other)); }
  DeviceLog& operator=(DeviceLog&& other) noexcept {
    if (this != &other) {
      reset();
      moveFrom(std::move(other));
    }
    return *this;
  }

  std::size_t gridSize() const { return grid_size_; }
  std::uint32_t blockThreads() const { return block_threads_; }
  std::uint32_t nWarpGroup4() const { return n_wg4_; }
  std::uint32_t mSingleWarp() const { return m_sw_; }
  std::uint32_t capacity() const { return capacity_; }

  DeviceLogView view() const {
    DeviceLogView v;
    v.capacity = capacity_;
    v.capacity_mask = capacity_ - 1u;
    v.n_wg4 = n_wg4_;
    v.m_sw = m_sw_;

    v.wg4.records = d_wg4_records_;
    v.wg4.write_idx = d_wg4_write_idx_;
    v.wg4.rings_per_block = n_wg4_;

    v.sw.records = d_sw_records_;
    v.sw.write_idx = d_sw_write_idx_;
    v.sw.rings_per_block = m_sw_;

    return v;
  }

  template <fixed_string Name>
  std::vector<std::uint64_t> getLogWarpGroup4(std::uint32_t block_id, std::uint32_t wg_id) const {
    if (block_id >= grid_size_) DEVICE_LOG_HOST_THROW("block_id out of range.");
    if (wg_id >= n_wg4_) DEVICE_LOG_HOST_THROW("wg_id out of range.");
    return getLogByTypeId(QueryDomain::WarpGroup4, block_id, wg_id, tag_type_id<Name>::value);
  }

  template <fixed_string Name>
  std::vector<std::uint64_t> getLogSingleWarp(std::uint32_t block_id, std::uint32_t sw_id) const {
    if (block_id >= grid_size_) DEVICE_LOG_HOST_THROW("block_id out of range.");
    if (sw_id >= m_sw_) DEVICE_LOG_HOST_THROW("sw_id out of range.");
    return getLogByTypeId(QueryDomain::SingleWarp, block_id, sw_id, tag_type_id<Name>::value);
  }

  std::vector<std::uint64_t> getLogWarpGroup4(std::uint32_t block_id, std::uint32_t wg_id, const std::string& name) const {
    return getLogByTypeId(QueryDomain::WarpGroup4, block_id, wg_id, fnv1a64_bytes(name.data(), name.size()));
  }
  std::vector<std::uint64_t> getLogSingleWarp(std::uint32_t block_id, std::uint32_t sw_id, const std::string& name) const {
    return getLogByTypeId(QueryDomain::SingleWarp, block_id, sw_id, fnv1a64_bytes(name.data(), name.size()));
  }

private:
  static bool isPowerOf2(std::uint32_t x) { return x && ((x & (x - 1u)) == 0u); }

  void allocateAll() {
    cudaError_t st = cudaSuccess;

    // WarpGroup4 domain
    if (n_wg4_ > 0) {
      const std::size_t total_rings = grid_size_ * static_cast<std::size_t>(n_wg4_);
      const std::size_t records_count = total_rings * static_cast<std::size_t>(capacity_);
      const std::size_t records_bytes = records_count * sizeof(Record);
      const std::size_t idx_bytes = total_rings * sizeof(std::uint32_t);

      st = cudaMalloc(reinterpret_cast<void**>(&d_wg4_records_), records_bytes);
      if (st != cudaSuccess) DEVICE_LOG_HOST_THROW("cudaMalloc(wg4.records) failed.");

      st = cudaMalloc(reinterpret_cast<void**>(&d_wg4_write_idx_), idx_bytes);
      if (st != cudaSuccess) DEVICE_LOG_HOST_THROW("cudaMalloc(wg4.write_idx) failed.");

      st = cudaMemset(d_wg4_write_idx_, 0, idx_bytes);
      if (st != cudaSuccess) DEVICE_LOG_HOST_THROW("cudaMemset(wg4.write_idx) failed.");

      st = cudaMemset(d_wg4_records_, 0, records_bytes);
      if (st != cudaSuccess) DEVICE_LOG_HOST_THROW("cudaMemset(wg4.records) failed.");
    }

    // SingleWarp domain
    if (m_sw_ > 0) {
      const std::size_t total_rings = grid_size_ * static_cast<std::size_t>(m_sw_);
      const std::size_t records_count = total_rings * static_cast<std::size_t>(capacity_);
      const std::size_t records_bytes = records_count * sizeof(Record);
      const std::size_t idx_bytes = total_rings * sizeof(std::uint32_t);

      st = cudaMalloc(reinterpret_cast<void**>(&d_sw_records_), records_bytes);
      if (st != cudaSuccess) DEVICE_LOG_HOST_THROW("cudaMalloc(sw.records) failed.");

      st = cudaMalloc(reinterpret_cast<void**>(&d_sw_write_idx_), idx_bytes);
      if (st != cudaSuccess) DEVICE_LOG_HOST_THROW("cudaMalloc(sw.write_idx) failed.");

      st = cudaMemset(d_sw_write_idx_, 0, idx_bytes);
      if (st != cudaSuccess) DEVICE_LOG_HOST_THROW("cudaMemset(sw.write_idx) failed.");

      st = cudaMemset(d_sw_records_, 0, records_bytes);
      if (st != cudaSuccess) DEVICE_LOG_HOST_THROW("cudaMemset(sw.records) failed.");
    }
  }

  std::vector<std::uint64_t> getLogByTypeId(QueryDomain domain,
                                            std::uint32_t block_id,
                                            std::uint32_t group_id,
                                            std::uint64_t type_id) const {
    if (capacity_ == 0) return {};

    std::uint64_t* d_out_values = nullptr;
    std::uint32_t* d_out_count = nullptr;

    cudaError_t st = cudaSuccess;
    st = cudaMalloc(reinterpret_cast<void**>(&d_out_values),
                    static_cast<std::size_t>(capacity_) * sizeof(std::uint64_t));
    if (st != cudaSuccess) DEVICE_LOG_HOST_THROW("cudaMalloc(d_out_values) failed.");

    st = cudaMalloc(reinterpret_cast<void**>(&d_out_count), sizeof(std::uint32_t));
    if (st != cudaSuccess) DEVICE_LOG_HOST_THROW("cudaMalloc(d_out_count) failed.");

    st = cudaMemset(d_out_count, 0, sizeof(std::uint32_t));
    if (st != cudaSuccess) DEVICE_LOG_HOST_THROW("cudaMemset(d_out_count) failed.");

    query_ring_type_kernel<<<1, 1>>>(view(), domain, block_id, group_id, type_id, d_out_values, d_out_count);
    st = cudaGetLastError();
    if (st != cudaSuccess) DEVICE_LOG_HOST_THROW("query_ring_type_kernel launch failed.");

    std::uint32_t h_count = 0;
    st = cudaMemcpy(&h_count, d_out_count, sizeof(std::uint32_t), cudaMemcpyDeviceToHost);
    if (st != cudaSuccess) DEVICE_LOG_HOST_THROW("cudaMemcpy(count) failed.");

    h_count = std::min<std::uint32_t>(h_count, capacity_);
    std::vector<std::uint64_t> out(h_count);

    if (h_count > 0) {
      st = cudaMemcpy(out.data(), d_out_values,
                      static_cast<std::size_t>(h_count) * sizeof(std::uint64_t),
                      cudaMemcpyDeviceToHost);
      if (st != cudaSuccess) DEVICE_LOG_HOST_THROW("cudaMemcpy(values) failed.");
    }

    cudaFree(d_out_values);
    cudaFree(d_out_count);
    return out;
  }

  void reset() noexcept {
    if (d_wg4_records_) cudaFree(d_wg4_records_);
    if (d_wg4_write_idx_) cudaFree(d_wg4_write_idx_);
    if (d_sw_records_) cudaFree(d_sw_records_);
    if (d_sw_write_idx_) cudaFree(d_sw_write_idx_);

    d_wg4_records_ = nullptr;
    d_wg4_write_idx_ = nullptr;
    d_sw_records_ = nullptr;
    d_sw_write_idx_ = nullptr;

    grid_size_ = 0;
    n_wg4_ = 0;
    m_sw_ = 0;
    capacity_ = 0;
    block_threads_ = 0;
  }

  void moveFrom(DeviceLog&& other) noexcept {
    d_wg4_records_ = other.d_wg4_records_;
    d_wg4_write_idx_ = other.d_wg4_write_idx_;
    d_sw_records_ = other.d_sw_records_;
    d_sw_write_idx_ = other.d_sw_write_idx_;

    grid_size_ = other.grid_size_;
    n_wg4_ = other.n_wg4_;
    m_sw_ = other.m_sw_;
    capacity_ = other.capacity_;
    block_threads_ = other.block_threads_;

    other.d_wg4_records_ = nullptr;
    other.d_wg4_write_idx_ = nullptr;
    other.d_sw_records_ = nullptr;
    other.d_sw_write_idx_ = nullptr;

    other.grid_size_ = 0;
    other.n_wg4_ = 0;
    other.m_sw_ = 0;
    other.capacity_ = 0;
    other.block_threads_ = 0;
  }

private:
  Record* d_wg4_records_ = nullptr;
  std::uint32_t* d_wg4_write_idx_ = nullptr;

  Record* d_sw_records_ = nullptr;
  std::uint32_t* d_sw_write_idx_ = nullptr;

  std::size_t grid_size_ = 0;
  std::uint32_t n_wg4_ = 0;
  std::uint32_t m_sw_ = 0;
  std::uint32_t capacity_ = 0;
  std::uint32_t block_threads_ = 0;
};

} // namespace device_log