#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <glx/diagonal_scan.cuh>
#include <kittens.cuh>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

namespace kt = kittens;
using glx::AddOp;
using glx::F32x2;
using glx::MMABuffer;
using glx::UnaryElement;

namespace {

__device__ __forceinline__ uint32_t shared_address(const void* pointer) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(pointer));
}

__device__ __forceinline__ void initialize_barrier(uint64_t* barrier) {
    uint32_t address = shared_address(barrier);
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n" :: "r"(address));
}

__device__ __forceinline__ void expect_bytes(uint64_t* barrier, int bytes) {
    uint32_t address = shared_address(barrier);
    asm volatile(
        "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n"
        :: "r"(address), "r"(bytes) : "memory");
}

__device__ __forceinline__ void wait_barrier(uint64_t* barrier) {
    uint32_t address = shared_address(barrier);
    asm volatile(
        "{\n"
        ".reg .pred ready;\n"
        "WAIT: mbarrier.try_wait.parity.shared::cta.b64 ready, [%0], 0;\n"
        "@!ready bra WAIT;\n"
        "}\n"
        :: "r"(address) : "memory");
}

__device__ __forceinline__ void tma_load_5d(
        const CUtensorMap* map, void* destination, uint64_t* barrier,
        int c0, int c1, int c2, int c3, int c4) {
    uint32_t dst = shared_address(destination);
    uint32_t bar = shared_address(barrier);
    asm volatile(
        "cp.async.bulk.tensor.5d.shared::cluster.global."
        "mbarrier::complete_tx::bytes "
        "[%0], [%1, {%3, %4, %5, %6, %7}], [%2];\n"
        :: "r"(dst), "l"(map), "r"(bar), "r"(c0), "r"(c1),
           "r"(c2), "r"(c3), "r"(c4) : "memory");
}

template<int N>
__host__ __device__ constexpr int logical_row_from_physical(int physical) {
    constexpr int unit = N / 8;
    int element = physical & 1;
    int lane_group = (physical >> 1) & 3;
    int column_register = physical >> 3;
    return column_register + lane_group * unit
         + element * (4 * unit);
}

template<int N, int K>
bool encode_permuted_b_map(CUtensorMap* map, void* pointer, int flattened_rows) {
    static_assert(N == 32 || N == 64);
    static_assert(K == 32 || K == 64 || K == 128);
    constexpr int swizzle_elements = K == 32 ? 32 : 64;
    constexpr int unit = N / 8;

    // TK rt accumulator slot [column/2].data[row + 2*(column&1)]
    // corresponds to physical MMA columns
    //   physical = 8*column + 2*lane_group + element.
    // The same register interpreted as GLX data[row][column] corresponds to
    //   logical = column + unit*lane_group + 4*unit*element.
    // TMA materializes exactly physical->logical in the B rows.
    const cuuint64_t dimensions[5]{
        swizzle_elements, 2, 4,
        static_cast<cuuint64_t>(flattened_rows), K / swizzle_elements};
    constexpr cuuint64_t strides[4]{
        (N / 2) * K * sizeof(__nv_bfloat16),
        unit * K * sizeof(__nv_bfloat16),
        K * sizeof(__nv_bfloat16),
        swizzle_elements * sizeof(__nv_bfloat16)};
    constexpr cuuint32_t box[5]{swizzle_elements, 2, 4, unit, 1};
    constexpr cuuint32_t element_strides[5]{1, 1, 1, 1, 1};
    constexpr auto swizzle = K == 32
        ? CU_TENSOR_MAP_SWIZZLE_64B : CU_TENSOR_MAP_SWIZZLE_128B;
    CUresult result = cuTensorMapEncodeTiled(
        map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 5, pointer,
        dimensions, strides, box, element_strides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if(result == CUDA_SUCCESS) return true;
    const char* message = nullptr;
    cuGetErrorString(result, &message);
    std::fprintf(stderr, "cuTensorMapEncodeTiled N=%d K=%d: %s\n",
                 N, K, message ? message : "unknown error");
    return false;
}

template<int N, int K>
__global__ void tma_mma_permute_kernel(
        const __nv_bfloat16* a,
        __grid_constant__ const CUtensorMap b_map,
        int row_base,
        __nv_bfloat16* loaded_b,
        float* c) {
    extern __shared__ __align__(128) unsigned char shared_bytes[];
    auto& a_tile = *reinterpret_cast<kt::st_bf<16, K>*>(shared_bytes);
    auto& b_tile = *reinterpret_cast<kt::st_bf<N, K>*>(
        shared_bytes + sizeof(a_tile));
    auto* barrier = reinterpret_cast<uint64_t*>(
        shared_bytes + sizeof(a_tile) + sizeof(b_tile));
    constexpr int swizzle_elements = K == 32 ? 32 : 64;

    if(threadIdx.x == 0) initialize_barrier(barrier);
    for(int index = threadIdx.x; index < 16 * K; index += blockDim.x) {
        int row = index / K;
        int column = index % K;
        a_tile[int2{row, column}] = a[index];
    }
    __syncthreads();

    if(threadIdx.x == 0) {
        expect_bytes(barrier, sizeof(b_tile));
        #pragma unroll
        for(int segment = 0; segment < K / swizzle_elements; ++segment) {
            tma_load_5d(
                &b_map,
                b_tile.data + segment * N * swizzle_elements,
                barrier, 0, 0, 0, row_base, segment);
        }
    }
    wait_barrier(barrier);
    __syncthreads();

    // Check the logical shared view independently of MMA.
    for(int index = threadIdx.x; index < N * K; index += blockDim.x) {
        int physical_row = index / K;
        int column = index % K;
        loaded_b[index] = b_tile[int2{physical_row, column}];
    }

    kt::rt_bf<16, K> a_register;
    kt::rt_bf<N, K> b_register;
    kt::rt_fl<16, N> accumulator{0.f};
    kt::warp::load(a_register, a_tile);
    kt::warp::load(b_register, b_tile);
    kt::warp::wmma::mma_ABt(
        accumulator, a_register, b_register, accumulator);

    using Buffer = MMABuffer<16, N, UnaryElement, AddOp, F32x2>;
    Buffer glx_buffer;
    constexpr int unit = N / 8;
    #pragma unroll
    for(int row_block = 0; row_block < 2; ++row_block) {
        #pragma unroll
        for(int column = 0; column < unit; ++column) {
            // Same register selection as sxdiag::TightMMABuffer::from_rt.
            auto value = accumulator.tiles[0][column / 2]
                .data[row_block + 2 * (column & 1)];
            glx_buffer.data[row_block][column].value =
                F32x2{value.x, value.y};
        }
    }

    #pragma unroll
    for(int row_block = 0; row_block < Buffer::ROW_BLOCKS; ++row_block) {
        #pragma unroll
        for(int column = 0; column < Buffer::COL_BLOCKS; ++column) {
            auto p0 = Buffer::layout(row_block, column, 0);
            auto p1 = Buffer::layout(row_block, column, 1);
            auto value = glx_buffer.data[row_block][column].value;
            c[p0.first * N + p0.second] = value.u0;
            c[p1.first * N + p1.second] = value.u1;
        }
    }
}

bool check_cuda(cudaError_t status, const char* operation) {
    if(status == cudaSuccess) return true;
    std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(status));
    return false;
}

template<int N, int K, int ROW_BASE>
bool run_case() {
    constexpr int a_count = 16 * K;
    constexpr int b_count = (N + ROW_BASE) * K;
    constexpr int c_count = 16 * N;
    std::mt19937 rng(0x9e3779b9u + N * 131u + K);
    std::uniform_int_distribution<int> distribution(-3, 3);
    std::vector<__nv_bfloat16> a(a_count), b(b_count);
    for(auto& value : a) value = __float2bfloat16(float(distribution(rng)));
    for(auto& value : b) value = __float2bfloat16(float(distribution(rng)));

    __nv_bfloat16 *device_a = nullptr, *device_b = nullptr, *device_loaded = nullptr;
    float* device_c = nullptr;
    bool okay = check_cuda(cudaMalloc(&device_a, sizeof(a[0]) * a_count), "cudaMalloc A")
             && check_cuda(cudaMalloc(&device_b, sizeof(b[0]) * b_count), "cudaMalloc B")
             && check_cuda(cudaMalloc(&device_loaded, sizeof(b[0]) * b_count), "cudaMalloc loaded B")
             && check_cuda(cudaMalloc(&device_c, sizeof(float) * c_count), "cudaMalloc C");
    if(!okay) return false;
    cudaMemcpy(device_a, a.data(), sizeof(a[0]) * a_count, cudaMemcpyHostToDevice);
    cudaMemcpy(device_b, b.data(), sizeof(b[0]) * b_count, cudaMemcpyHostToDevice);

    CUtensorMap map{};
    okay &= encode_permuted_b_map<N, K>(&map, device_b, N + ROW_BASE);
    constexpr size_t shared_bytes =
        sizeof(kt::st_bf<16, K>) + sizeof(kt::st_bf<N, K>) + sizeof(uint64_t);
    if(okay) {
        tma_mma_permute_kernel<N, K><<<1, 32, shared_bytes>>>(
            device_a, map, ROW_BASE, device_loaded, device_c);
        okay &= check_cuda(cudaGetLastError(), "kernel launch")
             && check_cuda(cudaDeviceSynchronize(), "kernel execution");
    }

    std::vector<__nv_bfloat16> loaded(N * K);
    std::vector<float> result(c_count);
    if(okay) {
        cudaMemcpy(loaded.data(), device_loaded, sizeof(loaded[0]) * N * K,
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(result.data(), device_c, sizeof(result[0]) * c_count,
                   cudaMemcpyDeviceToHost);
    }

    int load_errors = 0;
    float max_abs = 0.f;
    if(okay) {
        for(int physical = 0; physical < N; ++physical) {
            int logical = logical_row_from_physical<N>(physical);
            int row_errors = 0;
            for(int k = 0; k < K; ++k) {
                auto lhs = reinterpret_cast<const uint16_t*>(loaded.data())[
                    physical * K + k];
                auto rhs = reinterpret_cast<const uint16_t*>(b.data())[
                    (ROW_BASE + logical) * K + k];
                row_errors += lhs != rhs;
            }
            load_errors += row_errors;
            if(row_errors && physical < 8) {
                int best = -1, best_errors = K + 1;
                for(int candidate = 0; candidate < N; ++candidate) {
                    int errors = 0;
                    for(int k = 0; k < K; ++k) {
                        auto lhs = reinterpret_cast<const uint16_t*>(loaded.data())[
                            physical * K + k];
                        auto rhs = reinterpret_cast<const uint16_t*>(b.data())[
                            (ROW_BASE + candidate) * K + k];
                        errors += lhs != rhs;
                    }
                    if(errors < best_errors) {
                        best_errors = errors;
                        best = candidate;
                    }
                }
                std::fprintf(stderr,
                    "N=%d K=%d physical=%d expected=%d observed=%d row_errors=%d\n",
                    N, K, physical, logical, best, row_errors);
            }
        }
        for(int i = 0; i < 16; ++i) {
            for(int j = 0; j < N; ++j) {
                float reference = 0.f;
                for(int k = 0; k < K; ++k)
                    reference += __bfloat162float(a[i * K + k])
                               * __bfloat162float(
                                   b[(ROW_BASE + j) * K + k]);
                max_abs = std::max(max_abs,
                    std::abs(reference - result[i * N + j]));
            }
        }
    }
    okay &= load_errors == 0 && max_abs == 0.f;
    std::fprintf(stdout,
        "TMA-permuted B + MMA 16x%d K=%d base=%d: load_errors=%d max_abs=%g [%s]\n",
        N, K, ROW_BASE, load_errors, max_abs, okay ? "PASS" : "FAIL");

    cudaFree(device_c);
    cudaFree(device_loaded);
    cudaFree(device_b);
    cudaFree(device_a);
    return okay;
}

} // namespace

int main() {
    bool okay = true;
    okay &= run_case<32, 32, 0>();
    okay &= run_case<32, 64, 7>();
    okay &= run_case<32, 128, 0>();
    okay &= run_case<64, 32, 7>();
    okay &= run_case<64, 64, 0>();
    okay &= run_case<64, 128, 7>();
    return okay ? 0 : 1;
}
