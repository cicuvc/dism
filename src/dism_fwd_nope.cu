#define KITTENS_RTX_BLACKWELL

#include <cstdio>
#include <cassert>
#include <ct_math.hpp>
#include <dism_baseline_nope.hpp>
#include <kittens.cuh>
#include <sxdiag_new.cuh>
#include <common/debug.cuh>

namespace {

namespace kt = kittens;

namespace details{

template<kt::ducks::st::all ST, int WARP_SKIP_ROWS>
struct TmaWarpgroupStrided {
    using identifier = kt::ducks::tma::wrapper::identifier;
    using dtype = ST::dtype;
    using T_ = ST;
    static constexpr int value = 1; // NOLINT
    static constexpr bool swizzle_flag = true; // NOLINT
    static constexpr uint32_t num_elements = ST::num_elements; // NOLINT
    static constexpr uint32_t bytes = num_elements * sizeof(dtype); // NOLINT

    ST data; // NOLINT

    template<typename T, int AXIS, bool ENABLE_SWIZZLE = true>
    __host__ static void create_tensor_map(CUtensorMap *tma_map, const typename ST::dtype *src, int batch, int seq, int head, int channel){ // NOLINT
        static_assert(AXIS == 1, "Currently only axis over DEPTH is supported");
        static_assert(ENABLE_SWIZZLE, "Swizzle must be enabled");
        
        uint64_t gmem_shape [5] = {0, 0, 0, 0, 0};
        uint64_t gmem_stride[4] = {0, 0, 0, 0};
        uint32_t smem_shape [5] = {0, 0, 0, 0, 0};
        uint32_t smem_stride[5] = {1, 1, 1, 1, 1};

        constexpr int swizzle_elements = ST::swizzle_bytes / sizeof(dtype);
        constexpr uint64_t shared_tile_height = ST::rows; 
        constexpr uint64_t shared_tile_width  = ST::cols;
        constexpr uint64_t warps_per_wg = kt::WARPGROUP_WARPS;
        constexpr uint64_t warp_tile_rows = shared_tile_height / warps_per_wg;

        assert(channel % swizzle_elements == 0); // Irregular channel size is not yet supported

        // batch, seq, head, channel
        smem_shape[0] = swizzle_elements;
        smem_shape[1] = warp_tile_rows;
        smem_shape[2] = warps_per_wg;
        smem_shape[3] = shared_tile_width / swizzle_elements;
        smem_shape[4] = 1;

        gmem_stride[0] = (uint64_t)channel * head * sizeof(dtype); // seq
        gmem_stride[1] = (uint64_t)WARP_SKIP_ROWS * channel * head * sizeof(dtype); // 0
        gmem_stride[2] = (uint64_t)ST::swizzle_bytes; // (shared_tile_width / swizzle_elements) * head + channel % (shared_tile_width / swizzle_elements)
        gmem_stride[3] = (uint64_t)seq * head * channel * sizeof(dtype); // * batch

        gmem_shape[0] = swizzle_elements;
        gmem_shape[1] = (uint64_t)seq;
        gmem_shape[2] = (uint64_t)warps_per_wg;
        gmem_shape[3] = (uint64_t)(channel / swizzle_elements) * head;
        gmem_shape[4] = (uint64_t)batch;

        constexpr uint32_t  tma_dim = ENABLE_SWIZZLE ? 5 : 4;
        void *global_addr = (void*)(src);

        constexpr CUtensorMapDataType     tma_format      = (
            std::is_same_v<dtype, kt::bf16>  ? CU_TENSOR_MAP_DATA_TYPE_BFLOAT16 :
            std::is_same_v<dtype, kt::half>  ? CU_TENSOR_MAP_DATA_TYPE_FLOAT16 :
            std::is_same_v<dtype, float> ? CU_TENSOR_MAP_DATA_TYPE_FLOAT32 :
        #ifdef KITTENS_FEATURE_FP8
            std::is_same_v<dtype, kt::fp8e4m3> ? CU_TENSOR_MAP_DATA_TYPE_UINT8 :
            std::is_same_v<dtype, kt::fp8e5m2> ? CU_TENSOR_MAP_DATA_TYPE_UINT8 :
        #endif
        #ifdef KITTENS_FEATURE_UE8
            std::is_same_v<dtype, kt::fp8e8m0> ? CU_TENSOR_MAP_DATA_TYPE_UINT8 :
        #endif
        #ifdef KITTENS_FEATURE_FP4
            std::is_same_v<dtype, kt::fp4_2> ? CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B :
        #endif
            CUtensorMapDataType(-1)
        );
        constexpr CUtensorMapInterleave   tma_interleave  = CU_TENSOR_MAP_INTERLEAVE_NONE;
        constexpr CUtensorMapL2promotion  tma_l2Promotion = CU_TENSOR_MAP_L2_PROMOTION_NONE;
        constexpr CUtensorMapFloatOOBfill tma_oobFill     = CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE;
        constexpr CUtensorMapSwizzle      tma_swizzle     = ENABLE_SWIZZLE ? (
            ST::swizzle_bytes == 32  ? CU_TENSOR_MAP_SWIZZLE_32B  :
            ST::swizzle_bytes == 64  ? CU_TENSOR_MAP_SWIZZLE_64B  :
            ST::swizzle_bytes == 128 ? CU_TENSOR_MAP_SWIZZLE_128B : 
            CU_TENSOR_MAP_SWIZZLE_NONE
        ) : CU_TENSOR_MAP_SWIZZLE_NONE;

        CUresult result = cuTensorMapEncodeTiled(
            tma_map,
            tma_format,
            tma_dim,
            global_addr,
            gmem_shape,
            gmem_stride, 
            smem_shape,
            smem_stride,
            tma_interleave,
            tma_swizzle,
            tma_l2Promotion,
            tma_oobFill);
            
        const char *error_string;
        CUresult res = cuGetErrorString(result, &error_string);
        if (result != CUDA_SUCCESS) {
            std::string error_msg = kt::detail::tma::format_tma_error(
                "tile", error_string,
                batch, seq, head, channel,
                tma_map, tma_format, tma_dim, global_addr,
                gmem_shape, gmem_stride,
                smem_shape, smem_stride,
                5, 4, 5, 5,
                tma_interleave, tma_swizzle, tma_l2Promotion, tma_oobFill,
                "ST::rows: " + std::to_string(ST::rows) + "\n  ST::cols: " + std::to_string(ST::cols)
            );
            throw std::runtime_error(error_msg);
        }
    }

    template<int AXIS, kt::ducks::gl::all GL, typename COORD>
    static __device__ int4 get_coords(const COORD& coord, const GL& src){ // NOLINT
        // coord: batch, seq, head, channel
        auto batch = coord.b, seq = coord.d, head = coord.r, channel = coord.c;
        constexpr int warp_rows = (ST::rows / kt::WARPGROUP_WARPS);
        constexpr int swizzle_elements = ST::swizzle_bytes / sizeof(dtype);
        constexpr uint64_t shared_tile_height = ST::rows; 
        constexpr uint64_t shared_tile_width  = ST::cols;
        constexpr uint32_t col_atom_size = (shared_tile_width / swizzle_elements);
        
        return {seq, 0, int(col_atom_size * head + (channel % col_atom_size)), batch };
    }
};


template<kt::ducks::st::all ST>
struct TmaColumnGrouped {
    using identifier = kt::ducks::tma::wrapper::identifier;
    using dtype = ST::dtype;
    using T_ = ST;
    static constexpr int value = 1; // NOLINT
    static constexpr bool swizzle_flag = true; // NOLINT
    static constexpr uint32_t num_elements = ST::num_elements; // NOLINT
    static constexpr uint32_t bytes = num_elements * sizeof(dtype); // NOLINT

    ST data; // NOLINT

    template<typename T, int AXIS, bool ENABLE_SWIZZLE = true>
    __host__ static void create_tensor_map(CUtensorMap *tma_map, const typename ST::dtype *src, int batch, int seq, int head, int channel){ // NOLINT
        static_assert(AXIS == 1, "Currently only axis over DEPTH is supported");
        static_assert(ENABLE_SWIZZLE, "Swizzle must be enabled");
        
        uint64_t gmem_shape [5] = {0, 0, 0, 0, 0};
        uint64_t gmem_stride[4] = {0, 0, 0, 0};
        uint32_t smem_shape [5] = {0, 0, 0, 0, 0};
        uint32_t smem_stride[5] = {1, 1, 1, 1, 1};

        constexpr int swizzle_elements = ST::swizzle_bytes / sizeof(dtype);
        constexpr uint64_t shared_tile_height = ST::rows; 
        constexpr uint64_t shared_tile_width  = ST::cols;
        constexpr uint64_t mma_col_threads = 4; // 4 threads for each row of standard MMA layout
        constexpr uint64_t elements_per_thread = shared_tile_height / mma_col_threads;

        assert(channel % swizzle_elements == 0); // Irregular channel size is not yet supported
        assert(seq % 2 == 0 || batch == 1);

        // batch, seq, head, channel
        smem_shape[0] = swizzle_elements;
        smem_shape[1] = 2;
        smem_shape[2] = 4;
        smem_shape[3] = shared_tile_height / 8;
        smem_shape[4] = shared_tile_width / swizzle_elements;

        gmem_stride[0] = (uint64_t)channel * head * sizeof(dtype);
        gmem_stride[1] = (uint64_t)(shared_tile_height / 4) * channel * head * sizeof(dtype); 
        gmem_stride[2] = (uint64_t)2 * channel * head * sizeof(dtype);
        gmem_stride[3] = (uint64_t)ST::swizzle_bytes;

        gmem_shape[0] = swizzle_elements;
        gmem_shape[1] = seq % 2; // k
        gmem_shape[2] = seq / (shared_tile_height / 4); // j
        gmem_shape[3] = (seq % (shared_tile_height / 4)) / 2; // i
        gmem_shape[4] = (uint64_t)head * (shared_tile_width / swizzle_elements); // *

        constexpr uint32_t tma_dim = ENABLE_SWIZZLE ? 5 : 4;
        void *global_addr = (void*)(src);

        constexpr CUtensorMapDataType     tma_format      = (
            std::is_same_v<dtype, kt::bf16>  ? CU_TENSOR_MAP_DATA_TYPE_BFLOAT16 :
            std::is_same_v<dtype, kt::half>  ? CU_TENSOR_MAP_DATA_TYPE_FLOAT16 :
            std::is_same_v<dtype, float> ? CU_TENSOR_MAP_DATA_TYPE_FLOAT32 :
        #ifdef KITTENS_FEATURE_FP8
            std::is_same_v<dtype, kt::fp8e4m3> ? CU_TENSOR_MAP_DATA_TYPE_UINT8 :
            std::is_same_v<dtype, kt::fp8e5m2> ? CU_TENSOR_MAP_DATA_TYPE_UINT8 :
        #endif
        #ifdef KITTENS_FEATURE_UE8
            std::is_same_v<dtype, kt::fp8e8m0> ? CU_TENSOR_MAP_DATA_TYPE_UINT8 :
        #endif
        #ifdef KITTENS_FEATURE_FP4
            std::is_same_v<dtype, kt::fp4_2> ? CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B :
        #endif
            CUtensorMapDataType(-1)
        );
        constexpr CUtensorMapInterleave   tma_interleave  = CU_TENSOR_MAP_INTERLEAVE_NONE;
        constexpr CUtensorMapL2promotion  tma_l2Promotion = CU_TENSOR_MAP_L2_PROMOTION_NONE;
        constexpr CUtensorMapFloatOOBfill tma_oobFill     = CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE;
        constexpr CUtensorMapSwizzle      tma_swizzle     = ENABLE_SWIZZLE ? (
            ST::swizzle_bytes == 32  ? CU_TENSOR_MAP_SWIZZLE_32B  :
            ST::swizzle_bytes == 64  ? CU_TENSOR_MAP_SWIZZLE_64B  :
            ST::swizzle_bytes == 128 ? CU_TENSOR_MAP_SWIZZLE_128B : 
            CU_TENSOR_MAP_SWIZZLE_NONE
        ) : CU_TENSOR_MAP_SWIZZLE_NONE;

        CUresult result = cuTensorMapEncodeTiled(
            tma_map,
            tma_format,
            tma_dim,
            global_addr,
            gmem_shape,
            gmem_stride, 
            smem_shape,
            smem_stride,
            tma_interleave,
            tma_swizzle,
            tma_l2Promotion,
            tma_oobFill);
            
        const char *error_string;
        CUresult res = cuGetErrorString(result, &error_string);
        if (result != CUDA_SUCCESS) {
            std::string error_msg = kt::detail::tma::format_tma_error(
                "tile", error_string,
                batch, seq, head, channel,
                tma_map, tma_format, tma_dim, global_addr,
                gmem_shape, gmem_stride,
                smem_shape, smem_stride,
                5, 4, 5, 5,
                tma_interleave, tma_swizzle, tma_l2Promotion, tma_oobFill,
                "ST::rows: " + std::to_string(ST::rows) + "\n  ST::cols: " + std::to_string(ST::cols)
            );
            throw std::runtime_error(error_msg);
        }
    }

    template<int AXIS, kt::ducks::gl::all GL, typename COORD>
    static __device__ int4 get_coords(const COORD& coord, const GL& src){ // NOLINT
        constexpr int swizzle_elements = ST::swizzle_bytes / sizeof(dtype);
        constexpr uint64_t shared_tile_height = ST::rows; 
        constexpr uint64_t shared_tile_width  = ST::cols;

        // coord: batch, seq, head, channel
        auto batch_idx = coord.b, seq_idx = coord.d, head_idx = coord.r, channel_idx = coord.c;
        auto batch = src.batch(), seq = src.depth(), head = src.rows(), channel = src.cols();

        auto offset1 = seq_idx % 2;
        auto offset2 = seq_idx / (shared_tile_height / 4);
        auto offset3 = (seq_idx % (shared_tile_height / 4)) / 2; 
        auto offset4 = head_idx * (shared_tile_width / swizzle_elements);
        
        return { int(offset1), 0, 0, int(offset4) };
    }
};

template<int WARPS, bool INIT, int... ENDS_IDX>
struct CommEndInfo{
    static constexpr bool INITIAL = INIT;
    
    static constexpr int N_WARPS = WARPS * sizeof...(ENDS_IDX);
    static constexpr int LOG2_WARPS = __builtin_ffs(N_WARPS) - 1;
    static constexpr int ENDS[]  = {ENDS_IDX...};

    static_assert(N_WARPS == (1 << LOG2_WARPS), "Warp number must be power of 2");

    static constexpr bool hasEnd(int end_idx) {
        for(int i = 0; i < (int)sizeof...(ENDS_IDX); i++) if(ENDS[i] == end_idx) return true;
        return false;
    }
};

template<bool INIT, int... ENDIDX>
constexpr CommEndInfo<1, INIT, ENDIDX...> ProducerWarp = {};

template<bool INIT, int... ENDIDX>
constexpr CommEndInfo<4, INIT, ENDIDX...> ConsumerWarpGroup = {};

template<typename... Ts>
struct CommEndCollection{
    static constexpr int MAX_BITS = std::max({Ts::LOG2_WARPS...});
    inline static constexpr int findEnd(int end_idx){
        bool masks[] = {Ts::hasEnd(end_idx)...};
        for(int i = 0; i < (int)sizeof...(Ts); i++){
            if(masks[i]) return i;
        }
        return -1;
    }
    static constexpr int getWarpBits(int end_idx){
        int warps[] = {Ts::LOG2_WARPS...};
        return warps[findEnd(end_idx)];
    }
    static constexpr bool isInitial(int end_idx){
        return ((Ts::hasEnd(end_idx) && Ts::INITIAL) || ...);
    }
};


template<int N_BUFFERS, int N_ENDS, typename TLayout, typename... Ts>
struct MbarrierMultiEndSwitcher{
    kt::semaphore (&Barriers)[N_ENDS][N_BUFFERS];
    TLayout (&LayoutStorage)[N_BUFFERS];
    uint32_t PhaseState;
    uint32_t Slot, CurrIdx, ArriveCount;
    CommEndCollection<Ts...> EndDesc;

    __forceinline__ __device__ MbarrierMultiEndSwitcher(kt::shared_allocator<>& alloc, TLayout (&layoutStorage_)[N_BUFFERS], uint32_t role_idx, Ts... ends)
    : Barriers(alloc.template allocate<kt::semaphore, N_ENDS, N_BUFFERS>())
    , LayoutStorage(layoutStorage_), PhaseState(0), Slot(0) {
        static_assert(sizeof...(Ts) == N_ENDS, "The number of communication ends mismatches!");
        

        uint32_t wait_count = 1u << EndDesc.MAX_BITS;

        CurrIdx = EndDesc.findEnd(role_idx);
        ArriveCount = 1u << (EndDesc.MAX_BITS - EndDesc.getWarpBits(role_idx));

        PhaseState = EndDesc.isInitial(role_idx) ? ((1u << N_BUFFERS) - 1) : 0;
        if(kt::warpid() == 0){
            #pragma unroll
            for(int i = 0; i < N_BUFFERS; i++){
                #pragma unroll
                for(int j = 0; j < N_ENDS; j++){
                    kt::init_semaphore(Barriers[j][i], wait_count);
                }
            }
        }
    }

    __forceinline__ __device__ void setup(uint32_t slot = 0){Slot = slot;}

    struct Handle{
        MbarrierMultiEndSwitcher* PipeRef;
        uint32_t SlotIdx, NextEnd;

        __forceinline__ __device__ TLayout* operator->() const { return &PipeRef->LayoutStorage[SlotIdx]; }

        __forceinline__ __device__ kt::semaphore& getBarrier() const { return PipeRef->Barriers[PipeRef->EndDesc.findEnd(NextEnd)][SlotIdx]; }

        __forceinline__ __device__ void submitToNextAndTrigger(bool next_slot = true){
            PipeRef->PhaseState ^= 1u << (PipeRef->Slot);
            if(next_slot) PipeRef->Slot = (PipeRef->Slot + 1) % N_BUFFERS;
            kt::warp::arrive(getBarrier(), PipeRef->ArriveCount);
        }
        __forceinline__ __device__ int getArrivalCount() const {
            return PipeRef->ArriveCount;
        }
    };

    __forceinline__ __device__ Handle getNullHandle(){
        return {this,0,0};
    }

    __forceinline__ __device__ Handle waitBuffer(uint32_t next_end){
        kt::warp::wait(Barriers[CurrIdx][Slot], (PhaseState >> (Slot)) & 1);
        return { this, Slot, next_end };
    }
};

template<int N_BUFFERS, typename TLayout, typename... Ts>
MbarrierMultiEndSwitcher(kt::shared_allocator<>& alloc, TLayout (&layoutStorage_)[N_BUFFERS], uint32_t role_idx, Ts... endWarps) -> MbarrierMultiEndSwitcher<N_BUFFERS, sizeof...(Ts), TLayout, Ts...>;



namespace common{
    template<typename T>
    struct FallbackTuple{ T x, y; };

    template<typename T>
    using T2Tuple = std::conditional_t<std::is_same_v<T, float>, float2, FallbackTuple<T>>;

    template<typename TX, typename TY>
    struct Tuple2{ TX x; TY y; };

    __device__  __forceinline__ float2 pack_shfl_sync(const uint32_t mask, const float2 value, const int idx, const int lane = 32){
    return float2 { __shfl_sync(mask, value.x, idx, lane), __shfl_sync(mask, value.y, idx, lane) };
}

__device__  __forceinline__ float2 pack_shfl_xor_sync(const uint32_t mask, const float2 value, const int idx, const int lane = 32){
    return float2 { __shfl_xor_sync(mask, value.x, idx, lane), __shfl_xor_sync(mask, value.y, idx, lane) };
}

__device__  __forceinline__ float pack_shfl_sync(const uint32_t mask, const float value, const int idx, const int lane = 32){
    return __shfl_sync(mask, value, idx, lane);
}

__device__  __forceinline__ float pack_shfl_xor_sync(const uint32_t mask, const float value, const int idx, const int lane = 32){
    return __shfl_xor_sync(mask, value, idx, lane);
}

__device__  __forceinline__ half2 pack_shfl_sync(const uint32_t mask, const half2 value, const int idx, const int lane = 32){
    return __shfl_sync(mask, value, idx, lane);
}

__device__  __forceinline__ half2 pack_shfl_xor_sync(const uint32_t mask, const half2 value, const int idx, const int lane = 32){
    return __shfl_xor_sync(mask, value, idx, lane);
}

template<int A, int B>
static inline constexpr int ct_max_v = (A > B ? A : B);
template<int A, int B>
static inline constexpr int ct_min_v = (A < B ? A : B);
template<int N, class F>
static __host__ __device__ inline void static_for(F&& f) {
    []<size_t... IDX>(std::index_sequence<IDX...>, F&& f2) {
        (f2(std::integral_constant<int, (int)IDX>{}), ...);
    }(std::make_index_sequence<N>{}, (F&&)f);
}

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wformat-security"
template<typename... Ts>
__device__ void print(const char *fmt, Ts&&... args){
    if((threadIdx.x & 0x1f) == 0) printf(fmt, std::forward<Ts>(args)...);
}
#pragma clang diagnostic pop

} // namespace common


namespace ops{

struct Log2Map{
    template<typename T>
    __forceinline__ __device__ static T op(T x);
};

template<>
__forceinline__ __device__ float Log2Map::op(float x){
    float y;
    asm volatile(
        "lg2.approx.ftz.f32 %0, %1;\n":
        "=f"(y):"f"(x));
    return y;
}

template<>
__forceinline__ __device__ float2 Log2Map::op(float2 x){
    float2 y;
    asm volatile(
        "lg2.approx.ftz.f32 %0, %2;\n"
        "lg2.approx.ftz.f32 %1, %3;\n":
        "=f"(y.x),"=f"(y.y):"f"(x.x),"f"(x.y));
    return y;
}


struct FpAddLseOpSlow{
    template<typename T>
    __device__ static T op(T x, T y);
    template<typename T>
    __device__ static T get_zero();
};

template<>
__device__ std::tuple<float, float> FpAddLseOpSlow::op(std::tuple<float, float> lhs, std::tuple<float, float> rhs){
    float x = std::get<1>(lhs) + std::get<0>(rhs), y = std::get<1>(rhs);
    float nabs = -abs(x - y), mx = max(x, y); 

    float p = max(nabs, -4.3125f); 
    float logs = nabs + p * (-2.793604998770852221e-01f + p * (-5.756650101889571047e-02f + p * (-4.553042169205498771e-03f))), res;
    asm volatile("ex2.approx.ftz.f32 %0, %1;\n":"=f"(res): "f"(logs));
    return { std::get<0>(lhs) + std::get<0>(lhs), mx + res};
}

template<>
constexpr __device__ std::tuple<float, float> FpAddLseOpSlow::get_zero(){
    return {0.f, -99.f};
}

static __forceinline__ __device__ float2 operator+(float2 lhs, float2 rhs) { return {lhs.x + rhs.x, lhs.y + rhs.y}; }
static __forceinline__ __device__ float2 operator+(float lhs, float2 rhs) { return {lhs + rhs.x, lhs + rhs.y}; }
static __forceinline__ __device__ float2 operator-(float2 lhs, float2 rhs) { return {lhs.x - rhs.x, lhs.y - rhs.y}; }
static __forceinline__ __device__ float2 operator*(float lhs, float2 rhs) { return {lhs * rhs.x, lhs * rhs.y}; }
static __forceinline__ __device__ float2 operator*(float2 lhs, float rhs) { return {lhs.x * rhs, lhs.y * rhs}; }
static __forceinline__ __device__ float2 operator*(float2 lhs, float2 rhs) { return {lhs.x * rhs.x, lhs.y * rhs.y}; }
static __forceinline__ __device__ float2 operator-(float2 v) { return {-v.x, -v.y}; }
static __forceinline__ __device__ float2 absf2(float2 v) { return float2 { abs(v.x), abs(v.y)}; }
static __forceinline__ __device__ float2 maxf2(float2 lhs, float2 rhs) { return float2 { max(lhs.x, rhs.x), max(lhs.y, rhs.y)}; }
static __forceinline__ __device__ float2 maxf2(float2 lhs, float rhs) { return float2 { max(lhs.x, rhs), max(lhs.y, rhs)}; }

static __forceinline__ __device__ half2 absf2(half2 v) { 
    asm volatile("or.b32 %0, %0, 0x80008000;\n":"+r"(reinterpret_cast<uint32_t&>(v)));
    return v;
}
static __forceinline__ __device__ half2 maxf2(half2 lhs, half2 rhs) { return __hmax2(lhs, rhs); }
static __forceinline__ __device__ half2 maxf2(half2 lhs, float rhs) { return __hmax2(lhs, __float2half2_rn(rhs)); }

template<>
__device__ std::tuple<float2, float2> FpAddLseOpSlow::op(std::tuple<float2, float2> lhs, std::tuple<float2, float2> rhs){
    float2 x = std::get<1>(lhs) + std::get<0>(rhs), y = std::get<1>(rhs);
    float2 nabs = -absf2(x - y), mx = maxf2(x, y); 

    float2 p = maxf2(nabs, -4.3125f); 
    float2 logs = nabs + p * (-2.793604998770852221e-01f + p * (-5.756650101889571047e-02f + p * (-4.553042169205498771e-03f))), res;

    asm volatile("ex2.approx.ftz.f32 %0, %2;\nex2.approx.ftz.f32 %1, %3;\n":"=f"(res.x),"=f"(res.y): "f"(logs.x),"f"(logs.y));
    return { std::get<0>(lhs) + std::get<0>(rhs), mx + res};
}

template<>
__device__ std::tuple<float2, float2> FpAddLseOpSlow::get_zero(){
    return {{0.f,0.f}, {-99.f,-99.f}};
}

/*template<>
__device__ common::T2Tuple<half2> FpAddLseOpSlow::op(common::T2Tuple<half2> lhs, common::T2Tuple<half2> rhs){
    half2 x = lhs.y + rhs.x, y = rhs.y;
    half2 nabs = -absf2(x - y), mx = maxf2(x, y); 

    half2 p = maxf2(nabs, -4.3125f); 
    half2 logs = nabs + p * (__float2half2_rn(-2.793604998770852221e-01f) + p * (__float2half2_rn(-5.756650101889571047e-02f) + p * __float2half2_rn(-4.553042169205498771e-03f))), res;
    asm("ex2.approx.f16x2 %0, %1;" : "=r"(res) : "r"(logs));
    return common::T2Tuple<half2>{ lhs.x + rhs.x, mx + res};
}

template<>
__device__ common::T2Tuple<half> FpAddLseOpSlow::op(common::T2Tuple<half> lhs, common::T2Tuple<half> rhs){
    half x = lhs.y + rhs.x, y = rhs.y;
    half nabs = -__habs(x - y), mx = __hmax(x, y); 

    half p = __hmax(nabs, -4.3125f); 
    half logs = nabs + p * (__float2half_rn(-2.793604998770852221e-01f) + p * (__float2half_rn(-5.756650101889571047e-02f) + p * __float2half_rn(-4.553042169205498771e-03f))), res;
    asm("ex2.approx.f16 %0, %1;" : "=r"(res) : "r"(logs));
    return common::T2Tuple<half>{ lhs.x + rhs.x, mx + res};
}*/


} // namespace ops

} // namespace details

namespace sm90{

struct PreprocessKernelSm90Config {
    int head_dim, qk_dim, k_stages;
    float qk_eps, qk_bias, qk_log_offset;

    static constexpr int WARPGROUPS = 2;
    static constexpr int K_PREFETCH_SIZE = 1;
    int warp_q_size, warp_k_size;

    constexpr PreprocessKernelSm90Config(int head_dim, int qk_dim, int k_stages, int warp_q_size, int k_size, float eps)
        :head_dim(head_dim), qk_dim(qk_dim), k_stages(k_stages), warp_q_size(warp_q_size), warp_k_size(k_size), qk_eps(eps){
        qk_bias = util::log2f(1 - qk_eps * qk_dim);
        qk_log_offset = qk_eps / (1 - qk_eps * qk_dim);
    }

    constexpr int getCheckpointSize() const { return WARPGROUPS * warp_q_size; }
    constexpr int getQBlockSize() const { return getCheckpointSize() * kt::WARPGROUP_WARPS; }
    constexpr int getSkipSize() const { return getCheckpointSize(); }
};

constexpr PreprocessKernelSm90Config DEFAULT_CONFIG = PreprocessKernelSm90Config(64, 64, 3, 16, 64, 1e-3f);

template <PreprocessKernelSm90Config CONFIG = DEFAULT_CONFIG>
struct Globals {
    // Input layout: [B, N, H, C]/[1, N, H, C]
    static constexpr auto C = CONFIG;
    kt::gl<kt::half, -1, -1, -1, C.qk_dim, details::TmaWarpgroupStrided<kt::st_hf<64, CONFIG.qk_dim>, CONFIG.getCheckpointSize()>> Q;
    kt::gl<kt::half, -1, -1, -1, C.qk_dim, details::TmaColumnGrouped<kt::st_hf<C.warp_k_size, C.qk_dim>>> K;
    kt::gl<float, 1, 1, 1, -1> RcpTau;

    kt::gl<float, -1, -1, -1, C.warp_k_size> VBuffer; 
    kt::gl<float, -1, -1, -1, C.warp_k_size> HBuffer;
};

// Scheduler for non-varlen attention
template<PreprocessKernelSm90Config CONFIG>
struct FixedLengthScheduler{
    static_assert(CONFIG.getQBlockSize() >= CONFIG.warp_k_size, "Cases for Q<K not implemented!!");

    struct TaskInfo{
        int Batch, QStart, KStart, Head;
        int KBlocks, VHStart;
    };

    int CurrentTaskIdx, Batch, SeqlenQBlocks, Head;

    __device__ FixedLengthScheduler(const Globals<CONFIG>& args){
        CurrentTaskIdx = blockIdx.x;
        Batch = args.Q.batch();
        SeqlenQBlocks = (args.Q.depth() + CONFIG.getQBlockSize() - 1) / CONFIG.getQBlockSize(); 
        Head = args.Q.rows();
    }

    __forceinline__ __device__ bool getNextTask(TaskInfo& out){ 
        auto wid = kt::warpgroup::warpid();

        auto Idx = CurrentTaskIdx;
        auto Total = SeqlenQBlocks * Batch * Head;
        if(Idx >= Total) return false;
        out.Batch = Idx / (Head * SeqlenQBlocks);
        Idx -= out.Batch * (Head * SeqlenQBlocks);
        out.Head = Idx / SeqlenQBlocks;
        Idx -= out.Head * SeqlenQBlocks;

        out.QStart = CONFIG.getQBlockSize() * Idx;
        out.KStart = 0;
        out.KBlocks = (out.QStart + CONFIG.getQBlockSize() + CONFIG.warp_k_size - 1) / CONFIG.warp_k_size;

        out.VHStart = 4 * (CONFIG.getQBlockSize() / CONFIG.warp_k_size) * Idx * (Idx + 1) / 2 + wid * out.KBlocks;
        
        CurrentTaskIdx += gridDim.x;
        return true; 
    }
};

template<PreprocessKernelSm90Config CONFIG>
union LoadSharedMemoryLayouts{
    struct DefaultLayout {
        details::TmaColumnGrouped<kt::st_hf<CONFIG.warp_k_size, CONFIG.qk_dim>> k_buffer;
    } Default[CONFIG.k_stages];
    struct PrefetchLayout {
        DefaultLayout preflight_k[CONFIG.K_PREFETCH_SIZE];
        details::TmaWarpgroupStrided<kt::st_hf<CONFIG.warp_q_size * 4, CONFIG.qk_dim>, CONFIG.getSkipSize()> q_buffer[CONFIG.WARPGROUPS];
    } Prefetch[1];
};

template<PreprocessKernelSm90Config CONFIG>
struct IntermediateSharedMemoryLayouts{
    struct{
        sxdiag::SharedTopBottomVec<CONFIG.warp_k_size, float> h_buffer;
        sxdiag::SharedTopBottomVec<CONFIG.warp_k_size, float> v_buffer;
    } defaults[CONFIG.k_stages];
};

template<uint32_t N_UNROLL, typename TFunctor>
__forceinline__ __device__ void unroll_helper(TFunctor&& fn, uint32_t N){
    uint32_t extra = N % N_UNROLL;
    #pragma unroll 1
    for(uint32_t i = 0; i < N / N_UNROLL; i++) {
        #pragma unroll
        for(uint32_t j = 0; j < N_UNROLL; j++) fn(i * N_UNROLL + j);
    }
}

static __launch_bounds__(256 + 32,1) __global__ void kernel(const __grid_constant__ Globals<DEFAULT_CONFIG> args) {
    constexpr PreprocessKernelSm90Config CONFIG = DEFAULT_CONFIG;
    using Scheduler = FixedLengthScheduler<CONFIG>;
    using wg = kt::warpgroup;

    extern __shared__ int shmem[];
    kt::shared_allocator<> alloc{shmem};
    LoadSharedMemoryLayouts<CONFIG> &load_layouts = alloc.template allocate<LoadSharedMemoryLayouts<CONFIG>>();
    IntermediateSharedMemoryLayouts<CONFIG> &vh_layouts = alloc.template allocate<IntermediateSharedMemoryLayouts<CONFIG>>();


    uint32_t wgid = kt::warpgroup::groupid() % CONFIG.WARPGROUPS;
    constexpr uint32_t CONSUMER_A = 0, CONSUMER_B = 1, PRODUCER = 2;
    constexpr uint32_t PRODUCER_LOAD = 0, PRODUCER_STORE = 1;

    details::MbarrierMultiEndSwitcher k_load_pipe( alloc, load_layouts.Default, wgid, details::ProducerWarp<true, PRODUCER>, details::ConsumerWarpGroup<false, CONSUMER_A, CONSUMER_B>);
    details::MbarrierMultiEndSwitcher prefetch_load_pipe( alloc, load_layouts.Prefetch, wgid, details::ProducerWarp<true, PRODUCER>, details::ConsumerWarpGroup<false, CONSUMER_A, CONSUMER_B>);
    details::MbarrierMultiEndSwitcher vh_swap_pipe( alloc, vh_layouts.defaults, wgid, details::ConsumerWarpGroup<true, CONSUMER_A>, details::ConsumerWarpGroup<false, CONSUMER_B>);

    Scheduler scheduler(args);
    typename Scheduler::TaskInfo task;

    if(wgid == PRODUCER){ // producer groups
        while(scheduler.getNextTask(task)){
            if(kt::warpgroup::warpid() == PRODUCER_LOAD){ // loader
                //kt::print_utils::print("Task info: cta = %u, batch = %u, head = %u, n_kblocks = %u, vh_start = %u, qstart = %u\n", blockIdx.x, task.Batch, task.Head, task.KBlocks, task.VHStart, task.QStart);
                // first load Q and first K block
                prefetch_load_pipe.setup();
                auto prefetch_packet = prefetch_load_pipe.waitBuffer(CONSUMER_A); // just to match details::ConsumerWarpGroup<false, CONSUMER_A, CONSUMER_B>
                ([&]<size_t... IDX_Q, size_t... IDX_K>(std::index_sequence<IDX_Q...>, std::index_sequence<IDX_K...>){
                    kt::warp::tma::expect(prefetch_packet.getBarrier(), prefetch_packet->preflight_k[IDX_K]..., prefetch_packet->q_buffer[IDX_Q]...);
                    (kt::warp::tma::load_async<1, kt::cache_policy::NORMAL>(prefetch_packet->q_buffer[IDX_Q], args.Q, { task.Batch, task.QStart + int(IDX_Q * 16), task.Head, 0}, prefetch_packet.getBarrier()), ...);
                    (kt::warp::tma::load_async<1, kt::cache_policy::NORMAL>(prefetch_packet->preflight_k[IDX_K].k_buffer, args.K, { task.Batch, task.KStart + int(IDX_K), task.Head, 0}, prefetch_packet.getBarrier()), ...); // NOTE: AXIS MAY NOT CORRECT!!!
                })(std::make_index_sequence<CONFIG.WARPGROUPS>{}, std::make_index_sequence<CONFIG.K_PREFETCH_SIZE>{});
                
                prefetch_packet.submitToNextAndTrigger();
                    
                prefetch_load_pipe.waitBuffer(CONSUMER_A); // wait for Q is consumed and switch layout
                

                k_load_pipe.setup(); 

                #pragma unroll
                for(int i = 0; i < CONFIG.K_PREFETCH_SIZE; i++) k_load_pipe.waitBuffer(CONSUMER_A).submitToNextAndTrigger(); // already loaded
                
                #pragma unroll 1
                for(int iter = CONFIG.K_PREFETCH_SIZE; iter < task.KBlocks; iter++){
                    auto load_packet = k_load_pipe.waitBuffer(CONSUMER_A);
                    //kt::print_utils::print("Producer emit load %u (slot = %u)\n", iter, load_packet.SlotIdx);
                    kt::warp::tma::expect(load_packet.getBarrier(), load_packet->k_buffer);
                    kt::warp::tma::load_async<1, kt::cache_policy::NORMAL>(load_packet->k_buffer, args.K, { task.Batch, task.KStart + iter, task.Head, 0}, load_packet.getBarrier()); // NOTE: AXIS AND OFFSET IS INCORRECT
                    load_packet.submitToNextAndTrigger(); 
                }
            } 
        }
    } else {
        using MainOp = details::ops::FpAddLseOpSlow;
        constexpr auto ZERO = MainOp::get_zero<std::tuple<float,float>>();
        uint32_t warpid = kt::warpgroup::warpid();

        kt::rt_hf<CONFIG.warp_q_size, CONFIG.qk_dim> q_tile;
        kt::rt_fl<CONFIG.warp_q_size, CONFIG.warp_k_size> acc_front;

        auto vh_ptr = task.VHStart;
        auto storeVHVecs = [&](const decltype(vh_swap_pipe)::Handle& handle){
            handle->v_buffer.store(args.VBuffer, {task.Batch, task.Head, vh_ptr, 0});
            handle->h_buffer.store(args.HBuffer, {task.Batch, task.Head, vh_ptr, 0});
            vh_ptr++;
        };

        

        while(scheduler.getNextTask(task)){
            sxdiag::LeftRightVec<CONFIG.warp_q_size, float, float> lr_state{std::get<0>(ZERO),std::get<1>(ZERO)};
            sxdiag::TopBottomVec<CONFIG.warp_k_size, float, float> tb_state;

            uint32_t consumer_idx = wg::groupid() % CONFIG.WARPGROUPS;
            float rtau = args.RcpTau.raw_ptr[task.Head] + CONFIG.qk_bias;

            prefetch_load_pipe.setup();
            auto init_block = prefetch_load_pipe.waitBuffer(PRODUCER);
            //kt::print_utils_group<4>::print("Consumer %u get Q tile\n", wgid);
            

            typename decltype(k_load_pipe)::Handle k_handle = k_load_pipe.getNullHandle();

            kt::warpgroup::load(q_tile, init_block->q_buffer[consumer_idx].data);
            
            init_block.submitToNextAndTrigger();

            k_load_pipe.setup();
            vh_swap_pipe.setup();

            
            #pragma unroll 1
            for(int i = 0; i < task.KBlocks; i++){
                auto tb_handle = vh_swap_pipe.waitBuffer((wgid + 1) % CONFIG.WARPGROUPS);
                k_handle = k_load_pipe.waitBuffer(PRODUCER);
                wg::emulated_wgmma::mm_ABt(acc_front, q_tile, k_handle->k_buffer.data);

                if(wgid == CONSUMER_A){
                    tb_state = ZERO;
                    storeVHVecs(tb_handle);
                } else {
                    tb_state.load<0>(tb_handle->h_buffer);
                    tb_state.load<1>(tb_handle->v_buffer);
                }

                kt::warp::rt_maps::unary_map<details::ops::Log2Map>(acc_front, acc_front + CONFIG.qk_log_offset);
                auto scan_buffer = sxdiag::TightMMABuffer<float, CONFIG.warp_q_size, CONFIG.warp_k_size>::from_rt(acc_front + rtau);

                auto u = scan_buffer.expand(std::get<0>(ZERO));
                auto v = u.with_padding(std::get<1>(ZERO));

                sxdiag::ScanTile<CONFIG.warp_q_size, CONFIG.warp_k_size, float, float> tile(u, v);
                
                auto red_res = tile.diagScanOrReduce<false, false, MainOp>(sxdiag::StatePair<CONFIG.warp_q_size, CONFIG.warp_k_size, float, float>::from(lr_state, tb_state));

                lr_state = red_res.lr;

                red_res.tb.store<0>(tb_handle->h_buffer);
                red_res.tb.store<1>(tb_handle->v_buffer);
                
                tb_handle.submitToNextAndTrigger();

            }
            



            /*if(wgid == CONSUMER_A){ // store last
                auto tb_handle = vh_swap_pipe.waitBuffer((wgid + 1) % CONFIG.WARPGROUPS);
                storeVHVecs(tb_handle);
                // maybe we need to release tb_handle here?
                
            } else {
                if(kt::warp::laneid() == 31){
                    args.HBuffer[kt::coord<>{task.Batch, task.Head, task.VHStart,0}] = std::get<0>(lr_state.data[lr_state.ROW_UNITS - 1]);
                    args.VBuffer[kt::coord<>{task.Batch, task.Head, task.VHStart,0}] = std::get<1>(lr_state.data[lr_state.ROW_UNITS - 1]);
                }
            }*/
        }
    }
}


} // namespace sm90

//using TMAType = details::TmaColumnGrouped<kt::st_bf<64, 32>>;
/*__global__ void testTMA(const __grid_constant__ kt::gl<kt::bf16, -1, -1, -1, 32, TMAType> gm, int b, int s, int n){
    __shared__ TMAType buffer;
    __shared__ kt::semaphore sem;

    kt::warp::init_semaphore(sem, 1);
    __syncthreads();
    
    kt::warp::tma::expect(sem, buffer);
    kt::warp::tma::load_async<1, kt::cache_policy::NORMAL>(buffer, gm, {b,s,n,0}, sem);

    kt::warp::arrive(sem);
    kt::warp::wait(sem, 0);

    kt::print_utils::print(buffer.data);
}*/

} // namespace

extern void invokeTMA(void *buffer, size_t batch, size_t seq, size_t head, int b, int s, int n){
    //testTMA<<<1, 32>>>({(kt::bf16*)buffer, batch, seq, head, 0}, b, s, n);
    cudaDeviceSynchronize();
}

template <int N_HEADDIM, int N_KEYDIM>
void BaselineNoPEAttnStateImpl<N_HEADDIM, N_KEYDIM>::invokeFwdPreprocess(const FwdPreprocessArgs& args){
    using sm90::DEFAULT_CONFIG;
    using Globals = sm90::Globals<DEFAULT_CONFIG>;

    dim3 grid(1); // single CTA for test
    dim3 block(kt::WARPGROUP_THREADS * DEFAULT_CONFIG.WARPGROUPS + kt::WARP_THREADS * 2);
    auto shape = getVHBufferShape(args.Batch, args.Head, args.Seqlen);

    size_t shared_memory = 49152;

    sm90::kernel<<<grid, block, shared_memory>>>(Globals{
        { args.Q, args.Batch, args.Seqlen, args.Head, N_KEYDIM },
        { args.K, args.Batch, args.Seqlen, args.Head, N_KEYDIM },
        { args.RcpTau, 0, 0, 0, args.Head },
        { args.VBuffer, shape[0], shape[1], shape[2], 0 },
        { args.HBuffer, shape[0], shape[1], shape[2], 0 }
    });

    cudaDeviceSynchronize();
    //printf("Preproess kernel exit status: %s\n", cudaGetErrorString(cudaGetLastError()));
}

template <int N_HEADDIM, int N_KEYDIM>
std::array<size_t, 4> BaselineNoPEAttnStateImpl<N_HEADDIM, N_KEYDIM>::getVHBufferShape(size_t batch, size_t head, size_t seqlen){
    constexpr int q_block_size = sm90::DEFAULT_CONFIG.getQBlockSize();
    constexpr int qblock_vh_size = 4 * q_block_size / sm90::DEFAULT_CONFIG.warp_k_size;
    size_t q_blocks = (seqlen + q_block_size - 1) / q_block_size;
    size_t vh_elems = qblock_vh_size * (q_blocks + 1) * q_blocks / 2;
    return {batch, head, vh_elems, sm90::DEFAULT_CONFIG.warp_k_size};
}

template struct BaselineNoPEAttnStateImpl<64, 64>;