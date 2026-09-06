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

struct TmaExtension{
    template<bool COMMIT = false, kt::ducks::gl::all GL, kt::ducks::sv::all SV, typename COORD = kt::coord<SV>>
    __forceinline__ __device__ static void storeAsync(GL& dst, const SV& src, COORD idx){
        constexpr uint32_t transfer_size = SV::length * sizeof(typename SV::dtype);
        typename GL::dtype *dst_ptr = (typename GL::dtype*)&dst[(idx.template unit_coord<-1, 3>())];
        uint32_t src_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&src.data[0]));
        
        asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;\n"::"l"(dst_ptr),"r"(src_ptr),"n"(transfer_size));
        if constexpr(COMMIT) asm volatile("cp.async.bulk.commit_group;\n");
    }

    template<bool COMMIT = false, kt::ducks::gl::all GL, kt::ducks::sv::all SV, typename COORD = kt::coord<SV>>
    __forceinline__ __device__ static void loadAsync(SV& dst, GL& src, COORD idx, kt::semaphore& sem){
        constexpr uint32_t transfer_size = SV::length * sizeof(typename SV::dtype);
        typename GL::dtype *src_ptr = (typename GL::dtype*)&src[(idx.template unit_coord<-1, 3>())];
        uint32_t dst_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&dst.data[0]));
        uint32_t sem_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&sem));

        asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"::"r"(dst_ptr),"l"(src_ptr),"n"(transfer_size),"r"(sem_ptr));
        if constexpr(COMMIT) asm volatile("cp.async.bulk.commit_group;\n");
    }
};


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
            std::cerr << "Error at creating tensor map for " << __PRETTY_FUNCTION__ << std::endl;
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


template<typename TData, int ROWS, int COLS>
struct TmaColumnGrouped {
    using identifier = kt::ducks::tma::wrapper::identifier;
    using ST = kt::st<TData, ROWS, COLS>;
    using dtype = TData;
    using T_ = ST;
    static constexpr int value = 1; // NOLINT
    static constexpr bool swizzle_flag = true; // NOLINT
    static constexpr uint32_t num_elements = ST::num_elements; // NOLINT
    static constexpr uint32_t bytes = num_elements * sizeof(dtype); // NOLINT

    kt::st<TData, ROWS, COLS> data; // NOLINT

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

        assert(channel % swizzle_elements == 0); // Irregular channel size is not yet supported
        assert(seq % (shared_tile_height / 8) == 0);

        // batch, seq, head, channel
        smem_shape[0] = swizzle_elements;
        smem_shape[1] = 8;
        smem_shape[2] = shared_tile_height / 8;
        smem_shape[3] = shared_tile_width / swizzle_elements;
        smem_shape[4] = 1;

        gmem_stride[0] = (uint64_t)(shared_tile_height / 8) * channel * head * sizeof(dtype);
        gmem_stride[1] = (uint64_t)channel * head * sizeof(dtype); 
        gmem_stride[2] = (uint64_t)ST::swizzle_bytes;
        gmem_stride[3] = (uint64_t)seq * head * channel * sizeof(dtype);

        gmem_shape[0] = swizzle_elements;
        gmem_shape[1] = seq / (shared_tile_height/8); 
        gmem_shape[2] = shared_tile_height / 8 * 2; 
        gmem_shape[3] = (shared_tile_width / swizzle_elements) * head; 
        gmem_shape[4] = (uint64_t)batch; 

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

        auto offset1 = (seq_idx) / (shared_tile_height / 8);
        auto offset2 = head_idx * (shared_tile_width / swizzle_elements);
        auto offset3 = (seq_idx) % (shared_tile_height / 8);
        return { int(offset1), int(offset3), int(offset2), int(batch_idx) };
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


struct SigmoidFast{
    template<typename T>
    __forceinline__ __device__ static T op(T x);
};

template<>
__forceinline__ __device__ float SigmoidFast::op(float x){
    float y;
    asm volatile(
        "mul.f32 %1, %1, 0f3EB17218;\n"
        "tanh.approx.f32 %1, %1;\n"
        "fma.rn.f32 %0, %1, 0.5, 0.5;\n":
        "=f"(y):"f"(x));
    return y;
}

template<>
__forceinline__ __device__ float2 SigmoidFast::op(float2 x){
    float2 y;
    asm volatile(
        "mul.f32 %2, %2, 0f3EB17218;\n"
        "mul.f32 %3, %3, 0f3EB17218;\n"
        "tanh.approx.f32 %2, %2;\n"
        "tanh.approx.f32 %3, %3;\n"
        "fma.rn.f32 %0, %2, 0.5, 0.5;\n"
        "fma.rn.f32 %1, %3, 0.5, 0.5;\n":
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
    return { std::get<0>(lhs) + std::get<0>(rhs), mx + res};
}

struct FpFMAAffineOp{
    template<typename T>
    __device__ static T op(T x, T y);
    template<typename T>
    __device__ static T get_zero();
};

template<>
__device__ std::tuple<float, float> FpFMAAffineOp::op(std::tuple<float, float> lhs, std::tuple<float, float> rhs){
    return { std::get<0>(lhs) * std::get<0>(rhs), std::get<1>(lhs) * std::get<0>(rhs) + std::get<1>(rhs) };
}

template<>
constexpr __device__ std::tuple<float, float> FpFMAAffineOp::get_zero(){
    return {1.f, 0.f};
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
__device__ std::tuple<float2, float2> FpFMAAffineOp::op(std::tuple<float2, float2> lhs, std::tuple<float2, float2> rhs){
    return { std::get<0>(lhs) * std::get<0>(rhs), std::get<1>(lhs) * std::get<0>(rhs) + std::get<1>(rhs) };
}


template<>
__device__ std::tuple<float2, float2> FpAddLseOpSlow::get_zero(){
    return {{0.f,0.f}, {-99.f,-99.f}};
}


struct Logsumexp1Approx {
    template<typename T>
    __forceinline__ __device__ static T op(T x);
};

template<>
__forceinline__ __device__ float Logsumexp1Approx::op(float x){
    float nabs = -abs(x), mx = max(x, 0.f); 

    float p = max(nabs, -4.3125f); 
    float logs = nabs + p * (-2.793604998770852221e-01f + p * (-5.756650101889571047e-02f + p * (-4.553042169205498771e-03f))), res;
    asm volatile("ex2.approx.ftz.f32 %0, %1;\n":"=f"(res): "f"(logs));
    return mx + res;
}

template<>
__forceinline__ __device__ float2 Logsumexp1Approx::op(float2 x){
    float2 nabs = -absf2(x), mx = maxf2(x, 0.f); 

    float2 p = maxf2(nabs, -4.3125f); 
    float2 logs = nabs + p * (-2.793604998770852221e-01f + p * (-5.756650101889571047e-02f + p * (-4.553042169205498771e-03f))), res;
    asm volatile("ex2.approx.ftz.f32 %0, %2;\nex2.approx.ftz.f32 %1, %3;\n":"=f"(res.x),"=f"(res.y): "f"(logs.x),"f"(logs.y));
    return mx + res;
}

struct RsqrtOp{
    template<typename T>
    static __device__ inline T op(const T &v);
};

template<>
__device__ inline float RsqrtOp::op(const float &v){
    float y0;
    asm volatile(
        "rsqrt.approx.ftz.f32 %0, %1;\n":
        "=f"(y0):"f"(v));
    return y0;
}

template<>
__device__ inline float2 RsqrtOp::op(const float2 &v){
    float y0, y1;
    asm volatile(
        "rsqrt.approx.ftz.f32 %0, %2;\n"
        "rsqrt.approx.ftz.f32 %1, %3;\n":
        "=f"(y0),"=f"(y1):"f"(v.x),"f"(v.y));
    return {y0,y1};
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
    static constexpr int VH_PIPE_SIZE = 2;
    static constexpr float OUT_EPS = 1e-4;
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
    kt::gl<kt::half, -1, -1, -1, C.qk_dim, details::TmaColumnGrouped<kt::half, C.warp_k_size, C.qk_dim>> K;
    kt::gl<kt::bf16, -1, -1, -1, C.head_dim, details::TmaColumnGrouped<kt::bf16, C.warp_k_size, C.qk_dim>> V;
    kt::gl<kt::bf16, -1, -1, -1, C.head_dim> O;
    kt::gl<float, 1, 1, 1, -1> RcpTau;

    kt::gl<float, -1, -1, -1, C.warp_k_size> VBuffer; 
    kt::gl<float, -1, -1, -1, C.warp_k_size> HBuffer;
};

template <PreprocessKernelSm90Config CONFIG>
struct GlobalsFwd {
    // Input layout: [B, N, H, C]/[1, N, H, C]
    static constexpr auto C = CONFIG;
    kt::gl<kt::half, -1, -1, -1, C.qk_dim, details::TmaWarpgroupStrided<kt::st_hf<64, CONFIG.qk_dim>, CONFIG.getCheckpointSize()>> Q;
    kt::gl<kt::half, -1, -1, -1, C.qk_dim, details::TmaColumnGrouped<kt::half, C.warp_k_size, C.qk_dim>> K;
    kt::gl<kt::bf16, -1, -1, -1, C.head_dim, details::TmaColumnGrouped<kt::bf16, C.warp_k_size, C.head_dim>> V;
    kt::gl<kt::bf16, -1, -1, -1, C.head_dim, details::TmaWarpgroupStrided<kt::st_bf<64, CONFIG.head_dim>, CONFIG.getCheckpointSize()>> O;
    kt::gl<float, 1, 1, 1, -1> RcpTau;

    kt::gl<float, -1, -1, -1, C.warp_k_size> VBuffer; 
    kt::gl<float, -1, -1, -1, C.warp_k_size> HBuffer; 
    kt::gl<float, 1, -1, -1, -1> FwdMax;
};

template <PreprocessKernelSm90Config CONFIG>
struct GlobalsBwdPP {
    // Input layout: [B, N, H, C]/[1, N, H, C]
    static constexpr auto C = CONFIG;
    kt::gl<kt::half, -1, -1, -1, C.qk_dim, details::TmaColumnGrouped<kt::half, C.warp_k_size, C.qk_dim>> Q;
    kt::gl<kt::half, -1, -1, -1, C.qk_dim, details::TmaWarpgroupStrided<kt::st_hf<64, CONFIG.qk_dim>, CONFIG.getCheckpointSize()>> K;
    kt::gl<kt::bf16, -1, -1, -1, C.head_dim, details::TmaWarpgroupStrided<kt::st_bf<64, CONFIG.qk_dim>, CONFIG.getCheckpointSize()>> V;
    kt::gl<kt::bf16, -1, -1, -1, C.head_dim, details::TmaColumnGrouped<kt::bf16, C.warp_k_size, C.qk_dim>> dO;
    
    kt::gl<kt::bf16, -1, -1, -1, C.head_dim> dV;
    kt::gl<float, 1, 1, 1, -1> RcpTau;

    kt::gl<float, -1, -1, -1, C.warp_k_size> VBuffer; 
    kt::gl<float, -1, -1, -1, C.warp_k_size> HBuffer; 
    kt::gl<float, 1, -1, -1, -1> FwdMax;
};


template <PreprocessKernelSm90Config CONFIG>
struct GlobalsBwd {
    // Input layout: [B, N, H, C]/[1, N, H, C]
    static constexpr auto C = CONFIG;
    kt::gl<kt::half, -1, -1, -1, C.qk_dim, details::TmaColumnGrouped<kt::half, C.warp_k_size, C.qk_dim>> Q;
    kt::gl<kt::half, -1, -1, -1, C.qk_dim, details::TmaWarpgroupStrided<kt::st_hf<64, CONFIG.qk_dim>, CONFIG.getCheckpointSize()>> K;
    kt::gl<kt::bf16, -1, -1, -1, C.head_dim, details::TmaWarpgroupStrided<kt::st_bf<64, CONFIG.qk_dim>, CONFIG.getCheckpointSize()>> V;
    kt::gl<kt::bf16, -1, -1, -1, C.head_dim, details::TmaColumnGrouped<kt::bf16, C.warp_k_size, C.qk_dim>> dO;
    
    kt::gl<kt::half, -1, -1, -1, C.qk_dim> dQ;
    kt::gl<kt::half, -1, -1, -1, C.qk_dim> dK;
    kt::gl<float, 1, 1, 1, -1> RcpTau;

    kt::gl<float, -1, -1, -1, C.warp_k_size> VBuffer; 
    kt::gl<float, -1, -1, -1, C.warp_k_size> HBuffer; 
    kt::gl<float, 1, -1, -1, -1> FwdMax;
};

// Scheduler for non-varlen attention
template<PreprocessKernelSm90Config CONFIG, bool IS_PREPROCESS = true>
struct FixedLengthScheduler{
    static_assert(CONFIG.getQBlockSize() >= CONFIG.warp_k_size, "Cases for Q<K not implemented!!");

    struct TaskInfo{
        int Batch, QStart, KStart, Head, QIdx;
        int KBlocks, VHStart, VHBound, VHLoadStart, VHLoadBound;
    };

    int CurrentTaskIdx, Batch, SeqlenQBlocks, Head;

    __device__ FixedLengthScheduler(int batch, int seqlen, int head){
        CurrentTaskIdx = blockIdx.x;
        Batch = batch;
        SeqlenQBlocks = (seqlen + CONFIG.getQBlockSize() - 1) / CONFIG.getQBlockSize(); 
        Head = head;
    }

    __forceinline__ __device__ bool getNextTask(TaskInfo& out){ 
        auto wid = kt::warpgroup::warpid();
        auto wgid = kt::warpgroup::groupid();

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
        out.VHBound = out.VHStart + out.KBlocks;
        out.VHLoadStart = out.VHStart - (wid ? out.KBlocks : (out.KBlocks - (CONFIG.getQBlockSize() / CONFIG.warp_k_size)));
        out.VHLoadBound = out.VHLoadStart + (wid ? out.KBlocks : (out.KBlocks - (CONFIG.getQBlockSize() / CONFIG.warp_k_size)));
        out.QIdx = (out.QStart / 8) + (wid * CONFIG.WARPGROUPS + wgid) * (CONFIG.warp_q_size / 8);
        
        CurrentTaskIdx += gridDim.x;
        return true; 
    }
};


template<PreprocessKernelSm90Config CONFIG, bool IS_PREPROCESS = true>
struct FixedLengthBwdScheduler{
    static_assert(CONFIG.getQBlockSize() >= CONFIG.warp_k_size, "Cases for Q<K not implemented!!");

    struct TaskInfo{
        int Batch, KBegin, QEnd, Head, KIdx;
        int InitCurrentQBlock, CurrentQBlock, HInitIdx, VInitIdx;
        __forceinline__ __device__ int getCurrentQOffset() const { return CurrentQBlock * CONFIG.warp_k_size; }
        __forceinline__ __device__ bool VVecAvail() const { return VInitIdx > 0; }
        __forceinline__ __device__ bool iterateNext(){
            constexpr int BLOCK_STEPS = CONFIG.getQBlockSize() / CONFIG.warp_k_size;

            auto wid = kt::warpgroup::warpid();

            CurrentQBlock -= 1;

            //VHLineStart = getLineStart(CurrentQBlock);
            HInitIdx--;
            VInitIdx = getLineStart(CurrentQBlock - 1) + wid + (KBegin / CONFIG.getQBlockSize()) * BLOCK_STEPS;

            return CurrentQBlock >= QEnd;
        }
    };

    int CurrentTaskIdx, Batch, SeqlenKBlocks, Head;

    __device__ FixedLengthBwdScheduler(int batch, int seqlen, int head){
        CurrentTaskIdx = blockIdx.x;
        Batch = batch;
        SeqlenKBlocks = (seqlen + CONFIG.getQBlockSize() - 1) / CONFIG.getQBlockSize(); 
        Head = head;
    }

    static __forceinline__ __device__ int getLineStart(int QBlockIdx){
        constexpr int BLOCK_STEPS = CONFIG.getQBlockSize() / CONFIG.warp_k_size;
        //return ((QBlockIdx / BLOCK_STEPS) * (QBlockIdx / BLOCK_STEPS + 1) / 2) * 4 * BLOCK_STEPS + (QBlockIdx % BLOCK_STEPS) * (QBlockIdx / BLOCK_STEPS + 1) * BLOCK_STEPS;
        return (QBlockIdx / BLOCK_STEPS + 1) * BLOCK_STEPS * ((QBlockIdx / BLOCK_STEPS) * 2 + (QBlockIdx % BLOCK_STEPS));
    }

    __forceinline__ __device__ bool getNextTask(TaskInfo& out){ 
        constexpr int BLOCK_STEPS = CONFIG.getQBlockSize() / CONFIG.warp_k_size;

        auto wid = kt::warpgroup::warpid();
        auto wgid = kt::warpgroup::groupid();

        auto Idx = CurrentTaskIdx;
        auto Total = SeqlenKBlocks * Batch * Head;
        if(Idx >= Total) return false;
        out.Batch = Idx / (Head * SeqlenKBlocks);
        Idx -= out.Batch * (Head * SeqlenKBlocks);
        out.Head = Idx / SeqlenKBlocks;
        Idx -= out.Head * SeqlenKBlocks;

        out.KBegin = CONFIG.getQBlockSize() * Idx;
        out.KIdx = (Idx * CONFIG.getQBlockSize() + wid * CONFIG.getSkipSize() + wgid * CONFIG.warp_q_size) / 8;
        out.QEnd = Idx * BLOCK_STEPS;
        out.InitCurrentQBlock = out.CurrentQBlock = SeqlenKBlocks * BLOCK_STEPS;
        out.HInitIdx = (2 * SeqlenKBlocks - Idx + 1) * Idx * 4 / 2 * BLOCK_STEPS + wid * (SeqlenKBlocks - Idx) * BLOCK_STEPS;
        
        CurrentTaskIdx += gridDim.x;
        return true; 
    }
};


template<PreprocessKernelSm90Config CONFIG>
union LoadSharedMemoryLayouts{
    struct DefaultLayout {
        details::TmaColumnGrouped<kt::half, CONFIG.warp_k_size, CONFIG.qk_dim> k_buffer;
    } Default[CONFIG.k_stages];
    struct PrefetchLayout {
        DefaultLayout preflight_k[CONFIG.K_PREFETCH_SIZE];
        details::TmaWarpgroupStrided<kt::st_hf<CONFIG.warp_q_size * 4, CONFIG.qk_dim>, CONFIG.getSkipSize()> q_buffer[CONFIG.WARPGROUPS];
    } Prefetch[1];
};

template<PreprocessKernelSm90Config CONFIG>
union FwdLoadSharedMemoryLayouts{
    struct DefaultLayout {
        details::TmaColumnGrouped<kt::half, CONFIG.warp_k_size, CONFIG.qk_dim> k_buffer;
        details::TmaColumnGrouped<kt::bf16, CONFIG.warp_k_size, CONFIG.head_dim> v_buffer;
    } Default[CONFIG.k_stages];
    struct PrefetchLayout {
        DefaultLayout preflight_kv[CONFIG.K_PREFETCH_SIZE];
        details::TmaWarpgroupStrided<kt::st_hf<CONFIG.warp_q_size * 4, CONFIG.qk_dim>, CONFIG.getSkipSize()> q_buffer[CONFIG.WARPGROUPS];
        details::TmaWarpgroupStrided<kt::st_bf<CONFIG.warp_q_size * 4, CONFIG.head_dim>, CONFIG.getSkipSize()> in_buffer[CONFIG.WARPGROUPS];
    } Prefetch[1];
    kt::st_bf<CONFIG.warp_q_size, CONFIG.head_dim> out_buffer[CONFIG.WARPGROUPS * kt::WARPGROUP_WARPS];
};


template<PreprocessKernelSm90Config CONFIG>
union BwdPPLoadSharedMemoryLayouts{
    struct DefaultLayout {
        details::TmaColumnGrouped<kt::half, CONFIG.warp_k_size, CONFIG.qk_dim> q_buffer;
        details::TmaColumnGrouped<kt::bf16, CONFIG.warp_k_size, CONFIG.head_dim> dO_buffer;
        kt::sv_fl<CONFIG.getQBlockSize()> v_buffer;
        kt::sv_fl<CONFIG.warp_k_size> fwd_max;
    } Default[CONFIG.k_stages];
    struct PrefetchLayout {
        DefaultLayout preflight_qdO[CONFIG.K_PREFETCH_SIZE];
        details::TmaWarpgroupStrided<kt::st_hf<CONFIG.warp_q_size * 4, CONFIG.qk_dim>, CONFIG.getSkipSize()> k_buffer[CONFIG.WARPGROUPS];
        details::TmaWarpgroupStrided<kt::st_bf<CONFIG.warp_q_size * 4, CONFIG.head_dim>, CONFIG.getSkipSize()> v_buffer[CONFIG.WARPGROUPS];
    } Prefetch[1];
    kt::st_bf<CONFIG.warp_q_size, CONFIG.head_dim> dV_out_buffer[CONFIG.WARPGROUPS * kt::WARPGROUP_WARPS];
};

template<PreprocessKernelSm90Config CONFIG>
struct IntermediateSharedMemoryLayouts{
    struct{
        sxdiag::SharedTopBottomVec<CONFIG.warp_k_size, float> h_buffer[kt::WARPGROUP_WARPS];
        sxdiag::SharedTopBottomVec<CONFIG.warp_k_size, float> v_buffer[kt::WARPGROUP_WARPS];
    } defaults[CONFIG.VH_PIPE_SIZE];
};

template<PreprocessKernelSm90Config CONFIG>
struct BwdPPIntermediateSharedMemoryLayouts{
    struct{
        sxdiag::SharedTopBottomVec<CONFIG.warp_k_size, float> h_buffer[kt::WARPGROUP_WARPS];
        //sxdiag::SharedTopBottomVec<CONFIG.warp_k_size, float> v_buffer[kt::WARPGROUP_WARPS];

        sxdiag::SharedTopBottomVec<CONFIG.warp_k_size, float> bwd_h_buffer[CONFIG.WARPGROUPS][kt::WARPGROUP_WARPS];
        sxdiag::SharedTopBottomVec<CONFIG.warp_k_size, float> bwd_v_buffer[CONFIG.WARPGROUPS][kt::WARPGROUP_WARPS];
    } defaults[CONFIG.VH_PIPE_SIZE];
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

static __launch_bounds__(256 + 32,1) __global__ void preprocess_kernel(const __grid_constant__ Globals<DEFAULT_CONFIG> args) {
    constexpr PreprocessKernelSm90Config CONFIG = DEFAULT_CONFIG;
    using Scheduler = FixedLengthScheduler<CONFIG>;
    using wg = kt::warpgroup;

    extern __shared__ int shmem[];
    kt::shared_allocator<> alloc{shmem};
    LoadSharedMemoryLayouts<CONFIG> &load_layouts = alloc.template allocate<LoadSharedMemoryLayouts<CONFIG>>();
    IntermediateSharedMemoryLayouts<CONFIG> &vh_layouts = alloc.template allocate<IntermediateSharedMemoryLayouts<CONFIG>>();

    uint32_t wgid = kt::warpgroup::groupid();
    uint32_t warpid = kt::warpgroup::warpid();
    constexpr uint32_t CONSUMER_A = 0, CONSUMER_B = 1, PRODUCER = 2;
    constexpr uint32_t PRODUCER_LOAD = 0, PRODUCER_STORE = 1;

    details::MbarrierMultiEndSwitcher k_load_pipe( alloc, load_layouts.Default, wgid, details::ProducerWarp<true, PRODUCER>, details::ConsumerWarpGroup<false, CONSUMER_A, CONSUMER_B>);
    details::MbarrierMultiEndSwitcher prefetch_load_pipe( alloc, load_layouts.Prefetch, wgid, details::ProducerWarp<true, PRODUCER>, details::ConsumerWarpGroup<false, CONSUMER_A, CONSUMER_B>);
    details::MbarrierMultiEndSwitcher vh_swap_pipe( alloc, vh_layouts.defaults, wgid, details::ConsumerWarpGroup<true, CONSUMER_A>, details::ConsumerWarpGroup<false, CONSUMER_B>);

    Scheduler scheduler(args.Q.batch(), args.Q.depth(), args.Q.rows());
    typename Scheduler::TaskInfo task;

    if(wgid == PRODUCER && warpid == PRODUCER_LOAD && kt::warp::elect_leader()){ // producer groups
        while(scheduler.getNextTask(task)){
            //printf("Task info: cta = %u, batch = %u, head = %u, n_kblocks = %u, vh_start = %u, qstart = %u\n", blockIdx.x, task.Batch, task.Head, task.KBlocks, task.VHStart, task.QStart);
            // first load Q and first K block

            prefetch_load_pipe.setup();
            auto prefetch_packet = prefetch_load_pipe.waitBuffer(CONSUMER_A); // just to match details::ConsumerWarpGroup<false, CONSUMER_A, CONSUMER_B>
            ([&]<size_t... IDX_Q, size_t... IDX_K>(std::index_sequence<IDX_Q...>, std::index_sequence<IDX_K...>){
                kt::tma::expect_bytes(prefetch_packet.getBarrier(), kt::size_bytes<decltype(prefetch_packet->preflight_k[IDX_K])..., decltype(prefetch_packet->q_buffer[IDX_Q])...>);
                (kt::tma::load_async<1, kt::cache_policy::NORMAL>(prefetch_packet->q_buffer[IDX_Q], args.Q, { task.Batch, task.QStart + int(IDX_Q * CONFIG.warp_q_size), task.Head, 0}, prefetch_packet.getBarrier()), ...);
                (kt::tma::load_async<1, kt::cache_policy::NORMAL>(prefetch_packet->preflight_k[IDX_K].k_buffer, args.K, { task.Batch, task.KStart + int(IDX_K) * CONFIG.warp_k_size, task.Head, 0}, prefetch_packet.getBarrier()), ...); // NOTE: AXIS MAY NOT CORRECT!!!
            })(std::make_index_sequence<CONFIG.WARPGROUPS>{}, std::make_index_sequence<CONFIG.K_PREFETCH_SIZE>{});
            
            prefetch_packet.submitToNextAndTrigger();
                
            prefetch_load_pipe.waitBuffer(CONSUMER_A); // wait for Q is consumed and switch layout
            
            k_load_pipe.setup(); 

            #pragma unroll
            for(int i = 0; i < CONFIG.K_PREFETCH_SIZE; i++) k_load_pipe.waitBuffer(CONSUMER_A).submitToNextAndTrigger(); // already loaded
            
            #pragma unroll 1
            for(int iter = CONFIG.K_PREFETCH_SIZE; iter < task.KBlocks; iter++){
                auto load_packet = k_load_pipe.waitBuffer(CONSUMER_A);
                kt::tma::expect_bytes(load_packet.getBarrier(), sizeof(load_packet->k_buffer));
                kt::tma::load_async<1, kt::cache_policy::NORMAL>(load_packet->k_buffer, args.K, { task.Batch, task.KStart + iter * CONFIG.warp_k_size, task.Head, 0}, load_packet.getBarrier()); // NOTE: AXIS AND OFFSET IS INCORRECT
                load_packet.submitToNextAndTrigger(); 
            }

            __syncthreads();
        }
    } else if(wgid < PRODUCER){
        using MainOp = details::ops::FpAddLseOpSlow;
        constexpr auto ZERO = MainOp::get_zero<std::tuple<float,float>>();
        uint32_t warpid = kt::warpgroup::warpid();

        kt::rt_hf<CONFIG.warp_q_size, CONFIG.qk_dim> q_tile;
        kt::rt_fl<CONFIG.warp_q_size, CONFIG.warp_k_size> acc_front;

        

        while(scheduler.getNextTask(task)){
            int vh_start = task.VHStart;
            auto storeVHVecs = [&](const decltype(vh_swap_pipe)::Handle& handle){
                handle->v_buffer[warpid].store(args.VBuffer, {task.Batch, task.Head, vh_start, 0});
                handle->h_buffer[warpid].store(args.HBuffer, {task.Batch, task.Head, vh_start, 0});
                vh_start++;
            };

            //kt::print_utils_group<4>::print("Task info: cta = %u, batch = %u, head = %u, n_kblocks = %u, vh_start = %u, qstart = %u\n", blockIdx.x, task.Batch, task.Head, task.KBlocks, task.VHStart, task.QStart);
            sxdiag::LeftRightVec<CONFIG.warp_q_size, float, float> lr_state = ZERO;
            sxdiag::TopBottomVec<CONFIG.warp_k_size, float, float> tb_state;

            uint32_t consumer_idx = wgid % CONFIG.WARPGROUPS;
            float rtau = args.RcpTau.raw_ptr[task.Head] + CONFIG.qk_bias;

            prefetch_load_pipe.setup();
            auto init_block = prefetch_load_pipe.waitBuffer(PRODUCER);
            //kt::print_utils_group<4>::print("Consumer %u get Q tile\n", wgid);
            

            kt::warpgroup::load(q_tile, init_block->q_buffer[consumer_idx].data);

            init_block.submitToNextAndTrigger();

            

            k_load_pipe.setup();
            vh_swap_pipe.setup();

            
            #pragma unroll 1
            for(int i = 0; i < task.KBlocks; i++){
                
                auto k_handle = k_load_pipe.waitBuffer(PRODUCER);

                wg::emulated_wgmma::mm_ABt(acc_front, q_tile, k_handle->k_buffer.data);
                k_handle.submitToNextAndTrigger();

                //if(wgid==0) kt::print_utils_group<4>::print(acc_front);
            
                auto tb_handle = vh_swap_pipe.waitBuffer((wgid + 1) % CONFIG.WARPGROUPS);
                
                if(wgid == CONSUMER_A){
                    tb_state = ZERO;
                } else {
                    tb_state.load<0>(tb_handle->h_buffer[warpid]);
                    tb_state.load<1>(tb_handle->v_buffer[warpid]);
                }
                
                kt::warp::rt_maps::unary_map<details::ops::Log2Map>(acc_front, acc_front + CONFIG.qk_log_offset);
                auto scan_buffer = sxdiag::TightMMABuffer<float, CONFIG.warp_q_size, CONFIG.warp_k_size>::from_rt(acc_front + rtau);
                
                auto u = scan_buffer.expand(std::get<0>(ZERO));
                auto v = u.with_padding(std::get<1>(ZERO));

                sxdiag::ScanTile<CONFIG.warp_q_size, CONFIG.warp_k_size, float, float> tile(u, v);
                
                auto red_res = tile.diagScanOrReduce<false, false, MainOp>(sxdiag::StatePair<CONFIG.warp_q_size, CONFIG.warp_k_size, float, float>::from(lr_state, tb_state));

                lr_state = red_res.lr;
                
                red_res.tb.store<0>(tb_handle->h_buffer[warpid]);
                red_res.tb.store<1>(tb_handle->v_buffer[warpid]);

                if(wgid == CONSUMER_B) storeVHVecs(tb_handle);
                
                tb_handle.submitToNextAndTrigger();

            }

            if(wgid == CONSUMER_B && kt::warp::laneid() == 31){
                args.HBuffer[kt::coord<>{task.Batch, task.Head, task.VHStart,0}] = std::get<0>(lr_state.data[lr_state.ROW_UNITS - 1]);
                args.VBuffer[kt::coord<>{task.Batch, task.Head, task.VHStart,0}] = std::get<1>(lr_state.data[lr_state.ROW_UNITS - 1]);
            }

            __syncthreads();

        }
    }
}

constexpr PreprocessKernelSm90Config DEFAULT_CONFIG_FWD = PreprocessKernelSm90Config(64, 64, 3, 16, 32, 1e-3f);

template<int ROWS, int COLS>
__forceinline__ __device__ void make_causal(kt::rt_fl<ROWS, COLS>& dst, int row_start, int col_start, float ZERO){
    static_assert(COLS % 32 == 0);

    using RT = kt::rt_fl<ROWS, COLS>;
    uint32_t tid = threadIdx.x & 0x1f;
    uint32_t tidm4 = tid & 0x3;
    uint32_t tidr3 = tid >> 2;

    #pragma unroll
    for(int i = 0; i < RT::height; i++){
        #pragma unroll
        for(int i_rt = 0; i_rt < 2; i_rt++){
            if(((i * 2 + i_rt) + row_start) >= (col_start + 4 * (COLS / 32))){ // should be an uniform branch
                continue;
            }

            int tile_row = (i * 2 + i_rt + row_start) * 8 + tidr3;
            int tile_col_base = (col_start + tidm4 * (COLS / 32)) * 8;
            #pragma unroll
            for(int j = 0; j < RT::width; j++){
                #pragma unroll
                for(int j_rt = 0; j_rt < 2; j_rt++){
                    int tile_col_x = tile_col_base + (j * 2 + j_rt);
                    int tile_col_y = tile_col_base + (j * 2 + j_rt) + (COLS / 8);
                    dst.tiles[i][j].data[i_rt + j_rt * 2].x = (tile_row >= tile_col_x) ? dst.tiles[i][j].data[i_rt + j_rt * 2].x : ZERO;
                    dst.tiles[i][j].data[i_rt + j_rt * 2].y = (tile_row >= tile_col_y) ? dst.tiles[i][j].data[i_rt + j_rt * 2].y : ZERO;
                }
            }
        }
    }
}

template<int ROWS, int COLS>
__forceinline__ __device__ void make_causal_triu(kt::rt_fl<ROWS, COLS>& dst, int row_start, int col_start, float ZERO){
    static_assert(COLS % 32 == 0);

    using RT = kt::rt_fl<ROWS, COLS>;
    uint32_t tid = threadIdx.x & 0x1f;
    uint32_t tidm4 = tid & 0x3;
    uint32_t tidr3 = tid >> 2;

    #pragma unroll
    for(int i = 0; i < RT::height; i++){
        #pragma unroll
        for(int i_rt = 0; i_rt < 2; i_rt++){
            if((i * 2 + i_rt + row_start + 1) <= (col_start)){ // should be an uniform branch
                continue;
            }

            int tile_row = (i * 2 + i_rt + row_start) * 8 + tidr3;
            int tile_col_base = (col_start + tidm4 * (COLS / 32)) * 8;
            #pragma unroll
            for(int j = 0; j < RT::width; j++){
                #pragma unroll
                for(int j_rt = 0; j_rt < 2; j_rt++){
                    int tile_col_x = tile_col_base + (j * 2 + j_rt);
                    int tile_col_y = tile_col_base + (j * 2 + j_rt) + (COLS / 8);
                    dst.tiles[i][j].data[i_rt + j_rt * 2].x = (tile_row <= tile_col_x) ? dst.tiles[i][j].data[i_rt + j_rt * 2].x : ZERO;
                    dst.tiles[i][j].data[i_rt + j_rt * 2].y = (tile_row <= tile_col_y) ? dst.tiles[i][j].data[i_rt + j_rt * 2].y : ZERO;
                }
            }
        }
    }
}


static __launch_bounds__(256+128,1) __global__ void fwd_kernel(const __grid_constant__ GlobalsFwd<DEFAULT_CONFIG_FWD> args){
    constexpr PreprocessKernelSm90Config CONFIG = DEFAULT_CONFIG_FWD;
    using Scheduler = FixedLengthScheduler<CONFIG, false>;
    using wg = kt::warpgroup;

    extern __shared__ int shmem[];
    kt::shared_allocator<> alloc{shmem};
    FwdLoadSharedMemoryLayouts<CONFIG> &load_layouts = alloc.template allocate<FwdLoadSharedMemoryLayouts<CONFIG>>();
    IntermediateSharedMemoryLayouts<CONFIG> &vh_layouts = alloc.template allocate<IntermediateSharedMemoryLayouts<CONFIG>>();

    uint32_t wgid = kt::warpgroup::groupid();
    uint32_t warpid = kt::warpgroup::warpid();
    constexpr uint32_t CONSUMER_A = 0, CONSUMER_B = 1, PRODUCER = 2;
    constexpr uint32_t PRODUCER_LOAD = 0, PRODUCER_STORE = 1;

    details::MbarrierMultiEndSwitcher k_load_pipe( alloc, load_layouts.Default, wgid, details::ProducerWarp<true, PRODUCER>, details::ConsumerWarpGroup<false, CONSUMER_A, CONSUMER_B>);
    details::MbarrierMultiEndSwitcher prefetch_load_pipe( alloc, load_layouts.Prefetch, wgid, details::ProducerWarp<true, PRODUCER>, details::ConsumerWarpGroup<false, CONSUMER_A, CONSUMER_B>);
    details::MbarrierMultiEndSwitcher vh_swap_pipe( alloc, vh_layouts.defaults, wgid, details::ConsumerWarpGroup<true, CONSUMER_A>, details::ConsumerWarpGroup<false, CONSUMER_B>);

    Scheduler scheduler(args.Q.batch(), args.Q.depth(), args.Q.rows());
    typename Scheduler::TaskInfo task;

    if(wgid == PRODUCER){ // producer groups
        kt::warp::decrease_registers<40>();
        if(warpid == PRODUCER_LOAD && kt::warp::elect_leader()){
            while(scheduler.getNextTask(task)){
                //printf("Task info: cta = %u, batch = %u, head = %u, n_kblocks = %u, vh_start = %u, qstart = %u\n", blockIdx.x, task.Batch, task.Head, task.KBlocks, task.VHStart, task.QStart);
                // first load Q and first K block

                prefetch_load_pipe.setup();
                auto prefetch_packet = prefetch_load_pipe.waitBuffer(CONSUMER_A); // just to match details::ConsumerWarpGroup<false, CONSUMER_A, CONSUMER_B>
                ([&]<size_t... IDX_Q, size_t... IDX_K>(std::index_sequence<IDX_Q...>, std::index_sequence<IDX_K...>){
                    kt::tma::expect_bytes(prefetch_packet.getBarrier(), kt::size_bytes<decltype(prefetch_packet->preflight_kv[IDX_K])..., decltype(prefetch_packet->q_buffer[IDX_Q])..., decltype(prefetch_packet->in_buffer[IDX_Q])...>);
                    (kt::tma::load_async<1, kt::cache_policy::NORMAL>(prefetch_packet->q_buffer[IDX_Q], args.Q, { task.Batch, task.QStart + int(IDX_Q * CONFIG.warp_q_size), task.Head, 0}, prefetch_packet.getBarrier()), ...);
                    (kt::tma::load_async<1, kt::cache_policy::NORMAL>(prefetch_packet->in_buffer[IDX_Q], args.O, {task.Batch, task.QStart + int(IDX_Q * CONFIG.warp_q_size), task.Head, 0}, prefetch_packet.getBarrier()),... );
                    (kt::tma::load_async<1, kt::cache_policy::NORMAL>(prefetch_packet->preflight_kv[IDX_K].k_buffer, args.K, { task.Batch, task.KStart + int(IDX_K) * CONFIG.warp_k_size, task.Head, 0}, prefetch_packet.getBarrier()), ...); // NOTE: AXIS MAY NOT CORRECT!!!
                    (kt::tma::load_async<1, kt::cache_policy::NORMAL>(prefetch_packet->preflight_kv[IDX_K].v_buffer, args.V, { task.Batch, task.KStart + int(IDX_K) * CONFIG.warp_k_size, task.Head, 0}, prefetch_packet.getBarrier()), ...); // NOTE: AXIS MAY NOT CORRECT!!!
                })(std::make_index_sequence<CONFIG.WARPGROUPS>{}, std::make_index_sequence<CONFIG.K_PREFETCH_SIZE>{});
                
                prefetch_packet.submitToNextAndTrigger();
                    
                prefetch_load_pipe.waitBuffer(CONSUMER_A); // wait for Q is consumed and switch layout
                
                k_load_pipe.setup(); 

                #pragma unroll
                for(int i = 0; i < CONFIG.K_PREFETCH_SIZE; i++) k_load_pipe.waitBuffer(CONSUMER_A).submitToNextAndTrigger(); // already loaded
                
                #pragma unroll 1
                for(int iter = CONFIG.K_PREFETCH_SIZE; iter < task.KBlocks; iter++){
                    auto load_packet = k_load_pipe.waitBuffer(CONSUMER_A);
                    kt::tma::expect_bytes(load_packet.getBarrier(), sizeof(load_packet->k_buffer) + sizeof(load_packet->v_buffer));
                    kt::tma::load_async<1, kt::cache_policy::NORMAL>(load_packet->k_buffer, args.K, { task.Batch, task.KStart + iter * CONFIG.warp_k_size, task.Head, 0}, load_packet.getBarrier()); // NOTE: AXIS AND OFFSET IS INCORRECT
                    kt::tma::load_async<1, kt::cache_policy::NORMAL>(load_packet->v_buffer, args.V, { task.Batch, task.KStart + iter * CONFIG.warp_k_size, task.Head, 0}, load_packet.getBarrier()); // NOTE: AXIS AND OFFSET IS INCORRECT
                    load_packet.submitToNextAndTrigger(); 
                }

                __syncthreads();
            }
        }
    } else {
        static_assert(CONFIG.getCheckpointSize() == CONFIG.warp_k_size); // for simplicity

        kt::warp::increase_registers<232>();
        using MainOp = details::ops::FpAddLseOpSlow;
        constexpr auto ZERO = MainOp::get_zero<std::tuple<float,float>>();
        uint32_t warpid = kt::warpgroup::warpid();

        kt::rt_hf<CONFIG.warp_q_size, CONFIG.qk_dim> q_tile;
        kt::rt_fl<CONFIG.warp_q_size, CONFIG.warp_k_size> acc_front;

        kt::rt_fl<CONFIG.warp_q_size, CONFIG.head_dim> acc_tile;
        decltype(acc_tile)::col_vec row_max;

        while(scheduler.getNextTask(task)){
            kt::rv_fl<CONFIG.warp_q_size> row_max_naive;
            kt::warp::load(row_max_naive, args.FwdMax, kt::coord<>{0, task.Batch, task.Head, task.QIdx / (CONFIG.warp_q_size / 8)});
            kt::warp::copy(row_max, row_max_naive);

            int vh_start = task.VHLoadStart;
            int h_start = task.QIdx * 8 / CONFIG.getSkipSize();

            //kt::print_utils::print("Task info: cta = %u, wgid = %u, warpid = %u, batch = %u, head = %u, n_kblocks = %u, vh_start = %u, qstart = %u\n", blockIdx.x, wgid, warpid, task.Batch, task.Head, task.KBlocks, task.VHStart, task.QStart);
            sxdiag::LeftRightVec<CONFIG.warp_q_size, float, float> lr_state = ZERO;
            sxdiag::TopBottomVec<CONFIG.warp_k_size, float, float> tb_state;

            uint32_t consumer_idx = wgid % CONFIG.WARPGROUPS;
            float rtau = args.RcpTau.raw_ptr[task.Head] + CONFIG.qk_bias;

            prefetch_load_pipe.setup();
            auto init_block = prefetch_load_pipe.waitBuffer(PRODUCER);
            
            kt::warpgroup::load(q_tile, init_block->q_buffer[consumer_idx].data);
            kt::warpgroup::load(acc_tile, init_block->in_buffer[consumer_idx].data);
            
            init_block.submitToNextAndTrigger();

            k_load_pipe.setup();
            vh_swap_pipe.setup();

            float last_tb_init = 0.f;

            #pragma unroll 1
            for(int i = 0; i < task.KBlocks; i++){
                
                auto k_handle = k_load_pipe.waitBuffer(PRODUCER);

                wg::emulated_wgmma::mm_ABt(acc_front, q_tile, k_handle->k_buffer.data);
                
            
                auto tb_handle = vh_swap_pipe.waitBuffer((wgid + 1) % CONFIG.WARPGROUPS);
                
                tb_state.set<0>(std::get<0>(ZERO));

                bool should_load = wgid == CONSUMER_A && vh_start < task.VHLoadBound;
                bool init_avail = (vh_start < task.VHLoadBound) || wgid != CONSUMER_A;
                bool last_block = wgid == CONSUMER_A && warpid == 0 && vh_start == task.VHLoadBound;
                if(should_load){ // should load
                    tb_handle->v_buffer[warpid].load(args.VBuffer, { task.Batch, task.Head, vh_start, 0});
                    vh_start++;
                } 
                
                if(init_avail){
                    tb_state.load<1>(tb_handle->v_buffer[warpid]);
                } else {
                    tb_state.set<1>(std::get<1>(ZERO));
                }
                
                if(wgid == CONSUMER_A && i == 0){
                    last_tb_init = std::get<1>(tb_state.data[0]);
                    if(kt::warp::laneid() == 28) std::get<1>(tb_state.data[0]) = -INFINITY;
                } else if(last_block){
                    if(kt::warp::laneid() == 28) std::get<1>(tb_state.data[0]) = last_tb_init;
                    vh_start++;
                }
                
                
                kt::warp::rt_maps::unary_map<details::ops::Log2Map>(acc_front, acc_front + CONFIG.qk_log_offset);
                acc_front = acc_front + rtau;
                auto scan_buffer = sxdiag::TightMMABuffer<float, CONFIG.warp_q_size, CONFIG.warp_k_size>::from_rt(acc_front);
                
                // for better register allocation schedule
                make_causal(acc_front, task.QIdx, i * (CONFIG.warp_k_size / 8), -INFINITY);

                auto u = scan_buffer.expand(std::get<0>(ZERO));
                auto v = u.with_padding(std::get<1>(ZERO));

                sxdiag::ScanTile<CONFIG.warp_q_size, CONFIG.warp_k_size, float, float> tile(u, v);

                auto red_res = tile.diagScanOrReduce<false, true, MainOp>(sxdiag::StatePair<CONFIG.warp_q_size, CONFIG.warp_k_size, float, float>::from(lr_state, tb_state));

                auto v_scan = std::get<1>(tile.buffers).fold_to_rt();
                kt::warp::rt_maps::unary_map<details::ops::Logsumexp1Approx>(v_scan, v_scan);
                v_scan += acc_front;

                lr_state = red_res.lr;
                
                do{ // save lr states
                    constexpr int WARP_QK_RATIO = (CONFIG.getQBlockSize() / CONFIG.warp_k_size);

                    // NOTE: this assmues seqlen % CONFIG.getQBlockSize() == 0
                    bool is_last_q = (task.QStart == (scheduler.SeqlenQBlocks - 1) * CONFIG.getQBlockSize()) && wgid == 1 && warpid == 3;
                    
                    uint32_t tid = threadIdx.x & 0x1f;
                    uint32_t tidm4 = tid & 0x3, tidr3 = tid >> 2;
                    float *slot_ptr = (&args.HBuffer[kt::coord<>{task.Batch, task.Head, h_start, 0}]) + int(wgid * CONFIG.warp_q_size);
                    #pragma unroll
                    for(int j = 0; j < CONFIG.warp_q_size / 8; j++){
                        if(tidm4 == 3 && !(is_last_q && tid == 31 && j == (CONFIG.warp_q_size / 8 - 1))) slot_ptr[j * 8 + tidr3 + 1] = std::get<1>(lr_state.data[j]);
                    }
                    h_start += ((scheduler.SeqlenQBlocks * kt::WARPGROUP_WARPS - 2 - i) / (WARP_QK_RATIO) + 1) * kt::WARPGROUP_WARPS;
                }while(0);


                red_res.tb.store<1>(tb_handle->v_buffer[warpid]);
                
                tb_handle.submitToNextAndTrigger();

                decltype(acc_tile)::col_vec new_max, acc_rescale;
                kt::warp::rt_reductions::row_max(new_max, v_scan, row_max);

                kt::warp::rv_maps::exp2(acc_rescale, row_max - new_max);
                kt::warp::rt_maps::mul_row(acc_tile, acc_tile, acc_rescale);
                
                kt::rt_bf<CONFIG.warp_q_size, CONFIG.warp_k_size> v_scan_exp{v_scan - new_max};
                kt::warp::rt_maps::exp2(v_scan_exp, v_scan_exp);
                //kt::rt_bf<CONFIG.warp_q_size, CONFIG.warp_k_size> v_scan_exp_mma{v_scan_exp};

                kt::warpgroup::emulated_wgmma::mma_AB(acc_tile, v_scan_exp, k_handle->v_buffer.data);
                
                k_handle.submitToNextAndTrigger();


                row_max = new_max;
            }

            kt::group<CONFIG.WARPGROUPS * 4>::sync(1);


            decltype(acc_tile)::col_vec output_norm;
            kt::warp::rt_reductions::row_sum(output_norm, acc_tile * acc_tile);
            kt::warp::rv_maps::unary_op<details::ops::RsqrtOp>(output_norm, output_norm * (1.f / CONFIG.head_dim) + CONFIG.OUT_EPS);
            kt::warp::rt_maps::mul_row(acc_tile, acc_tile, output_norm);

            kt::warp::store(load_layouts.out_buffer[warpid * CONFIG.WARPGROUPS + wgid], acc_tile);

            //if(wgid==0&&warpid==0)kt::print_utils::print(row_max);

            __syncwarp();

            kt::warp::copy(row_max_naive, row_max);
            kt::warp::store(args.FwdMax,row_max_naive, kt::coord<>{0, task.Batch, task.Head, task.QIdx / (CONFIG.warp_q_size / 8)});
            kt::warp::store<1, true>(args.O, load_layouts.out_buffer[warpid * CONFIG.WARPGROUPS + wgid], kt::coord<>{task.Batch, task.QIdx * 8, task.Head, 0});

            __syncthreads();
        }
        //kt::print_utils::print("Consumer warpid = %u, wgid = %u died\n", warpid, wgid);
    }
}

template<typename lg, kt::ducks::rt::all RT, kt::ducks::st::all ST>
__device__ inline static void row_inv_load(RT &dst, const ST &src) {
    constexpr int height = ST::height;
    constexpr int warp_height = RT::height;
    static_assert(sizeof(typename ST::dtype) == 2, "Currently only support 16-bit types for inverted group load / store.");
    static_assert(height%lg::GROUP_WARPS == 0, "Group load / store requires tile height to be a multiple of GROUP_WARPS.");
    static_assert(height%warp_height == 0, "Group load / store requires tile height to be a multiple of the RT height.");
    static_assert(ST::width==RT::width, "Group load / store requires tile widths to match.");
    int local_warpid;
    if constexpr(lg::GROUP_WARPS % 4 == 0) local_warpid = (lg::warpid()/4+(lg::warpid()%4)*(lg::GROUP_WARPS/4));
    else local_warpid = lg::warpid();
    using T2 = RT::dtype;
    using U  = ST::dtype;
    using T  = kt::base_types::packing<T2>::unpacked_type;
    using U2 = kt::base_types::packing<U>::packed_type;
    int warp_laneid = ::kittens::laneid();

    // convert to shared state space
    uint32_t shared_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&src.data[0]));

    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            if constexpr (sizeof(typename ST::dtype) == 2) {
                // handle the row-major layout for 16-bit types
                U2 tmp[4];
                int row = (local_warpid*warp_height + i)*dst.tile_size_row + (warp_laneid % 16);
                int col = j*dst.tile_size_col + (warp_laneid / 16) * 8;
                if constexpr (std::is_same_v<typename RT::layout, kt::ducks::rt_layout::row>) {
                    kt::move<U2>::ldsm4(tmp[0], tmp[1], tmp[2], tmp[3], src.idx(shared_addr, {7^row, col}));
                }
                else {
                    kt::move<U2>::ldsm4t(tmp[0], tmp[2], tmp[1], tmp[3], src.idx(shared_addr, {7^row, col}));
                }
                dst.tiles[dst.height - 1 - i][j].data[0] = kt::base_types::convertor<T2, U2>::convert(tmp[1]);
                dst.tiles[dst.height - 1 - i][j].data[1] = kt::base_types::convertor<T2, U2>::convert(tmp[0]);
                dst.tiles[dst.height - 1 - i][j].data[2] = kt::base_types::convertor<T2, U2>::convert(tmp[3]);
                dst.tiles[dst.height - 1 - i][j].data[3] = kt::base_types::convertor<T2, U2>::convert(tmp[2]);
            }
        }
    }
}

static __launch_bounds__(256+128,1) __global__ void bwd_preprocess_kernel(const __grid_constant__ GlobalsBwdPP<DEFAULT_CONFIG_FWD> args){
    constexpr PreprocessKernelSm90Config CONFIG = DEFAULT_CONFIG_FWD;
    using Scheduler = FixedLengthBwdScheduler<CONFIG, false>;
    using wg = kt::warpgroup;

    extern __shared__ int shmem[];
    kt::shared_allocator<> alloc{shmem};
    BwdPPLoadSharedMemoryLayouts<CONFIG> &load_layouts = alloc.template allocate<BwdPPLoadSharedMemoryLayouts<CONFIG>>();
    BwdPPIntermediateSharedMemoryLayouts<CONFIG> &vh_layouts = alloc.template allocate<BwdPPIntermediateSharedMemoryLayouts<CONFIG>>();

    uint32_t wgid = kt::warpgroup::groupid();
    uint32_t warpid = kt::warpgroup::warpid();
    constexpr uint32_t CONSUMER_A = 0, CONSUMER_B = 1, PRODUCER = 2;
    constexpr uint32_t PRODUCER_LOAD = 0, PRODUCER_STORE = 1;

    details::MbarrierMultiEndSwitcher qo_load_pipe( alloc, load_layouts.Default, wgid, details::ProducerWarp<true, PRODUCER>, details::ConsumerWarpGroup<false, CONSUMER_A, CONSUMER_B>);
    details::MbarrierMultiEndSwitcher prefetch_load_pipe( alloc, load_layouts.Prefetch, wgid, details::ProducerWarp<true, PRODUCER>, details::ConsumerWarpGroup<false, CONSUMER_A, CONSUMER_B>);
    details::MbarrierMultiEndSwitcher vh_swap_pipe( alloc, vh_layouts.defaults, wgid, details::ConsumerWarpGroup<true, CONSUMER_A>, details::ConsumerWarpGroup<false, CONSUMER_B>);

    Scheduler scheduler(args.Q.batch(), args.Q.depth(), args.Q.rows());
    typename Scheduler::TaskInfo task;

    if(wgid == PRODUCER){ // producer groups
        //kt::warp::decrease_registers<40>();
        if(warpid == PRODUCER_LOAD && kt::warp::elect_leader()){
            while(scheduler.getNextTask(task)){
                static_assert(CONFIG.K_PREFETCH_SIZE == 1);
                printf("Task info: cta = %u, batch = %u, head = %u, q_end = %u, k_begin = %u, CurrentQBlock = %u\n", blockIdx.x, task.Batch, task.Head, task.QEnd, task.KBegin, task.CurrentQBlock);
                prefetch_load_pipe.setup();
                auto prefetch_packet = prefetch_load_pipe.waitBuffer(CONSUMER_A); // just to match details::ConsumerWarpGroup<false, CONSUMER_A, CONSUMER_B>
                
                task.iterateNext(); // NOTE: here we assume K_PREFETCH_SIZE = 1. Fix it later
                ([&]<size_t... IDX_KV>(std::index_sequence<IDX_KV...>){
                    details::TmaExtension::loadAsync(prefetch_packet->preflight_qdO[0].fwd_max, args.FwdMax, kt::coord<>{0,task.Batch, task.Head, task.CurrentQBlock * CONFIG.warp_k_size}, prefetch_packet.getBarrier());
                    details::TmaExtension::loadAsync(prefetch_packet->preflight_qdO[0].v_buffer, args.VBuffer, kt::coord<>{task.Batch, task.Head, task.VInitIdx, 0}, prefetch_packet.getBarrier());
                    
                    kt::tma::expect_bytes(prefetch_packet.getBarrier(), kt::size_bytes<decltype(prefetch_packet->preflight_qdO[0]), decltype(prefetch_packet->k_buffer[IDX_KV])..., decltype(prefetch_packet->v_buffer[IDX_KV])...>);
                    (kt::tma::load_async<1, kt::cache_policy::NORMAL>(prefetch_packet->k_buffer[IDX_KV], args.K, { task.Batch, task.KBegin + int(IDX_KV * CONFIG.warp_q_size), task.Head, 0}, prefetch_packet.getBarrier()), ...);
                    (kt::tma::load_async<1, kt::cache_policy::NORMAL>(prefetch_packet->v_buffer[IDX_KV], args.V, {task.Batch, task.KBegin + int(IDX_KV * CONFIG.warp_q_size), task.Head, 0}, prefetch_packet.getBarrier()),... );
                    kt::tma::load_async<1, kt::cache_policy::NORMAL>(prefetch_packet->preflight_qdO[0].q_buffer, args.Q, { task.Batch, task.getCurrentQOffset(), task.Head, 0}, prefetch_packet.getBarrier()); // NOTE: AXIS MAY NOT CORRECT!!!
                    kt::tma::load_async<1, kt::cache_policy::NORMAL>(prefetch_packet->preflight_qdO[0].dO_buffer, args.dO, { task.Batch, task.getCurrentQOffset(), task.Head, 0}, prefetch_packet.getBarrier()); // NOTE: AXIS MAY NOT CORRECT!!!
                    
                })(std::make_index_sequence<CONFIG.WARPGROUPS>{});
                
               
                prefetch_packet.submitToNextAndTrigger();
                    
                prefetch_load_pipe.waitBuffer(CONSUMER_A); // wait for Q is consumed and switch layout
                
                qo_load_pipe.setup(); 

                #pragma unroll
                for(int i = 0; i < CONFIG.K_PREFETCH_SIZE; i++) qo_load_pipe.waitBuffer(CONSUMER_A).submitToNextAndTrigger(); // already loaded
                
                while(task.iterateNext()){
                    auto load_packet = qo_load_pipe.waitBuffer(CONSUMER_A);

                    bool v_init_avail = task.VInitIdx >= 0;

                    kt::tma::expect_bytes(load_packet.getBarrier(), sizeof(load_packet->q_buffer) + sizeof(load_packet->dO_buffer) + sizeof(load_packet->fwd_max) + (v_init_avail ? sizeof(load_packet->v_buffer) : 0));

                    if(v_init_avail){
                        details::TmaExtension::loadAsync(load_packet->v_buffer, args.VBuffer, kt::coord<>{task.Batch, task.Head, task.VInitIdx, 0}, load_packet.getBarrier());
                    }
                    details::TmaExtension::loadAsync(load_packet->fwd_max, args.FwdMax, kt::coord<>{0,task.Batch, task.Head, task.CurrentQBlock * CONFIG.warp_k_size}, load_packet.getBarrier());
                    
                    kt::tma::load_async<1, kt::cache_policy::NORMAL>(load_packet->q_buffer, args.Q, { task.Batch, task.CurrentQBlock * CONFIG.warp_k_size, task.Head, 0}, load_packet.getBarrier()); // NOTE: AXIS AND OFFSET IS INCORRECT
                    kt::tma::load_async<1, kt::cache_policy::NORMAL>(load_packet->dO_buffer, args.dO, { task.Batch, task.CurrentQBlock * CONFIG.warp_k_size, task.Head, 0}, load_packet.getBarrier()); // NOTE: AXIS AND OFFSET IS INCORRECT
                    
                    
                    load_packet.submitToNextAndTrigger(); 
                }

                __syncthreads();
            }
        }
    } else {
        //kt::warp::increase_registers<232>();
        using MainOp = details::ops::FpAddLseOpSlow;
        using AuxOp = details::ops::FpFMAAffineOp;

        constexpr auto ZERO = MainOp::get_zero<std::tuple<float,float>>();
        uint32_t warpid = kt::warpgroup::warpid();

        kt::rt_hf<CONFIG.warp_q_size, CONFIG.qk_dim> k_tile;
        kt::rt_bf<CONFIG.warp_q_size, CONFIG.head_dim> v_tile;
        kt::rt_fl<CONFIG.warp_q_size, CONFIG.warp_k_size> acc_front;

        kt::rt_fl<CONFIG.warp_q_size, CONFIG.head_dim> dV_tile;
        decltype(dV_tile)::row_vec col_max;

        while(scheduler.getNextTask(task)){
            kt::warp::rt_maps::zero(dV_tile);

            //kt::print_utils::print("Task info: cta = %u, wgid = %u, warpid = %u, batch = %u, head = %u, n_kblocks = %u, vh_start = %u, qstart = %u\n", blockIdx.x, wgid, warpid, task.Batch, task.Head, task.KBlocks, task.VHStart, task.QStart);
            sxdiag::LeftRightVec<CONFIG.warp_q_size, float, float> lr_state;
            sxdiag::TopBottomVec<CONFIG.warp_k_size, float, float> tb_state;


            sxdiag::LeftRightVec<CONFIG.warp_q_size, float, float> bwd_lr_state = ZERO;
            sxdiag::TopBottomVec<CONFIG.warp_k_size, float, float> bwd_tb_state;


            uint32_t consumer_idx = wgid % CONFIG.WARPGROUPS;
            float rtau = args.RcpTau.raw_ptr[task.Head] + CONFIG.qk_bias;

            prefetch_load_pipe.setup();
            auto init_block = prefetch_load_pipe.waitBuffer(PRODUCER);
            
            kt::warpgroup::load(k_tile, init_block->k_buffer[consumer_idx].data);
            kt::warpgroup::load(v_tile, init_block->v_buffer[consumer_idx].data);
            
            init_block.submitToNextAndTrigger();

            qo_load_pipe.setup();
            vh_swap_pipe.setup();

            while(task.iterateNext()){
                auto qo_handle = qo_load_pipe.waitBuffer(PRODUCER);

                // load colmax
                kt::rv_fl<CONFIG.warp_k_size, kt::ducks::rv_layout::align> col_max;
                
                kt::rv_fl<CONFIG.warp_k_size> col_max_naive;
                do{
                    static_assert(CONFIG.warp_k_size == 32);
                    uint32_t tid = threadIdx.x & 0x1f;
                    col_max_naive.data[0][0] = qo_handle->fwd_max.data[(tid / 8) + (tid % 8) * 4];
                    kt::warp::copy(col_max, col_max_naive);
                } while(0);
                
                

                wg::emulated_wgmma::mm_ABt(acc_front, k_tile, qo_handle->q_buffer.data);
                
                auto tb_handle = vh_swap_pipe.waitBuffer((wgid + 1) % CONFIG.WARPGROUPS);

                bool hinit_avail = wgid != CONSUMER_A || (warpid != 0 || task.KIdx != 0);
                if(wgid == CONSUMER_A){
                    tb_handle->h_buffer[warpid].load(args.HBuffer, { task.Batch, task.Head, task.HInitIdx, 0});
                }
                
                tb_state.set<0>(std::get<0>(ZERO));
                lr_state.set<0>(std::get<0>(ZERO));
                if(hinit_avail){
                    tb_state.load<1>(tb_handle->h_buffer[warpid]);
                } else {
                    tb_state.set<1>(std::get<1>(ZERO));
                }
                
                lr_state.load_left<1>(qo_handle->v_buffer, (warpid * 2 + wgid) * CONFIG.warp_q_size + 1);

                kt::warp::rt_maps::unary_map<details::ops::Log2Map>(acc_front, acc_front + CONFIG.qk_log_offset);
                acc_front = acc_front + rtau;
                auto scan_buffer = sxdiag::TightMMABuffer<float, CONFIG.warp_q_size, CONFIG.warp_k_size>::from_rt(acc_front);
                
                // for better register allocation schedule
                
                
                make_causal_triu(acc_front, task.KIdx, task.CurrentQBlock * (CONFIG.warp_k_size / 8), -INFINITY);

                auto u = scan_buffer.expand(std::get<0>(ZERO));
                auto v = u.with_padding(std::get<1>(ZERO));

                sxdiag::ScanTile<CONFIG.warp_q_size, CONFIG.warp_k_size, float, float> tile(u, v);

                auto red_res = tile.diagScanOrReduce<false, true, MainOp>(sxdiag::StatePair<CONFIG.warp_q_size, CONFIG.warp_k_size, float, float>::from(lr_state, tb_state));

                auto v_scan = std::get<1>(tile.buffers).fold_to_rt();
                kt::warp::rt_maps::unary_map<details::ops::Logsumexp1Approx>(v_scan, v_scan);
                v_scan += acc_front;

                


                red_res.tb.store<1>(tb_handle->h_buffer[warpid]);
                
                kt::rt_fl<CONFIG.warp_q_size, CONFIG.warp_k_size> v_scan_exp;
                kt::warp::rt_maps::exp2(v_scan_exp, v_scan - col_max);
                kt::warpgroup::emulated_wgmma::mma_AB(dV_tile, kt::rt_bf<CONFIG.warp_q_size, CONFIG.warp_k_size>{v_scan_exp}, qo_handle->dO_buffer.data);
                
                /*if(wgid == 0 && warpid == 0) { 
                    kt::print_utils::print(qo_handle->fwd_max);
                    //kt::print_utils::print(qo_handle->v_buffer);
                    kt::print_utils::print("task.VInitIdx = %d, HInitIdx = %d\n", task.VInitIdx, task.HInitIdx);
                    //sxdiag::TightMMABuffer<float, CONFIG.warp_q_size, CONFIG.warp_k_size>::from_rt(v_scan_exp).print_tile();
                }*/

                kt::rt_fl<CONFIG.warp_q_size, CONFIG.warp_k_size> add_terms;
                kt::warpgroup::emulated_wgmma::mm_ABt(add_terms, v_tile, qo_handle->dO_buffer.data);
                add_terms *= v_scan_exp;

                kt::rt_fl<CONFIG.warp_q_size, CONFIG.warp_k_size> mul_terms;
                kt::warp::rt_maps::unary_map<details::ops::SigmoidFast>(mul_terms, v_scan);

                sxdiag::ScanTile<CONFIG.warp_q_size, CONFIG.warp_k_size, float, float> tile2(
                    sxdiag::TightMMABuffer<float, CONFIG.warp_q_size, CONFIG.warp_k_size>::from_rt(mul_terms).expand(1.f),
                    sxdiag::TightMMABuffer<float, CONFIG.warp_q_size, CONFIG.warp_k_size>::from_rt(add_terms).expand(0.f)
                );

                auto bwd_res = tile2.diagScanOrReduce<true, false, AuxOp>(sxdiag::StatePair<CONFIG.warp_q_size, CONFIG.warp_k_size, float, float>::from(bwd_lr_state, bwd_tb_state));
                
                bwd_lr_state = bwd_res.lr;
                bwd_res.tb.store<0>(tb_handle->bwd_h_buffer[consumer_idx][warpid]);
                bwd_res.tb.store<1>(tb_handle->bwd_v_buffer[consumer_idx][warpid]);
                
                tb_handle.submitToNextAndTrigger();
                qo_handle.submitToNextAndTrigger();
            }

            kt::group<CONFIG.WARPGROUPS * 4>::sync(1);

            kt::warp::store(load_layouts.dV_out_buffer[warpid * CONFIG.WARPGROUPS + wgid], dV_tile);

            __syncwarp();

            kt::warp::store<1, true>(args.dV, load_layouts.dV_out_buffer[warpid * CONFIG.WARPGROUPS + wgid], kt::coord<>{task.Batch, task.KIdx * 8, task.Head, 0});

            __syncthreads();
        }
        //kt::print_utils::print("Consumer warpid = %u, wgid = %u died\n", warpid, wgid);
    }
}

static __global__ void bwd_kernel(){
    
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

    dim3 grid(4); // single CTA for test
    dim3 block(kt::WARPGROUP_THREADS * DEFAULT_CONFIG.WARPGROUPS + kt::WARP_THREADS * 1);
    auto shape = getVHBufferShape(args.Batch, args.Head, args.Seqlen);

    size_t shared_memory = 65536;

    cudaFuncSetAttribute(sm90::preprocess_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory);

    auto kernel_args = Globals{
        { args.Q, args.Batch, args.Seqlen, args.Head, 0 },
        { args.K, args.Batch, args.Seqlen, args.Head, 0 },
        { args.V, args.Batch, args.Seqlen, args.Head, 0 },
        { args.O, args.Batch, args.Seqlen, args.Head, 0 },
        { args.RcpTau, 0, 0, 0, args.Head },
        { args.VBuffer, shape[0], shape[1], shape[2], 0 },
        { args.HBuffer, shape[0], shape[1], shape[2], 0 }
    };
    sm90::preprocess_kernel<<<grid, block, shared_memory>>>(kernel_args);

    cudaDeviceSynchronize();
    printf("Preproess kernel exit status: %s\n", cudaGetErrorString(cudaGetLastError()));
}


template <int N_HEADDIM, int N_KEYDIM>
void BaselineNoPEAttnStateImpl<N_HEADDIM, N_KEYDIM>::invokeFwd(const FwdPreprocessArgs& args){
    using sm90::DEFAULT_CONFIG_FWD;
    using Globals = sm90::GlobalsFwd<DEFAULT_CONFIG_FWD>;

    dim3 grid(2); // single CTA for test
    dim3 block(kt::WARPGROUP_THREADS * DEFAULT_CONFIG_FWD.WARPGROUPS + kt::WARP_THREADS * 4);
    auto shape = getVHBufferShape(args.Batch, args.Head, args.Seqlen);

    size_t shared_memory = 65536;

    cudaFuncSetAttribute(sm90::fwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory);

    auto kernel_args = Globals{
        { args.Q, args.Batch, args.Seqlen, args.Head, 0 },
        { args.K, args.Batch, args.Seqlen, args.Head, 0 },
        { args.V, args.Batch, args.Seqlen, args.Head, 0 },
        { args.O, args.Batch, args.Seqlen, args.Head, 0 },
        { args.RcpTau, 0, 0, 0, args.Head },
        { args.VBuffer, shape[0], shape[1], shape[2] * (sm90::DEFAULT_CONFIG.warp_k_size / DEFAULT_CONFIG_FWD.warp_k_size), 0 },
        { args.HBuffer, shape[0], shape[1], shape[2] * (sm90::DEFAULT_CONFIG.warp_k_size / DEFAULT_CONFIG_FWD.warp_k_size), 0 },
        { args.FwdMax, 0, args.Batch, args.Head, args.Seqlen }
    };
    sm90::fwd_kernel<<<grid, block, shared_memory>>>(kernel_args);

    cudaDeviceSynchronize();
    printf("Preproess kernel exit status: %s\n", cudaGetErrorString(cudaGetLastError()));
}

template <int N_HEADDIM, int N_KEYDIM>
void BaselineNoPEAttnStateImpl<N_HEADDIM, N_KEYDIM>::invokeBwdPreprocess(const BwdPreprocessArgs& args){
    using sm90::DEFAULT_CONFIG_FWD;
    using Globals = sm90::GlobalsBwdPP<DEFAULT_CONFIG_FWD>;

    dim3 grid(1); // single CTA for test
    dim3 block(kt::WARPGROUP_THREADS * DEFAULT_CONFIG_FWD.WARPGROUPS + kt::WARP_THREADS * 4);
    auto shape = getVHBufferShape(args.Batch, args.Head, args.Seqlen);

    size_t shared_memory = 65536;

    cudaFuncSetAttribute(sm90::bwd_preprocess_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory);

    auto kernel_args = Globals{
        { args.Q, args.Batch, args.Seqlen, args.Head, 0 },
        { args.K, args.Batch, args.Seqlen, args.Head, 0 },
        { args.V, args.Batch, args.Seqlen, args.Head, 0 },
        { args.dO, args.Batch, args.Seqlen, args.Head, 0 },
        { args.dV, args.Batch, args.Seqlen, args.Head, 0 },
        { args.RcpTau, 0, 0, 0, args.Head },
        { args.VBuffer, shape[0], shape[1], shape[2] * (sm90::DEFAULT_CONFIG.warp_k_size / DEFAULT_CONFIG_FWD.warp_k_size), 0 },
        { args.HBuffer, shape[0], shape[1], shape[2] * (sm90::DEFAULT_CONFIG.warp_k_size / DEFAULT_CONFIG_FWD.warp_k_size), 0 },
        { args.FwdMax, 0, args.Batch, args.Head, args.Seqlen }
    };
    sm90::bwd_preprocess_kernel<<<grid, block, shared_memory>>>(kernel_args);

    cudaDeviceSynchronize();
    printf("Preproess kernel exit status: %s\n", cudaGetErrorString(cudaGetLastError()));
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