#include <utility>
#define KITTENS_RTX_BLACKWELL
// We use emulated wgmma mode in RTX Blackwell in development
#include <dism_baseline_nope.hpp>
#include <kittens.cuh>

namespace kt = kittens;

namespace {

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

} // namespace details

namespace baseline::preprocess {

struct PreprocessKernelSm90Config {
    int HeadDim, QkDim;
    int QBlockSize, KBlockSize;
    int NStages;
    float QkEps;

    static constexpr int WARPGROUPS = 2;
    int WarpQSize, WarpKSize, SkipRows;

    constexpr PreprocessKernelSm90Config(int headDim_, int qkDim_, int qBlockSize_, int kBlockSize_, int nStage_, float qkEps_){
        HeadDim = headDim_, QkDim = qkDim_;
        QBlockSize = qBlockSize_, KBlockSize = kBlockSize_;
        NStages = nStage_;
        QkEps = qkEps_;

        WarpQSize = (qBlockSize_ / WARPGROUPS / kt::WARPGROUP_WARPS);
        WarpKSize = kBlockSize_;
        SkipRows = qBlockSize_ / kt::WARPGROUP_WARPS;
    }
};

namespace sm90 {

static constexpr PreprocessKernelSm90Config DEFAULT_CONFIG = {32, 32, 128, 32, 2, 0.4f};

template <PreprocessKernelSm90Config CONFIG = DEFAULT_CONFIG>
struct Globals {
    // Input layout: [B, N, H, C]/[1, N, H, C]
    static constexpr auto C = CONFIG;
    kt::gl<kt::bf16, -1, -1, -1, C.QkDim> Q;
    kt::gl<kt::bf16, -1, -1, -1, C.QkDim> K;
    kt::gl<kt::bf16, -1, -1, -1, C.HeadDim> V;
    kt::gl<float, 1, 1, 1, -1> RcpTau;

    kt::gl<kt::half, -1, -1, -1, C.QkDim> QSoft;
    kt::gl<kt::half, -1, -1, -1, C.QkDim> KSoft;

    kt::gl<float, -1, -1, -1, C.WarpKSize> VBuffer;
    kt::gl<float, -1, -1, -1, C.WarpKSize> HBuffer;

    uint64_t *QOffset, *VHOffset;
};

template<typename TGlobals>
struct NonVarlenScheduler{
    static std::pair<uint32_t, uint32_t> operator()(const TGlobals& args, uint32_t i_iter){

        return std::make_pair(0, 0);
    }
};

template <PreprocessKernelSm90Config CONFIG = DEFAULT_CONFIG>
static __global__ void kernel(const __grid_constant__ Globals<CONFIG> args) {
    using wg = kt::warpgroup;

    constexpr int N_CONSUMER_WARPS = CONFIG.WARPGROUPS * kt::WARPGROUP_WARPS;

    extern int shmem[];
    kt::shared_allocator<> allocator{shmem};

    if(kt::warpid() == N_CONSUMER_WARPS){ // producer groups

    }
}

static __global__ void testTmaKernel(const __grid_constant__ kt::gl<kt::bf16, -1, -1, -1, 32, details::TmaWarpgroupStrided<kt::st_bf<64, 32>, 32>> gm, int b, int s, int h, int c){
    __shared__ details::TmaWarpgroupStrided<kt::st_bf<64, 32>, 32> buffer;
    kt::warp::st_maps::one(buffer.data);

    kt::warp::tma::store_async<1>(gm, buffer, { b, s, h, c });
    kt::warp::tma::store_async_wait();

    __syncthreads();
}



} // namespace sm90
} // namespace baseline::preprocess

} // namespace

extern void invokeTestTma(void *ptr, size_t b, size_t s, size_t h, int ib, int is, int ih, int ic){
    baseline::preprocess::sm90::testTmaKernel<<<1, 32>>>({(kt::bf16*)ptr, b, s, h, 0}, ib, is, ih, ic);
}