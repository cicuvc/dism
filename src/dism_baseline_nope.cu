#define KITTENS_RTX_BLACKWELL
// We use emulated wgmma mode in RTX Blackwell in development

#include "common/util.cuh"
#include <dism_baseline_nope.hpp>
#include <kittens.cuh>

namespace kt = kittens;

namespace {

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
    static constexpr auto C = CONFIG;
    kt::gl<kt::bf16, -1, -1, -1, C.QkDim> Q;
    kt::gl<kt::bf16, -1, -1, -1, C.QkDim> K;
    kt::gl<kt::bf16, -1, -1, -1, C.HeadDim> V;
    kt::gl<float, 1, 1, 1, -1> RcpTau;

    kt::gl<kt::half, -1, -1, -1, C.QkDim> QSoft;
    kt::gl<kt::half, -1, -1, -1, C.QkDim> KSoft;

    kt::gl<float, -1, -1, -1, C.WarpKSize> VBuffer;
    kt::gl<float, -1, -1, -1, C.WarpKSize> HBuffer;
};

template <PreprocessKernelSm90Config CONFIG = DEFAULT_CONFIG>
static __global__ void kernel(const __grid_constant__ Globals<CONFIG> args) {
    extern int shmem[];
    kt::shared_allocator<> allocator{shmem};

    
}
} // namespace sm90
} // namespace baseline::preprocess

} // namespace