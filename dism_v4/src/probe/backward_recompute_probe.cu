#include "backward/recompute.cuh"

#include "variant.cuh"
namespace DISM_VARIANT {

namespace {
using namespace dism_backward;
// Dense diagnostic output only: production backward never materializes this.
__global__ void recompute_probe(const __grid_constant__ RecomputeArgs args, float* output) {
    __shared__ KeyTile key;
    __shared__ QueryTile query;
    __shared__ kt::semaphore ready;
    int bh = blockIdx.z;
    int k0 = blockIdx.y * K, q0 = blockIdx.x * Q;
    bool leader = kt::warp::elect_leader();
    if (leader) kt::init_semaphore(ready,1);
    __syncthreads();
    if (leader) {
        kt::tma::expect_bytes<true>(ready,sizeof(KeyTile)+sizeof(QueryTile));
        kt::tma::load_async(key,args.k,{bh/args.heads,bh%args.heads,k0,0},ready);
        kt::tma::load_async(query,args.q,{bh/args.heads,q0,bh%args.heads,0},ready);
    }
    kt::wait(ready,0);
    kt::rt_bf<K,D> keys;
    kt::warp::load(keys,key);
    auto score = recompute(args,keys,query,bh,k0,q0);
#pragma unroll
    for (int r = 0; r < 2; ++r) {
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            auto pos = Scalar::layout(r,c,0);
            int k = k0+pos.first, q = q0+pos.second;
            if (k<args.n && q<args.n)
                output[(int64_t(bh)*args.n+q)*args.n+k] = score.data[r][c].value.u0;
            if (k<args.n && q+4<args.n)
                output[(int64_t(bh)*args.n+q+4)*args.n+k] = score.data[r][c].value.u1;
        }
    }
}
}



const void* recompute_probe_address0() { return reinterpret_cast<const void*>(recompute_probe); }

} // namespace DISM_VARIANT
