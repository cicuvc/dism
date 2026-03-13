#include <diag_scan.cuh>
#include <common/debug.cuh>


namespace{
    struct Fp32Add{
        __forceinline__ __device__ static float op(float x, float y) { return x + y; }
    };
    struct NegLseOpSlow{
        __forceinline__ __device__ static float op(float x, float y) {
            // This approximately calculates -log2(exp2(-x)+exp2(-y)) safely.
            // out = -log2(1+exp2(-|x-y|)) + min(x,y)
            // Trivial approach: use MUFU to calculate exp2(-|x-y|), and 3 order polynomial to fit -log2(1+x) in (0,1]
            // Here: let -log2(1+exp2(x)) = -exp2(f(max(LIMIT, x)) + x), so f(x) = log2(log2(1+exp2(max(LIMIT, x)))) - x
            // and fit log2(log2(1+exp2(t))) in (LIMIT, 0]. Fit error of polynomial and cut-off error can be reduced by 
            // the small derivative of exp2 at negative interval.

            // SASS code:
            // FADD DIFF, X, -Y
            // FMNMX.MIN MI, X, Y
            // FMNMX.MAX P, -|DIFF|, -5.f
            // FFMA X1, P, -0.004380030100270f, -0.056839571752722f
            // FFMA X2, P, X1, -0.278691685310180f
            // FFMA LOGS, P, X2, -|DIFF|
            // MUFU.EX2 RES, LOGS
            // FADD OUT, MI, -RES
            float nabs = -abs(x - y), mi = min(x, y); 
            float p = max(nabs, -4.f); 
            float logs = nabs + p * (-2.793604998770852221e-01f + p * (-5.756650101889571047e-02f + p * (-4.553042169205498771e-03f))), res;
            asm volatile("ex2.approx.ftz.f32 %0, %1;\n":"=f"(res): "f"(logs));
            return mi - res;
        }
    };
    __global__ void diagScanReference(const __grid_constant__ kt::gl<float, 1, 1, 16, 16> logM, const __grid_constant__ kt::gl<float, 1, 1, 2, 16> tl_init){
        __shared__ kt::sv_fl<16> ti, li;
        
        kt::rt_fl<16, 16> rt_logM;
        kt::warp::load(ti, tl_init, {0,0,0,0});
        kt::warp::load(li, tl_init, {0,0,1,0});
        kt::warp::load(rt_logM, logM, {0,0,0,0});

        diagscan::ScanLrState<16, diagscan::details::DownVec> l_state;
        diagscan::ScanTbState<16, diagscan::details::DownVec> t_state;
        l_state.load(li), t_state.load(ti);

        auto init = diagscan::DiagScanHelpers::makeState<diagscan::details::Input, diagscan::details::DownVec, false, false, 16, 16>(-l_state, -t_state);

        auto acc_state = diagscan::DiagScanHelpers::diagScanRef<Fp32Add, 0.f>(rt_logM);
        

        auto res = diagscan::DiagScanHelpers::diagReduce<NegLseOpSlow, 9999.f, diagscan::details::DownVec, diagscan::details::Output>(rt_logM, init);
        
        diagscan::ScanLrState<16, diagscan::details::DownVec>(acc_state.LrState - res.LrState).store(li);
        diagscan::ScanTbState<16, diagscan::details::DownVec>(acc_state.TbState - res.TbState).store(ti);

        kt::warp::store(tl_init, ti, {0,0,0,0});
        kt::warp::store(tl_init, li, {0,0,1,0});
    }
} // namespace

extern void invokeDiagScan(void *logM, void *tl_init){
    diagScanReference<<<1,32>>>({(float*)logM, 0,0,0,0}, {(float*)tl_init, 0,0,0,0});
    cudaDeviceSynchronize();
}