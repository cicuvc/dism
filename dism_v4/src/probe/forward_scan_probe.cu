#include "forward/scan.cuh"

#include "variant.cuh"
namespace DISM_VARIANT {

template <bool Exact> struct ProbeLogAffine {
    template <typename T>
    __host__ __device__ static pscore::BinaryElement<T> identity() {
        return LogAffineOp::identity<T>();
    }
    __device__ __forceinline__ static float exact_lse(float a, float b) {
        return fmaxf(a, b) + log1pf(exp2f(-fabsf(a - b))) * 1.4426950408889634f;
    }
    template <typename T>
    __device__ __forceinline__ static pscore::BinaryElement<T> apply(
            const pscore::BinaryElement<T>& a, const pscore::BinaryElement<T>& b) {
        if constexpr (!Exact) return LogAffineOp::apply(a, b);
        else {
            pscore::BinaryElement<T> result;
            result.first = a.first + b.first;
            result.second.u0 = exact_lse(a.second.u0 + b.first.u0, b.second.u0);
            if constexpr (requires { result.second.u1; }) {
                result.second.u1 = exact_lse(a.second.u1 + b.first.u1, b.second.u1);
            }
            return result;
        }
    }
};

// Independent16x64 bring-up: keep incoming HState intact across prescan,
// expose its output before postscan, and carry VState across key tiles.
template <bool Exact>
__global__ void forward_scan_probe_kernel(const float* scores, const float* top,
                                          float* output, float* bottom, int columns) {
    using namespace dism_forward;
    using ProbeScan = pscore::AltLayoutSplitScanBuffer<16, WarpKSize, pscore::BinaryElement,
                                                     ProbeLogAffine<Exact>>;
    typename ProbeScan::VState left;
#pragma unroll 1
    for (int start = 0; start < columns; start += WarpKSize) {
        Scalar scalar;
#pragma unroll
        for (int r = 0; r < 2; ++r) {
#pragma unroll
            for (int c = 0; c < Scalar::COL_BLOCKS; ++c) {
                auto pos = Scalar::layout(r, c, 0);
                int offset = pos.first * columns + start + pos.second;
                scalar.data[r][c].value = {scores[offset], scores[offset + Scalar::COL_BLOCKS]};
            }
        }
        scalar.roll();
        ProbeScan scan;
#pragma unroll
        for (int r = 0; r < 2; ++r) {
#pragma unroll
            for (int c = 0; c < Scalar::COL_BLOCKS; ++c) {
                scan.data[r][c] = {scalar.data[r][c].value, scalar.data[r][c].value};
            }
        }
        typename ProbeScan::HState incoming;
        int col = start + boundary_column();
        incoming.init[0].first = {0.f, 0.f};
        if (boundary_owner())
            incoming.init[0].second = {top[col], top[col + Scalar::COL_BLOCKS]};
        auto state = scan.inclusive_prescan(left, incoming);
        left = state.vertical;
        if (boundary_owner()) {
            bottom[col] = state.horizontal.init[0].second.u0;
            bottom[col + Scalar::COL_BLOCKS] = state.horizontal.init[0].second.u1;
        }
        auto values = finish_scan(scan, incoming, state.intermediate);
#pragma unroll
        for (int r = 0; r < 2; ++r) {
#pragma unroll
            for (int c = 0; c < Scalar::COL_BLOCKS; ++c) {
                auto pos = Scalar::layout(r, c, 0);
                int offset = pos.first * columns + start + pos.second;
                output[offset] = values.data[r][c].value.u0;
                output[offset + Scalar::COL_BLOCKS] = values.data[r][c].value.u1;
            }
        }
    }
}



const void* forward_scan_probe_kernel_address0() { return reinterpret_cast<const void*>(forward_scan_probe_kernel<true>); }
const void* forward_scan_probe_kernel_address1() { return reinterpret_cast<const void*>(forward_scan_probe_kernel<false>); }

} // namespace DISM_VARIANT
