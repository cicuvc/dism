#include "summary/primitives.cuh"

#include "variant.cuh"
namespace DISM_VARIANT {

namespace {
using Reverse = pscore::AltLayoutSplitScanBuffer<16,32,pscore::BinaryElement,
                                                pscore::AffineComposeOp>;
using Scalar = pscore::AltLayoutSplitScanBuffer<16,32,pscore::UnaryElement>;

// Tests native reverse ownership, both affine components and early publication.
__global__ void reverse_probe(const float2* pairs, const float* bottom,
                              const float* right, float2* result, float2* edge) {
    Reverse scan;
    int lane = threadIdx.x & 31;
    int row_lane = lane / 4;
#pragma unroll
    for (int r = 0; r < 2; ++r) {
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            auto pos = Scalar::layout(r,c,0);
            auto x = pairs[pos.first * 32 + pos.second];
            auto y = pairs[pos.first * 32 + pos.second + 4];
            scan.data[r][c] = {{x.x,y.x},{x.y,y.y}};
        }
    }
    scan.reverse_roll();
    Reverse::HState incoming;
    Reverse::VState side;
    int column = 8 * (lane & 3) + 7 - row_lane;
    if (row_lane >= 4)
        incoming.init[0] = {{1.f,1.f},{bottom[column],bottom[column+4]}};
    if ((lane & 3) == 0) {
#pragma unroll
        for (int r = 0; r < 2; ++r)
            side.init[r] = {{1.f},{right[r*8+row_lane]}}; // physical rows1..16
    }
    auto early = scan.reverse_inclusive_prescan(side,incoming);
    if (row_lane >= 4) {
        auto value = early.horizontal.init[0];
        edge[column] = {value.first.u0,value.second.u0};
        edge[column+4] = {value.first.u1,value.second.u1};
    }
    scan.reverse_inclusive_postscan(incoming,early.intermediate);
    scan.reverse_roll<false>();
#pragma unroll
    for (int r = 0; r < 2; ++r) {
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            auto pos = Scalar::layout(r,c,0);
            auto value = scan.data[r][c];
            result[pos.first*32+pos.second] = {value.first.u0,value.second.u0};
            result[pos.first*32+pos.second+4] = {value.first.u1,value.second.u1};
        }
    }
}
} // namespace



const void* reverse_probe_address0() { return reinterpret_cast<const void*>(reverse_probe); }

} // namespace DISM_VARIANT
