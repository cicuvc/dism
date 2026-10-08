#pragma once
#include "forward/affine.cuh"
#include "dimensions.cuh"

#include "variant.cuh"
namespace DISM_VARIANT {

namespace dism_backward {
constexpr int D = ActiveConfig::D;
constexpr int R = ActiveConfig::R;
constexpr int DV = ActiveConfig::DV;
constexpr int K = 16;
constexpr int Q = 32;
using Scalar = pscore::AltLayoutSplitScanBuffer<K,Q,pscore::UnaryElement>;
using Forward = pscore::AltLayoutSplitScanBuffer<K,Q,pscore::BinaryElement,
                                               dism_forward::ForwardLogAffineOp>;
using Reverse = pscore::AltLayoutSplitScanBuffer<K,Q,pscore::BinaryElement,
                                               pscore::AffineComposeOp>;
using KeyTile = kt::st_bf<K,D>;
using QueryTile = KPermutationSharedBuffer<kt::bf16,Q,D>;
using KeyGlobal = kt::gl<kt::bf16,-1,-1,-1,D,KeyTile>;
using QueryGlobal = kt::gl<kt::bf16,-1,-1,-1,D,QueryTile>;

struct RecomputeArgs {
    int batch, heads, n, padded;
    QueryGlobal q;
    KeyGlobal k;
    const float *q_lse, *k_lse, *tau, *vertical, *horizontal;
    const int *q_label, *k_label;
    const uint8_t *hard, *direction;
    const float *gate_delta = nullptr;
};

// Natural-log bias is seeded before MMA, as in the saved forward computation.
// This initial correctness helper deliberately keeps metadata access explicit;
// the WS producer will cache it in the same input slot as its query tile.
template<class ScoreArgs>
__device__ __forceinline__ Scalar recompute(
        const ScoreArgs& args, const kt::rt_bf<K,D>& keys,
        const QueryTile& query_shared, int bh, int k0, int q0,
        const float* row_gate=nullptr) {
    constexpr float LOG2E = 1.4426950408889634f;
    kt::rt_fl<K,Q> score;
    int64_t offset = int64_t(bh) * args.padded;
    bool query_lse = args.direction[bh];
#pragma unroll
    for (int r = 0; r < 2; ++r) {
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            auto pos = Scalar::layout(r,c,0);
            float key_bias = -args.k_lse[offset+k0+pos.first];
            float x = query_lse ? -args.q_lse[offset+q0+pos.second] : key_bias;
            float y = query_lse ? -args.q_lse[offset+q0+pos.second+4] : key_bias;
            score.tiles[0][c/2].data[r+2*(c&1)] = {x,y};
        }
    }
    {
        kt::rt_bf<Q,D> queries;
        kt::warp::load<true>(queries,query_shared.payload);
        [[clang::always_inline]] kt::warp::wmma::mma_ABt(score,keys,queries,score);
    }
    Scalar values;
#pragma unroll
    for (int r = 0; r < 2; ++r) {
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            auto pos = Scalar::layout(r,c,0);
            auto dot = score.tiles[0][c/2].data[r+2*(c&1)];
            int key = k0 + pos.first;
            int label = args.k_label[offset+key];
#pragma unroll
            for (int e = 0; e < 2; ++e) {
                int query = q0 + pos.second + 4*e;
                float value = (e ? dot.y : dot.x) * LOG2E;
                if (args.hard[offset+query])
                    value = args.q_label[offset+query] == label ? args.tau[bh%args.heads]*LOG2E : LOG_ZERO;
                if (query >= args.n || key >= args.n || key > query) value = LOG_ZERO;
                if (e) values.data[r][c].value.u1 = value;
                else values.data[r][c].value.u0 = value;
            }
        }
    }
    values.roll();
    Forward scan;
#pragma unroll
    for (int r = 0; r < 2; ++r) {
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            scan.data[r][c] = {values.data[r][c].value,values.data[r][c].value};
            if (row_gate) {
                auto pos=Scalar::layout(r,c,0);
                scan.data[r][c].first.u0 -= row_gate[pos.second];
                scan.data[r][c].first.u1 -= row_gate[pos.second+4];
            }
        }
    }
    int lane = kt::warp::laneid();
    Forward::HState top;
    Forward::VState left;
    if (k0 > 0 && k0 < args.n && lane/4 < 4) {
        int query = q0 + 8*(lane&3) + 3-lane/4;
        int64_t base = (int64_t(bh)*((args.n-1)/16)+k0/16-1)*args.padded;
        top.init[0].second = {args.vertical[base+query],args.vertical[base+query+4]};
    }
    if (q0 > 0 && (lane&3) == 3) {
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            int key = k0 + r*8 + lane/4 - 1;
            int64_t base = (int64_t(bh)*((args.n-1)/32)+q0/32-1)*args.padded;
            if (key >= 0 && key < args.n && key < q0)
                left.init[r].second.u0 = args.horizontal[base+key];
        }
    }
    auto state = scan.inclusive_prescan(left,top);
    scan.inclusive_postscan(top,state.intermediate);
#pragma unroll
    for (int r = 0; r < 2; ++r) {
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            values.data[r][c].value = scan.data[r][c].second;
        }
    }
    values.roll<false>();
    return values;
}

#if DISM_HOST_API
inline RecomputeArgs make_recompute_args(const std::vector<at::Tensor>& operands,
                                         const at::Tensor& vertical, int n) {
    TORCH_CHECK((operands.size()==13 || operands.size()==14),"expected saved forward operands");
    auto q = operands[0], k = operands[1];
    TORCH_CHECK(q.is_cuda() && q.dim()==4 && q.scalar_type()==at::kBFloat16 &&
                q.size(3)==D && q.is_contiguous() && k.sizes()==q.sizes() &&
                k.device()==q.device() && k.scalar_type()==at::kBFloat16 && k.is_contiguous(),
                "backward requires contiguous BF16 Q/K matching compiled D");
    int batch=q.size(0), padded=q.size(1), heads=q.size(2);
    TORCH_CHECK(n>0 && n<=padded && padded%256==0,"invalid padded sequence length");
    auto check = [&](const at::Tensor& x,at::ScalarType type,at::IntArrayRef shape) {
        TORCH_CHECK(x.device()==q.device() && x.scalar_type()==type && x.is_contiguous() &&
                    x.sizes()==shape,"invalid saved backward metadata");
    };
    check(operands[5],at::kFloat,{batch,heads,padded});
    check(operands[6],at::kFloat,{batch,heads,padded});
    check(operands[7],at::kInt,{batch,heads,padded});
    check(operands[8],at::kInt,{batch,heads,padded});
    check(operands[9],at::kByte,{batch,heads,padded});
    check(operands[10],at::kByte,{batch,heads});
    check(operands[11],at::kFloat,{heads});
    check(operands[12],at::kFloat,{batch,heads,(n-1)/32,padded});
    check(vertical,at::kFloat,{batch,heads,(n-1)/16,padded});
    if(operands.size()==14) check(operands[13],at::kFloat,{batch,heads,padded});
    return {batch,heads,n,padded,
        QueryGlobal(reinterpret_cast<kt::bf16*>(q.data_ptr()),batch,padded,heads,D),
        KeyGlobal(reinterpret_cast<kt::bf16*>(k.data_ptr()),batch,heads,padded,D,
                  kt::gl_strides{size_t(padded*heads*D),D,size_t(heads*D)}),
        operands[5].data_ptr<float>(),operands[6].data_ptr<float>(),operands[11].data_ptr<float>(),
        vertical.data_ptr<float>(),operands[12].data_ptr<float>(),operands[7].data_ptr<int>(),
        operands[8].data_ptr<int>(),operands[9].data_ptr<uint8_t>(),operands[10].data_ptr<uint8_t>(),
        operands.size()==14 ? operands[13].data_ptr<float>() : nullptr};
}
#endif
} // namespace dism_backward

} // namespace DISM_VARIANT
