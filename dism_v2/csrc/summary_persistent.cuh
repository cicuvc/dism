// Persistent summary: Q input is independent of the K ring and pair mailboxes.
// Included inside dism_v2; output core is deliberately unchanged.
template <int D> struct SummaryShared {
    static constexpr int SLOTS = D == 128 ? 1 : 3;
    kt::st_bf<128, D> q;
    Slot<D, 32, false> slot[SLOTS];
    uint64_t qready, qfree;
    uint64_t ready[SLOTS], free[SLOTS];
    // Four payloads, but ONE ready/free epoch per warpgroup and slot.
    uint64_t mail_ready[SLOTS], mail_free[SLOTS];
    Buffer::HState::SharedStorage mail[4][SLOTS];
};
template <int D> CUtensorMap summary_q_map(const Args &p) {
    constexpr int S = D == 32 ? 32 : 64;
    const cuuint64_t dims[]{S, cuuint64_t(p.n), cuuint64_t(p.batch_heads), D / S, 1};
    const cuuint64_t strides[]{D * 2, cuuint64_t(p.n) * D * 2, S * 2, S * 2};
    // A single transfer covers both all query rows and all swizzle panels.
    // Physical Q rows stay natural; warp subtiles select the interleaved rows.
    const cuuint32_t box[]{S, 128, 1, D / S, 1}, elem[]{1, 1, 1, 1, 1};
    CUtensorMap map{};
    auto e = cuTensorMapEncodeTiled(&map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 5, const_cast<void *>(p.a), dims,
                                    strides, box, elem, CU_TENSOR_MAP_INTERLEAVE_NONE,
                                    D == 32 ? CU_TENSOR_MAP_SWIZZLE_64B : CU_TENSOR_MAP_SWIZZLE_128B,
                                    CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (e != CUDA_SUCCESS)
        throw std::runtime_error("summary Q TMA map failed");
    return map;
}
__device__ __forceinline__ void summary_tma(const CUtensorMap *map, void *dst, uint64_t *bar, int4 coord) {
    kt::tma::atoms::load_async_atom<kt::cache_policy::NORMAL>(smaddr(dst), reinterpret_cast<uint64_t>(map),
                                                              coord, *reinterpret_cast<kt::semaphore *>(bar));
}
// Explicit predicate selection: no branch and no arithmetic with masked infinities.
__device__ __forceinline__ float summary_select(bool pred, float yes, float no) {
    float result;
    asm("{ .reg .pred p; setp.ne.u32 p, %3, 0; selp.f32 %0, %1, %2, p; }"
        : "=f"(result) : "f"(yes), "f"(no), "r"(int(pred)));
    return result;
}

template<int ROWS, int COLS>
__device__ __forceinline__ void load_rhs_tile(kt::rt_bf<COLS, ROWS, kt::ducks::rt_layout::col>& dst, const kt::st_bf<ROWS, COLS>& src){
    int warp_laneid = ::kittens::laneid();

    uint32_t shared_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&src.data[0]));

    #pragma unroll
    for (int c = 0; c < dst.height; c++) {
        #pragma unroll
        for (int r = 0; r < dst.width; r++) {
            kittens::bf16_2 tmp[4];

            int row = 16 * r + (warp_laneid / 16) * 8 + (warp_laneid % 8);
            int col = 16 * c + ((warp_laneid / 8) % 2) * 8;

            kt::move<kt::bf16_2>::ldsm4(tmp[0], tmp[2], tmp[1], tmp[3], src.idx(shared_addr, {row, col}));
            dst.tiles[c][r].data[0] = tmp[0];
            dst.tiles[c][r].data[1] = tmp[1];
            dst.tiles[c][r].data[2] = tmp[2];
            dst.tiles[c][r].data[3] = tmp[3];
        }
    }
}
// MODE: 0 soft, 1 hard, 2 mixed. Label width is a host dispatch decision.
template <int D, bool COLUMN_LSE, int MODE, typename Label>
__global__ __launch_bounds__(384, 1) void summary_persistent(__grid_constant__ const Args p,
                                                             __grid_constant__ const CUtensorMap qm,
                                                             __grid_constant__ const CUtensorMap km) {
    constexpr int SLOTS = SummaryShared<D>::SLOTS;
    extern __shared__ __align__(128) unsigned char bytes[];
    auto &shared = *reinterpret_cast<SummaryShared<D> *>(bytes);
    int warp = threadIdx.x / 32, lane = threadIdx.x & 31;
    int blocks = (p.n + 127) / 128, total = blocks * p.batch_heads;
    if (warp == 0 && kt::warp::elect_leader()) {
        init_bar(&shared.qready, 1);
        init_bar(&shared.qfree, 256);
#pragma unroll
        for (int s = 0; s < SLOTS; ++s) {
            init_bar(&shared.ready[s], 32);
            init_bar(&shared.free[s], 256);
            init_bar(&shared.mail_ready[s], 128);
            init_bar(&shared.mail_free[s], 128);
        }
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    }
    __syncthreads();
    int tile = 0, task_round = 0;
    if (warp >= 8) {
        kt::warpgroup::decrease_registers<40>();
        if (warp == 8) {
            bool leader = kt::warp::elect_leader();
#pragma unroll 1
            for (int task = blockIdx.x; task < total; task += gridDim.x, ++task_round) {
                int bh = task / blocks, base = (task % blocks) * 128;
                int key_end = min(p.padded_n, base + 128);
                // Producer runs ahead: this Q and K0 launch while consumers
                // finish the previous workload. No task-end CTA barrier.
                if (leader) {
                    if (task_round)
                        wait(&shared.qfree, (task_round - 1) & 1);
                    expect(&shared.qready, 8 * 16 * D * 2);
                    summary_tma(&qm, shared.q.data, &shared.qready, {base, bh, 0, 0});
                }
#pragma unroll 1
                for (int t = 0; t * 64 < key_end; ++t, ++tile) {
                    int s = tile % SLOTS;
                    if (tile >= SLOTS)
                        wait(&shared.free[s], ((tile / SLOTS) - 1) & 1);
                    auto &slot = shared.slot[s];
                    if (t * 64 + 64 <= p.n) {
                        if (leader) {
                            expect(&shared.ready[s], sizeof(slot.k));
                            summary_tma(&km, slot.k.data, &shared.ready[s],
                                        {0, 0, bh * p.n + t * 64, 0});
                        } else arrive(&shared.ready[s]);
                    } else {
// Safe sequence-local tail; no flattened-map overread.
#pragma unroll
                        for (int x = lane; x < 64 * D; x += 32) {
                            int j = t * 64 + logical_row(x / D);
                            slot.k[int2{x / D, x % D}] = j < p.n
                                                             ? static_cast<const __nv_bfloat16 *>(
                                                                   p.b)[(int64_t(bh) * p.n + j) * D + x % D]
                                                             : __float2bfloat16(0);
                        }
                        __syncwarp();
                        arrive(&shared.ready[s]);
                    }
                }
            }
        }
    } else {
        kt::warpgroup::increase_registers<232>();
#pragma unroll 1
        for (int task = blockIdx.x; task < total; task += gridDim.x, ++task_round) {
            int bh = task / blocks, base = (task % blocks) * 128;
            int key_end = min(p.padded_n, base + 128);
            int checkpoint = (task % blocks) * 4 + (warp & 3);
            int qbase = base + (warp & 3) * 32 + (warp / 4) * 16;
            wait(&shared.qready, task_round & 1);
            kt::rt_bf<16, D> qreg;
            auto qview = shared.q.template subtile<16, D>({(warp & 3) * 2 + warp / 4, 0});
            kt::warp::load(qreg, qview);
            arrive(&shared.qfree);
            // One lane per query row generates a decision, before the key loop.
            bool hard[2];
            if constexpr(MODE!=2) {hard[0]=hard[1]=MODE==1;}
            else if(p.hard_bits) {
                uint32_t bits=qbase<p.n?p.hard_bits[int64_t(bh)*((p.n+31)/32)+qbase/32]:0;
                hard[0]=(bits>>((qbase%32)+lane/4))&1;
                hard[1]=(bits>>((qbase%32)+8+lane/4))&1;
            } else {
                int decision=0;
                if(lane<16 && qbase+lane<p.n)
                    decision=row_hard(p.seed,p.offset,uint64_t(bh)*p.n+qbase+lane,p.hard_prob);
                hard[0]=__shfl_sync(0xffffffff,decision,lane/4,32);
                hard[1]=__shfl_sync(0xffffffff,decision,8+lane/4,32);
            }
            // Cache query-row invariants before the key loop. Column LSE remains
            // indexed by key inside score(); the two directions are not equivalent.
            const float tau = p.tau[bh % p.heads];
            const float tau2 = tau * LOG2E, scale2 = p.scale * LOG2E;
            float row_bias2[2];
            Label cached_label[2];
#pragma unroll
            for (int r = 0; r < 2; ++r) {
                int i = qbase + r * 8 + lane / 4;
                if constexpr (!COLUMN_LSE && MODE!=1)
                    row_bias2[r] = i < p.n ? (tau-p.lse[int64_t(bh)*p.n+i])*LOG2E : 0.f;
                if constexpr(MODE!=0)
                    cached_label[r] = i < p.n ? reinterpret_cast<const Label*>(p.q_label)[int64_t(bh) * p.n + i] : -1;
            }
            Buffer::VState left;
#pragma unroll 1
            for (int t = 0; t * 64 < key_end; ++t, ++tile) {
                int s = tile % SLOTS, phase = (tile / SLOTS) & 1;
                Scalar scalar;
                {
                    // Distributed registers: lane l owns columns l and l+32.
                    // Launch metadata loads BEFORE the input wait and MMA;
                    // no global load is hidden inside per-element selection.
                    Label key_label[2];
                    float key_bias2[2];
                    #pragma unroll
                    for (int e=0;e<2;++e) {
                        int j=t*64+lane+e*32;
                        if constexpr(MODE!=0)
                            key_label[e]=j<p.n ? reinterpret_cast<const Label*>(p.k_label)[int64_t(bh)*p.n+j] : -1;
                        if constexpr(COLUMN_LSE && MODE!=1)
                            key_bias2[e]=j<p.n ? (tau-p.lse[int64_t(bh)*p.n+j])*LOG2E : 0.f;
                    }
                    wait(&shared.ready[s], phase);
                    kt::rt_bf<D, 64, kt::ducks::rt_layout::col> kreg;
                    kt::rt_fl<16, 64> accum{0.f};
                    load_rhs_tile(kreg, shared.slot[s].k);
                    kt::warp::wmma::mma_AB(accum, qreg, kreg, accum);
#pragma unroll
                    for (int r = 0; r < 2; ++r) {
#pragma unroll
                        for (int c = 0; c < 8; ++c) {
                            auto x = accum.tiles[0][c / 2].data[r + 2 * (c & 1)];
                            const int col=c+8*(lane&3), i=qbase+r*8+lane/4;
                            float values[2];
                            #pragma unroll
                            for(int e=0;e<2;++e) {
                                float score;
                                if constexpr(MODE!=1) {
                                    float bias;
                                    if constexpr(COLUMN_LSE) bias=__shfl_sync(0xffffffff,key_bias2[e],col, 32);
                                    else bias=row_bias2[r];
                                    score=fmaf(e==0?x.x:x.y,scale2,bias);
                                }
                                if constexpr(MODE!=0) {
                                    Label label=__shfl_sync(0xffffffff,key_label[e],col, 32);
                                    float hs=summary_select(cached_label[r]==label,tau2,LOG_ZERO);
                                    if constexpr(MODE==1) score=hs;
                                    else score=summary_select(hard[r],hs,score);
                                }
                                int j=t*64+col+e*32;
                                values[e]=summary_select(i<p.n && j<p.n && j<=i,
                                    score,LOG_ZERO);
                            }
                            scalar.data[r][c].value={values[0],values[1]};
                        }
                    }
                }
                arrive(&shared.free[s]);
                scalar.roll();
                Buffer data;
#pragma unroll
                for (int r = 0; r < 2; ++r) {
#pragma unroll
                    for (int c = 0; c < 8; ++c) {
                        auto x = scalar.data[r][c].value;
                        data.data[r][c] = {x, x};
                        int i = qbase + (r * 8 + lane / 4 - (7 - c) + 16) % 16;
                        int j = t * 64 + c + (lane & 3) * 8;
                        if (i >= p.n || j >= p.n) {
                            data.data[r][c].first.u0 = 0;
                            data.data[r][c].second.u0 = LOG_ZERO;
                        }
                        if (i >= p.n || j + 32 >= p.n) {
                            data.data[r][c].first.u1 = 0;
                            data.data[r][c].second.u1 = LOG_ZERO;
                        }
                    }
                }
                Buffer::HState top;
                if (warp >= 4) {
                    wait(&shared.mail_ready[s], phase);
                    top = Buffer::HState::load_shared(shared.mail[warp - 4][s]);
                    arrive(&shared.mail_free[s]);
                }
                Buffer::StatePair result;
                result = data.reduce_forward(left, top);
                left = result.first;
                if (warp < 4) {
                    if (tile >= SLOTS)
                        wait(&shared.mail_free[s], phase ^ 1);
                    result.second.store_shared(shared.mail[warp][s]);
                    arrive(&shared.mail_ready[s]);
                } else {
                    if (checkpoint < p.checkpoints) {
                        int j = 8 * (lane & 3) + 6 - lane / 4;
                        auto h = result.second.init[0];
                        auto *dest = reinterpret_cast<float2 *>(p.summary) +
                                     (int64_t(bh) * p.checkpoints + checkpoint) * p.padded_n + t * 64;
                        if (j >= 0)
                            dest[j] = make_float2(h.first.u0, h.second.u0);
                        dest[j + 32] = make_float2(h.first.u1, h.second.u1);
                        if (lane == 31)
                            dest[63] = make_float2(left.init[1].first.u0, left.init[1].second.u0);
                    }
                }
            }
            if (warp >= 4 && checkpoint < p.checkpoints) {
#pragma unroll 1
                for (int j = base + 128 + lane; j < p.padded_n; j += 32) {
                    int64_t off = (int64_t(bh) * p.checkpoints + checkpoint) * p.padded_n + j;
                    reinterpret_cast<float2 *>(p.summary)[off] =
                        make_float2(j >= p.n + 31 ? 0.f : LOG_ZERO, LOG_ZERO);
                }
            }
        }
    }
    __syncthreads(); // Final drain only, not a workload boundary.
}
template <int D, bool COLUMN_LSE, int MODE, typename Label> void launch_persistent_summary_variant(const Args &p, cudaStream_t stream) {
    auto qm = summary_q_map<D>(p), km = permuted_map<D, true>(p.b, p.batch_heads * p.n);
    constexpr int sm = sizeof(SummaryShared<D>);
    auto e = cudaFuncSetAttribute(summary_persistent<D,COLUMN_LSE,MODE,Label>, cudaFuncAttributeMaxDynamicSharedMemorySize, sm);
    if (e != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(e));
    int device, sms;
    e = cudaGetDevice(&device);
    if (e != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(e));
    e = cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
    if (e != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(e));
    int tasks = ((p.n + 127) / 128) * p.batch_heads;
    summary_persistent<D,COLUMN_LSE,MODE,Label><<<std::min(tasks, sms), 384, sm, stream>>>(p, qm, km);
}
template<int D,bool C,typename Label> void launch_persistent_summary_mode(const Args& p,cudaStream_t s) {
    if(p.hard_prob==0) launch_persistent_summary_variant<D,C,0,int>(p,s);
    else if(p.hard_prob==1) launch_persistent_summary_variant<D,C,1,Label>(p,s);
    else launch_persistent_summary_variant<D,C,2,Label>(p,s);
}
template<int D,bool C> void launch_persistent_summary_direction(const Args& p,cudaStream_t s) {
    if(p.label32) launch_persistent_summary_mode<D,C,int>(p,s);
    else launch_persistent_summary_mode<D,C,long long>(p,s);
}
template <int D> void launch_persistent_summary(const Args &p, cudaStream_t stream) {
    if(p.column_lse) launch_persistent_summary_direction<D,true>(p,stream);
    else launch_persistent_summary_direction<D,false>(p,stream);
}
