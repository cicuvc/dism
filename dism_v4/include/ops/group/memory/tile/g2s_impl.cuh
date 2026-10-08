#pragma once

#include "../../group.cuh"

using kittens::group;

template <int GW>
template <bool assume_aligned, ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD>
__device__ inline void group<GW>::load(ST &dst, const GL &src, const COORD &idx) {
    using T = typename ST::dtype;
    const int row_stride = src.template stride<dim::ROW>();
    // we can handle this many rows each time we run a memcpy_async
    constexpr int elem_per_memcpy = sizeof(float4) / sizeof(typename ST::dtype);
    constexpr int memcpy_per_row = ST::cols / elem_per_memcpy;
    constexpr int total_calls = (ST::height * ST::width * kittens::TILE_ROW_DIM<T> * kittens::TILE_COL_DIM<T> +
                                 GROUP_THREADS * elem_per_memcpy - 1) /
                                (GROUP_THREADS * elem_per_memcpy); // round up
    constexpr int total_rows = ST::height * ST::width;

    coord<> unit_coord = coord<>(idx);
    typename GL::dtype *src_ptr = (typename GL::dtype *)&src[unit_coord];
    uint32_t dst_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&dst.data[0]));
    int laneid = threadIdx.x % GROUP_THREADS;

#pragma unroll
    for (int i = 0; i < total_calls; i++) {

        int load_idx = i * GROUP_THREADS + laneid;

        int row = load_idx / memcpy_per_row;
        int col = (load_idx * elem_per_memcpy) % dst.cols;

        if constexpr (assume_aligned) {
            float4 tmp;
            move<float4>::ldg(tmp, (float4 *)&src_ptr[row * row_stride + col]);
            move<float4>::sts(dst.idx(dst_ptr, {row, col}), tmp);
        } else {
            if (row + unit_coord.r < src.template shape<dim::ROW>()) {
                float4 tmp;
                move<float4>::ldg(tmp, (float4 *)&src_ptr[row * row_stride + col]);
                move<float4>::sts(dst.idx(dst_ptr, {row, col}), tmp);
            } else {
                float4 zeros = {0.f, 0.f, 0.f, 0.f};
                move<float4>::sts(dst.idx(dst_ptr, {row, col}),
                                  zeros); // use the default value
            }
        }
    }
}

template <int GW>
template <ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD>
__device__ inline void group<GW>::load(ST &dst, const GL &src, const COORD &idx) {
    load<false, ST, GL, COORD>(dst, src, idx);
}

/**
 * @brief Stores data from a shared memory tile into global memory.
 *
 * @tparam ST The type of the shared tile.
 * @param[out] dst The destination global memory array.
 * @param[in] src The source shared memory tile.
 * @param row_stride[in] The stride between rows in the destination array.
 */
template <int GW>
template <bool assume_aligned, ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD>
__device__ inline void group<GW>::store(const GL &dst, const ST &src, const COORD &idx) {
    using T = typename ST::dtype;
    const int row_stride = dst.template stride<dim::ROW>();
    // we can handle this many rows each time we run a memcpy_async
    constexpr int elem_per_memcpy = sizeof(float4) / sizeof(typename ST::dtype);
    constexpr int memcpy_per_row = ST::cols / elem_per_memcpy;
    constexpr int total_calls = (ST::height * ST::width * kittens::TILE_ROW_DIM<T> * kittens::TILE_COL_DIM<T> +
                                 GROUP_THREADS * elem_per_memcpy - 1) /
                                (GROUP_THREADS * elem_per_memcpy); // round up

    coord<> unit_coord = coord<>(idx);
    typename GL::dtype *dst_ptr = (typename GL::dtype *)&dst[unit_coord];
    uint32_t src_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&src.data[0]));
    int laneid = threadIdx.x % GROUP_THREADS;

#pragma unroll
    for (int i = 0; i < total_calls; i++) {

        int load_idx = i * GROUP_THREADS + laneid;

        int row = load_idx / memcpy_per_row;
        int col = (load_idx * elem_per_memcpy) % src.cols;

        if constexpr (assume_aligned) {
            float4 tmp;
            move<float4>::lds(tmp, src.idx(src_ptr, {row, col}));
            move<float4>::stg((float4 *)&dst_ptr[row * row_stride + col], tmp);
        } else {
            if (row + unit_coord.r < dst.template shape<dim::ROW>()) {
                float4 tmp;
                move<float4>::lds(tmp, src.idx(src_ptr, {row, col}));
                move<float4>::stg((float4 *)&dst_ptr[row * row_stride + col], tmp);
            }
        }
    }
}

template <int GW>
template <ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD>
__device__ inline void group<GW>::store(const GL &dst, const ST &src, const COORD &idx) {
    store<false, ST, GL, COORD>(dst, src, idx);
}

/**
 * @brief Asynchronously loads data from global memory into a shared memory
 * tile.
 *
 * @tparam ST The type of the shared tile.
 * @param[out] dst The destination shared memory tile.
 * @param[in] src The source global memory array.
 *
 * @note This function expects 16-byte alignments. Otherwise, behavior is
 * undefined.
 * @note The caller must commit the issued copies with
 * load_async_commit_group() before waiting for them.
 */
template <int GW>
template <bool assume_aligned, ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD>
__device__ inline void group<GW>::load_async(ST &dst, const GL &src, const COORD &idx) {
    using T = typename ST::dtype;
    const int row_stride = src.template stride<dim::ROW>();
    // we can handle this many rows each time we run a memcpy_async
    constexpr int elem_per_memcpy = sizeof(float4) / sizeof(typename ST::dtype);
    constexpr int memcpy_per_row = ST::cols / elem_per_memcpy;
    constexpr int total_calls = (ST::height * ST::width * kittens::TILE_ROW_DIM<T> * kittens::TILE_COL_DIM<T> +
                                 GROUP_THREADS * elem_per_memcpy - 1) /
                                (GROUP_THREADS * elem_per_memcpy); // round up

    coord<> unit_coord = coord<>(idx);
    typename GL::dtype *src_ptr = (typename GL::dtype *)&src[unit_coord];
    uint32_t dst_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&dst.data[0]));
    int laneid = threadIdx.x % GROUP_THREADS;

    // For a complete tile, idx() is an F2-linear swizzle within each
    // 8*swizzle_bytes region. Evaluate its lane-dependent part once, then
    // toggle the row bits for each subsequent cp.async. This path requires
    // the shared allocation to be aligned to 8*ST::swizzle_bytes.
    constexpr int chunks_per_swizzle = ST::swizzle_bytes / sizeof(float4);
    constexpr int rows_per_call = GROUP_THREADS / memcpy_per_row;
    constexpr bool simple_swizzle = ST::rows == ST::underlying_rows && ST::cols == ST::underlying_cols &&
                                    GROUP_THREADS >= memcpy_per_row && GROUP_THREADS % memcpy_per_row == 0 &&
                                    rows_per_call > 0 && (rows_per_call & (rows_per_call - 1)) == 0;
    if constexpr (assume_aligned && simple_swizzle) {
        const int base_row = laneid / memcpy_per_row;
        const int base_chunk = laneid % memcpy_per_row;
        const int outer_idx = base_chunk / chunks_per_swizzle;
        const int inner_col = (base_chunk % chunks_per_swizzle) * elem_per_memcpy;
        const uint32_t sbase = dst.idx(0u, {base_row, inner_col});

#pragma unroll
        for (int i = 0; i < total_calls; i++) {
            const int load_idx = i * GROUP_THREADS + laneid;
            const int row = load_idx / memcpy_per_row;
            const int col = (load_idx * elem_per_memcpy) % dst.cols;
            const uint32_t row_delta = i * rows_per_call * ST::swizzle_bytes;
            const uint32_t swizzled_delta = row_delta ^ (((row_delta % (8 * ST::swizzle_bytes)) >> 7) << 4);
            const uint32_t swizzled_addr =
                dst_ptr + (sbase ^ swizzled_delta) + outer_idx * ST::rows * ST::swizzle_bytes;

            asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n" ::"r"(swizzled_addr),
                         "l"(&src_ptr[row * row_stride + col])
                         : "memory");
        }
    } else {
#pragma unroll
        for (int i = 0; i < total_calls; i++) {

            int load_idx = i * GROUP_THREADS + laneid;

            int row = load_idx / memcpy_per_row;
            int col = (load_idx * elem_per_memcpy) % dst.cols;

            if constexpr (assume_aligned) {
                asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n" ::"r"(dst.idx(dst_ptr, {row, col})),
                             "l"(&src_ptr[row * row_stride + col])
                             : "memory");
            } else {
                if (row + unit_coord.r < src.template shape<dim::ROW>()) {
                    asm volatile(
                        "cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n" ::"r"(dst.idx(dst_ptr, {row, col})),
                        "l"(&src_ptr[row * row_stride + col])
                        : "memory");
                } else {
                    // printf("thread %d skipping async load on row %d, col %d\n",
                    // threadIdx.x, row + unit_coord.r, col);
                    float4 zeros = {0.f, 0.f, 0.f, 0.f};
                    move<float4>::sts(dst.idx(dst_ptr, {row, col}),
                                      zeros); // use the default value
                }
            }
        }
    }
}
template <int GW>
template <ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD>
__device__ inline void group<GW>::load_async(ST &dst, const GL &src, const COORD &idx) {
    load_async<false, ST, GL, COORD>(dst, src, idx);
}
