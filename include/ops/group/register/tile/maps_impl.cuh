#pragma once
#include "../../group.cuh"

template <int GW>
struct group<GW>::rt_maps {

    template <ducks::rt::all RT>
    __device__ static inline bool hasnan(const RT &src);

    template <typename op, ducks::rt::all T>
    __device__ static inline void unary_map(T &dst, const T &src);

    template <typename op, ducks::rt::all T>
    __device__ static inline void bin_map(T &dst, const T &src, const typename T::dtype &param);

    template <typename op, ducks::rt::all T>
    __device__ static inline void bin_map(T &dst, const T &src,
                                          const typename base_types::packing<typename T::dtype>::unpacked_type &param);

    template <typename op, ducks::rt::all T>
    __device__ static inline void bin_map(T &dst, const T &lhs, const T &rhs);

    template <ducks::rt::all RT, typename Lambda>
    __device__ static inline void apply(RT &dst, const RT &src, Lambda &&lambda);

    template <ducks::rt::all RT, typename Lambda>
    __device__ static inline RT apply(const RT &src, Lambda &&lambda);

    template <typename op, ducks::rt::row_layout T, ducks::rv::all V>
    __device__ static inline void row_map(T &dst, const T &src, const V &row_values);

    template <typename op, ducks::rt::col_layout T, ducks::rv::all V>
    __device__ static inline void row_map(T &dst, const T &src, const V &row_values);

    template <typename op, ducks::rt::row_layout T, ducks::rv::all V>
    __device__ static inline void row_map(T &dst, const T &a, const T &b, const V &row_values);

    template <typename op, ducks::rt::col_layout T, ducks::rv::all V>
    __device__ static inline void row_map(T &dst, const T &a, const T &b, const V &row_values);

    template <typename op, ducks::rt::row_layout T, ducks::rv::all V>
    __device__ static inline void col_map(T &dst, const T &src, const V &col_values);

    template <typename op, ducks::rt::col_layout T, ducks::rv::all V>
    __device__ static inline void col_map(T &dst, const T &src, const V &col_values);

    template <typename op, ducks::rt::row_layout T, ducks::rv::all V>
    __device__ static inline void col_map(T &dst, const T &a, const T &b, const V &col_values);

    template <typename op, ducks::rt::col_layout T, ducks::rv::all V>
    __device__ static inline void col_map(T &dst, const T &a, const T &b, const V &col_values);

    template <ducks::rt::all T>
    __device__ static inline void zero(T &dst);

    template <ducks::rt::all T>
    __device__ static inline void one(T &dst);

    template <ducks::rt::all T>
    __device__ static inline void pos_infty(T &dst);

    template <ducks::rt::all T>
    __device__ static inline void neg_infty(T &dst);

    template <ducks::rt::all T>
    __device__ static inline void exp(T &dst, const T &src);

    template <ducks::rt::all T>
    __device__ static inline T exp(const T &src);

    template <ducks::rt::all T>
    __device__ static inline void exp2(T &dst, const T &src);

    template <ducks::rt::all T>
    __device__ static inline T exp2(const T &src);

    template <ducks::rt::all T>
    __device__ static inline void log(T &dst, const T &src);

    template <ducks::rt::all T>
    __device__ static inline T log(const T &src);

    template <ducks::rt::all T>
    __device__ static inline void log2(T &dst, const T &src);

    template <ducks::rt::all T>
    __device__ static inline T log2(const T &src);

    template <ducks::rt::all T>
    __device__ static inline void abs(T &dst, const T &src);

    template <ducks::rt::all T>
    __device__ static inline T abs(const T &src);

    template <ducks::rt::all T>
    __device__ static inline void relu(T &dst, const T &src);

    template <ducks::rt::all T>
    __device__ static inline T relu(const T &src);

    template <ducks::rt::all T, typename U>
    __device__ static inline void copy(T &dst, const U &src);

    template <ducks::rt::all T, typename U>
    __device__ static inline void max(T &dst, const T &lhs, const U &rhs);

    template <ducks::rt::all T, typename U>
    __device__ static inline T max(const T &lhs, const U &rhs);

    template <ducks::rt::all T, typename U>
    __device__ static inline void min(T &dst, const T &lhs, const U &rhs);

    template <ducks::rt::all T, typename U>
    __device__ static inline T min(const T &lhs, const U &rhs);

    template <ducks::rt::all T, typename U>
    __device__ static inline void add(T &dst, const T &lhs, const U &rhs);

    template <ducks::rt::all T, typename U>
    __device__ static inline void sub(T &dst, const T &lhs, const U &rhs);

    template <ducks::rt::all T, typename U>
    __device__ static inline void mul(T &dst, const T &lhs, const U &rhs);

    template <ducks::rt::all T, typename U>
    __device__ static inline void div(T &dst, const T &lhs, const U &rhs);

    template <ducks::rt::all T, ducks::rv::all V>
    __device__ static inline void add_row(T &dst, const T &src, const V &row_values);

    template <ducks::rt::all T, ducks::rv::all V>
    __device__ static inline void sub_row(T &dst, const T &src, const V &row_values);

    template <ducks::rt::all T, ducks::rv::all V>
    __device__ static inline void mul_row(T &dst, const T &src, const V &row_values);

    template <ducks::rt::all T, ducks::rv::all V>
    __device__ static inline void div_row(T &dst, const T &src, const V &row_values);

    template <ducks::rt::all T, ducks::rv::all V>
    __device__ static inline void broadcast_row(T &dst, const V &row_values);

    template <ducks::rt::all T, ducks::rv::all V>
    __device__ static inline T broadcast_row(const V &row_values);

    template <ducks::rt::all T, ducks::rv::all V>
    __device__ static inline void add_col(T &dst, const T &src, const V &col_values);

    template <ducks::rt::all T, ducks::rv::all V>
    __device__ static inline void sub_col(T &dst, const T &src, const V &col_values);

    template <ducks::rt::all T, ducks::rv::all V>
    __device__ static inline void mul_col(T &dst, const T &src, const V &col_values);

    template <ducks::rt::all T, ducks::rv::all V>
    __device__ static inline void div_col(T &dst, const T &src, const V &col_values);

    template <ducks::rt::all T, ducks::rv::all V>
    __device__ static inline void broadcast_col(T &dst, const V &col_values);

    template <ducks::rt::all T, ducks::rv::all V>
    __device__ static inline T broadcast_col(const V &col_values);

    template <ducks::rt::all RT>
    __device__ static inline void tril(RT &dst, const RT &src, int diagonal = 0,
                                       const typename base_types::packing<typename RT::dtype>::unpacked_type &val = 0);

    template <ducks::rt::all RT>
    __device__ static inline void triu(RT &dst, const RT &src, int diagonal = 0,
                                       const typename base_types::packing<typename RT::dtype>::unpacked_type &val = 0);
};

/* ----------  Uniform tile maps (independent of layout)  ---------- */

template <int GW>
template <ducks::rt::all RT>
__device__ inline bool group<GW>::rt_maps::hasnan(const RT &src) {
    KITTENS_CHECK_WARP
    bool nan_detected = false;
#pragma unroll
    for (int i = 0; i < RT::height; i++) {
#pragma unroll
        for (int j = 0; j < RT::width; j++) {
#pragma unroll
            for (int k = 0; k < RT::packed_per_tile; k++) {
                if constexpr (std::is_same_v<typename RT::T, float>) {
                    if (isnan(src.tiles[i][j].data[k].x) || isnan(src.tiles[i][j].data[k].y)) {
                        nan_detected = true;
                    }
                } else if constexpr (std::is_same_v<typename RT::T, bf16>) {
                    if (isnan(__bfloat162float(src.tiles[i][j].data[k].x)) ||
                        isnan(__bfloat162float(src.tiles[i][j].data[k].y))) {
                        nan_detected = true;
                    }
                } else if constexpr (std::is_same_v<typename RT::T, half>) {
                    if (isnan(__half2float(src.tiles[i][j].data[k].x)) ||
                        isnan(__half2float(src.tiles[i][j].data[k].y))) {
                        nan_detected = true;
                    }
                } else {
                    static_assert(sizeof(typename RT::T) == 999, "Unsupported dtype");
                }
            }
        }
    }
    // Ballot across the warp to see if any lane detected a nan
    return (__ballot_sync(0xffffffff, nan_detected) != 0);
}

/**
 * @brief Applies a unary operation to each element of a tile.
 *
 * @tparam op Unary operation to apply.
 * @tparam T Tile type.
 * @param dst[out] Destination tile where the result is stored.
 * @param src[in] Source tile to apply the operation on.
 */
template <int GW>
template <typename op, ducks::rt::all T>
__device__ __forceinline__ void group<GW>::rt_maps::unary_map(T &dst, const T &src) {
#pragma unroll
    for (int i = 0; i < dst.height; i++) {
#pragma unroll
        for (int j = 0; j < dst.width; j++) {
#pragma unroll
            for (int k = 0; k < dst.packed_per_tile; k++) {
                dst.tiles[i][j].data[k] = op::template op<typename T::dtype>(src.tiles[i][j].data[k]);
            }
        }
    }
}

/**
 * @brief Applies a binary operation to each element of a tile with a scalar
 * parameter.
 *
 * @tparam op Binary operation to apply.
 * @tparam T Tile type.
 * @param dst[out] Destination tile where the result is stored.
 * @param src[in] Source tile to apply the operation on.
 * @param param[in] Scalar parameter for the binary operation.
 */
template <int GW>
template <typename op, ducks::rt::all T>
__device__ inline void group<GW>::rt_maps::bin_map(T &dst, const T &src, const typename T::dtype &param) {
#pragma unroll
    for (int i = 0; i < dst.height; i++) {
#pragma unroll
        for (int j = 0; j < dst.width; j++) {
#pragma unroll
            for (int k = 0; k < dst.packed_per_tile; k++) {
                dst.tiles[i][j].data[k] = op::template op<typename T::dtype>(src.tiles[i][j].data[k], param);
            }
        }
    }
}
/**
 * @brief Applies a binary operation to each element of a tile with an unpacked
 * scalar parameter.
 *
 * @tparam op Binary operation to apply.
 * @tparam T Tile type.
 * @param dst[out] Destination tile where the result is stored.
 * @param src[in] Source tile to apply the operation on.
 * @param param[in] Unpacked scalar parameter for the binary operation.
 */
template <int GW>
template <typename op, ducks::rt::all T>
__device__ inline void
group<GW>::rt_maps::bin_map(T &dst, const T &src,
                            const typename base_types::packing<typename T::dtype>::unpacked_type &param) {
    // The optimizing compiler should eliminate this pack in the 32-bit case but
    // not in the 16-bit case
    bin_map<op, T>(dst, src, base_types::packing<typename T::dtype>::pack(param));
}
/**
 * @brief Applies a binary operation element-wise between two tiles.
 *
 * @tparam op Binary operation to apply.
 * @tparam T Tile type.
 * @param dst[out] Destination tile where the result is stored.
 * @param lhs[in] Left-hand side source tile for the operation.
 * @param rhs[in] Right-hand side source tile for the operation.
 */
template <int GW>
template <typename op, ducks::rt::all T>
__device__ inline void group<GW>::rt_maps::bin_map(T &dst, const T &lhs, const T &rhs) {
#pragma unroll
    for (int i = 0; i < dst.height; i++) {
#pragma unroll
        for (int j = 0; j < dst.width; j++) {
#pragma unroll
            for (int k = 0; k < dst.packed_per_tile; k++) {
                dst.tiles[i][j].data[k] =
                    op::template op<typename T::dtype>(lhs.tiles[i][j].data[k], rhs.tiles[i][j].data[k]);
            }
        }
    }
}

template <int GW>
template <ducks::rt::all RT, typename Lambda>
__device__ inline void group<GW>::rt_maps::apply(RT &dst, const RT &src, Lambda &&lambda) {
    int row_offset = 0;
    if constexpr (GROUP_WARPS > 1) {
        row_offset = warpid() * RT::height;
    }
    static_assert(sizeof(RT::T) != 1, "Cannot apply lambda to 8-bit types");
    if constexpr (ducks::rt::row_layout<RT>) {
#pragma unroll
        for (int i = 0; i < dst.height; i++) {
#pragma unroll
            for (int j = 0; j < dst.width; j++) {
#pragma unroll
                for (int k = 0; k < dst.packed_per_tile; k++) {
                    int row = row_offset + i * TILE_ROW_DIM<typename RT::T> +
                              (k % 2) * (TILE_ROW_DIM<typename RT::T> / 2) + ::kittens::laneid() / 4;
                    int col = j * TILE_COL_DIM<typename RT::T> + (k / 2) * (TILE_COL_DIM<typename RT::T> / 2) +
                              (::kittens::laneid() % 4) * 2;
                    dst.tiles[i][j].data[k].x = lambda(row, col + 0, src.tiles[i][j].data[k].x);
                    dst.tiles[i][j].data[k].y = lambda(row, col + 1, src.tiles[i][j].data[k].y);
                }
            }
        }
    } else {
#pragma unroll
        for (int i = 0; i < dst.height; i++) {
#pragma unroll
            for (int j = 0; j < dst.width; j++) {
#pragma unroll
                for (int k = 0; k < dst.packed_per_tile; k++) {
                    int row = row_offset + i * TILE_ROW_DIM<typename RT::T> +
                              (k / 2) * (TILE_ROW_DIM<typename RT::T> / 2) + (::kittens::laneid() % 4) * 2;
                    int col = j * TILE_COL_DIM<typename RT::T> + (k % 2) * (TILE_COL_DIM<typename RT::T> / 2) +
                              ::kittens::laneid() / 4;
                    dst.tiles[i][j].data[k].x = lambda(row + 0, col, src.tiles[i][j].data[k].x);
                    dst.tiles[i][j].data[k].y = lambda(row + 1, col, src.tiles[i][j].data[k].y);
                }
            }
        }
    }
}
template <int GW>
template <ducks::rt::all RT, typename Lambda>
__device__ inline RT group<GW>::rt_maps::apply(const RT &src, Lambda &&lambda) {
    RT dst;
    apply<RT, Lambda>(dst, src, std::forward<Lambda>(lambda));
    return dst;
}

/* ----------  Row tile maps  ----------*/

/**
 * @brief Applies an operation across the rows of a tile in a row-major layout.
 *
 * @tparam op Operation to apply.
 * @tparam T Tile type with row-major layout.
 * @tparam V Column vector type.
 * @param dst[out] Destination tile where the result is stored.
 * @param src[in] Source tile to apply the operation on.
 * @param row_values[in] Column vector containing values to apply across each
 * row.
 */
template <int GW>
template <typename op, ducks::rt::row_layout T, ducks::rv::all V>
__device__ inline void group<GW>::rt_maps::row_map(T &dst, const T &src, const V &row_values) {

    static_assert(
        std::is_same_v<typename V::layout,
                       typename rt_base<typename T::T, typename T::layout>::col_vec_layout>); // compatible layout
    static_assert(std::is_same_v<typename V::dtype,
                                 typename T::dtype>); // compatible type
    static_assert(V::outer_dim == T::height);         // compatible size

    using dtype = T::dtype;

#pragma unroll
    for (int i = 0; i < dst.height; i++) {
        dtype packed_top_row = base_types::packing<dtype>::pack(row_values[i][0].x);    //  first value in eager mode
        dtype packed_bottom_row = base_types::packing<dtype>::pack(row_values[i][0].y); // second value in eager mode
#pragma unroll
        for (int j = 0; j < dst.width; j++) {
#pragma unroll
            for (int k = 0; k < dst.packed_per_tile; k += 2) {
                dst.tiles[i][j].data[k + 0] = op::template op<dtype>(src.tiles[i][j].data[k + 0], packed_top_row);
                dst.tiles[i][j].data[k + 1] = op::template op<dtype>(src.tiles[i][j].data[k + 1], packed_bottom_row);
            }
        }
    }
}
/**
 * @brief Applies an operation across the rows of a tile in a column-major
 * layout.
 *
 * @tparam op Operation to apply.
 * @tparam T Tile type with column-major layout.
 * @tparam V Column vector type.
 * @param dst[out] Destination tile where the result is stored.
 * @param src[in] Source tile to apply the operation on.
 * @param row_values[in] Column vector containing values to apply across each
 * row.
 */
template <int GW>
template <typename op, ducks::rt::col_layout T, ducks::rv::all V>
__device__ inline void group<GW>::rt_maps::row_map(T &dst, const T &src, const V &row_values) {

    static_assert(std::is_same_v<typename V::dtype,
                                 typename T::dtype>); // compatible type
    static_assert(
        std::is_same_v<typename V::layout,
                       typename rt_base<typename T::T, typename T::layout>::col_vec_layout>); // compatible layout
    static_assert(V::outer_dim == T::height);                                                 // compatible size

    using dtype = T::dtype;

#pragma unroll
    for (int i = 0; i < dst.height; i++) {
#pragma unroll
        for (int j = 0; j < dst.width; j++) {
#pragma unroll
            for (int k = 0; k < dst.packed_per_tile / 2; k++) {
                dst.tiles[i][j].data[k + 0] = op::template op<dtype>(src.tiles[i][j].data[k + 0], row_values[i][0]);
                dst.tiles[i][j].data[k + 2] = op::template op<dtype>(src.tiles[i][j].data[k + 2], row_values[i][1]);
            }
        }
    }
}

// Three-operand row map. Mostly useful for FMA instructions.

/**
 * @brief Applies an operation across the rows of two tiles in a row-major
 * layout, using a third operand.
 *
 * @tparam op Operation to apply.
 * @tparam T Tile type with row-major layout.
 * @tparam V Column vector type.
 * @param dst[out] Destination tile where the result is stored.
 * @param a[in] First source tile to apply the operation on.
 * @param b[in] Second source tile to apply the operation on.
 * @param row_values[in] Column vector containing values to apply across each
 * row.
 */
template <int GW>
template <typename op, ducks::rt::row_layout T, ducks::rv::all V>
__device__ inline void group<GW>::rt_maps::row_map(T &dst, const T &a, const T &b, const V &row_values) {

    static_assert(
        std::is_same_v<typename V::layout,
                       typename rt_base<typename T::T, typename T::layout>::col_vec_layout>); // compatible layout
    static_assert(std::is_same_v<typename V::dtype,
                                 typename T::dtype>); // compatible type
    static_assert(V::outer_dim == T::height);         // compatible size

    using dtype = T::dtype;

#pragma unroll
    for (int i = 0; i < dst.height; i++) {
        dtype packed_top_row = base_types::packing<dtype>::pack(row_values[i][0].x);    //  first value in eager mode
        dtype packed_bottom_row = base_types::packing<dtype>::pack(row_values[i][0].y); // second value in eager mode
#pragma unroll
        for (int j = 0; j < dst.width; j++) {
#pragma unroll
            for (int k = 0; k < dst.packed_per_tile; k += 2) {
                dst.tiles[i][j].data[k + 0] =
                    op::template op<dtype>(a.tiles[i][j].data[k + 0], b.tiles[i][j].data[k + 0], packed_top_row);
                dst.tiles[i][j].data[k + 1] =
                    op::template op<dtype>(a.tiles[i][j].data[k + 1], b.tiles[i][j].data[k + 1], packed_bottom_row);
            }
        }
    }
}
/**
 * @brief Applies an operation across the rows of two tiles in a column-major
 * layout, using a third operand.
 *
 * @tparam op Operation to apply.
 * @tparam T Tile type with column-major layout.
 * @tparam V Column vector type.
 * @param dst[out] Destination tile where the result is stored.
 * @param a[in] First source tile to apply the operation on.
 * @param b[in] Second source tile to apply the operation on.
 * @param row_values[in] Column vector containing values to apply across each
 * row.
 */
template <int GW>
template <typename op, ducks::rt::col_layout T, ducks::rv::all V>
__device__ inline void group<GW>::rt_maps::row_map(T &dst, const T &a, const T &b, const V &row_values) {

    static_assert(
        std::is_same_v<typename V::layout,
                       typename rt_base<typename T::T, typename T::layout>::col_vec_layout>); // compatible layout
    static_assert(std::is_same_v<typename V::dtype,
                                 typename T::dtype>); // compatible type
    static_assert(V::outer_dim == T::height);         // compatible size

    using dtype = T::dtype;

#pragma unroll
    for (int i = 0; i < dst.height; i++) {
#pragma unroll
        for (int j = 0; j < dst.width; j++) {
#pragma unroll
            for (int k = 0; k < dst.packed_per_tile / 2; k++) {
                dst.tiles[i][j].data[k + 0] =
                    op::template op<dtype>(a.tiles[i][j].data[k + 0], b.tiles[i][j].data[k + 0], row_values[i][0]);
                dst.tiles[i][j].data[k + 2] =
                    op::template op<dtype>(a.tiles[i][j].data[k + 2], b.tiles[i][j].data[k + 2], row_values[i][1]);
            }
        }
    }
}

/* ----------  Col major tile maps  ----------*/

/**
 * @brief Applies an operation across the columns of a tile in a row-major
 * layout.
 *
 * @tparam op Operation to apply.
 * @tparam T Tile type with row-major layout.
 * @tparam V Row vector type.
 * @param dst[out] Destination tile where the result is stored.
 * @param src[in] Source tile to apply the operation on.
 * @param col_values[in] Row vector containing values to apply across each
 * column.
 */
template <int GW>
template <typename op, ducks::rt::row_layout T, ducks::rv::all V>
__device__ inline void group<GW>::rt_maps::col_map(T &dst, const T &src, const V &col_values) {
    KITTENS_CHECK_WARP

    static_assert(
        std::is_same_v<typename V::layout,
                       typename rt_base<typename T::T, typename T::layout>::row_vec_layout>); // compatible layout
    static_assert(std::is_same_v<typename V::dtype,
                                 typename T::dtype>); // compatible type
    static_assert(V::outer_dim == T::width);          // compatible size

    using dtype = T::dtype;

#pragma unroll
    for (int j = 0; j < dst.width; j++) {
#pragma unroll
        for (int i = 0; i < dst.height; i++) {
#pragma unroll
            for (int k = 0; k < dst.packed_per_tile / 2; k++) {
                dst.tiles[i][j].data[k + 0] = op::template op<dtype>(src.tiles[i][j].data[k + 0], col_values[j][0]);
                dst.tiles[i][j].data[k + 2] = op::template op<dtype>(src.tiles[i][j].data[k + 2], col_values[j][1]);
            }
        }
    }
}
/**
 * @brief Applies an operation across the columns of a tile in a column-major
 * layout.
 *
 * @tparam op Operation to apply.
 * @tparam T Tile type with column-major layout.
 * @tparam V Row vector type.
 * @param dst[out] Destination tile where the result is stored.
 * @param src[in] Source tile to apply the operation on.
 * @param col_values[in] Row vector containing values to apply across each
 * column.
 */
template <int GW>
template <typename op, ducks::rt::col_layout T, ducks::rv::all V>
__device__ inline void group<GW>::rt_maps::col_map(T &dst, const T &src, const V &col_values) {
    KITTENS_CHECK_WARP

    static_assert(
        std::is_same_v<typename V::layout,
                       typename rt_base<typename T::T, typename T::layout>::row_vec_layout>); // compatible layout
    static_assert(std::is_same_v<typename V::dtype,
                                 typename T::dtype>); // compatible type
    static_assert(V::outer_dim == T::width);          // compatible size

    using dtype = T::dtype;

#pragma unroll
    for (int j = 0; j < dst.width; j++) {
        dtype packed_left_col = base_types::packing<dtype>::pack(col_values[j][0].x);  //  first value in eager mode
        dtype packed_right_col = base_types::packing<dtype>::pack(col_values[j][0].y); // second value in eager mode
#pragma unroll
        for (int i = 0; i < dst.height; i++) {
#pragma unroll
            for (int k = 0; k < dst.packed_per_tile; k += 2) {
                dst.tiles[i][j].data[k + 0] = op::template op<dtype>(src.tiles[i][j].data[k + 0], packed_left_col);
                dst.tiles[i][j].data[k + 1] = op::template op<dtype>(src.tiles[i][j].data[k + 1], packed_right_col);
            }
        }
    }
}

// Three-operand col map
/**
 * @brief Applies an operation across the columns of two tiles in a row-major
 * layout, using a third operand.
 *
 * @tparam op Operation to apply.
 * @tparam T Tile type with row-major layout.
 * @tparam V Row vector type.
 * @param dst[out] Destination tile where the result is stored.
 * @param a[in] First source tile to apply the operation on.
 * @param b[in] Second source tile to apply the operation on.
 * @param col_values[in] Row vector containing values to apply across each
 * column.
 */
template <int GW>
template <typename op, ducks::rt::row_layout T, ducks::rv::all V>
__device__ inline void group<GW>::rt_maps::col_map(T &dst, const T &a, const T &b, const V &col_values) {
    KITTENS_CHECK_WARP

    static_assert(
        std::is_same_v<typename V::layout,
                       typename rt_base<typename T::T, typename T::layout>::row_vec_layout>); // compatible layout
    static_assert(std::is_same_v<typename V::dtype,
                                 typename T::dtype>); // compatible type
    static_assert(V::outer_dim == T::width);          // compatible size

    using dtype = T::dtype;

#pragma unroll
    for (int j = 0; j < dst.width; j++) {
#pragma unroll
        for (int i = 0; i < dst.height; i++) {
#pragma unroll
            for (int k = 0; k < dst.packed_per_tile / 2; k++) {
                dst.tiles[i][j].data[k + 0] =
                    op::template op<dtype>(a.tiles[i][j].data[k + 0], b.tiles[i][j].data[k + 0], col_values[j][0]);
                dst.tiles[i][j].data[k + 2] =
                    op::template op<dtype>(a.tiles[i][j].data[k + 2], b.tiles[i][j].data[k + 2], col_values[j][1]);
            }
        }
    }
}
/**
 * @brief Applies an operation across the columns of two tiles in a column-major
 * layout, using a third operand.
 *
 * @tparam op Operation to apply.
 * @tparam T Tile type with column-major layout.
 * @tparam V Row vector type.
 * @param dst[out] Destination tile where the result is stored.
 * @param a[in] First source tile to apply the operation on.
 * @param b[in] Second source tile to apply the operation on.
 * @param col_values[in] Row vector containing values to apply across each
 * column.
 */
template <int GW>
template <typename op, ducks::rt::col_layout T, ducks::rv::all V>
__device__ inline void group<GW>::rt_maps::col_map(T &dst, const T &a, const T &b, const V &col_values) {
    KITTENS_CHECK_WARP

    static_assert(std::is_same_v<typename V::dtype,
                                 typename T::dtype>); // compatible type
    static_assert(
        std::is_same_v<typename V::layout,
                       typename rt_base<typename T::T, typename T::layout>::row_vec_layout>); // compatible layout
    static_assert(V::outer_dim == T::width);                                                  // compatible size

    using dtype = T::dtype;
#pragma unroll
    for (int j = 0; j < dst.width; j++) {
        dtype packed_left_col = base_types::packing<dtype>::pack(col_values[j][0].x);  //  first value in eager mode
        dtype packed_right_col = base_types::packing<dtype>::pack(col_values[j][0].y); // second value in eager mode
#pragma unroll
        for (int i = 0; i < dst.height; i++) {
#pragma unroll
            for (int k = 0; k < dst.packed_per_tile; k += 2) {
                dst.tiles[i][j].data[k + 0] =
                    op::template op<dtype>(a.tiles[i][j].data[k + 0], b.tiles[i][j].data[k + 0], packed_left_col);
                dst.tiles[i][j].data[k + 1] =
                    op::template op<dtype>(a.tiles[i][j].data[k + 1], b.tiles[i][j].data[k + 1], packed_right_col);
            }
        }
    }
}

/* ----------  WRAPPERS FOR PRETTINESS  ---------- */

// All of the annoying qualifiers *should* be automatically inferred during
// compile-time. So, syntax should just be kittens::add_row(tile, colvec);

/**
 * @brief Sets all elements of a tile to zero.
 *
 * @tparam T Tile type.
 * @param dst[out] Destination tile where the result is stored.
 */
template <int GW>
template <ducks::rt::all T>
__device__ inline void group<GW>::rt_maps::zero(T &dst) {
    unary_map<base_ops::zero, T>(dst, dst);
}
/**
 * @brief Sets all elements of a tile to one.
 *
 * @tparam T Tile type.
 * @param dst[out] Destination tile where the result is stored.
 */
template <int GW>
template <ducks::rt::all T>
__device__ inline void group<GW>::rt_maps::one(T &dst) {
    unary_map<base_ops::one, T>(dst, dst);
}
/**
 * @brief Sets all elements of a tile to positive infinity.
 *
 * @tparam T Tile type.
 * @param dst[out] Destination tile where the result is stored.
 */
template <int GW>
template <ducks::rt::all T>
__device__ inline void group<GW>::rt_maps::pos_infty(T &dst) {
    unary_map<base_ops::pos_infty, T>(dst, dst);
}
/**
 * @brief Sets all elements of a tile to negative infinity.
 *
 * @tparam T Tile type.
 * @param dst[out] Destination tile where the result is stored.
 */
template <int GW>
template <ducks::rt::all T>
__device__ inline void group<GW>::rt_maps::neg_infty(T &dst) {
    unary_map<base_ops::neg_infty, T>(dst, dst);
}

/**
 * @brief Applies the exponential function to each element of a tile.
 *
 * @tparam T Tile type.
 * @param dst[out] Destination tile where the result is stored.
 * @param src[in] Source tile to apply the exponential function on.
 */
template <int GW>
template <ducks::rt::all T>
__device__ inline void group<GW>::rt_maps::exp(T &dst, const T &src) {
    unary_map<base_ops::exp, T>(dst, src);
}
template <int GW>
template <ducks::rt::all T>
__device__ inline T group<GW>::rt_maps::exp(const T &src) {
    T dst;
    exp(dst, src);
    return dst;
}

/**
 * @brief Applies the exponential function to each element of a tile, in base 2.
 *
 * @tparam T Tile type.
 * @param dst[out] Destination tile where the result is stored.
 * @param src[in] Source tile to apply the exponential function on.
 */
template <int GW>
template <ducks::rt::all T>
__device__ inline void group<GW>::rt_maps::exp2(T &dst, const T &src) {
    unary_map<base_ops::exp2, T>(dst, src);
}
template <int GW>
template <ducks::rt::all T>
__device__ inline T group<GW>::rt_maps::exp2(const T &src) {
    T dst;
    exp2(dst, src);
    return dst;
}

/**
 * @brief Applies the natural logarithm function to each element of a tile.
 *
 * @tparam T Tile type.
 * @param dst[out] Destination tile where the result is stored.
 * @param src[in] Source tile to apply the natural logarithm function on.
 */
template <int GW>
template <ducks::rt::all T>
__device__ inline void group<GW>::rt_maps::log(T &dst, const T &src) {
    unary_map<base_ops::log, T>(dst, src);
}
template <int GW>
template <ducks::rt::all T>
__device__ inline T group<GW>::rt_maps::log(const T &src) {
    T dst;
    log(dst, src);
    return dst;
}

/**
 * @brief Applies the logarithm base 2 function to each element of a tile.
 *
 * @tparam T Tile type.
 * @param dst[out] Destination tile where the result is stored.
 * @param src[in] Source tile to apply the logarithm base 2 function on.
 */
template <int GW>
template <ducks::rt::all T>
__device__ inline void group<GW>::rt_maps::log2(T &dst, const T &src) {
    unary_map<base_ops::log2, T>(dst, src);
}
template <int GW>
template <ducks::rt::all T>
__device__ inline T group<GW>::rt_maps::log2(const T &src) {
    T dst;
    log2(dst, src);
    return dst;
}

/**
 * @brief Applies the absolute value function to each element of a tile.
 *
 * @tparam T Tile type.
 * @param dst[out] Destination tile where the result is stored.
 * @param src[in] Source tile to apply the absolute value function on.
 */
template <int GW>
template <ducks::rt::all T>
__device__ inline void group<GW>::rt_maps::abs(T &dst, const T &src) {
    unary_map<base_ops::abs, T>(dst, src);
}
template <int GW>
template <ducks::rt::all T>
__device__ inline T group<GW>::rt_maps::abs(const T &src) {
    T dst;
    abs(dst, src);
    return dst;
}

/**
 * @brief Applies the rectified linear unit (ReLU) function to each element of a
 * tile.
 *
 * @tparam T Tile type.
 * @param dst[out] Destination tile where the result is stored.
 * @param src[in] Source tile to apply the ReLU function on.
 */
template <int GW>
template <ducks::rt::all T>
__device__ inline void group<GW>::rt_maps::relu(T &dst, const T &src) {
    unary_map<base_ops::relu, T>(dst, src);
}
template <int GW>
template <ducks::rt::all T>
__device__ inline T group<GW>::rt_maps::relu(const T &src) {
    T dst;
    relu(dst, src);
    return dst;
}

/**
 * @brief Copies the elements from one tile to another.
 *
 * @tparam T Destination tile type.
 * @tparam U Source tile type.
 * @param dst[out] Destination tile where the result is stored.
 * @param src[in] Source tile to copy from.
 */
template <int GW>
template <ducks::rt::all T, typename U>
__device__ inline void group<GW>::rt_maps::copy(T &dst, const U &src) {
    bin_map<base_ops::copy2, T>(dst, src);
}

/**
 * @brief Applies the max operation element-wise between two tiles or a tile and
 * a scalar.
 *
 * @tparam T Tile type.
 * @tparam U Second operand type, which can be a tile or a scalar.
 * @param dst[out] Destination tile where the result is stored.
 * @param lhs[in] Left-hand side source tile for the operation.
 * @param rhs[in] Right-hand side source tile or scalar for the operation.
 */
template <int GW>
template <ducks::rt::all T, typename U>
__device__ inline void group<GW>::rt_maps::max(T &dst, const T &lhs, const U &rhs) {
    bin_map<base_ops::max, T>(dst, lhs, rhs);
}
template <int GW>
template <ducks::rt::all T, typename U>
__device__ inline T group<GW>::rt_maps::max(const T &lhs, const U &rhs) {
    T dst;
    max(dst, lhs, rhs);
    return dst;
}

/**
 * @brief Applies the min operation element-wise between two tiles or a tile and
 * a scalar.
 *
 * @tparam T Tile type.
 * @tparam U Second operand type, which can be a tile or a scalar.
 * @param dst[out] Destination tile where the result is stored.
 * @param lhs[in] Left-hand side source tile for the operation.
 * @param rhs[in] Right-hand side source tile or scalar for the operation.
 */
template <int GW>
template <ducks::rt::all T, typename U>
__device__ inline void group<GW>::rt_maps::min(T &dst, const T &lhs, const U &rhs) {
    bin_map<base_ops::min, T>(dst, lhs, rhs);
}
template <int GW>
template <ducks::rt::all T, typename U>
__device__ inline T group<GW>::rt_maps::min(const T &lhs, const U &rhs) {
    T dst;
    min(dst, lhs, rhs);
    return dst;
}

/**
 * @brief Adds two tiles element-wise or adds a scalar to each element of a
 * tile.
 *
 * @tparam T Tile type.
 * @tparam U Second operand type, which can be a tile or a scalar.
 * @param dst[out] Destination tile where the result is stored.
 * @param lhs[in] Left-hand side source tile for the addition.
 * @param rhs[in] Right-hand side source tile or scalar for the addition.
 */
template <int GW>
template <ducks::rt::all T, typename U>
__device__ inline void group<GW>::rt_maps::add(T &dst, const T &lhs, const U &rhs) {
    bin_map<base_ops::sum, T>(dst, lhs, rhs);
}

/**
 * @brief Subtracts two tiles element-wise or subtracts a scalar from each
 * element of a tile.
 *
 * @tparam T Tile type.
 * @tparam U Second operand type, which can be a tile or a scalar.
 * @param dst[out] Destination tile where the result is stored.
 * @param lhs[in] Left-hand side source tile for the subtraction.
 * @param rhs[in] Right-hand side source tile or scalar for the subtraction.
 */
template <int GW>
template <ducks::rt::all T, typename U>
__device__ inline void group<GW>::rt_maps::sub(T &dst, const T &lhs, const U &rhs) {
    bin_map<base_ops::sub, T>(dst, lhs, rhs);
}
/**
 * @brief Multiplies two tiles element-wise or multiplies each element of a tile
 * by a scalar.
 *
 * @tparam T Tile type.
 * @tparam U Second operand type, which can be a tile or a scalar.
 * @param dst[out] Destination tile where the result is stored.
 * @param lhs[in] Left-hand side source tile for the multiplication.
 * @param rhs[in] Right-hand side source tile or scalar for the multiplication.
 */
template <int GW>
template <ducks::rt::all T, typename U>
__device__ inline void group<GW>::rt_maps::mul(T &dst, const T &lhs, const U &rhs) {
    bin_map<base_ops::mul, T>(dst, lhs, rhs);
}

/**
 * @brief Divides two tiles element-wise or divides each element of a tile by a
 * scalar.
 *
 * @tparam T Tile type.
 * @tparam U Second operand type, which can be a tile or a scalar.
 * @param dst[out] Destination tile where the result is stored.
 * @param lhs[in] Left-hand side source tile for the division.
 * @param rhs[in] Right-hand side source tile or scalar for the division.
 */
template <int GW>
template <ducks::rt::all T, typename U>
__device__ inline void group<GW>::rt_maps::div(T &dst, const T &lhs, const U &rhs) {
    bin_map<base_ops::div, T>(dst, lhs, rhs);
}

/**
 * @brief Adds row values to each row of a tile.
 *
 * @tparam T Tile type.
 * @tparam V Column vector type.
 * @param dst[out] Destination tile where the result is stored.
 * @param src[in] Source tile to apply the addition on.
 * @param row_values[in] Column vector containing values to add to each row.
 */
template <int GW>
template <ducks::rt::all T, ducks::rv::all V>
__device__ inline void group<GW>::rt_maps::add_row(T &dst, const T &src, const V &row_values) {
    row_map<base_ops::sum, T, V>(dst, src, row_values);
}

/**
 * @brief Subtracts row values from each row of a tile.
 *
 * @tparam T Tile type.
 * @tparam V Column vector type.
 * @param dst[out] Destination tile where the result is stored.
 * @param src[in] Source tile to apply the subtraction on.
 * @param row_values[in] Column vector containing values to subtract from each
 * row.
 */
template <int GW>
template <ducks::rt::all T, ducks::rv::all V>
__device__ inline void group<GW>::rt_maps::sub_row(T &dst, const T &src, const V &row_values) {
    row_map<base_ops::sub, T, V>(dst, src, row_values);
}

/**
 * @brief Multiplies each row of a tile by row values.
 *
 * @tparam T Tile type.
 * @tparam V Column vector type.
 * @param dst[out] Destination tile where the result is stored.
 * @param src[in] Source tile to apply the multiplication on.
 * @param row_values[in] Column vector containing values to multiply each row
 * by.
 */
template <int GW>
template <ducks::rt::all T, ducks::rv::all V>
__device__ inline void group<GW>::rt_maps::mul_row(T &dst, const T &src, const V &row_values) {
    row_map<base_ops::mul, T, V>(dst, src, row_values);
}

/**
 * @brief Divides each row of a tile by row values.
 *
 * @tparam T Tile type.
 * @tparam V Column vector type.
 * @param dst[out] Destination tile where the result is stored.
 * @param src[in] Source tile to apply the division on.
 * @param row_values[in] Column vector containing values to divide each row by.
 */
template <int GW>
template <ducks::rt::all T, ducks::rv::all V>
__device__ inline void group<GW>::rt_maps::div_row(T &dst, const T &src, const V &row_values) {
    row_map<base_ops::div, T, V>(dst, src, row_values);
}

/**
 * @brief Broadcast a vector into into a tile's rows.
 *
 * @tparam T Tile type.
 * @tparam V Column vector type.
 * @param dst[out] Destination tile where the result is stored.
 * @param row_values[in] Column vector containing values to broadcast into rows.
 */
template <int GW>
template <ducks::rt::all T, ducks::rv::all V>
__device__ inline void group<GW>::rt_maps::broadcast_row(T &dst, const V &row_values) {
    row_map<base_ops::copy2, T, V>(dst, dst, row_values);
}
template <int GW>
template <ducks::rt::all T, ducks::rv::all V>
__device__ inline T group<GW>::rt_maps::broadcast_row(const V &row_values) {
    T dst;
    broadcast_row(dst, row_values);
    return dst;
}

// col maps
/**
 * @brief Adds column values to each column of a tile.
 *
 * @tparam T Tile type.
 * @tparam V Row vector type.
 * @param dst[out] Destination tile where the result is stored.
 * @param src[in] Source tile to apply the addition on.
 * @param col_values[in] Row vector containing values to add to each column.
 */
template <int GW>
template <ducks::rt::all T, ducks::rv::all V>
__device__ inline void group<GW>::rt_maps::add_col(T &dst, const T &src, const V &col_values) {
    col_map<base_ops::sum, T, V>(dst, src, col_values);
}

/**
 * @brief Subtracts column values from each column of a tile.
 *
 * @tparam T Tile type.
 * @tparam V Row vector type.
 * @param dst[out] Destination tile where the result is stored.
 * @param src[in] Source tile to apply the subtraction on.
 * @param col_values[in] Row vector containing values to subtract from each
 * column.
 */
template <int GW>
template <ducks::rt::all T, ducks::rv::all V>
__device__ inline void group<GW>::rt_maps::sub_col(T &dst, const T &src, const V &col_values) {
    col_map<base_ops::sub, T, V>(dst, src, col_values);
}

/**
 * @brief Multiplies each column of a tile by column values.
 *
 * @tparam T Tile type.
 * @tparam V Row vector type.
 * @param dst[out] Destination tile where the result is stored.
 * @param src[in] Source tile to apply the multiplication on.
 * @param col_values[in] Row vector containing values to multiply each column
 * by.
 */
template <int GW>
template <ducks::rt::all T, ducks::rv::all V>
__device__ inline void group<GW>::rt_maps::mul_col(T &dst, const T &src, const V &col_values) {
    col_map<base_ops::mul, T, V>(dst, src, col_values);
}

/**
 * @brief Divides each column of a tile by column values.
 *
 * @tparam T Tile type.
 * @tparam V Row vector type.
 * @param dst[out] Destination tile where the result is stored.
 * @param src[in] Source tile to apply the division on.
 * @param col_values[in] Row vector containing values to divide each column by.
 */
template <int GW>
template <ducks::rt::all T, ducks::rv::all V>
__device__ inline void group<GW>::rt_maps::div_col(T &dst, const T &src, const V &col_values) {
    col_map<base_ops::div, T, V>(dst, src, col_values);
}

/**
 * @brief Broadcast a vector into into a tile's columns.
 *
 * @tparam T Tile type.
 * @tparam V Row vector type.
 * @param dst[out] Destination tile where the result is stored.
 * @param row_values[in] Row vector containing values to broadcast into cols.
 */
template <int GW>
template <ducks::rt::all T, ducks::rv::all V>
__device__ inline void group<GW>::rt_maps::broadcast_col(T &dst, const V &col_values) {
    col_map<base_ops::copy2, T, V>(dst, dst, col_values);
}
template <int GW>
template <ducks::rt::all T, ducks::rv::all V>
__device__ inline T group<GW>::rt_maps::broadcast_col(const V &col_values) {
    T dst;
    broadcast_col(dst, col_values);
    return dst;
}

// Triangular masks
template <int GW>
template <ducks::rt::all RT>
__device__ inline void
group<GW>::rt_maps::tril(RT &dst, const RT &src, int diagonal,
                         const typename base_types::packing<typename RT::dtype>::unpacked_type &val) {
    apply(dst, src, [val, diagonal] __device__(int row, int col, auto &src_val) {
        return col <= row + diagonal ? src_val : val;
    });
}
template <int GW>
template <ducks::rt::all RT>
__device__ inline void
group<GW>::rt_maps::triu(RT &dst, const RT &src, int diagonal,
                         const typename base_types::packing<typename RT::dtype>::unpacked_type &val) {
    apply(dst, src, [val, diagonal] __device__(int row, int col, auto &src_val) {
        return col >= row + diagonal ? src_val : val;
    });
}