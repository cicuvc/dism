#pragma once
#include "../../group.cuh"

template <int GW>
struct group<GW>::st_maps {
    template <ducks::st::all ST>
    __device__ static inline bool hasnan(const ST &src);
    template <typename op, ducks::st::all T> // T2, w, h can be inferred from
                                             // dst as long as op is specialized
    __device__ static inline void unary_map(T &dst, const T &src);

    template <typename op, ducks::st::all T>
    __device__ static inline void bin_map(T &dst, const T &src, const typename T::dtype &param);

    template <typename op, ducks::st::all T>
    __device__ static inline void bin_map(T &dst, const T &lhs, const T &rhs);

    template <typename op, ducks::st::all T, ducks::sv::all V>
    __device__ static inline void row_map(T &dst, const T &src, const V &vec);

    template <typename op, ducks::st::all T, ducks::sv::all V>
    __device__ static inline void col_map(T &dst, const T &src, const V &vec);

    template <ducks::st::all T>
    __device__ static inline void zero(T &dst);

    template <ducks::st::all T>
    __device__ static inline void one(T &dst);

    template <ducks::st::all T>
    __device__ static inline void pos_infty(T &dst);

    template <ducks::st::all T>
    __device__ static inline void neg_infty(T &dst);

    template <ducks::st::all T>
    __device__ static inline void exp(T &dst, const T &src);

    template <ducks::st::all T>
    __device__ static inline void exp2(T &dst, const T &src);

    template <ducks::st::all T>
    __device__ static inline void log(T &dst, const T &src);

    template <ducks::st::all T>
    __device__ static inline void log2(T &dst, const T &src);

    template <ducks::st::all T>
    __device__ static inline void abs(T &dst, const T &src);

    template <ducks::st::all T>
    __device__ static inline void relu(T &dst, const T &src);

    template <ducks::st::all T, typename U>
    __device__ static inline void copy(T &dst, const U &src);

    template <ducks::st::all T, typename U>
    __device__ static inline void max(T &dst, const T &lhs, const U &rhs);

    template <ducks::st::all T, typename U>
    __device__ static inline void min(T &dst, const T &lhs, const U &rhs);

    template <ducks::st::all T, typename U>
    __device__ static inline void add(T &dst, const T &lhs, const U &rhs);

    template <ducks::st::all T, typename U>
    __device__ static inline void sub(T &dst, const T &lhs, const U &rhs);

    template <ducks::st::all T, typename U>
    __device__ static inline void mul(T &dst, const T &lhs, const U &rhs);

    template <ducks::st::all T, typename U>
    __device__ static inline void div(T &dst, const T &lhs, const U &rhs);

    template <ducks::st::all T, ducks::sv::all V>
    __device__ static inline void add_row(T &dst, const T &src, const V &row_values);

    template <ducks::st::all T, ducks::sv::all V>
    __device__ static inline void sub_row(T &dst, const T &src, const V &row_values);

    template <ducks::st::all T, ducks::sv::all V>
    __device__ static inline void mul_row(T &dst, const T &src, const V &row_values);

    template <ducks::st::all T, ducks::sv::all V>
    __device__ static inline void div_row(T &dst, const T &src, const V &row_values);

    template <ducks::st::all T, ducks::sv::all V>
    __device__ static inline void broadcast_row(T &dst, const V &row_values);

    template <ducks::st::all T, ducks::sv::all V>
    __device__ static inline void add_col(T &dst, const T &src, const V &col_values);

    template <ducks::st::all T, ducks::sv::all V>
    __device__ static inline void sub_col(T &dst, const T &src, const V &col_values);

    template <ducks::st::all T, ducks::sv::all V>
    __device__ static inline void mul_col(T &dst, const T &src, const V &col_values);

    template <ducks::st::all T, ducks::sv::all V>
    __device__ static inline void div_col(T &dst, const T &src, const V &col_values);

    template <ducks::st::all T, ducks::sv::all V>
    __device__ static inline void broadcast_col(T &dst, const V &col_values);

    template <int axis, ducks::st::all T, ducks::sv::all V>
    __device__ static inline void add(T &dst, const T &src, const V &col_values);

    template <int axis, ducks::st::all T, ducks::sv::all V>
    __device__ static inline void sub(T &dst, const T &src, const V &col_values);

    template <int axis, ducks::st::all T, ducks::sv::all V>
    __device__ static inline void mul(T &dst, const T &src, const V &col_values);

    template <int axis, ducks::st::all T, ducks::sv::all V>
    __device__ static inline void div(T &dst, const T &src, const V &col_values);

    template <int axis, ducks::st::all T, ducks::sv::all V>
    __device__ static inline void broadcast(T &dst, const V &col_values);
};

template <int GW>
template <ducks::st::all ST>
__device__ inline bool group<GW>::st_maps::hasnan(const ST &src) {
    KITTENS_CHECK_WARP
    bool nan_detected = false;
#pragma unroll
    for (int i = laneid(); i < ST::num_elements; i += GROUP_THREADS) {
        if constexpr (std::is_same_v<typename ST::T, float>) {
            if (isnan(src[i])) {
                nan_detected = true;
            }
        } else if constexpr (std::is_same_v<typename ST::T, bf16>) {
            if (isnan(__bfloat162float(src[i]))) {
                nan_detected = true;
            }
        } else if constexpr (std::is_same_v<typename ST::T, half>) {
            if (isnan(__half2float(src[i]))) {
                nan_detected = true;
            }
        } else {
            static_assert(sizeof(typename ST::T) == 999, "Unsupported dtype");
        }
    }
    // Ballot across the warp to see if any lane detected a nan
    return (__ballot_sync(0xffffffff, nan_detected) != 0);
}

template <int GW>
template <typename op, ducks::st::all T> // T2, w, h can be inferred from dst as
                                         // long as op is specialized
__device__ inline void group<GW>::st_maps::unary_map(T &dst, const T &src) {
#pragma unroll
    for (int i = laneid(); i < dst.num_elements; i += GROUP_THREADS) {
        dst.data[i] = op::template op<typename T::dtype>(src.data[i]);
    }
}

template <int GW>
template <typename op, ducks::st::all T>
__device__ inline void group<GW>::st_maps::bin_map(T &dst, const T &src, const typename T::dtype &param) {
#pragma unroll
    for (int i = laneid(); i < dst.num_elements; i += GROUP_THREADS) {
        dst.data[i] = op::template op<typename T::dtype>(src.data[i], param);
    }
}

template <int GW>
template <typename op, ducks::st::all T>
__device__ inline void group<GW>::st_maps::bin_map(T &dst, const T &lhs, const T &rhs) {
#pragma unroll
    for (int i = laneid(); i < dst.num_elements; i += GROUP_THREADS) {
        dst.data[i] = op::template op<typename T::dtype>(lhs.data[i], rhs.data[i]);
    }
}

template <int GW>
template <typename op, ducks::st::all T, ducks::sv::all V>
__device__ inline void group<GW>::st_maps::row_map(T &dst, const T &src, const V &vec) {
    static_assert(std::is_same<typename T::dtype, typename V::dtype>::value,
                  "Tile and vector must have the same data type");
    static_assert(V::length == T::rows, "Vector length must match the number of rows in the tile");
#pragma unroll
    for (int i = laneid(); i < dst.num_elements; i += GROUP_THREADS) {
        int row = i / dst.cols, col = i % dst.cols;
        dst[{row, col}] = op::template op<typename T::dtype>(src[{row, col}], vec[row]);
    }
}

template <int GW>
template <typename op, ducks::st::all T, ducks::sv::all V>
__device__ inline void group<GW>::st_maps::col_map(T &dst, const T &src, const V &vec) {
    static_assert(std::is_same<typename T::dtype, typename V::dtype>::value,
                  "Tile and vector must have the same data type");
    static_assert(V::length == T::cols, "Vector length must match the number of columns in the tile");
#pragma unroll
    for (int i = laneid(); i < dst.num_elements; i += GROUP_THREADS) {
        int row = i / dst.cols, col = i % dst.cols;
        dst[{row, col}] = op::template op<typename T::dtype>(src[{row, col}], vec[col]);
    }
}

/* ----------  WRAPPERS FOR PRETTINESS  ---------- */

// All of the annoying qualifiers *should* be automatically inferred during
// compile-time. So, syntax should just be kittens::add_row(tile, colvec);

// const maps

template <int GW>
template <ducks::st::all T>
__device__ inline void group<GW>::st_maps::zero(T &dst) {
    unary_map<base_ops::zero, T>(dst, dst);
}

template <int GW>
template <ducks::st::all T>
__device__ inline void group<GW>::st_maps::one(T &dst) {
    unary_map<base_ops::one, T>(dst, dst);
}

template <int GW>
template <ducks::st::all T>
__device__ inline void group<GW>::st_maps::pos_infty(T &dst) {
    unary_map<base_ops::pos_infty, T>(dst, dst);
}

template <int GW>
template <ducks::st::all T>
__device__ inline void group<GW>::st_maps::neg_infty(T &dst) {
    unary_map<base_ops::neg_infty, T>(dst, dst);
}

// unary maps

template <int GW>
template <ducks::st::all T>
__device__ inline void group<GW>::st_maps::exp(T &dst, const T &src) {
    unary_map<base_ops::exp, T>(dst, src);
}

template <int GW>
template <ducks::st::all T>
__device__ inline void group<GW>::st_maps::exp2(T &dst, const T &src) {
    unary_map<base_ops::exp2, T>(dst, src);
}

template <int GW>
template <ducks::st::all T>
__device__ inline void group<GW>::st_maps::log(T &dst, const T &src) {
    unary_map<base_ops::log, T>(dst, src);
}

template <int GW>
template <ducks::st::all T>
__device__ inline void group<GW>::st_maps::log2(T &dst, const T &src) {
    unary_map<base_ops::log2, T>(dst, src);
}

template <int GW>
template <ducks::st::all T>
__device__ inline void group<GW>::st_maps::abs(T &dst, const T &src) {
    unary_map<base_ops::abs, T>(dst, src);
}

template <int GW>
template <ducks::st::all T>
__device__ inline void group<GW>::st_maps::relu(T &dst, const T &src) {
    unary_map<base_ops::relu, T>(dst, src);
}

template <int GW>
template <ducks::st::all T, typename U>
__device__ inline void group<GW>::st_maps::copy(T &dst, const U &src) {
    bin_map<base_ops::copy, T>(dst, src);
}

// uniform binary maps

template <int GW>
template <ducks::st::all T, typename U>
__device__ inline void group<GW>::st_maps::max(T &dst, const T &lhs, const U &rhs) {
    bin_map<base_ops::max, T>(dst, lhs, rhs);
}

template <int GW>
template <ducks::st::all T, typename U>
__device__ inline void group<GW>::st_maps::min(T &dst, const T &lhs, const U &rhs) {
    bin_map<base_ops::min, T>(dst, lhs, rhs);
}

template <int GW>
template <ducks::st::all T, typename U>
__device__ inline void group<GW>::st_maps::add(T &dst, const T &lhs, const U &rhs) {
    bin_map<base_ops::sum, T>(dst, lhs, rhs);
}

template <int GW>
template <ducks::st::all T, typename U>
__device__ inline void group<GW>::st_maps::sub(T &dst, const T &lhs, const U &rhs) {
    bin_map<base_ops::sub, T>(dst, lhs, rhs);
}

template <int GW>
template <ducks::st::all T, typename U>
__device__ inline void group<GW>::st_maps::mul(T &dst, const T &lhs, const U &rhs) {
    bin_map<base_ops::mul, T>(dst, lhs, rhs);
}

template <int GW>
template <ducks::st::all T, typename U>
__device__ inline void group<GW>::st_maps::div(T &dst, const T &lhs, const U &rhs) {
    bin_map<base_ops::div, T>(dst, lhs, rhs);
}

// Row and col maps

template <int GW>
template <ducks::st::all T, ducks::sv::all V>
__device__ inline void group<GW>::st_maps::add_row(T &dst, const T &src, const V &row_values) {
    row_map<base_ops::sum, T, V>(dst, src, row_values);
}

template <int GW>
template <ducks::st::all T, ducks::sv::all V>
__device__ inline void group<GW>::st_maps::sub_row(T &dst, const T &src, const V &row_values) {
    row_map<base_ops::sub, T, V>(dst, src, row_values);
}

template <int GW>
template <ducks::st::all T, ducks::sv::all V>
__device__ inline void group<GW>::st_maps::mul_row(T &dst, const T &src, const V &row_values) {
    row_map<base_ops::mul, T, V>(dst, src, row_values);
}

template <int GW>
template <ducks::st::all T, ducks::sv::all V>
__device__ inline void group<GW>::st_maps::div_row(T &dst, const T &src, const V &row_values) {
    row_map<base_ops::div, T, V>(dst, src, row_values);
}

template <int GW>
template <ducks::st::all T, ducks::sv::all V>
__device__ inline void group<GW>::st_maps::broadcast_row(T &dst, const V &row_values) {
    row_map<base_ops::copy2, T, V>(dst, dst, row_values);
}

// col maps

template <int GW>
template <ducks::st::all T, ducks::sv::all V>
__device__ inline void group<GW>::st_maps::add_col(T &dst, const T &src, const V &col_values) {
    col_map<base_ops::sum, T, V>(dst, src, col_values);
}

template <int GW>
template <ducks::st::all T, ducks::sv::all V>
__device__ inline void group<GW>::st_maps::sub_col(T &dst, const T &src, const V &col_values) {
    col_map<base_ops::sub, T, V>(dst, src, col_values);
}

template <int GW>
template <ducks::st::all T, ducks::sv::all V>
__device__ inline void group<GW>::st_maps::mul_col(T &dst, const T &src, const V &col_values) {
    col_map<base_ops::mul, T, V>(dst, src, col_values);
}

template <int GW>
template <ducks::st::all T, ducks::sv::all V>
__device__ inline void group<GW>::st_maps::div_col(T &dst, const T &src, const V &col_values) {
    col_map<base_ops::div, T, V>(dst, src, col_values);
}

template <int GW>
template <ducks::st::all T, ducks::sv::all V>
__device__ inline void group<GW>::st_maps::broadcast_col(T &dst, const V &col_values) {
    col_map<base_ops::copy2, T, V>(dst, dst, col_values);
}

// Templated versions of each

template <int GW>
template <int axis, ducks::st::all T, ducks::sv::all V>
__device__ inline void group<GW>::st_maps::add(T &dst, const T &src, const V &col_values) {
    if constexpr (axis == axis::COL)
        add_col(dst, src, col_values);
    else
        add_row(dst, src, col_values);
}

template <int GW>
template <int axis, ducks::st::all T, ducks::sv::all V>
__device__ inline void group<GW>::st_maps::sub(T &dst, const T &src, const V &col_values) {
    if constexpr (axis == axis::COL)
        sub_col(dst, src, col_values);
    else
        sub_row(dst, src, col_values);
}

template <int GW>
template <int axis, ducks::st::all T, ducks::sv::all V>
__device__ inline void group<GW>::st_maps::mul(T &dst, const T &src, const V &col_values) {
    if constexpr (axis == axis::COL)
        mul_col(dst, src, col_values);
    else
        mul_row(dst, src, col_values);
}

template <int GW>
template <int axis, ducks::st::all T, ducks::sv::all V>
__device__ inline void group<GW>::st_maps::div(T &dst, const T &src, const V &col_values) {
    if constexpr (axis == axis::COL)
        div_col(dst, src, col_values);
    else
        div_row(dst, src, col_values);
}

template <int GW>
template <int axis, ducks::st::all T, ducks::sv::all V>
__device__ inline void group<GW>::st_maps::broadcast(T &dst, const V &col_values) {
    if constexpr (axis == axis::COL)
        broadcast_col(dst, col_values);
    else
        broadcast_row(dst, col_values);
}