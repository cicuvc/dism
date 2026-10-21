#pragma once

#include "../../group.cuh"

template <int GW>
struct group<GW>::sv_maps {
    template <ducks::sv::all SV>
    __device__ static inline bool hasnan(const SV &src);
    template <typename op, ducks::sv::all T>
    __device__ static inline void unary_op(T &dst, const T &src);

    template <typename op, ducks::sv::all T>
    __device__ static inline void bin_op(T &dst, const T &lhs, const T &rhs);

    template <typename op, ducks::sv::all T>
    __device__ static inline void bin_op(T &dst, const T &src, const typename T::dtype &param);

    template <ducks::sv::all T>
    __device__ static inline void zero(T &dst);

    template <ducks::sv::all T>
    __device__ static inline void one(T &dst);

    template <ducks::sv::all T>
    __device__ static inline void pos_infty(T &dst);

    template <ducks::sv::all T>
    __device__ static inline void neg_infty(T &dst);

    template <ducks::sv::all T, typename U>
    __device__ static inline void copy(T &dst, const U &src);

    template <ducks::sv::all T>
    __device__ static inline void exp(T &dst, const T &src);

    template <ducks::sv::all T>
    __device__ static inline void exp2(T &dst, const T &src);

    template <ducks::sv::all T>
    __device__ static inline void log(T &dst, const T &src);

    template <ducks::sv::all T>
    __device__ static inline void log2(T &dst, const T &src);

    template <ducks::sv::all T>
    __device__ static inline void abs(T &dst, const T &src);

    template <ducks::sv::all T>
    __device__ static inline void relu(T &dst, const T &src);

    template <ducks::sv::all T, typename U>
    __device__ static inline void max(T &dst, const T &lhs, const U &rhs);

    template <ducks::sv::all T, typename U>
    __device__ static inline void min(T &dst, const T &lhs, const U &rhs);

    template <ducks::sv::all T, typename U>
    __device__ static inline void add(T &dst, const T &lhs, const U &rhs);

    template <ducks::sv::all T, typename U>
    __device__ static inline void sub(T &dst, const T &lhs, const U &rhs);

    template <ducks::sv::all T, typename U>
    __device__ static inline void mul(T &dst, const T &lhs, const U &rhs);

    template <ducks::sv::all T, typename U>
    __device__ static inline void div(T &dst, const T &lhs, const U &rhs);
};

template <int GW>
template <ducks::sv::all SV>
__device__ inline bool group<GW>::sv_maps::hasnan(const SV &src) {
    KITTENS_CHECK_WARP
    bool nan_detected = false;
#pragma unroll
    for (int i = laneid(); i < SV::length; i += GROUP_THREADS) {
        if constexpr (std::is_same_v<typename SV::T, float>) {
            if (isnan(src[i])) {
                nan_detected = true;
            }
        } else if constexpr (std::is_same_v<typename SV::T, bf16>) {
            if (isnan(__bfloat162float(src[i]))) {
                nan_detected = true;
            }
        } else if constexpr (std::is_same_v<typename SV::T, half>) {
            if (isnan(__half2float(src[i]))) {
                nan_detected = true;
            }
        } else {
            static_assert(sizeof(typename SV::T) == 999, "Unsupported dtype");
        }
    }
    // Ballot across the warp to see if any lane detected a nan
    return (__ballot_sync(0xffffffff, nan_detected) != 0);
}

/**
 * @brief Applies a unary operation to each element of a shared memory vector.
 *
 * @tparam op Unary operation type.
 * @tparam T Shared memory vector type.
 * @param dst[out] Destination vector in which to store the result.
 * @param src[in] Source vector to apply the unary operation.
 */
template <int GW>
template <typename op, ducks::sv::all T>
__device__ inline void group<GW>::sv_maps::unary_op(T &dst, const T &src) {
    uint32_t tid = laneid();
#pragma unroll
    for (auto cur = 0; cur + tid < T::length; cur += GROUP_THREADS) {
        dst[cur + tid] = op::template op<typename T::dtype>(src[cur + tid]);
    }
}
/**
 * @brief Perform a binary operation on two shared vectors.
 *
 * @tparam op The binary operation to perform.
 * @tparam T The type of the vectors.
 * @param dst[out] The destination vector where the result is stored.
 * @param lhs[in] The left-hand side vector for the operation.
 * @param rhs[in] The right-hand side vector for the operation.
 */
template <int GW>
template <typename op, ducks::sv::all T>
__device__ inline void group<GW>::sv_maps::bin_op(T &dst, const T &lhs, const T &rhs) {
#pragma unroll
    for (auto cur = laneid(); cur < T::length; cur += GROUP_THREADS) {
        dst[cur] = op::template op<typename T::dtype>(lhs[cur], rhs[cur]);
    }
}
/**
 * @brief Perform a binary operation on a shared vector and a scalar.
 *
 * @tparam op The binary operation to perform.
 * @tparam T The type of the vector.
 * @param dst[out] The destination vector where the result is stored.
 * @param src[in] The source vector for the operation.
 * @param param[in] The scalar parameter for the operation.
 */
template <int GW>
template <typename op, ducks::sv::all T>
__device__ inline void group<GW>::sv_maps::bin_op(T &dst, const T &src, const typename T::dtype &param) {
#pragma unroll
    for (auto cur = laneid(); cur < T::length; cur += GROUP_THREADS) {
        dst[cur] = op::template op<typename T::dtype>(src[cur], param);
    }
}

/* ----------  WRAPPERS FOR PRETTINESS  ---------- */

// ---- const ops ----

/**
 * @brief Sets all elements of a shared memory vector to zero.
 *
 * @tparam T Shared memory vector type.
 * @param dst[out] Destination vector to be set to zero.
 */
template <int GW>
template <ducks::sv::all T>
__device__ inline void group<GW>::sv_maps::zero(T &dst) {
    unary_op<base_ops::zero, T>(dst, dst);
}
/**
 * @brief Sets all elements of a shared memory vector to one.
 *
 * @tparam T Shared memory vector type.
 * @param dst[out] Destination vector to be set to one.
 */
template <int GW>
template <ducks::sv::all T>
__device__ inline void group<GW>::sv_maps::one(T &dst) {
    unary_op<base_ops::one, T>(dst, dst);
}
/**
 * @brief Sets all elements of a shared memory vector to positive infinity.
 *
 * @tparam T Shared memory vector type.
 * @param dst[out] Destination vector to be set to positive infinity.
 */
template <int GW>
template <ducks::sv::all T>
__device__ inline void group<GW>::sv_maps::pos_infty(T &dst) {
    unary_op<base_ops::pos_infty, T>(dst, dst);
}
/**
 * @brief Sets all elements of a shared memory vector to negative infinity.
 *
 * @tparam T Shared memory vector type.
 * @param dst[out] Destination vector to be set to negative infinity.
 */
template <int GW>
template <ducks::sv::all T>
__device__ inline void group<GW>::sv_maps::neg_infty(T &dst) {
    unary_op<base_ops::neg_infty, T>(dst, dst);
}

// ---- unary ops ----

/**
 * @brief Copies the elements from one shared vector to another.
 *
 * @tparam T Shared vector type.
 * @tparam U Type of the source vector.
 * @param dst[out] Destination vector where the elements will be copied to.
 * @param src[in] Source vector to copy the elements from.
 */
template <int GW>
template <ducks::sv::all T, typename U>
__device__ inline void group<GW>::sv_maps::copy(T &dst, const U &src) {
    bin_op<base_ops::copy2, T>(dst, dst,
                               src); // the second arg is ignored here.
}
/**
 * @brief Applies the exponential function element-wise to a shared vector.
 *
 * @tparam T Shared vector type.
 * @param dst[out] Destination vector where the exponential values will be
 * stored.
 * @param src[in] Source vector to apply the exponential function to.
 */
template <int GW>
template <ducks::sv::all T>
__device__ inline void group<GW>::sv_maps::exp(T &dst, const T &src) {
    unary_op<base_ops::exp, T>(dst, src);
}
/**
 * @brief Applies the exponential function element-wise to a shared vector, in
 * base 2.
 *
 * @tparam T Shared vector type.
 * @param dst[out] Destination vector where the exponential values will be
 * stored.
 * @param src[in] Source vector to apply the exponential function to.
 */
template <int GW>
template <ducks::sv::all T>
__device__ inline void group<GW>::sv_maps::exp2(T &dst, const T &src) {
    unary_op<base_ops::exp2, T>(dst, src);
}
/**
 * @brief Applies the natural logarithm function element-wise to a shared
 * vector.
 *
 * @tparam T Shared vector type.
 * @param dst[out] Destination vector where the logarithm values will be stored.
 * @param src[in] Source vector to apply the logarithm function to.
 */
template <int GW>
template <ducks::sv::all T>
__device__ inline void group<GW>::sv_maps::log(T &dst, const T &src) {
    unary_op<base_ops::log, T>(dst, src);
}
/**
 * @brief Applies the logarithm base 2 function element-wise to a shared vector.
 *
 * @tparam T Shared vector type.
 * @param dst[out] Destination vector where the logarithm base 2 values will be
 * stored.
 * @param src[in] Source vector to apply the logarithm base 2 function to.
 */
template <int GW>
template <ducks::sv::all T>
__device__ inline void group<GW>::sv_maps::log2(T &dst, const T &src) {
    unary_op<base_ops::log2, T>(dst, src);
}
/**
 * @brief Applies the absolute value function element-wise to a shared vector.
 *
 * @tparam T Shared vector type.
 * @param dst[out] Destination vector where the absolute values will be stored.
 * @param src[in] Source vector to apply the absolute value function to.
 */
template <int GW>
template <ducks::sv::all T>
__device__ inline void group<GW>::sv_maps::abs(T &dst, const T &src) {
    unary_op<base_ops::abs, T>(dst, src);
}
/**
 * @brief Applies the rectified linear unit (ReLU) function element-wise to a
 * shared vector.
 *
 * @tparam T Shared vector type.
 * @param dst[out] Destination vector where the ReLU values will be stored.
 * @param src[in] Source vector to apply the ReLU function to.
 */
template <int GW>
template <ducks::sv::all T>
__device__ inline void group<GW>::sv_maps::relu(T &dst, const T &src) {
    unary_op<base_ops::relu, T>(dst, src);
}

// ---- binary ops ----

/**
 * @brief Computes the element-wise maximum of two shared vectors.
 *
 * @tparam T Shared vector type.
 * @tparam U Type of the second vector.
 * @param dst[out] Destination vector where the maximum values will be stored.
 * @param lhs[in] First vector for the maximum operation.
 * @param rhs[in] Second vector for the maximum operation.
 */
template <int GW>
template <ducks::sv::all T, typename U>
__device__ inline void group<GW>::sv_maps::max(T &dst, const T &lhs, const U &rhs) {
    bin_op<base_ops::max, T>(dst, lhs, rhs);
}
/**
 * @brief Computes the element-wise minimum of two shared vectors.
 *
 * @tparam T Shared vector type.
 * @tparam U Type of the second vector.
 * @param dst[out] Destination vector where the minimum values will be stored.
 * @param lhs[in] First vector for the minimum operation.
 * @param rhs[in] Second vector for the minimum operation.
 */
template <int GW>
template <ducks::sv::all T, typename U>
__device__ inline void group<GW>::sv_maps::min(T &dst, const T &lhs, const U &rhs) {
    bin_op<base_ops::min, T>(dst, lhs, rhs);
}
/**
 * @brief Computes the element-wise sum of two shared vectors.
 *
 * @tparam T Shared vector type.
 * @tparam U Type of the second vector.
 * @param dst[out] Destination vector where the sum values will be stored.
 * @param lhs[in] First vector for the sum operation.
 * @param rhs[in] Second vector for the sum operation.
 */
template <int GW>
template <ducks::sv::all T, typename U>
__device__ inline void group<GW>::sv_maps::add(T &dst, const T &lhs, const U &rhs) {
    bin_op<base_ops::sum, T>(dst, lhs, rhs);
}
/**
 * @brief Computes the element-wise difference of two shared vectors.
 *
 * @tparam T Shared vector type.
 * @tparam U Type of the second vector.
 * @param dst[out] Destination vector where the difference values will be
 * stored.
 * @param lhs[in] First vector for the difference operation.
 * @param rhs[in] Second vector for the difference operation.
 */
template <int GW>
template <ducks::sv::all T, typename U>
__device__ inline void group<GW>::sv_maps::sub(T &dst, const T &lhs, const U &rhs) {
    bin_op<base_ops::sub, T>(dst, lhs, rhs);
}
/**
 * @brief Computes the element-wise product of two shared vectors.
 *
 * @tparam T Shared vector type.
 * @tparam U Type of the second vector.
 * @param dst[out] Destination vector where the product values will be stored.
 * @param lhs[in] First vector for the product operation.
 * @param rhs[in] Second vector for the product operation.
 */
template <int GW>
template <ducks::sv::all T, typename U>
__device__ inline void group<GW>::sv_maps::mul(T &dst, const T &lhs, const U &rhs) {
    bin_op<base_ops::mul, T>(dst, lhs, rhs);
}
/**
 * @brief Computes the element-wise division of two shared vectors.
 *
 * @tparam T Shared vector type.
 * @tparam U Type of the second vector.
 * @param dst[out] Destination vector where the division values will be stored.
 * @param lhs[in] First vector for the division operation.
 * @param rhs[in] Second vector for the division operation.
 */
template <int GW>
template <ducks::sv::all T, typename U>
__device__ inline void group<GW>::sv_maps::div(T &dst, const T &lhs, const U &rhs) {
    bin_op<base_ops::div, T>(dst, lhs, rhs);
}