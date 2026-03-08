/**
 * @file
 * @brief A collection of all of the operations that ThunderKittens defines.
 */

#pragma once

#include "device/device.cuh"
#include "group/group.cuh"
#include "thread/thread.cuh"

namespace kittens {

// Operator overloading, which defaults to warp scope.

// Tile operators

template <ducks::rt::all T, typename U>
__device__ static inline T operator+(const T &lhs, const U &rhs) {
    T dst;
    warp::rt_maps::add(dst, lhs, rhs);
    return dst;
}
template <ducks::rt::all T, typename U>
__device__ static inline void operator+=(T &lhs, const U &rhs) {
    warp::rt_maps::add(lhs, lhs, rhs);
}
template <ducks::rt::all T, typename U>
__device__ static inline T operator-(const T &lhs, const U &rhs) {
    T dst;
    warp::rt_maps::sub(dst, lhs, rhs);
    return dst;
}
template <ducks::rt::all T, typename U>
__device__ static inline void operator-=(T &lhs, const U &rhs) {
    warp::rt_maps::sub(lhs, lhs, rhs);
}
template <ducks::rt::all T, typename U>
__device__ static inline T operator*(const T &lhs, const U &rhs) {
    T dst;
    warp::rt_maps::mul(dst, lhs, rhs);
    return dst;
}
template <ducks::rt::all T, typename U>
__device__ static inline void operator*=(T &lhs, const U &rhs) {
    warp::rt_maps::mul(lhs, lhs, rhs);
}
template <ducks::rt::all T, typename U>
__device__ static inline T operator/(const T &lhs, const U &rhs) {
    T dst;
    warp::rt_maps::div(dst, lhs, rhs);
    return dst;
}
template <ducks::rt::all T, typename U>
__device__ static inline void operator/=(T &lhs, const U &rhs) {
    warp::rt_maps::div(lhs, lhs, rhs);
}
template <ducks::rt::row_layout T, ducks::rv::ortho_layout V>
__device__ static inline T operator+(const T &src, const V &row_values) {
    T dst;
    warp::rt_maps::add_row(dst, src, row_values);
    return dst;
}
template <ducks::rt::col_layout T, ducks::rv::align_layout V>
__device__ static inline T operator+(const T &src, const V &row_values) {
    T dst;
    warp::rt_maps::add_row(dst, src, row_values);
    return dst;
}
template <ducks::rt::row_layout T, ducks::rv::ortho_layout V>
__device__ static inline void operator+=(T &lhs, const V &row_values) {
    warp::rt_maps::add_row(lhs, lhs, row_values);
}
template <ducks::rt::col_layout T, ducks::rv::align_layout V>
__device__ static inline void operator+=(T &lhs, const V &row_values) {
    warp::rt_maps::add_row(lhs, lhs, row_values);
}
template <ducks::rt::row_layout T, ducks::rv::ortho_layout V>
__device__ static inline T operator-(const T &src, const V &row_values) {
    T dst;
    warp::rt_maps::sub_row(dst, src, row_values);
    return dst;
}
template <ducks::rt::col_layout T, ducks::rv::align_layout V>
__device__ static inline T operator-(const T &src, const V &row_values) {
    T dst;
    warp::rt_maps::sub_row(dst, src, row_values);
    return dst;
}
template <ducks::rt::row_layout T, ducks::rv::ortho_layout V>
__device__ static inline void operator-=(T &lhs, const V &row_values) {
    warp::rt_maps::sub_row(lhs, lhs, row_values);
}
template <ducks::rt::col_layout T, ducks::rv::align_layout V>
__device__ static inline void operator-=(T &lhs, const V &row_values) {
    warp::rt_maps::sub_row(lhs, lhs, row_values);
}
template <ducks::rt::row_layout T, ducks::rv::ortho_layout V>
__device__ static inline T operator*(const T &src, const V &row_values) {
    T dst;
    warp::rt_maps::mul_row(dst, src, row_values);
    return dst;
}
template <ducks::rt::col_layout T, ducks::rv::align_layout V>
__device__ static inline T operator*(const T &src, const V &row_values) {
    T dst;
    warp::rt_maps::mul_row(dst, src, row_values);
    return dst;
}
template <ducks::rt::row_layout T, ducks::rv::ortho_layout V>
__device__ static inline void operator*=(T &lhs, const V &row_values) {
    warp::rt_maps::mul_row(lhs, lhs, row_values);
}
template <ducks::rt::col_layout T, ducks::rv::align_layout V>
__device__ static inline void operator*=(T &lhs, const V &row_values) {
    warp::rt_maps::mul_row(lhs, lhs, row_values);
}
template <ducks::rt::row_layout T, ducks::rv::ortho_layout V>
__device__ static inline T operator/(const T &src, const V &row_values) {
    T dst;
    warp::rt_maps::div_row(dst, src, row_values);
    return dst;
}
template <ducks::rt::col_layout T, ducks::rv::align_layout V>
__device__ static inline T operator/(const T &src, const V &row_values) {
    T dst;
    warp::rt_maps::div_row(dst, src, row_values);
    return dst;
}
template <ducks::rt::row_layout T, ducks::rv::ortho_layout V>
__device__ static inline void operator/=(T &lhs, const V &row_values) {
    warp::rt_maps::div_row(lhs, lhs, row_values);
}
template <ducks::rt::col_layout T, ducks::rv::align_layout V>
__device__ static inline void operator/=(T &lhs, const V &row_values) {
    warp::rt_maps::div_row(lhs, lhs, row_values);
}
template <ducks::rt::row_layout T, ducks::rv::align_layout V>
__device__ static inline T operator+(const T &src, const V &col_values) {
    T dst;
    warp::rt_maps::add_col(dst, src, col_values);
    return dst;
}
template <ducks::rt::col_layout T, ducks::rv::ortho_layout V>
__device__ static inline T operator+(const T &src, const V &col_values) {
    T dst;
    warp::rt_maps::add_col(dst, src, col_values);
    return dst;
}
template <ducks::rt::row_layout T, ducks::rv::align_layout V>
__device__ static inline void operator+=(T &lhs, const V &col_values) {
    warp::rt_maps::add_col(lhs, lhs, col_values);
}
template <ducks::rt::col_layout T, ducks::rv::ortho_layout V>
__device__ static inline void operator+=(T &lhs, const V &col_values) {
    warp::rt_maps::add_col(lhs, lhs, col_values);
}
template <ducks::rt::row_layout T, ducks::rv::align_layout V>
__device__ static inline T operator-(const T &src, const V &col_values) {
    T dst;
    warp::rt_maps::sub_col(dst, src, col_values);
    return dst;
}
template <ducks::rt::col_layout T, ducks::rv::ortho_layout V>
__device__ static inline T operator-(const T &src, const V &col_values) {
    T dst;
    warp::rt_maps::sub_col(dst, src, col_values);
    return dst;
}
template <ducks::rt::row_layout T, ducks::rv::align_layout V>
__device__ static inline void operator-=(T &lhs, const V &col_values) {
    warp::rt_maps::sub_col(lhs, lhs, col_values);
}
template <ducks::rt::col_layout T, ducks::rv::ortho_layout V>
__device__ static inline void operator-=(T &lhs, const V &col_values) {
    warp::rt_maps::sub_col(lhs, lhs, col_values);
}
template <ducks::rt::row_layout T, ducks::rv::align_layout V>
__device__ static inline T operator*(const T &src, const V &col_values) {
    T dst;
    warp::rt_maps::mul_col(dst, src, col_values);
    return dst;
}
template <ducks::rt::col_layout T, ducks::rv::ortho_layout V>
__device__ static inline T operator*(const T &src, const V &col_values) {
    T dst;
    warp::rt_maps::mul_col(dst, src, col_values);
    return dst;
}
template <ducks::rt::row_layout T, ducks::rv::align_layout V>
__device__ static inline void operator*=(T &lhs, const V &col_values) {
    warp::rt_maps::mul_col(lhs, lhs, col_values);
}
template <ducks::rt::col_layout T, ducks::rv::ortho_layout V>
__device__ static inline void operator*=(T &lhs, const V &col_values) {
    warp::rt_maps::mul_col(lhs, lhs, col_values);
}
template <ducks::rt::row_layout T, ducks::rv::align_layout V>
__device__ static inline T operator/(const T &src, const V &col_values) {
    T dst;
    warp::rt_maps::div_col(dst, src, col_values);
    return dst;
}
template <ducks::rt::col_layout T, ducks::rv::ortho_layout V>
__device__ static inline T operator/(const T &src, const V &col_values) {
    T dst;
    warp::rt_maps::div_col(dst, src, col_values);
    return dst;
}
template <ducks::rt::row_layout T, ducks::rv::align_layout V>
__device__ static inline void operator/=(T &lhs, const V &col_values) {
    warp::rt_maps::div_col(lhs, lhs, col_values);
}
template <ducks::rt::col_layout T, ducks::rv::ortho_layout V>
__device__ static inline void operator/=(T &lhs, const V &col_values) {
    warp::rt_maps::div_col(lhs, lhs, col_values);
}

// Vector operators

template <ducks::rv::all T, typename U>
__device__ static inline T operator+(const T &lhs, const U &rhs) {
    T dst;
    warp::rt_maps::add(dst, lhs, rhs);
    return dst;
}
template <ducks::rv::all T, typename U>
__device__ static inline void operator+=(T &lhs, const U &rhs) {
    warp::rt_maps::add(lhs, lhs, rhs);
}
template <ducks::rv::all T, typename U>
__device__ static inline T operator-(const T &lhs, const U &rhs) {
    T dst;
    warp::rt_maps::sub(dst, lhs, rhs);
    return dst;
}
template <ducks::rv::all T, typename U>
__device__ static inline void operator-=(T &lhs, const U &rhs) {
    warp::rt_maps::sub(lhs, lhs, rhs);
}
template <ducks::rv::all T, typename U>
__device__ static inline T operator*(const T &lhs, const U &rhs) {
    T dst;
    warp::rt_maps::mul(dst, lhs, rhs);
    return dst;
}
template <ducks::rv::all T, typename U>
__device__ static inline void operator*=(T &lhs, const U &rhs) {
    warp::rt_maps::mul(lhs, lhs, rhs);
}
template <ducks::rv::all T, typename U>
__device__ static inline T operator/(const T &lhs, const U &rhs) {
    T dst;
    warp::rt_maps::div(dst, lhs, rhs);
    return dst;
}
template <ducks::rv::all T, typename U>
__device__ static inline void operator/=(T &lhs, const U &rhs) {
    warp::rt_maps::div(lhs, lhs, rhs);
}

} // namespace kittens
