#pragma once

#include "../../common/base_concepts.cuh"
#include "../register/register.cuh"
#include <cstddef>
#include <type_traits>

namespace kittens {

/** Element strides for the three non-innermost dimensions of a gl. */
struct gl_strides {
    size_t batch;
    size_t depth;
    size_t row;
};

namespace tma {
/**
 * Host-side description passed to tensor-map encoders. Shapes and strides use
 * logical B/D/R/C order; strides are measured in elements and COL is always 1.
 */
template <typename T> struct global_tensor_view {
    T *data;
    uint64_t shape[4];
    uint64_t stride[4];
};
} // namespace tma

namespace ducks {
namespace gl {

template <int d>
concept cdim = (d > 0); // represents a compile-time dimension
template <int d>
concept rdim = (d == -1); // represents a runtime dimension

struct dummy_arg {
    __host__ __device__ inline dummy_arg(size_t){};
};
template <int _v> struct compiled_dim {
    static_assert(cdim<_v>, "Invalid compile-time dimension value");
    static constexpr size_t v = _v;
    __host__ __device__ inline compiled_dim(const std::nullptr_t &_) {}
    __host__ __device__ inline compiled_dim(const dummy_arg &_) {}
    __host__ __device__ inline constexpr operator size_t() const { return v; }
};
struct runtime_dim {
    size_t v;
    __host__ __device__ inline runtime_dim(const size_t &_v) : v(_v) {}
    __host__ __device__ inline operator size_t() const { return v; }
};
template <int d> using make_dim_t = std::conditional_t<rdim<d>, runtime_dim, compiled_dim<d>>;
template <int d>
using make_arg_t = std::conditional_t<rdim<d>, size_t,
                                      dummy_arg>; // we pass runtime dims as size_t, comptime
                                                  // dims as nullptr_t
} // namespace gl
} // namespace ducks

namespace detail {
template <typename T>
concept tile = ducks::st::all<T> || ducks::rt::all<T> || ducks::cst::all<T> || ducks::crt::all<T>;
template <typename T>
concept vec = ducks::sv::all<T> || ducks::rv::all<T> || ducks::csv::all<T> || ducks::crv::all<T>;
} // namespace detail

template <typename _T = ducks::default_type> struct coord { // essentially a named int4 for tensor coordinates.
    using identifier = ducks::coord::identifier;
    using BASE = _T; // category tag only; all fields are logical element offsets.
    static_assert(std::is_same_v<BASE, ducks::default_type> || detail::tile<BASE> ||
                  detail::vec<BASE>); // ensure BASE is a valid type
    int b, d, r, c;
    __device__ inline coord(int _b, int _d, int _r, int _c) : b(_b), d(_d), r(_r), c(_c) {}
    __device__ inline coord(int _d, int _r, int _c) : b(0), d(_d), r(_r), c(_c) {}
    __device__ inline coord(int _r, int _c) : b(0), d(0), r(_r), c(_c) {}
    __device__ inline coord(int _c) : b(0), d(0), r(0), c(_c) {}
    __device__ inline coord() : b(0), d(0), r(0), c(0) {}
    template <typename U>
    __device__ inline coord(const coord<U> &other) : b(other.b), d(other.d), r(other.r), c(other.c) {}
    __device__ inline coord(const int4 &other) : b(other.x), d(other.y), r(other.z), c(other.w) {}
    __device__ inline operator int4() const { return int4(b, d, r, c); }
    template <int axis> __device__ inline int dim() const {
        static_assert(axis >= 0 && axis <= 3, "axis must be between 0 and 3");
        if constexpr (axis == 0) {
            return b;
        } else if constexpr (axis == 1) {
            return d;
        } else if constexpr (axis == 2) {
            return r;
        } else {
            return c;
        }
    }
};
namespace ducks {
namespace coord {
/**
 * @brief Concept for all coordinate types.
 * @tparam T The type to check against the concept requirements.
 *
 * Requires:
 * - T has a nested type identifier that is the same as
 * ducks::coord::identifier.
 */
template <typename T>
concept all =
    requires {
        typename T::identifier; // Checks if T::identifier exists
    } && std::is_same_v<typename T::identifier,
                        identifier>; // Checks if T::identifier is
                                     // ducks::coord::identifier
template <typename T>
concept tile = all<T> && (std::is_same_v<typename T::BASE, ducks::default_type> || detail::tile<typename T::BASE>);
template <typename T>
concept vec = all<T> && (std::is_same_v<typename T::BASE, ducks::default_type> || detail::vec<typename T::BASE>);
} // namespace coord
} // namespace ducks
} // namespace kittens
