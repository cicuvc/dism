#pragma once

namespace kittens::ducks {

namespace rt_layout {

struct row {}; // for most matrices

struct col {}; // for the B-matrix of MMA ops.

template <typename T>
concept all = std::is_same_v<T, row> || std::is_same_v<T, col>;

template <all L>
struct transpose {
    using type = col;
};
template <>
struct transpose<col> {
    using type = row;
};

} // namespace rt_layout

namespace rv_layout {

/**
 * @brief A dummy type used to identify an aligned (8x replicated) layout.
 */
struct align {
    constexpr static int inner_dim = 2;
};
/**
 * @brief A dummy type used to identify an orthogonal (4x replicated) layout.
 */
struct ortho {
    constexpr static int inner_dim = 1;
};

struct naive {
    constexpr static int inner_dim = 1;
};

template <typename T>
concept all = std::is_same_v<T, align> || std::is_same_v<T, ortho> || std::is_same_v<T, naive>;

} // namespace rv_layout

namespace ipc {
namespace handle {

struct identifier {};

template <typename T>
concept all = requires { typename T::identifier; } && std::is_same_v<typename T::identifier, identifier>;

} // namespace handle
} // namespace ipc

namespace pgl {

struct identifier {};

template <typename T>
concept all = requires { typename T::identifier; } && std::is_same_v<typename T::identifier, identifier>;

} // namespace pgl

namespace cgl {
struct identifier {};
} // namespace cgl

namespace tt {

struct identifier {};

template <typename T>
concept all =
    requires {
        typename T::identifier; // Checks if T::identifier exists
    } && std::is_same_v<typename T::identifier,
                        identifier>; // Checks if T::identifier is
                                     // ducks::tt::identifier
template <typename T>
concept half = all<T> && T::rows == 64;
template <typename T>
concept full = all<T> && T::rows == 128;
} // namespace tt

namespace tensor_allocator {

struct identifier {};

template <typename T>
concept all =
    requires {
        typename T::identifier; // Checks if T::identifier exists
    } && std::is_same_v<typename T::identifier,
                        identifier>; // Checks if T::identifier is
                                     // ducks::tt::identifier
} // namespace tensor_allocator

namespace sv {

struct identifier {};

template <typename T>
concept all =
    requires {
        typename T::identifier; // Checks if T::identifier exists
    } && std::is_same_v<typename T::identifier,
                        identifier>; // Checks if T::identifier is
                                     // ducks::sv::identifier
} // namespace sv

namespace st {

struct identifier {};

template <typename T>
concept all =
    requires {
        typename T::identifier; // Checks if T::identifier exists
    } && std::is_same_v<typename T::identifier,
                        identifier>; // Checks if T::identifier is
                                     // ducks::st::identifier
} // namespace st

namespace csv {

struct identifier {};

template <typename T>
concept all = requires { typename T::identifier; } && std::is_same_v<typename T::identifier, identifier> &&
              ducks::sv::all<typename T::component>;

} // namespace csv

namespace cst {

struct identifier {};

template <typename T>
concept all = requires { typename T::identifier; } && std::is_same_v<typename T::identifier, identifier> &&
              ducks::st::all<typename T::component>;

} // namespace cst

namespace rv {

struct identifier {};

template <typename T>
concept all =
    requires {
        typename T::identifier; // Checks if T::identifier exists
    } && std::is_same_v<typename T::identifier,
                        identifier>; // Checks if T::identifier is
                                     // ducks::rv::identifier.

template <typename T>
concept naive_layout = all<T> && std::is_same_v<typename T::layout, ducks::rv_layout::naive>;
template <typename T>
concept align_layout = all<T> && std::is_same_v<typename T::layout, ducks::rv_layout::align>;
template <typename T>
concept ortho_layout = all<T> && std::is_same_v<typename T::layout, ducks::rv_layout::ortho>;
template <typename T>
concept tile_layout = align_layout<T> || ortho_layout<T>; // vector layouts for interacting with tiles.
} // namespace rv

namespace rt {

struct identifier {};

template <typename T>
concept all =
    requires {
        typename T::identifier; // Checks if T::identifier exists
    } && std::is_same_v<typename T::identifier,
                        identifier>; // Checks if T::identifier is
                                     // ducks::rt::identifier

template <typename T>
concept row_layout = all<T> && std::is_same_v<typename T::layout, ducks::rt_layout::row>;

template <typename T>
concept col_layout = all<T> && std::is_same_v<typename T::layout, ducks::rt_layout::col>;
} // namespace rt

/**
 * @namespace rt_base
 *
 * @brief The namespace where concepts and abstract types for register base
 * (16x16) tiles live.
 */
namespace rt_base {

struct identifier {};
} // namespace rt_base

namespace tma {
namespace descriptor {
struct identifier {};
template <typename T>
concept all = requires { typename T::identifier; } && std::is_same_v<typename T::identifier, identifier>;
} // namespace descriptor

} // namespace tma

namespace crt {

struct identifier {};

template <typename T>
concept all = requires { typename T::identifier; } && std::is_same_v<typename T::identifier, identifier> &&
              ducks::rt::all<typename T::component>;

template <typename T>
concept row_layout = all<T> && std::is_same_v<typename T::layout, ducks::rt_layout::row>;

template <typename T>
concept col_layout = all<T> && std::is_same_v<typename T::layout, ducks::rt_layout::col>;
} // namespace crt

namespace crv {

struct identifier {};

template <typename T>
concept all =
    requires {
        typename T::identifier; // Checks if T::identifier exists
    } && std::is_same_v<typename T::identifier,
                        identifier>; // Checks if T::identifier is
                                     // ducks::rv::identifier.

template <typename T>
concept naive_layout = all<T> && std::is_same_v<typename T::layout, ducks::rv_layout::naive>;
template <typename T>
concept align_layout = all<T> && std::is_same_v<typename T::layout, ducks::rv_layout::align>;
template <typename T>
concept ortho_layout = all<T> && std::is_same_v<typename T::layout, ducks::rv_layout::ortho>;
template <typename T>
concept tile_layout = align_layout<T> || ortho_layout<T>; // vector layouts for interacting with tiles.
} // namespace crv

namespace gl {
struct identifier {};
} // namespace gl

namespace st_descriptor {
struct identifier {};
// input refers to either an ST directly or to a pre-generated descriptor, which
// can save cycles in certain situations.
template <typename T>
concept input = ducks::st::all<T> || (requires { typename T::identifier; } &&
                                      std::is_same_v<typename T::identifier, ducks::st_descriptor::identifier>);
template <typename T>
concept complex_input = ducks::cst::all<T>;
namespace detail {
template <typename T>
struct st_getter {
    using type = typename T::ST;
};
template <ducks::st::all T>
struct st_getter<T> {
    using type = T;
};
template <ducks::cst::all T>
struct st_getter<T> {
    using type = T::component;
};
template <typename T>
using get_st = typename st_getter<T>::type;
} // namespace detail
} // namespace st_descriptor

namespace coord {
struct identifier {};
} // namespace coord
} // namespace kittens::ducks
