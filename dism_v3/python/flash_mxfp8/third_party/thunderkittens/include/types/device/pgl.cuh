/**
 * @file
 * @brief Templated layouts for parallel global memory.
 */

#pragma once

#include "../../common/common.cuh"
#include "../global/global.cuh"
#include "../shared/shared.cuh"

namespace kittens {

/* ----------  Parallel global layout descriptor  ---------- */

/**
 * @brief Parallel global layout. Represents a region of data spread across
 multiple devices.
 * @tparam GL The underlying global layout on each device.
 * @tparam NUM_DEVICES The number of GPU devices.
 * @tparam MULTICAST Whether the multicast object should be initialized by the
 caller.
 * @tparam TMA_Types The types of TMA descriptors to use for the multicast
 locations. Only valid if MULTICAST is true.
 */
template <kittens::ducks::gl::all _GL, int NUM_DEVICES = 8, bool MULTICAST = true, typename... TMA_Types> struct pgl {
    using identifier = ducks::pgl::identifier;
    using GL = _GL;
    using T = GL::dtype;
    using dtype = T;

    static constexpr int num_devices = NUM_DEVICES;
    static constexpr bool multicast = MULTICAST;

    T *mc_ptr; // multicast pointer; nullptr if MULTICAST is false
    GL gls[NUM_DEVICES];

    detail::descriptor_dict<TMA_Types...> tma_descs;

    __host__ __device__ const GL &operator[](int idx) const { return gls[idx]; }
    __device__ inline T *mc_ptr_at(const coord<ducks::default_type> &idx) const {
        static_assert(MULTICAST, "Multicast is not enabled for this PGL.");
        const GL &gl = gls[0]; // all gls have the same shape
        return &mc_ptr[size_t(idx.b) * gl.template stride<dim::BATCH>() +
                       size_t(idx.d) * gl.template stride<dim::DEPTH>() +
                       size_t(idx.r) * gl.template stride<dim::ROW>() + size_t(idx.c)];
    }

    __host__ inline pgl(T **_data, // an array of NUM_DEVICES pointers to the
                                   // data on each device
                        ducks::gl::make_arg_t<GL::__b__> _batch, ducks::gl::make_arg_t<GL::__d__> _depth,
                        ducks::gl::make_arg_t<GL::__r__> _rows, ducks::gl::make_arg_t<GL::__c__> _cols)
        : pgl(std::make_index_sequence<NUM_DEVICES>{}, _data, _batch, _depth, _rows, _cols) {}

    __host__ inline pgl(T **_data, ducks::gl::make_arg_t<GL::__b__> _batch, ducks::gl::make_arg_t<GL::__d__> _depth,
                        ducks::gl::make_arg_t<GL::__r__> _rows, ducks::gl::make_arg_t<GL::__c__> _cols,
                        gl_strides _strides)
        : pgl(std::make_index_sequence<NUM_DEVICES>{}, _data, _batch, _depth, _rows, _cols, _strides) {}

    __host__ inline pgl(T *_mc_ptr, // multicast pointer, initialized by the caller
                        T **_data,  // an array of NUM_DEVICES pointers to the data on each
                                    // device
                        ducks::gl::make_arg_t<GL::__b__> _batch, ducks::gl::make_arg_t<GL::__d__> _depth,
                        ducks::gl::make_arg_t<GL::__r__> _rows, ducks::gl::make_arg_t<GL::__c__> _cols)
        : pgl(std::make_index_sequence<NUM_DEVICES>{}, _mc_ptr, _data, _batch, _depth, _rows, _cols) {}

    __host__ inline pgl(T *_mc_ptr, T **_data, ducks::gl::make_arg_t<GL::__b__> _batch,
                        ducks::gl::make_arg_t<GL::__d__> _depth, ducks::gl::make_arg_t<GL::__r__> _rows,
                        ducks::gl::make_arg_t<GL::__c__> _cols, gl_strides _strides)
        : pgl(std::make_index_sequence<NUM_DEVICES>{}, _mc_ptr, _data, _batch, _depth, _rows, _cols, _strides) {}

    template <size_t... I>
    __host__ inline pgl(std::index_sequence<I...>, T **_data, ducks::gl::make_arg_t<GL::__b__> _batch,
                        ducks::gl::make_arg_t<GL::__d__> _depth, ducks::gl::make_arg_t<GL::__r__> _rows,
                        ducks::gl::make_arg_t<GL::__c__> _cols)
        : mc_ptr(nullptr), gls{GL(_data[I], _batch, _depth, _rows, _cols)...} {
        static_assert(!MULTICAST, "Multicast pointer not passed to multicast-enabled PGL.");
    }

    template <size_t... I>
    __host__ inline pgl(std::index_sequence<I...>, T **_data, ducks::gl::make_arg_t<GL::__b__> _batch,
                        ducks::gl::make_arg_t<GL::__d__> _depth, ducks::gl::make_arg_t<GL::__r__> _rows,
                        ducks::gl::make_arg_t<GL::__c__> _cols, gl_strides _strides)
        : mc_ptr(nullptr), gls{GL(_data[I], _batch, _depth, _rows, _cols, _strides)...} {
        static_assert(!MULTICAST, "Multicast pointer not passed to multicast-enabled PGL.");
    }

    template <size_t... I>
    __host__ inline pgl(std::index_sequence<I...>, T *_mc_ptr, T **_data, ducks::gl::make_arg_t<GL::__b__> _batch,
                        ducks::gl::make_arg_t<GL::__d__> _depth, ducks::gl::make_arg_t<GL::__r__> _rows,
                        ducks::gl::make_arg_t<GL::__c__> _cols)
        : mc_ptr(_mc_ptr), gls{GL(_data[I], _batch, _depth, _rows, _cols)...} {
        static_assert(MULTICAST, "Multicast pointer passed to multicast-disabled PGL.");
        initialize_tma_descs();
    }

    template <size_t... I>
    __host__ inline pgl(std::index_sequence<I...>, T *_mc_ptr, T **_data, ducks::gl::make_arg_t<GL::__b__> _batch,
                        ducks::gl::make_arg_t<GL::__d__> _depth, ducks::gl::make_arg_t<GL::__r__> _rows,
                        ducks::gl::make_arg_t<GL::__c__> _cols, gl_strides _strides)
        : mc_ptr(_mc_ptr), gls{GL(_data[I], _batch, _depth, _rows, _cols, _strides)...} {
        static_assert(MULTICAST, "Multicast pointer passed to multicast-disabled PGL.");
        initialize_tma_descs();
    }

    template <typename U> __device__ inline const CUtensorMap *get_tma() const { return tma_descs.template get<U>(); }

    __host__ __device__ inline auto batch() const { return gls[0].batch(); }
    __host__ __device__ inline auto depth() const { return gls[0].depth(); }
    __host__ __device__ inline auto rows() const { return gls[0].rows(); }
    __host__ __device__ inline auto cols() const { return gls[0].cols(); }
    __host__ __device__ inline size_t numel() const { return static_cast<size_t>(batch()) * depth() * rows() * cols(); }

    template <int axis> __device__ inline size_t shape() const { return gls[0].template shape<axis>(); }
    template <int axis> __device__ inline size_t stride() const { return gls[0].template stride<axis>(); }

  private:
    __host__ inline void initialize_tma_descs() {
        const GL &gl = gls[0];
        kittens::tma::global_tensor_view<T> view{
            mc_ptr,
            {uint64_t(gl.batch()), uint64_t(gl.depth()), uint64_t(gl.rows()), uint64_t(gl.cols())},
            {uint64_t(gl.template stride<dim::BATCH>()), uint64_t(gl.template stride<dim::DEPTH>()),
             uint64_t(gl.template stride<dim::ROW>()), 1}};
        tma_descs = detail::descriptor_dict<TMA_Types...>(view);
    }
};

template <ducks::pgl::all PGL, bool safe = true>
__host__ inline PGL make_pgl(uint64_t *data, int b, int d, int r, int c) {
    if constexpr (safe) {
        if (PGL::GL::__b__ > 0 && b != PGL::GL::__b__) {
            throw std::runtime_error("Batch dimension mismatch. Expected: " + std::to_string(PGL::GL::__b__) +
                                     ", Got: " + std::to_string(b));
        }
        if (PGL::GL::__d__ > 0 && d != PGL::GL::__d__) {
            throw std::runtime_error("Depth dimension mismatch. Expected: " + std::to_string(PGL::GL::__d__) +
                                     ", Got: " + std::to_string(d));
        }
        if (PGL::GL::__r__ > 0 && r != PGL::GL::__r__) {
            throw std::runtime_error("Row dimension mismatch. Expected: " + std::to_string(PGL::GL::__r__) +
                                     ", Got: " + std::to_string(r));
        }
        if (PGL::GL::__c__ > 0 && c != PGL::GL::__c__) {
            throw std::runtime_error("Column dimension mismatch. Expected: " + std::to_string(PGL::GL::__c__) +
                                     ", Got: " + std::to_string(c));
        }
    }
    return PGL(reinterpret_cast<typename PGL::dtype **>(data), make_unsafe_gl_arg<PGL::GL::__b__>(b),
               make_unsafe_gl_arg<PGL::GL::__d__>(d), make_unsafe_gl_arg<PGL::GL::__r__>(r),
               make_unsafe_gl_arg<PGL::GL::__c__>(c));
}

template <ducks::pgl::all PGL, bool safe = true>
__host__ inline PGL make_pgl(uint64_t *data, int b, int d, int r, int c, gl_strides strides) {
    if constexpr (safe) {
        if (PGL::GL::__b__ > 0 && b != PGL::GL::__b__)
            throw std::runtime_error("Batch dimension mismatch.");
        if (PGL::GL::__d__ > 0 && d != PGL::GL::__d__)
            throw std::runtime_error("Depth dimension mismatch.");
        if (PGL::GL::__r__ > 0 && r != PGL::GL::__r__)
            throw std::runtime_error("Row dimension mismatch.");
        if (PGL::GL::__c__ > 0 && c != PGL::GL::__c__)
            throw std::runtime_error("Column dimension mismatch.");
        if (strides.batch == 0 || strides.depth == 0 || strides.row == 0)
            throw std::runtime_error("Global layout strides must be positive.");
    }
    return PGL(reinterpret_cast<typename PGL::dtype **>(data), make_unsafe_gl_arg<PGL::GL::__b__>(b),
               make_unsafe_gl_arg<PGL::GL::__d__>(d), make_unsafe_gl_arg<PGL::GL::__r__>(r),
               make_unsafe_gl_arg<PGL::GL::__c__>(c), strides);
}

template <ducks::pgl::all PGL, bool safe = true>
__host__ inline PGL make_pgl(uint64_t mc_ptr, uint64_t *data, int b, int d, int r, int c) {
    if constexpr (safe) {
        if (PGL::GL::__b__ > 0 && b != PGL::GL::__b__) {
            throw std::runtime_error("Batch dimension mismatch. Expected: " + std::to_string(PGL::GL::__b__) +
                                     ", Got: " + std::to_string(b));
        }
        if (PGL::GL::__d__ > 0 && d != PGL::GL::__d__) {
            throw std::runtime_error("Depth dimension mismatch. Expected: " + std::to_string(PGL::GL::__d__) +
                                     ", Got: " + std::to_string(d));
        }
        if (PGL::GL::__r__ > 0 && r != PGL::GL::__r__) {
            throw std::runtime_error("Row dimension mismatch. Expected: " + std::to_string(PGL::GL::__r__) +
                                     ", Got: " + std::to_string(r));
        }
        if (PGL::GL::__c__ > 0 && c != PGL::GL::__c__) {
            throw std::runtime_error("Column dimension mismatch. Expected: " + std::to_string(PGL::GL::__c__) +
                                     ", Got: " + std::to_string(c));
        }
    }
    return PGL(reinterpret_cast<typename PGL::dtype *>(mc_ptr), reinterpret_cast<typename PGL::dtype **>(data),
               make_unsafe_gl_arg<PGL::GL::__b__>(b), make_unsafe_gl_arg<PGL::GL::__d__>(d),
               make_unsafe_gl_arg<PGL::GL::__r__>(r), make_unsafe_gl_arg<PGL::GL::__c__>(c));
}

template <ducks::pgl::all PGL, bool safe = true>
__host__ inline PGL make_pgl(uint64_t mc_ptr, uint64_t *data, int b, int d, int r, int c, gl_strides strides) {
    if constexpr (safe) {
        if (PGL::GL::__b__ > 0 && b != PGL::GL::__b__)
            throw std::runtime_error("Batch dimension mismatch.");
        if (PGL::GL::__d__ > 0 && d != PGL::GL::__d__)
            throw std::runtime_error("Depth dimension mismatch.");
        if (PGL::GL::__r__ > 0 && r != PGL::GL::__r__)
            throw std::runtime_error("Row dimension mismatch.");
        if (PGL::GL::__c__ > 0 && c != PGL::GL::__c__)
            throw std::runtime_error("Column dimension mismatch.");
        if (strides.batch == 0 || strides.depth == 0 || strides.row == 0)
            throw std::runtime_error("Global layout strides must be positive.");
    }
    return PGL(reinterpret_cast<typename PGL::dtype *>(mc_ptr), reinterpret_cast<typename PGL::dtype **>(data),
               make_unsafe_gl_arg<PGL::GL::__b__>(b), make_unsafe_gl_arg<PGL::GL::__d__>(d),
               make_unsafe_gl_arg<PGL::GL::__r__>(r), make_unsafe_gl_arg<PGL::GL::__c__>(c), strides);
}

} // namespace kittens
