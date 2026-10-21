/**
 * @file
 * @brief Group (collaborative warp) ops for loading shared tiles from and
 * storing to global memory.
 */

template <bool assume_aligned, ducks::cst::all CST, ducks::cgl::all CGL, ducks::coord::tile COORD = coord<CST>>
__device__ static inline void load(CST &dst, const CGL &src, const COORD &idx) {
    load<assume_aligned, typename CST::component, typename CGL::component, COORD>(dst.real, src.real, idx);
    load<assume_aligned, typename CST::component, typename CGL::component, COORD>(dst.imag, src.imag, idx);
}
template <ducks::cst::all CST, ducks::cgl::all CGL, ducks::coord::tile COORD = coord<CST>>
__device__ static inline void load(CST &dst, const CGL &src, const COORD &idx) {
    load<false, typename CST::component, typename CGL::component, COORD>(dst.real, src.real, idx);
    load<false, typename CST::component, typename CGL::component, COORD>(dst.imag, src.imag, idx);
}

template <bool assume_aligned, ducks::cst::all CST, ducks::cgl::all CGL, ducks::coord::tile COORD = coord<CST>>
__device__ static inline void store(CGL &dst, const CST &src, const COORD &idx) {
    store<assume_aligned, typename CST::component, typename CGL::component, COORD>(dst.real, src.real, idx);
    store<assume_aligned, typename CST::component, typename CGL::component, COORD>(dst.imag, src.imag, idx);
}
template <ducks::cst::all CST, ducks::cgl::all CGL, ducks::coord::tile COORD = coord<CST>>
__device__ static inline void store(CGL &dst, const CST &src, const COORD &idx) {
    store<false, typename CST::component, typename CGL::component, COORD>(dst.real, src.real, idx);
    store<false, typename CST::component, typename CGL::component, COORD>(dst.imag, src.imag, idx);
}

template <bool assume_aligned, ducks::cst::all CST, ducks::cgl::all CGL, ducks::coord::tile COORD = coord<CST>>
__device__ static inline void load_async(CST &dst, const CGL &src, const COORD &idx) {
    load_async<assume_aligned, typename CST::component, typename CGL::component, COORD>(dst.real, src.real, idx);
    load_async<assume_aligned, typename CST::component, typename CGL::component, COORD>(dst.imag, src.imag, idx);
}
template <ducks::cst::all CST, ducks::cgl::all CGL, ducks::coord::tile COORD = coord<CST>>
__device__ static inline void load_async(CST &dst, const CGL &src, const COORD &idx) {
    load_async<false, typename CST::component, typename CGL::component, COORD>(dst.real, src.real, idx);
    load_async<false, typename CST::component, typename CGL::component, COORD>(dst.imag, src.imag, idx);
}
