#pragma once
#include "../../group.cuh"
#include "../../mma/warp/warp_impl.cuh"

template <int GW>
struct group<GW>::emulated_wgmma {

    template <ducks::rt::row_layout D>
    __device__ static inline void mma_fence(D &dst) {}
    template <ducks::crt::row_layout D>
    __device__ static inline void mma_fence(D &dst) {}
    template <typename T = kittens::ducks::default_type>
    __device__ static inline void mma_fence() {}

    template <typename T = kittens::ducks::default_type>
    __device__ static inline void mma_commit_group() {}

    template <int N = 0>
    __device__ static inline void mma_async_wait() {
        sync(groupid());
    }

    template <ducks::rt::row_layout D, ducks::rt::row_layout A, ducks::st_descriptor::input B, int fence = 1,
              int accumulate = 1>
    __device__ static inline void mma_AB(D &d, const A &a, const B &b) {
        KITTENS_CHECK_WARPGROUP

        constexpr int M_DIV_4 = A::height;
        static_assert(D::height == M_DIV_4); // output register is correctly sized
        constexpr int N = B::width;
        constexpr int K = A::width;
        static_assert(B::height == K); // K dimension must match
        static_assert(std::is_same_v<typename A::T,
                                     typename B::T>); // A and B must match type.

        rt<typename B::T, B::height * 16, B::width * 16, ducks::rt_layout::col> b_reg;
        load(b_reg, b);

        wmma::mma_AB(d, a, b_reg, accumulate ? d : D{0.f});
    }
    template <ducks::rt::row_layout D, ducks::rt::row_layout A, ducks::st_descriptor::input B>
    __device__ static inline void mm_AB(D &d, const A &a, const B &b) {
        mma_AB<D, A, B, 1, 0>(d, a, b);
    }

    template <ducks::rt::row_layout D, ducks::st_descriptor::input A, ducks::st_descriptor::input B, int fence = 1,
              int accumulate = 1>
    __device__ static inline void mma_AB(D &d, const A &a, const B &b) {
        KITTENS_CHECK_WARPGROUP
        constexpr int M = A::height;
        static_assert(M == 4);
        static_assert(D::height == 1); // output register is correctly sized
        constexpr int N = B::width;
        constexpr int K = A::width;
        static_assert(B::height == K); // K dimension must match
        static_assert(std::is_same_v<typename A::T,
                                     typename B::T>); // A and B must match type.

        rt<typename A::T, A::height * 4, A::width * 16> a_reg;
        rt<typename B::T, B::height * 16, B::width * 16, ducks::rt_layout::col> b_reg;
        load(a_reg, a);
        load(b_reg, b);

        wmma::mma_AB(d, a_reg, b_reg, accumulate ? d : D{0.f});
    }
    template <ducks::rt::row_layout D, ducks::st_descriptor::input A, ducks::st_descriptor::input B>
    __device__ static inline void mm_AB(D &d, const A &a, const B &b) {
        mma_AB<D, A, B, 1, 0>(d, a, b);
    }

    template <ducks::rt::row_layout D, ducks::rt::row_layout A, ducks::st_descriptor::input B, int fence = 1,
              int accumulate = 1>
    __device__ static inline void mma_ABt(D &d, const A &a, const B &b) {
        // Checks
        KITTENS_CHECK_WARPGROUP
        constexpr int M_DIV_4 = A::height;
        static_assert(D::height == M_DIV_4); // output register is correctly sized
        constexpr int N = B::height;
        constexpr int K = A::width;
        static_assert(B::width == K); // K dimension must match
        static_assert(std::is_same_v<typename A::T,
                                     typename B::T>); // A and B must match type.

        rt<typename B::T, B::height * 16, B::width * 16> b_reg;
        warp::load(b_reg, b);

        warp::wmma::mma_ABt(d, a, b_reg, accumulate ? d : D{0.f});
    }
    template <ducks::rt::row_layout D, ducks::rt::row_layout A, ducks::st_descriptor::input B>
    __device__ static inline void mm_ABt(D &d, const A &a, const B &b) {
        mma_ABt<D, A, B, 1, 0>(d, a, b);
    }

    template <ducks::rt::row_layout D, ducks::st_descriptor::input A, ducks::st_descriptor::input B, int fence = 1,
              int accumulate = 1>
    __device__ static inline void mma_ABt(D &d, const A &a, const B &b) {
        KITTENS_CHECK_WARPGROUP
        constexpr int M = A::height;
        static_assert(M == 4);
        static_assert(D::height == 1); // output register is correctly sized
        constexpr int N = B::height;
        constexpr int K = A::width;
        static_assert(B::width == K); // K dimension must match
        static_assert(std::is_same_v<typename A::T,
                                     typename B::T>); // A and B must match type.

        rt<typename A::T, A::height * 4, A::width * 16> a_reg;
        rt<typename B::T, B::height * 16, B::width * 16> b_reg;
        load(a_reg, a);
        load(b_reg, b);

        warp::wmma::mma_ABt(d, a_reg, b_reg, accumulate ? d : D{0.f});
    }
    template <ducks::rt::row_layout D, ducks::st_descriptor::input A, ducks::st_descriptor::input B>
    __device__ static inline void mm_ABt(D &d, const A &a, const B &b) {
        mma_ABt<D, A, B, 1, 0>(d, a, b);
    }

    template <ducks::rt::row_layout D, ducks::st_descriptor::input A, ducks::st_descriptor::input B, int fence = 1,
              int accumulate = 1>
    __device__ static inline void mma_AtB(D &d, const A &a, const B &b) {
        KITTENS_CHECK_WARPGROUP
        constexpr int M = A::width;
        static_assert(M == 4);
        static_assert(D::height == 1); // output register is correctly sized
        constexpr int N = B::width;
        constexpr int K = A::height;
        static_assert(B::height == K); // K dimension must match
        static_assert(std::is_same_v<typename A::T,
                                     typename B::T>); // A and B must match type.

        rt<typename A::T, A::height * 16, A::width * 4, ducks::rt_layout::col> a_reg;
        rt<typename B::T, B::height * 16, B::width * 16, ducks::rt_layout::col> b_reg;
        load(a_reg, a);
        load(b_reg, b);

        wmma::mma_AtB(d, a_reg, b_reg, accumulate ? d : D{0.f});
    }
    template <ducks::rt::row_layout D, ducks::st_descriptor::input A, ducks::st_descriptor::input B>
    __device__ static inline void mm_AtB(D &d, const A &a, const B &b) {
        mma_AtB<D, A, B, 1, 0>(d, a, b);
    }

    template <ducks::rt::row_layout D, ducks::st_descriptor::input A, ducks::st_descriptor::input B, int fence = 1,
              int accumulate = 1>
    __device__ static inline void mma_AtBt(D &d, const A &a, const B &b) {
        KITTENS_CHECK_WARPGROUP
        constexpr int M = A::width;
        static_assert(M == 4);
        static_assert(D::height == 1); // output register is correctly sized
        constexpr int N = B::height;
        constexpr int K = A::height;
        static_assert(B::width == K); // K dimension must match
        static_assert(std::is_same_v<typename A::T,
                                     typename B::T>); // A and B must match type.

        rt<typename A::T, A::height * 16, A::width * 4, ducks::rt_layout::col> a_reg;
        rt<typename B::T, B::height * 16, B::width * 16, ducks::rt_layout::row> b_reg;
        load(a_reg, a);
        load(b_reg, b);

        wmma::mma_AtBt(d, a_reg, b_reg, accumulate ? d : D{0.f});
    }
    template <ducks::rt::row_layout D, ducks::st_descriptor::input A, ducks::st_descriptor::input B>
    __device__ static inline void mm_AtBt(D &d, const A &a, const B &b) {
        mma_AtBt<D, A, B, 1, 0>(d, a, b);
    }

    template <int trans_A, int trans_B, typename D, typename A, typename B>
    __device__ static inline void mma(D &d, const A &a, const B &b) {
        if constexpr (trans_A == transpose::T) {
            if constexpr (trans_B == transpose::T) {
                mma_AtBt(d, a, b);
            } else {
                mma_AtB(d, a, b);
            }
        } else {
            if constexpr (trans_B == transpose::T) {
                mma_ABt(d, a, b);
            } else {
                mma_AB(d, a, b);
            }
        }
    }

    template <int trans_A, int trans_B, typename D, typename A, typename B>
    __device__ static inline void mm(D &d, const A &a, const B &b) {
        if constexpr (trans_A == transpose::T) {
            if constexpr (trans_B == transpose::T) {
                mm_AtBt(d, a, b);
            } else {
                mm_AtB(d, a, b);
            }
        } else {
            if constexpr (trans_B == transpose::T) {
                mm_ABt(d, a, b);
            } else {
                mm_AB(d, a, b);
            }
        }
    }
};
