#pragma once

#include <concepts>
#include <kittens.cuh>
#include <utility>

#include "common/debug.cuh"

namespace kt = kittens;

namespace{
namespace diagscan{

// ============================================================================
// Type Definitions and Concepts
// ============================================================================
namespace details {
struct VecDirectionId {};
struct ScanResultId {};

struct UpVec : std::true_type {
    using identifier = VecDirectionId;
};
struct DownVec : std::false_type {
    using identifier = VecDirectionId;
};

template <typename T>
concept vec_direction = std::is_same_v<typename T::identifier, VecDirectionId>;

struct Input : std::false_type {
    using identifier = ScanResultId;
};
struct Output : std::true_type {
    using identifier = ScanResultId;
};

template <typename T>
concept scan_result = std::is_same_v<typename T::identifier, ScanResultId>;
}; // namespace details

// ============================================================================
// Utility Structs (Internal Implementation Details)
// ============================================================================

namespace details {

struct SelIntrinsics {
    static __forceinline__ __device__ float selRead(int tidm4, float v0, float v1, float v2, float v3) {
        return (tidm4 & 2) ? (tidm4 & 1 ? v3 : v2) : (tidm4 & 1 ? v1 : v0);
    }
    static __forceinline__ __device__ void selWrite(int tidm4, float &v0, float &v1, float &v2, float &v3, float val) {
        asm volatile("{\n"
                     ".reg .pred p0, p1, p2, p3;\n"
                     "setp.eq.s32 p0, %4, 0x0;\n"
                     "setp.eq.s32 p1, %4, 0x1;\n"
                     "setp.eq.s32 p2, %4, 0x2;\n"
                     "setp.eq.s32 p3, %4, 0x3;\n"
                     "@p0 mov.b32 %0, %5;\n"
                     "@p1 mov.b32 %1, %5;\n"
                     "@p2 mov.b32 %2, %5;\n"
                     "@p3 mov.b32 %3, %5;\n"
                     "}\n"
                     : "+r"(v0), "+r"(v1), "+r"(v2), "+r"(v3)
                     : "r"(tidm4 & 0x3), "r"(val));
    }

    static __forceinline__ __device__ void selWrite(int pred, float &vtrue, float &vfalse, float val) {
        asm volatile("{\n"
                     ".reg .pred p1;\n"
                     "setp.ne.s32 p1, %2, 0x0;\n"
                     "@!p1 mov.b32 %1, %3;\n"
                     "@p1 mov.b32 %0, %3;\n"
                     "}\n"
                     : "+r"(vtrue), "+r"(vfalse)
                     : "r"(pred), "r"(val));
    }
};

template <int STEP, int N>
__device__ void roll(float (&dst)[N]) {
    float result[N];
#pragma unroll
    for (int i = 0; i < N; i++)
        result[i] = dst[(N + i + STEP) % N];
#pragma unroll
    for (int i = 0; i < N; i++)
        dst[i] = result[i];
}

} // namespace details

// ============================================================================
// Scan State Types
// ============================================================================

template <int N, details::vec_direction Dir>
struct ScanState {
    float Data[N / 8];
    __device__ ScanState() : Data{0.f} {}
    __device__ ScanState(float zero) {
#pragma unroll
        for (int i = 0; i < N / 8; i++)
            Data[i] = zero;
    }

    __device__ bool operator==(const ScanState &rhs) const {
#pragma unroll
        for (int i = 0; i < N / 8; i++)
            if (std::abs(Data[i] - rhs.Data[i]) > 1e-5)
                return false;
        return true;
    }
    __device__ ScanState operator-(const ScanState &rhs) const {
        ScanState result;
#pragma unroll
        for (int i = 0; i < N / 8; i++)
            result.Data[i] = Data[i] - rhs.Data[i];
        return result;
    }
    __device__ ScanState operator-() const{
        ScanState result;
#pragma unroll
        for (int i = 0; i < N / 8; i++)
            result.Data[i] = -Data[i];
        return result;
    }

    __device__ void print() const { kt::print_utils::print(reinterpret_cast<const float (&)[N / 8][1]>(Data)); }
};

template <int N, details::vec_direction Dir>
struct ScanLrState : ScanState<N, Dir> {
    __device__ ScanLrState() : ScanState<N, Dir>{} {}
    __device__ ScanLrState(const ScanState<N, Dir> &x) : ScanState<N, Dir>{x} {}
    __device__ ScanLrState(float zero) : ScanState<N, Dir>{zero} {}

    template <size_t DN>
    __forceinline__ __device__ void store(kt::sv_fl<DN> &dst, int dst_offset = 0) {
        static_assert(DN >= N, "Space of destination vector is not large enough");

        uint32_t ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&dst.data[dst_offset]));

        uint32_t tid = threadIdx.x & 0x1f;
        uint32_t tidm4 = tid & 0x3;
        uint32_t tidr3 = tid >> 2;

        if constexpr (Dir::value) {
            bool offset4 = (tidr3 < 2 * tidm4);
            uint32_t pos4 = (3 - tidm4) + offset4;
            uint32_t ptr_offset4 = tidr3 + (offset4 ? 32 : 24) - 10 * tidm4;

            bool offset2 = (tidr3 < 2 * (tidm4));
            uint32_t pos2 = (1 - (tidm4 >> 1)) + offset2;
            uint32_t ptr_offset2 = tidr3 + (offset4 ? 16 : 8) - 6 * tidm4;

#pragma unroll
            for (int i = 0; i < N / 8; i += 4) {
                if (N / 8 - i >= 4) {
                    float v0 = pos4 & 0x2 ? (pos4 & 0x1 ? this->Data[3 + i] : this->Data[2 + i])
                                          : (pos4 & 0x1 ? this->Data[1 + i] : this->Data[0 + i]);
                    kt::move<float>::sts(ptr + sizeof(float) * (i * 8 + ptr_offset4), v0);
                } else {
                    float v0 = pos2 & 0x1 ? this->Data[1 + i] : this->Data[i];
                    if (!(tidm4 & 1)) {
                        kt::move<float>::sts(ptr + sizeof(float) * (i * 8 + ptr_offset2), v0);
                    }
                }
            }
        } else {
            bool offset4 = (tidr3 >= 2 * (tidm4 + 1));
            uint32_t pos4 = (3 - tidm4) - offset4;
            uint32_t ptr_offset4 = tidr3 + (offset4 ? 22 : 30) - 10 * tidm4;

            uint32_t pos2 = (1 - (tidm4 >> 1)) - offset4;
            uint32_t ptr_offset2 = tidr3 + (offset4 ? 10 : 18) - 6 * tidm4;

#pragma unroll
            for (int i = 0; i < N / 8; i += 4) {
                if (N / 8 - i >= 4) {
                    float v0 = pos4 & 0x2 ? (pos4 & 0x1 ? this->Data[3 + i] : this->Data[2 + i])
                                          : (pos4 & 0x1 ? this->Data[1 + i] : this->Data[0 + i]);
                    kt::move<float>::sts(ptr + sizeof(float) * (i * 8 + ptr_offset4), v0);
                } else {
                    float v0 = pos2 & 0x1 ? this->Data[1 + i] : this->Data[i];
                    if (tidm4 & 1) {
                        kt::move<float>::sts(ptr + sizeof(float) * (i * 8 + ptr_offset2), v0);
                    }
                }
            }
        }
    }

    template <size_t DN>
    __forceinline__ __device__ void load(kt::sv_fl<DN> &dst, int dst_offset = 0) {
        static_assert(DN >= N, "Space of destination vector is not large enough");
        uint32_t ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&dst.data[dst_offset]));

        uint32_t tid = threadIdx.x & 0x1f;
        uint32_t tidm4 = tid & 0x3;
        uint32_t tidr3 = tid >> 2;

        if constexpr (Dir::value) {
            bool offset4 = (tidr3 < 2 * tidm4);
            uint32_t pos4 = (3 - tidm4) + offset4;
            uint32_t ptr_offset4 = tidr3 + (offset4 ? 32 : 24) - 10 * tidm4;

#pragma unroll
            for (int i = 0; i < N / 8; i += 4) {
                if (N / 8 - i >= 4) {
                    float v[4];
                    kt::move<float>::lds(v[0], ptr + sizeof(float) * (i * 8 + ptr_offset4));

                    v[1] = __shfl_sync(~0u, v[0], tid + (tidm4 ? -9 : 27));
                    v[2] = __shfl_sync(~0u, v[0], tid + (tidm4 & 2 ? -18 : 18));
                    v[3] = __shfl_sync(~0u, v[0], tid + (tidm4 == 3 ? -27 : 9));

                    if (pos4 & 1)
                        details::roll<-1>(v);
                    if (pos4 & 2)
                        details::roll<-2>(v);

                    this->Data[1 + i] = v[1], this->Data[2 + i] = v[2], this->Data[3 + i] = v[3];

                    float sink;
                    details::SelIntrinsics::selWrite((tidr3 < 2 * tidm4), (i >= 4) ? this->Data[i + 4] : sink,
                                                       this->Data[i + 0], v[0]);
                } else {
                    float v0;
                    kt::move<float>::lds(v0, ptr + sizeof(float) * (i * 8 + (ptr_offset4 % 16)));

                    float v1 = __shfl_sync(~0u, v0, tid + (tidm4 & 0x2 ? -9 : 9), 32);

                    if (pos4 & 1)
                        std::swap(v0, v1);
                    this->Data[1 + i] = v1;

                    float sink;
                    details::SelIntrinsics::selWrite((tidr3 < 2 * tidm4), (i + 2 < N / 8) ? this->Data[i + 2] : sink,
                                                       this->Data[i + 0], v0);
                }
            }
        } else {
            bool offset4 = (tidr3 >= 2 * (tidm4 + 1));
            uint32_t pos4 = (3 - tidm4) - offset4;
            uint32_t ptr_offset4 = tidr3 + (offset4 ? 22 : 30) - 10 * tidm4;

#pragma unroll
            for (int i = 0; i < N / 8; i += 4) {
                if (N / 8 - i >= 4) {
                    float v[4];
                    kt::move<float>::lds(v[0], ptr + sizeof(float) * (i * 8 + ptr_offset4));

                    v[1] = __shfl_sync(~0u, v[0], tid + (tidm4 ? -9 : -5));
                    v[2] = __shfl_sync(~0u, v[0], tid + (tidm4 & 2 ? -18 : 18));
                    v[3] = __shfl_sync(~0u, v[0], tid + (tidm4 == 3 ? 5 : 9));

                    if (pos4 & 1)
                        details::roll<1>(v);
                    if (pos4 & 2)
                        details::roll<2>(v);

                    this->Data[0 + i] = v[0], this->Data[1 + i] = v[1], this->Data[2 + i] = v[2];

                    float sink;
                    details::SelIntrinsics::selWrite((tidr3 >= 2 * (tidm4 + 1)), (i >= 4) ? this->Data[i - 1] : sink,
                                                       this->Data[i + 3], v[3]);
                } else {
                    float v0;
                    kt::move<float>::lds(v0, ptr + sizeof(float) * (i * 8 + (ptr_offset4 % 16)));

                    float v1 = __shfl_sync(~0u, v0, tid + (tidm4 & 0x2 ? -9 : 9), 32);

                    if (pos4 & 1)
                        std::swap(v0, v1);
                    this->Data[0 + i] = v0;

                    float sink;
                    details::SelIntrinsics::selWrite((tidr3 >= 2 * (tidm4 + 1)), (i >= 1) ? this->Data[i - 1] : sink,
                                                       this->Data[i + 1], v1);
                }
            }
        }
    }
};

template <int N, details::vec_direction Dir>
struct ScanTbState : ScanState<N, Dir> {
    __device__ ScanTbState() : ScanState<N, Dir>{} {}
    __device__ ScanTbState(const ScanState<N, Dir> &x) : ScanState<N, Dir>{x} {}
    __device__ ScanTbState(float zero) : ScanState<N, Dir>{zero} {}

    template <size_t DN>
    __forceinline__ __device__ void store(kt::sv_fl<DN> &dst, int dst_offset = 0) {
        static_assert(DN >= N, "Space of destination vector is not large enough");
        uint32_t ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&dst.data[dst_offset]));

        uint32_t tid = threadIdx.x & 0x1f;
        uint32_t tidm4 = tid & 0x3;
        uint32_t tidr3 = tid >> 2;

        if constexpr (Dir::value) {
            bool offset4 = (tidr3 < 2 * tidm4);
            uint32_t pos4 = tidm4 - offset4;
            uint32_t ptr_offset4 = 10 * tidm4 - tidr3 + (offset4 ? -1 : 7);

            uint32_t pos2 = (tidm4 >> 1) - offset4;
            uint32_t ptr_offset2 = 6 * tidm4 - tidr3 + (offset4 ? -1 : 7);

#pragma unroll
            for (int i = 0; i < N / 8; i += 4) {
                if (N / 8 - i >= 4) {
                    float v0 = pos4 & 0x2 ? (pos4 & 0x1 ? this->Data[3 + i] : this->Data[2 + i])
                                          : (pos4 & 0x1 ? this->Data[1 + i] : this->Data[0 + i]);
                    kt::move<float>::sts(ptr + sizeof(float) * (i * 8 + ptr_offset4), v0);
                } else {
                    float v0 = pos2 ? this->Data[1 + i] : this->Data[0 + i];
                    if (!(tidm4 & 1))
                        kt::move<float>::sts(ptr + sizeof(float) * (i * 8 + ptr_offset2), v0);
                }
            }
        } else {
            bool offset4 = (tidr3 >= 2 * (tidm4 + 1));
            uint32_t pos4 = tidm4 + offset4;
            uint32_t ptr_offset4 = 10 * tidm4 - tidr3 + (offset4 ? 9 : 1);

            uint32_t pos2 = (tidm4 >> 1) + offset4;
            uint32_t ptr_offset2 = 6 * tidm4 - tidr3 + (offset4 ? 5 : -3);

#pragma unroll
            for (int i = 0; i < N / 8; i += 4) {
                if (N / 8 - i >= 4) {
                    float v0 = pos4 & 0x2 ? (pos4 & 0x1 ? this->Data[3 + i] : this->Data[2 + i])
                                          : (pos4 & 0x1 ? this->Data[1 + i] : this->Data[0 + i]);
                    kt::move<float>::sts(ptr + sizeof(float) * (i * 8 + ptr_offset4), v0);
                } else {
                    float v0 = pos2 ? this->Data[1 + i] : this->Data[0 + i];
                    if (tidm4 & 1)
                        kt::move<float>::sts(ptr + sizeof(float) * (i * 8 + ptr_offset2), v0);
                }
            }
        }
    }

    template <size_t DN>
    __forceinline__ __device__ void load(kt::sv_fl<DN> &dst, int dst_offset = 0) {
        uint32_t ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&dst.data[dst_offset]));

        uint32_t tid = threadIdx.x & 0x1f;
        uint32_t tidm4 = tid & 0x3;
        uint32_t tidr3 = tid >> 2;

        if constexpr (Dir::value) {
            bool offset4 = (tidr3 < 2 * tidm4);
            uint32_t pos4 = tidm4 - offset4;
            uint32_t ptr_offset4 = 10 * tidm4 - tidr3 + (offset4 ? -1 : 7);

#pragma unroll
            for (int i = 0; i < N / 8; i += 4) {
                if (N / 8 - i >= 4) {
                    float v[4];
                    kt::move<float>::lds(v[0], ptr + sizeof(float) * (i * 8 + ptr_offset4));

                    v[1] = __shfl_sync(~0u, v[0], tid + (tidm4 == 3 ? 5 : 9));
                    v[2] = __shfl_sync(~0u, v[0], tid + (tidm4 & 2 ? -18 : 18));
                    v[3] = __shfl_sync(~0u, v[0], tid + (tidm4 ? -9 : 27));

                    if ((pos4 & 1))
                        details::roll<-1>(v);
                    if ((pos4 & 2))
                        details::roll<-2>(v);
                    this->Data[0 + i] = v[0];
                    this->Data[1 + i] = v[1];
                    this->Data[2 + i] = v[2];

                    float sink;
                    details::SelIntrinsics::selWrite((tidr3 < 2 * tidm4), (i >= 1) ? this->Data[i - 1] : sink,
                                                       this->Data[3 + i], v[3]);
                } else {
                    float v0;
                    kt::move<float>::lds(v0, ptr + sizeof(float) * (i * 8 + (ptr_offset4 % 16)));

                    float v1 = __shfl_sync(~0u, v0, tid + (tidm4 & 0x2 ? -9 : 9));
                    if (pos4 & 1)
                        std::swap(v0, v1);
                    this->Data[0 + i] = v0;

                    float sink;
                    details::SelIntrinsics::selWrite((tidr3 < 2 * tidm4), (i >= 1) ? this->Data[i - 1] : sink,
                                                       this->Data[1 + i], v1);
                }
            }
        } else {
            bool offset4 = (tidr3 >= 2 * (tidm4 + 1));
            uint32_t pos4 = tidm4 + offset4;
            uint32_t ptr_offset4 = 10 * tidm4 - tidr3 + (offset4 ? 9 : 1);

#pragma unroll
            for (int i = 0; i < N / 8; i += 4) {

                if (N / 8 - i >= 4) {
                    float v[4];
                    kt::move<float>::lds(v[0], ptr + sizeof(float) * (i * 8 + ptr_offset4));

                    v[1] = __shfl_sync(~0u, v[0], tid + (tidm4 == 3 ? 5 : 9));
                    v[2] = __shfl_sync(~0u, v[0], tid + (tidm4 & 2 ? -18 : 18));
                    v[3] = __shfl_sync(~0u, v[0], tid + (tidm4 ? -9 : 27));

                    if ((pos4 & 1))
                        details::roll<-1>(v);
                    if ((pos4 & 2))
                        details::roll<-2>(v);
                    this->Data[1 + i] = v[1];
                    this->Data[2 + i] = v[2];
                    this->Data[3 + i] = v[3];

                    float sink;
                    details::SelIntrinsics::selWrite((tidr3 >= 2 * (tidm4 + 1)),
                                                       (i + 4 < N / 8) ? this->Data[i + 4] : sink, this->Data[i], v[0]);
                } else {
                    float v0;
                    kt::move<float>::lds(v0, ptr + sizeof(float) * (i * 8 + (ptr_offset4 % 16)));

                    float v1 = __shfl_sync(~0u, v0, tid + (tidm4 & 0x2 ? -9 : 9));
                    if (pos4 & 1)
                        std::swap(v0, v1);
                    this->Data[1 + i] = v1;

                    float sink;
                    details::SelIntrinsics::selWrite((tidr3 >= 2 * (tidm4 + 1)),
                                                       (i + 4 < N / 8) ? this->Data[i + 2] : sink, this->Data[i], v0);
                }
            }
        }
    }
};

template <int ROWS, int COLS, details::vec_direction Dir, details::scan_result ResultType>
struct DiagScanState {
    static constexpr uint32_t ROW_UNITS = ROWS / 8;
    static constexpr uint32_t COL_UNITS = COLS / 8;

    ScanLrState<ROWS, Dir> LrState;
    ScanTbState<COLS, Dir> TbState;

    __device__ bool operator==(const DiagScanState &rhs) const {
        return LrState == rhs.LrState && TbState == rhs.TbState;
    }

    __device__ static DiagScanState fromUdata(const float (&udata)[ROW_UNITS + COL_UNITS][1]) {
        DiagScanState result{{0.f}, {0.f}};
        if constexpr (!(Dir::value ^ ResultType::value)) {
#pragma unroll
            for (auto i = 0u; i < COL_UNITS; i++) {
                result.TbState.Data[COL_UNITS - i - 1] = udata[i][0];
            }
#pragma unroll
            for (auto i = 0u; i < ROW_UNITS; i++) {
                result.LrState.Data[i] = udata[COL_UNITS + i][0];
            }
        } else {
#pragma unroll
            for (auto i = 0u; i < ROW_UNITS; i++) {
                result.LrState.Data[i] = udata[i][0];
            }
#pragma unroll
            for (auto i = 0u; i < COL_UNITS; i++) {
                result.TbState.Data[COL_UNITS - 1 - i] = udata[ROW_UNITS + i][0];
            }
        }
        return result;
    }

    __device__ void toUdata(float (&udata)[ROW_UNITS + COL_UNITS][1]) const {
        if constexpr (!(Dir::value ^ ResultType::value)) {
#pragma unroll
            for (auto i = 0u; i < COL_UNITS; i++) {
                udata[i][0] = TbState.Data[COL_UNITS - i - 1];
            }
#pragma unroll
            for (auto i = 0u; i < ROW_UNITS; i++) {
                udata[COL_UNITS + i][0] = LrState.Data[i];
            }
        } else {
#pragma unroll
            for (auto i = 0u; i < ROW_UNITS; i++) {
                udata[i][0] = LrState.Data[i];
            }
#pragma unroll
            for (auto i = 0u; i < COL_UNITS; i++) {
                udata[ROW_UNITS + i][0] = TbState.Data[COL_UNITS - 1 - i];
            }
        }
    }

    __device__ void print() const {
        float udata[ROW_UNITS + COL_UNITS][1];
        toUdata(udata);
        kt::print_utils::print(udata);
    }

    __device__ DiagScanState<ROWS, COLS, Dir, std::conditional_t<ResultType::value, details::Input, details::Output>>
    cast() const {
        float udata[ROW_UNITS + COL_UNITS][1];
        toUdata(udata);
        return DiagScanState<
            ROWS, COLS, Dir,
            std::conditional_t<ResultType::value, details::Input, details::Output>>::fromUdata(udata);
    }

    __device__ DiagScanState operator-(const DiagScanState &rhs) const {
        return {LrState - rhs.LrState, TbState - rhs.TbState};
    }
};

// ============================================================================
// Scan Utilities and Helpers
// ============================================================================

namespace details {

struct ScanUtils {

    template <details::scan_result ResultType, bool DEBUG = false, int ROWS, int COLS>
    static __forceinline__ __device__ void propState(ScanLrState<ROWS, details::DownVec> &l_init,
                                                      ScanTbState<COLS, details::DownVec> &t_init) {
        static constexpr uint32_t row_units = ROWS / 8;
        static constexpr uint32_t col_units = COLS / 8;
        uint32_t tid = threadIdx.x & 0x1f;
        int tidr3 = tid >> 2;
        int tidm4 = tid & 0x3;

        float udata[row_units + col_units][1];

        DiagScanState<ROWS, COLS, details::DownVec, ResultType>{l_init, t_init}.toUdata(udata);

        if constexpr (DEBUG)
            kt::print_utils::print(udata);

        do {
            float remote_vals[row_units + col_units];
#pragma unroll
            for (auto i = 0u; i < row_units + col_units; i++) {
                remote_vals[i] = __shfl_sync(~0u, udata[i][0], (tidr3 - 2 * tidm4) * 4 + 27, 32);
            }
#pragma unroll
            for (auto i = 0u; i < row_units + col_units; i++) {
                float sink;
                details::SelIntrinsics::selWrite(tidr3 - 2 * tidm4 - 1 <= 0, udata[i][0],
                                                   i == 0 ? sink : udata[i - 1][0], remote_vals[i]);
            }
        } while (0);

        auto result = DiagScanState<ROWS, COLS, details::DownVec, ResultType>::fromUdata(udata);

        l_init = result.LrState, t_init = result.TbState;
    }

    template <details::scan_result ResultType, bool DEBUG = false, int ROWS, int COLS>
    static __forceinline__ __device__ void propState(ScanLrState<ROWS, details::UpVec> &l_init,
                                                      ScanTbState<COLS, details::UpVec> &t_init) {
        static constexpr uint32_t row_units = ROWS / 8;
        static constexpr uint32_t col_units = COLS / 8;
        uint32_t tid = threadIdx.x & 0x1f;
        int tidr3 = tid >> 2;
        int tidm4 = tid & 0x3;

        float udata[row_units + col_units][1];

        DiagScanState<ROWS, COLS, details::UpVec, ResultType>{l_init, t_init}.toUdata(udata);

        if constexpr (DEBUG)
            kt::print_utils::print(udata);

        do {
            float remote_vals[row_units + col_units];

#pragma unroll
            for (int i = 0; i < row_units + col_units; i++) {
                remote_vals[i] = __shfl_sync(~0u, udata[i][0], (tidr3 - 2 * tidm4) * 4, 32);
            }
#pragma unroll
            for (int i = 0; i < row_units + col_units; i++) {
                float sink;
                details::SelIntrinsics::selWrite(tidr3 - 2 * tidm4 >= 0, udata[i][0],
                                                   i == row_units + col_units - 1 ? sink : udata[i + 1][0],
                                                   remote_vals[i]);
            }
        } while (0);

        auto result = DiagScanState<ROWS, COLS, details::UpVec, ResultType>::fromUdata(udata);

        l_init = result.LrState, t_init = result.TbState;
    }
};
} // namespace details

// ============================================================================
// Helper Functions for Diagonal Scan Operations
// ============================================================================

template <typename T>
concept has_batch_reduction = requires(float x, float y, float z) {
    { T::batch_reduce(x, y, z) } -> std::same_as<float>;
};

namespace details {

template <typename AssocOp>
static __forceinline__ __device__ float binOp(float lhs, float rhs) {
    return AssocOp::op(lhs, rhs);
}

// binary fold tree
template <typename AssocOp, std::size_t L, std::size_t R, typename Tuple>
static constexpr float reduceRange(Tuple &&t) {
    static_assert(L < R);

    if constexpr (R - L == 1) {
        return std::get<L>(std::forward<Tuple>(t));
    } else {
        constexpr std::size_t M = L + (R - L) / 2;
        return details::binOp<AssocOp>(reduceRange<AssocOp, L, M>(std::forward<Tuple>(t)),
                                         reduceRange<AssocOp, M, R>(std::forward<Tuple>(t)));
    }
}

template <typename AssocOp, typename... Ts>
static __forceinline__ __device__ float batchOp(float init, Ts... xs) {
    if constexpr (has_batch_reduction<AssocOp>) {
        return AssocOp::batch_reduce(init, xs...);
    } else {
        auto tup = std::forward_as_tuple(init, std::forward<Ts>(xs)...);
        constexpr std::size_t N = 1 + sizeof...(Ts);
        return reduceRange<AssocOp, 0, N>(tup);
    }
}

template <int RU, int ROWS, int COLS>
static __forceinline__ __device__ void copyData(kt::rt_fl<ROWS, COLS> &dst, const float2 (&src)[RU][COLS / 8]) {
    constexpr int row_units = ROWS / 8;
    constexpr int col_units = COLS / 8;
    static_assert(RU >= row_units);

    using RT = kt::rt_fl<ROWS, COLS>;

#pragma unroll
    for (int i = 0; i < RT::height; i++) {
#pragma unroll
        for (int j = 0; j < RT::width; j++) {
            dst.tiles[i][j].data[0] = src[i * 2 + 0][j * 2 + 0];
            dst.tiles[i][j].data[1] = src[i * 2 + 1][j * 2 + 0];
            dst.tiles[i][j].data[2] = src[i * 2 + 0][j * 2 + 1];
            dst.tiles[i][j].data[3] = src[i * 2 + 1][j * 2 + 1];
        }
    }
}

template <int RU, float ZERO, int ROWS, int COLS>
static __forceinline__ __device__ void copyData(float2 (&dst)[RU][COLS / 8], const kt::rt_fl<ROWS, COLS> &src) {
    constexpr int row_units = ROWS / 8;
    constexpr int col_units = COLS / 8;
    using RT = kt::rt_fl<ROWS, COLS>;

#pragma unroll
    for (int i = 0; i < RT::height; i++) {
#pragma unroll
        for (int j = 0; j < RT::width; j++) {
            dst[i * 2 + 0][j * 2 + 0] = src.tiles[i][j].data[0];
            dst[i * 2 + 1][j * 2 + 0] = src.tiles[i][j].data[1];
            dst[i * 2 + 0][j * 2 + 1] = src.tiles[i][j].data[2];
            dst[i * 2 + 1][j * 2 + 1] = src.tiles[i][j].data[3];
        }
    }

#pragma unroll
    for (int i = row_units; i < RU; i++) {
#pragma unroll
        for (int j = 0; j < col_units; j++) {
            dst[i][j] = {ZERO, ZERO};
        }
    }
}

template <int RU, float ZERO, int ROWS, int COLS>
static __forceinline__ __device__ void invCopyData(float2 (&dst)[RU][COLS / 8], const kt::rt_fl<ROWS, COLS> &src) {
    constexpr int row_units = ROWS / 8;
    constexpr int col_units = COLS / 8;
    using RT = kt::rt_fl<ROWS, COLS>;
    constexpr int pad = RU - row_units;

#pragma unroll
    for (int i = 0; i < pad; i++) {
#pragma unroll
        for (int j = 0; j < col_units; j++) {
            dst[i][j] = {ZERO, ZERO};
        }
    }

#pragma unroll
    for (int i = 0; i < RT::height; i++) {
#pragma unroll
        for (int j = 0; j < RT::width; j++) {
            dst[pad + i * 2 + 0][j * 2 + 0] = src.tiles[i][j].data[0];
            dst[pad + i * 2 + 1][j * 2 + 0] = src.tiles[i][j].data[1];
            dst[pad + i * 2 + 0][j * 2 + 1] = src.tiles[i][j].data[2];
            dst[pad + i * 2 + 1][j * 2 + 1] = src.tiles[i][j].data[3];
        }
    }
}

template <int RU, float ZERO, int ROWS, int COLS>
static __forceinline__ __device__ void invCopyData(kt::rt_fl<ROWS, COLS> &dst, const float2 (&src)[RU][COLS / 8]) {
    using RT = kt::rt_fl<ROWS, COLS>;
    constexpr int row_units = ROWS / 8;
    constexpr int col_units = COLS / 8;
    constexpr int pad = RU - row_units;

#pragma unroll
    for (int i = 0; i < RT::height; i++) {
#pragma unroll
        for (int j = 0; j < RT::width; j++) {
            dst.tiles[i][j].data[0] = src[pad + i * 2 + 0][j * 2 + 0];
            dst.tiles[i][j].data[1] = src[pad + i * 2 + 1][j * 2 + 0];
            dst.tiles[i][j].data[2] = src[pad + i * 2 + 0][j * 2 + 1];
            dst.tiles[i][j].data[3] = src[pad + i * 2 + 1][j * 2 + 1];
        }
    }
}

} // namespace details

// ============================================================================
// Main Diagonal Scan Helpers (Public API)
// ============================================================================

struct DiagScanHelpers {
    // This helpers perform reductions/exclusive parallel scan on tiles with
    // standard MMA layout over diagonal lines with initial values (left-top to
    // right-bottom)

    template <details::scan_result ResultType, details::vec_direction Dir, bool SKIP_FIX = false, bool DEBUG = false,
              int ROWS, int COLS>
    __device__ static DiagScanState<ROWS, COLS, Dir, ResultType> makeState(const ScanLrState<ROWS, Dir> &lr,
                                                                               const ScanTbState<COLS, Dir> &tb) {
        DiagScanState<ROWS, COLS, Dir, ResultType> state{lr, tb};
        if constexpr (!SKIP_FIX)
            details::ScanUtils::propState<ResultType, DEBUG>(state.LrState, state.TbState);
        return state;
    }

    // ========================================================================
    // Diagonal Apply Operation
    // ========================================================================

    template <typename AssocOp, float ZERO, details::vec_direction Dir, details::scan_result ResultType, int ROWS,
              int COLS>
    static __forceinline__ __device__ void diagApply(kt::rt_fl<ROWS, COLS> &dst,
                                                      const DiagScanState<ROWS, COLS, Dir, ResultType> &init) {
        constexpr int row_units = ROWS / 8;
        constexpr int col_units = COLS / 8;
        // This procedure broadcasts initial values along diagonal lines of
        // tiles in standard MMA layout, where the updated value is assoc_op(old
        // value, broadcast value)
        uint32_t tid = threadIdx.x & 0x1f;
        uint32_t tidr3 = tid >> 2;

        float udata[row_units + col_units][1];
        float2 data[row_units + 1][col_units];
        details::copyData<row_units + 1, ZERO>(data, dst);

        init.toUdata(udata);

#pragma unroll
        for (int diag = 1; diag < row_units + col_units; diag++) {
            float current_y = udata[row_units + col_units - diag - 1][0];
            float current_x =
                __shfl_sync(~0u, (tidr3 != 0) ? current_y : udata[row_units + col_units - diag][0], tid + 4, 32);

            if (diag < row_units) {
#pragma unroll
                for (int irow = row_units - diag, icol = 0; irow < row_units + 1 && icol < col_units; irow++, icol++) {
                    data[irow][icol].x = details::binOp<AssocOp>(data[irow][icol].x, current_x);
                    data[irow][icol].y = details::binOp<AssocOp>(data[irow][icol].y, current_y);
                }
            } else {
#pragma unroll
                for (int irow = 0, icol = diag - row_units; irow < row_units + 1 && icol < col_units; irow++, icol++) {
                    data[irow][icol].x = details::binOp<AssocOp>(data[irow][icol].x, current_x);
                    data[irow][icol].y = details::binOp<AssocOp>(data[irow][icol].y, current_y);
                }
            }
        }

        details::copyData(dst, data);
    }

    // ========================================================================
    // Compile-Time Utilities
    // ========================================================================

    template <int A, int B>
    static inline constexpr int CT_MAX_V = (A > B ? A : B);
    template <int A, int B>
    static inline constexpr int CT_MIN_V = (A < B ? A : B);

    template <int N, class F>
    static __host__ __device__ inline void staticFor(F &&f) {
        []<size_t... IS>(std::index_sequence<IS...>, F &&f2) {
            (f2(std::integral_constant<int, (int)IS>{}), ...);
        }(std::make_index_sequence<N>{}, (F &&)f);
    }

    // ========================================================================
    // Diagonal Reduction Operations
    // ========================================================================

    template <typename AssocOp, float ZERO, details::vec_direction Dir, details::scan_result ResultType,
              details::scan_result InitResultType = details::Input, bool DEBUG = false, int ROWS, int COLS>
    static __forceinline__ __device__ DiagScanState<ROWS, COLS, Dir, ResultType>
    diagReduce(const kt::rt_fl<ROWS, COLS> &dst,
                const DiagScanState<ROWS, COLS, Dir, InitResultType> &init = {{ZERO}, {ZERO}}) {
        constexpr int row_units = ROWS / 8;
        constexpr int col_units = COLS / 8;

        // This procedure performs parallel reduction along diagonal lines of
        // tiles in standard MMA layout. Results are returned as a pair
        uint32_t tid = threadIdx.x & 0x1f;
        uint32_t tidr3 = tid >> 2;
        uint32_t tidm4 = tid & 0x3;
        bool pred3 = tidm4 == 3;
        bool pred1 = tidm4 & 1;

        float2 data[row_units + 1][col_units];
        float udata_x[row_units + col_units][1];
        float udata_y[row_units + col_units][1];
        if constexpr (Dir::value)
            details::invCopyData<row_units + 1, ZERO>(data, dst);
        else
            details::copyData<row_units + 1, ZERO>(data, dst);

        init.toUdata(Dir::value ? udata_x : udata_y);

        // propagation
        staticFor<row_units + col_units>([&](auto diagC) {
            constexpr int diag = decltype(diagC)::value;

            constexpr int irow0 = CT_MAX_V<0, row_units - diag>;
            constexpr int icol0 = CT_MAX_V<0, diag - row_units>;
            constexpr int Len = CT_MIN_V<(row_units + 1) - irow0, col_units - icol0>;

            constexpr int slot = (row_units + col_units - 1 - diag);

            float2 current = Dir::value ? float2{(!tidm4) ? udata_x[slot][0] : ZERO, ZERO}
                                        : float2{ZERO, pred3 ? udata_y[slot][0] : ZERO};

            current.x = [x = current.x]<int... KS>(std::integer_sequence<int, KS...>, auto &d) {
                return details::batchOp<AssocOp>(x, (d[irow0 + KS][icol0 + KS].x)...);
            }(std::make_integer_sequence<int, Len>{}, data);

            current.y = [y = current.y]<int... KS>(std::integer_sequence<int, KS...>, auto &d) {
                return details::batchOp<AssocOp>(y, (d[irow0 + KS][icol0 + KS].y)...);
            }(std::make_integer_sequence<int, Len>{}, data);

            udata_x[slot][0] = current.x;
            udata_y[slot][0] = current.y;
        });

        // x2 sync
        do {
            float new_vals[row_units + col_units];
            if constexpr (Dir::value) {
#pragma unroll
                for (int i = 0; i < row_units + col_units; i++) {
                    float local_val = ((tidr3 == 7) ? (i > 0 ? udata_x[i - 1][0] : ZERO) : udata_x[i][0]);
                    new_vals[i] = details::binOp<AssocOp>(__shfl_sync(~0u, udata_y[i][0], (tid + 4), 32), local_val);
                }

#pragma unroll
                for (int i = 0; i < row_units + col_units; i++) {
                    float sink;
                    details::SelIntrinsics::selWrite((tidr3 == 7), (i > 0 ? udata_x[i - 1][0] : sink), udata_x[i][0],
                                                       new_vals[i]);
                }
#pragma unroll
                for (int i = 0; i < row_units + col_units; i++)
                    udata_y[i][0] = udata_x[i][0];
            } else {
#pragma unroll
                for (int i = 0; i < row_units + col_units; i++) {
                    float local_val =
                        ((tidr3 < 1) ? (i < row_units + col_units - 1 ? udata_y[i + 1][0] : ZERO) : udata_y[i][0]);
                    new_vals[i] = details::binOp<AssocOp>(__shfl_sync(~0u, udata_x[i][0], (tid - 4), 32), local_val);
                }

#pragma unroll
                for (int i = 0; i < row_units + col_units; i++) {
                    float sink;
                    details::SelIntrinsics::selWrite((tidr3 < 1),
                                                       (i < row_units + col_units - 1 ? udata_y[i + 1][0] : sink),
                                                       udata_y[i][0], new_vals[i]);
                }
            }
        } while (0);

        // x4 sync
        do {
            float new_vals[row_units + col_units];
#pragma unroll
            for (int i = 0; i < row_units + col_units; i++) {
                float local_val =
                    (pred1 && (tidr3 < 2) ? (i < row_units + col_units - 1 ? udata_y[i + 1][0] : ZERO) : udata_y[i][0]);
                new_vals[i] = details::binOp<AssocOp>(
                    __shfl_xor_sync(~0u, local_val, (tid ^ (tid >> 3)) & 0x1 ? 25 : 9, 32), local_val);
            }
#pragma unroll
            for (int i = 0; i < row_units + col_units; i++) {
                float sink;
                details::SelIntrinsics::selWrite(pred1 && (tidr3 < 2),
                                                   (i < row_units + col_units - 1 ? udata_y[i + 1][0] : sink),
                                                   udata_y[i][0], new_vals[i]);
            }
        } while (0);

        // x8 down-scan
        do {
            float new_vals[row_units + col_units];
            if constexpr (Dir::value) {
#pragma unroll
                for (int i = 0; i < row_units + col_units; i++) {
                    float local_val = ((tidr3 >> 1) < (tidm4))
                                          ? (i != row_units + col_units - 1 ? udata_y[i + 1][0] : ZERO)
                                          : udata_y[i][0];
                    float remote_val = __shfl_sync(~0u, local_val, tidm4 == 1 ? (tid + 9) : (tid ^ 18), 32);
                    new_vals[i] = details::binOp<AssocOp>(remote_val, local_val);
                }
#pragma unroll
                for (int i = 0; i < row_units + col_units; i++) {
                    float sink;
                    details::SelIntrinsics::selWrite(((tidr3 >> 1) < (tidm4)),
                                                       (i != row_units + col_units - 1 ? udata_y[i + 1][0] : sink),
                                                       udata_y[i][0], new_vals[i]);
                }

            } else {
#pragma unroll
                for (int i = 0; i < row_units + col_units; i++) {
                    float local_val = ((tidr3 >> 1) > (tidm4)) ? (i > 0 ? udata_y[i - 1][0] : ZERO) : udata_y[i][0];
                    float remote_val = __shfl_sync(~0u, local_val, tidm4 == 2 ? (tid - 9) : tid ^ 18, 32);
                    new_vals[i] = details::binOp<AssocOp>(remote_val, local_val);
                }
#pragma unroll
                for (int i = 0; i < row_units + col_units; i++) {
                    float sink;
                    details::SelIntrinsics::selWrite(((tidr3 >> 1) > (tidm4)), (i > 0 ? udata_y[i - 1][0] : sink),
                                                       udata_y[i][0], new_vals[i]);
                }
            }

        } while (0);

        return DiagScanState<ROWS, COLS, Dir, ResultType>::fromUdata(udata_y);
    }

    // ========================================================================
    // Inverse Diagonal Scan (Reference Implementation)
    // ========================================================================

    template <typename AssocOp, float ZERO, bool SKIP_FIX = false, int ROWS, int COLS>
    static __forceinline__ __device__ DiagScanState<ROWS, COLS, details::UpVec, details::Output>
    invDiagScanRef(kt::rt_fl<ROWS, COLS> &dst,
                      const DiagScanState<ROWS, COLS, details::UpVec, details::Input> &init = {{ZERO}, {ZERO}}) {
        constexpr int row_units = ROWS / 8;
        constexpr int col_units = COLS / 8;
        // This procedure perform exclusive parallel scan on tiles with standard
        // MMA layout over diagonal lines with initial values in inversed order
        // (right-bottom to left-top)
        uint32_t tid = threadIdx.x & 0x1f;
        uint32_t tidr3 = tid >> 2;
        uint32_t tidm4 = tid & 0x3;

        float2 data[row_units + 1][col_units];

        details::invCopyData<row_units + 1, ZERO>(data, dst);

        // x2 up-scan
        do {
            float next_vals[row_units + 1][col_units];
#pragma unroll
            for (int i = 0; i < row_units + 1; i++) {
#pragma unroll
                for (int j = 0; j < col_units; j++) {
                    float local_val = (tidr3 < 1) ? (i < row_units ? data[i + 1][j].y : ZERO) : data[i][j].y;
                    next_vals[i][j] = details::binOp<AssocOp>(data[i][j].x, __shfl_sync(~0u, local_val, tid + 4, 32));
                }
            }
#pragma unroll
            for (int i = 0; i < row_units + 1; i++) {
#pragma unroll
                for (int j = 0; j < col_units; j++) {
                    data[i][j].x = next_vals[i][j];
                }
            }
        } while (false);

        // x4 up-scan
        bool pred1 = tid & 0x1;
        do {
            float next_vals[row_units + 1][col_units];
#pragma unroll
            for (int i = 0; i < row_units + 1; i++) {
#pragma unroll
                for (int j = 0; j < col_units; j++) {
                    float local_val = tidr3 < 2 ? (i < row_units ? data[i + 1][j].x : ZERO) : data[i][j].x;
                    float remote_val = __shfl_sync(~0u, local_val, (tid + 9), 32);
                    next_vals[i][j] = details::binOp<AssocOp>(data[i][j].x, remote_val);
                }
            }
#pragma unroll
            for (int i = 0; i < row_units + 1; i++) {
#pragma unroll
                for (int j = 0; j < col_units; j++) {
                    data[i][j].x = (!pred1) ? next_vals[i][j] : data[i][j].x;
                }
            }
        } while (false);

        // x8 up-scan
        bool pred3 = tidm4 & 0x3;
        do {
            float next_vals[row_units + 1][col_units];
#pragma unroll
            for (int i = 0; i < row_units + 1; i++) {
#pragma unroll
                for (int j = 0; j < col_units; j++) {
                    float local_val = tidr3 < 4 ? (i < row_units ? data[i + 1][j].x : ZERO) : data[i][j].x;
                    float remote_val = __shfl_sync(~0u, local_val, (tid + 18), 32);
                    next_vals[i][j] = details::binOp<AssocOp>(data[i][j].x, remote_val);
                }
            }
#pragma unroll
            for (int i = 0; i < row_units + 1; i++) {
#pragma unroll
                for (int j = 0; j < col_units; j++) {
                    data[i][j].x = (!pred3) ? next_vals[i][j] : data[i][j].x;
                }
            }
        } while (false);

        ScanLrState<ROWS, details::UpVec> lr_state;
        ScanTbState<COLS, details::UpVec> tb_state;

// propagation
#pragma unroll
        for (int diag = 0; diag < row_units + col_units; diag++) {
            float current;
            if (diag < row_units) {
                current = init.LrState.Data[diag];
#pragma unroll
                for (int irow = diag, icol = col_units - 1; irow >= 0 && icol >= 0; irow--, icol--) {
                    float next_val = details::binOp<AssocOp>(current, data[irow][icol].x);
                    data[irow][icol].x = (!pred3) ? current : data[irow][icol].x;
                    current = next_val;
                }
            } else {
                current = init.TbState.Data[row_units + col_units - 1 - diag];
#pragma unroll
                for (int irow = row_units, icol = row_units + col_units - 1 - diag; irow >= 0 && icol >= 0;
                     irow--, icol--) {
                    float next_val = details::binOp<AssocOp>(current, data[irow][icol].x);
                    data[irow][icol].x = (!pred3) ? current : data[irow][icol].x;
                    current = next_val;
                }
            }
            if (diag < col_units) {
                tb_state.Data[col_units - 1 - diag] = current;
            } else {
                lr_state.Data[diag - col_units] = current;
            }
        }

        // x8 down-scan
        do {
            float local_vals[row_units + 1][col_units];
            float remote_vals[row_units + 1][col_units];
            float next_vals[row_units + 1][col_units];
#pragma unroll
            for (int i = 0; i < row_units + 1; i++) {
#pragma unroll
                for (int j = 0; j < col_units; j++) {
                    local_vals[i][j] = ((!pred3) && (tidr3 >= 4) ? (i > 0 ? data[i - 1][j].x : ZERO) : data[i][j].x);
                    remote_vals[i][j] = __shfl_xor_sync(~0u, local_vals[i][j], 18, 32);
                    next_vals[i][j] = details::binOp<AssocOp>(local_vals[i][j], remote_vals[i][j]);
                }
            }
#pragma unroll
            for (int i = 0; i < row_units + 1; i++) {
#pragma unroll
                for (int j = 0; j < col_units; j++) {
                    float sink;
                    details::SelIntrinsics::selWrite(
                        (!pred3) && (tidr3 >= 4), (i > 0 ? data[i - 1][j].x : sink), data[i][j].x,
                        (!pred1) ? ((!pred3) ? next_vals[i][j] : remote_vals[i][j]) : local_vals[i][j]);
                }
            }
        } while (0);

        // x4 down-scan
        do {
            float remote_vals[row_units + 1][col_units];
            float next_vals[row_units + 1][col_units];
#pragma unroll
            for (int i = 0; i < row_units + 1; i++) {
#pragma unroll
                for (int j = 0; j < col_units; j++) {
                    float local_val = ((!pred1) && (tidr3 >= 6) ? (i > 0 ? data[i - 1][j].x : ZERO) : data[i][j].x);
                    remote_vals[i][j] = __shfl_xor_sync(~0u, local_val, (tid ^ (tid >> 3)) & 0x1 ? 25 : 9, 32);
                    next_vals[i][j] = details::binOp<AssocOp>(local_val, remote_vals[i][j]);
                }
            }
#pragma unroll
            for (int i = 0; i < row_units + 1; i++) {
#pragma unroll
                for (int j = 0; j < col_units; j++) {
                    float sink = ZERO;
                    details::SelIntrinsics::selWrite((!pred1) && (tidr3 >= 6), (i > 0 ? data[i - 1][j].x : sink),
                                                       data[i][j].x, (!pred1) ? next_vals[i][j] : remote_vals[i][j]);
                }
            }
        } while (0);

        do {
            float next_vals[row_units + 1][col_units];
            float remote_lo_vals[row_units + 1][col_units];
#pragma unroll
            for (int i = 0; i < row_units + 1; i++) {
#pragma unroll
                for (int j = 0; j < col_units; j++) {
                    float sink = ZERO;
                    float local_lo_val = tidr3 == 7 ? (i > 0 ? data[i - 1][j].x : sink) : data[i][j].x;
                    float local_hi_val = data[i][j].y;
                    float remote_hi_val = __shfl_sync(~0u, local_hi_val, (tid + 4), 32);
                    remote_lo_vals[i][j] = __shfl_sync(~0u, local_lo_val, (tid - 4), 32);
                    next_vals[i][j] = details::binOp<AssocOp>(remote_hi_val, local_lo_val);
                }
            }
#pragma unroll
            for (int i = 0; i < row_units + 1; i++) {
#pragma unroll
                for (int j = 0; j < col_units; j++) {
                    float sink;
                    details::SelIntrinsics::selWrite(tidr3 == 7, (i > 0 ? data[i - 1][j].x : sink), data[i][j].x,
                                                       next_vals[i][j]);
                    data[i][j].y = remote_lo_vals[i][j];
                }
            }
        } while (0);

        details::invCopyData<row_units + 1, ZERO>(dst, data);

        details::ScanUtils::propState<details::Output>(lr_state, tb_state);
        return {lr_state, tb_state};
    }

    // ========================================================================
    // Diagonal Scan (Reference Implementation)
    // ========================================================================

    template <typename AssocOp, float ZERO, bool SKIP_FIX = false, bool DEBUG = false, int ROWS, int COLS>
    static __forceinline__ __device__ DiagScanState<ROWS, COLS, details::DownVec, details::Output>
    diagScanRef(kt::rt_fl<ROWS, COLS> &dst,
                  const DiagScanState<ROWS, COLS, details::DownVec, details::Input> &init = {{ZERO}, {ZERO}}) {
        constexpr int row_units = ROWS / 8;
        constexpr int col_units = COLS / 8;
        // This procedure perform exclusive parallel scan on tiles with standard
        // MMA layout over diagonal lines with initial values (left-top to
        // right-bottom)

        uint32_t tid = threadIdx.x & 0x1f;
        uint32_t tidr3 = tid >> 2;
        uint32_t tidm4 = tid & 0x3;

        float2 data[row_units + 1][col_units]; // 2D matrix of (8x8) cells with an extra zero
                                               // line at last row (This means we have ROWS =
                                               // row_units * 8, COLS = col_units * 8)

        details::copyData<row_units + 1, ZERO>(data, dst);

        // x2 up-scan
        do {
            float local_vals[row_units + 1][col_units];
            float remote_vals[row_units + 1][col_units];
            float new_vals[row_units + 1][col_units];
#pragma unroll
            for (int i = 0; i < row_units + 1; i++) {
#pragma unroll
                for (int j = 0; j < col_units; j++) {
                    local_vals[i][j] = tidr3 >= 7 ? (i == 0 ? ZERO : data[i - 1][j].x) : data[i][j].x;
                    remote_vals[i][j] = __shfl_sync(~0u, local_vals[i][j], (tid - 4), 32);
                    new_vals[i][j] = details::binOp<AssocOp>(remote_vals[i][j], data[i][j].y);
                }
            }
#pragma unroll
            for (int i = 0; i < row_units + 1; i++) {
#pragma unroll
                for (int j = 0; j < col_units; j++) {
                    data[i][j].y = new_vals[i][j];
                }
            }
        } while (0);

        // x4 up-scan
        bool pred1 = tid & 0x1;
        do {
            float local_vals[row_units + 1][col_units];
            float remote_vals[row_units + 1][col_units];
            float next_vals[row_units + 1][col_units];
#pragma unroll
            for (int i = 0; i < row_units + 1; i++) {
#pragma unroll
                for (int j = 0; j < col_units; j++) {
                    local_vals[i][j] = tidr3 >= 6 ? (i == 0 ? ZERO : data[i - 1][j].y) : data[i][j].y;
                    remote_vals[i][j] = __shfl_sync(~0u, local_vals[i][j], (tid - 9), 32);
                    next_vals[i][j] = details::binOp<AssocOp>(data[i][j].y, remote_vals[i][j]);
                }
            }
#pragma unroll
            for (int i = 0; i < row_units + 1; i++) {
#pragma unroll
                for (int j = 0; j < col_units; j++) {
                    data[i][j].y = pred1 ? next_vals[i][j] : data[i][j].y;
                }
            }
        } while (0);

        // x8 up-scan
        bool pred3 = (tid & 0x3) == 0x3;
        do {
            float local_vals[row_units + 1][col_units];
            float remote_vals[row_units + 1][col_units];
            float next_vals[row_units + 1][col_units];
#pragma unroll
            for (int i = 0; i < row_units + 1; i++) {
#pragma unroll
                for (int j = 0; j < col_units; j++) {
                    local_vals[i][j] = tidr3 >= 4 ? (i == 0 ? ZERO : data[i - 1][j].y) : data[i][j].y;
                    remote_vals[i][j] = __shfl_sync(~0u, local_vals[i][j], (tid - 18), 32);
                    next_vals[i][j] = details::binOp<AssocOp>(data[i][j].y, remote_vals[i][j]);
                }
            }
#pragma unroll
            for (int i = 0; i < row_units + 1; i++) {
#pragma unroll
                for (int j = 0; j < col_units; j++) {
                    data[i][j].y = pred3 ? next_vals[i][j] : data[i][j].y;
                }
            }
        } while (0);

        ScanLrState<ROWS, details::DownVec> lr_state;
        ScanTbState<COLS, details::DownVec> tb_state;

// propagation
#pragma unroll
        for (int diag = 0; diag < row_units + col_units; diag++) {
            float current;
            if (diag < row_units) {
                current = init.LrState.Data[row_units - diag - 1];
#pragma unroll
                for (int irow = row_units - diag, icol = 0; irow < row_units + 1 && icol < col_units; irow++, icol++) {
                    float next_val = details::binOp<AssocOp>(current, data[irow][icol].y);
                    data[irow][icol].y = pred3 ? current : data[irow][icol].y;
                    current = next_val;
                }
            } else {
                current = init.TbState.Data[diag - row_units];
#pragma unroll
                for (int irow = 0, icol = diag - row_units; irow < row_units + 1 && icol < col_units; irow++, icol++) {
                    float next_val = details::binOp<AssocOp>(current, data[irow][icol].y);
                    data[irow][icol].y = pred3 ? current : data[irow][icol].y;
                    current = next_val;
                }
            }
            if (diag < col_units) {
                tb_state.Data[diag] = current;
            } else {
                lr_state.Data[row_units + col_units - 1 - diag] = current;
            }
        }

        // x8 down-scan
        do {
            float local_vals[row_units + 1][col_units];
            float remote_vals[row_units + 1][col_units];
            float new_vals[row_units + 1][col_units];
#pragma unroll
            for (int i = 0; i < row_units + 1; i++) {
#pragma unroll
                for (int j = 0; j < col_units; j++) {
                    local_vals[i][j] =
                        (pred3 && (tidr3 < 4) ? (i < row_units ? data[i + 1][j].y : ZERO) : data[i][j].y);
                    remote_vals[i][j] = __shfl_xor_sync(~0u, local_vals[i][j], 18, 32);
                    new_vals[i][j] = details::binOp<AssocOp>(remote_vals[i][j], local_vals[i][j]);
                }
            }
#pragma unroll
            for (int i = 0; i < row_units + 1; i++) {
#pragma unroll
                for (int j = 0; j < col_units; j++) {
                    float sink;
                    details::SelIntrinsics::selWrite(
                        pred3 && (tidr3 < 4), (i < row_units ? data[i + 1][j].y : sink), data[i][j].y,
                        pred1 ? (pred3 ? new_vals[i][j] : remote_vals[i][j]) : local_vals[i][j]);
                }
            }
        } while (0);

        // x4 down-scan
        do {
            float local_vals[row_units + 1][col_units];
            float remote_vals[row_units + 1][col_units];
            float new_vals[row_units + 1][col_units];
#pragma unroll
            for (int i = 0; i < row_units + 1; i++) {
#pragma unroll
                for (int j = 0; j < col_units; j++) {
                    local_vals[i][j] =
                        (pred1 && (tidr3 < 2) ? (i < row_units ? data[i + 1][j].y : ZERO) : data[i][j].y);
                    remote_vals[i][j] = __shfl_xor_sync(~0u, local_vals[i][j], (tid ^ (tid >> 3)) & 0x1 ? 25 : 9, 32);
                    new_vals[i][j] = details::binOp<AssocOp>(remote_vals[i][j], local_vals[i][j]);
                }
            }
#pragma unroll
            for (int i = 0; i < row_units + 1; i++) {
#pragma unroll
                for (int j = 0; j < col_units; j++) {
                    float sink;
                    details::SelIntrinsics::selWrite(pred1 && (tidr3 < 2), (i < row_units ? data[i + 1][j].y : sink),
                                                       data[i][j].y, (pred1 ? new_vals[i][j] : remote_vals[i][j]));
                }
            }
        } while (0);

        // x2 down-scan
        do {
            float local_hi_vals[row_units + 1][col_units];
            float local_lo_vals[row_units + 1][col_units];
            float remote_high_vals[row_units + 1][col_units];
            float remote_low_vals[row_units + 1][col_units];
            float new_hi_vals[row_units + 1][col_units];
#pragma unroll
            for (int i = 0; i < row_units + 1; i++) {
#pragma unroll
                for (int j = 0; j < col_units; j++) {
                    local_hi_vals[i][j] = tidr3 == 0 ? (i < row_units ? data[i + 1][j].y : ZERO) : data[i][j].y;
                    local_lo_vals[i][j] = data[i][j].x;
                    remote_high_vals[i][j] = __shfl_sync(~0u, local_hi_vals[i][j], (tid + 4), 32);
                    remote_low_vals[i][j] = __shfl_sync(~0u, local_lo_vals[i][j], (tid - 4), 32);
                    new_hi_vals[i][j] = details::binOp<AssocOp>(remote_low_vals[i][j], local_hi_vals[i][j]);
                }
            }
#pragma unroll
            for (int i = 0; i < row_units + 1; i++) {
#pragma unroll
                for (int j = 0; j < col_units; j++) {
                    float sink;
                    details::SelIntrinsics::selWrite(tidr3 == 0, (i < row_units ? data[i + 1][j].y : sink),
                                                       data[i][j].y, new_hi_vals[i][j]);
                    data[i][j].x = remote_high_vals[i][j];
                }
            }
        } while (0);

        if constexpr (DEBUG)
            kt::print_utils::print(data);

        details::copyData(dst, data);
        return makeState<details::Output, details::DownVec, SKIP_FIX>(lr_state, tb_state);
    }
};

} // namespace diagscan
} // namespace