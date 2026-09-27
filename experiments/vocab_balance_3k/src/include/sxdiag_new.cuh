#pragma once

#include <array>
#include <tuple>
#include <type_traits>
#include <utility>
#include <cstdio>
#include <cuda.h>
#include <cuda_fp16.h>
#include <kittens.cuh>

namespace{

namespace common{
template<typename T>
struct FallbackTuple{ T x, y; };

template<typename T>
using T2Tuple = std::conditional_t<std::is_same_v<T, float>, float2, FallbackTuple<T>>;

__device__  __forceinline__ float2 pack_shfl_sync(const uint32_t mask, const float2 value, const int idx, const int lane = 32){
    return float2 { __shfl_sync(mask, value.x, idx, lane), __shfl_sync(mask, value.y, idx, lane) };
}

__device__  __forceinline__ float2 pack_shfl_xor_sync(const uint32_t mask, const float2 value, const int idx, const int lane = 32){
    return float2 { __shfl_xor_sync(mask, value.x, idx, lane), __shfl_xor_sync(mask, value.y, idx, lane) };
}

__device__  __forceinline__ float pack_shfl_sync(const uint32_t mask, const float value, const int idx, const int lane = 32){
    return __shfl_sync(mask, value, idx, lane);
}

__device__  __forceinline__ float pack_shfl_xor_sync(const uint32_t mask, const float value, const int idx, const int lane = 32){
    return __shfl_xor_sync(mask, value, idx, lane);
}

__device__  __forceinline__ half2 pack_shfl_sync(const uint32_t mask, const half2 value, const int idx, const int lane = 32){
    return __shfl_sync(mask, value, idx, lane);
}

__device__  __forceinline__ half2 pack_shfl_xor_sync(const uint32_t mask, const half2 value, const int idx, const int lane = 32){
    return __shfl_xor_sync(mask, value, idx, lane);
}

template<int A, int B>
static inline constexpr int ct_max_v = (A > B ? A : B);
template<int A, int B>
static inline constexpr int ct_min_v = (A < B ? A : B);
template<int N, class F>
static __host__ __device__ inline void static_for(F&& f) {
    []<size_t... Is>(std::index_sequence<Is...>, F&& f2) {
        (f2(std::integral_constant<int, (int)Is>{}), ...);
    }(std::make_index_sequence<N>{}, (F&&)f);
}

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wformat-security"
template<typename... Ts>
__device__ void print(const char *fmt, Ts&&... args){
    if((threadIdx.x & 0x1f) == 0) printf(fmt, std::forward<Ts>(args)...);
}
#pragma clang diagnostic pop



    template<typename... Ts>
    __forceinline__ __device__ static std::tuple<Ts...> deref(const std::tuple<Ts&...>& src){
        return std::apply([](auto&... xs){ return std::tuple<Ts...>{xs...}; }, src);
    }

    template<typename... Ts>
    __forceinline__ __device__ static void assign(const std::tuple<Ts&...>& dst, const std::tuple<Ts...>& src){
        ([&]<size_t... IDX>(std::index_sequence<IDX...>){
            ((std::get<IDX>(dst) = std::get<IDX>(src)), ...);
        })(std::make_index_sequence<sizeof...(Ts)>{});
    }


} // namespace common

namespace sxdiag{

using common::assign, common::deref;
using common::T2Tuple;
using common::pack_shfl_xor_sync, common::pack_shfl_sync, common::print, common::static_for, common::ct_max_v, common::ct_min_v;

enum class Direction { Forward, Backward };


template<int x_offset, int y_offset, typename T>
__device__ __forceinline__ T2Tuple<T> packed_shfl_sync_pairs(const T2Tuple<T> &data){
    int tid = threadIdx.x;
    if constexpr(x_offset == y_offset){
        return pack_shfl_sync(~0u, data, tid + x_offset, 32);
    } else {
        return { pack_shfl_sync(~0u, data.x, tid + x_offset, 32), pack_shfl_sync(~0u, data.y, tid + y_offset, 32) };
    }
}

template<typename T, int ROWS, int COLS>
struct TightMMABuffer{
    static_assert(COLS >= 32, "Columns must be larger than 32");
    static_assert(COLS % 8 == 0 && ROWS % 8 == 0);

    static constexpr int ROW_UNITS = ROWS / 8;
    static constexpr int COL_UNITS = COLS / 8;
    static constexpr int COL_REG_BITS = __builtin_ffs(COL_UNITS) - 1;

    T2Tuple<T> data[ROW_UNITS][COL_UNITS];

    __forceinline__ __device__ TightMMABuffer(){}
    __forceinline__ __device__ TightMMABuffer(T ZERO){
        #pragma unroll
        for(int i = 0; i < ROW_UNITS; i++){
            #pragma unroll
            for(int j = 0; j < COL_UNITS; j++){
                data[i][j] = T2Tuple<T>{ZERO, ZERO};
            }
        }
    }

    __device__ kittens::rt<T, ROWS, COLS> to_rt() const {
        kittens::rt<T, ROWS, COLS> res;
        #pragma unroll
        for(int i = 0; i < ROW_UNITS; i++){
            #pragma unroll
            for(int j = 0; j < COL_UNITS; j++){
                res.tiles[i / 2][j / 2].data[(i % 2) + 2 * (j % 2)] = data[i][j];
            }
        }
        return res;
    }

    static __forceinline__ __device__ TightMMABuffer from_rt(const kittens::rt<T, ROWS, COLS>& reg_tile){
        TightMMABuffer res;
        #pragma unroll
        for(int i = 0; i < ROW_UNITS; i++){
            #pragma unroll
            for(int j = 0; j < COL_UNITS; j++){
                res.data[i][j] = reg_tile.tiles[i / 2][j / 2].data[(i % 2) + 2 * (j % 2)];
            }
        }
        return res;
    }

    __device__ TightMMABuffer(const T2Tuple<T> (&vals)[ROW_UNITS][COL_UNITS]){
        #pragma unroll
        for(int i = 0; i < ROW_UNITS; i++){
            #pragma unroll
            for(int j = 0; j < COL_UNITS; j++){
                data[i][j] = vals[i][j];
            }
        }
    }

    __device__ static TightMMABuffer fromMMALayout(const T2Tuple<T> (&vals)[ROW_UNITS][COL_UNITS]){
        TightMMABuffer buffer;

        uint32_t tid = threadIdx.x & 0x1f;
        uint32_t tidm4 = tid & 0x3;

        uint32_t j_swap = tidm4 << (COL_REG_BITS - 2);
        #pragma unroll
        for(int i = 0; i < ROW_UNITS; i++){
            #pragma unroll
            for(int j = 0; j < COL_UNITS; j++){
                buffer.data[i][j] = vals[i][j ^ j_swap];
            }
        }
        
        T2Tuple<T> remote_vals[ROW_UNITS][COL_UNITS];
        #pragma unroll
        for(int i = 0; i < ROW_UNITS; i++){
            #pragma unroll
            for(int j = 0; j < COL_UNITS; j++){
                remote_vals[i][j] = pack_shfl_xor_sync(~0u, buffer.data[i][j], j >> (COL_REG_BITS - 2));
            }
        }
        T2Tuple<T> remote_vals2[ROW_UNITS][COL_UNITS];
        #pragma unroll
        for(int i = 0; i < ROW_UNITS; i++){
            #pragma unroll
            for(int j = 0; j < COL_UNITS; j++){
                int dst_idx = (((j << 2) | (j >> (COL_REG_BITS - 2))) % COL_UNITS);
                remote_vals2[i][dst_idx] = remote_vals[i][j ^ j_swap];
            }
        }

        #pragma unroll
        for(int i = 0; i < ROW_UNITS; i++){
            #pragma unroll
            for(int j = 0; j < COL_UNITS; j++){
                buffer.data[i][j].x = (j&1) ? remote_vals2[i][j>>1].y : remote_vals2[i][j>>1].x;
                buffer.data[i][j].y = (j&1) ? remote_vals2[i][(COL_UNITS/2)+(j>>1)].y : remote_vals2[i][(COL_UNITS/2)+(j>>1)].x;
            }
        }

        return buffer;
    }

    __device__ void print_tile(){
        // Print column headers
        print("   "); // Padding for row indices
        for (int c = 0; c < COLS; c++) print("%10d", c);
        print("\n");
        
        // Print data rows
        #pragma unroll
        for (int r = 0; r < ROWS; r++) {
            if(r % 8 == 0){
                print("    +");
                for (int c = 0; c < COLS; c++) {
                    print("---------+");
                }
                print("\n");
            }

            print("%3d |", r); // Row index

            constexpr uint32_t hmask = 1 ^ ((1u << (COL_REG_BITS)) - 1);

            #pragma unroll
            for (int c = 0; c < COLS; c++) {
                T2Tuple<T> val = data[r / 8][(c) % (1u << (COL_REG_BITS))];
                T print_val = pack_shfl_sync(~0u, ((c>>COL_REG_BITS) & 1 ? val.y : val.x), (c >> (COL_REG_BITS + 1)) + (r % 8) * 4);
                if constexpr(std::is_same_v<T, float>) print("%8.4f |", print_val);
                if constexpr(std::is_same_v<T, float2>) print("%8.4f |", print_val.y);
            }
            print("\n");
        }
        print("\n");
    }


    __device__ __forceinline__ TightMMABuffer<T, ROWS + 8, COLS> expand(T ZERO) const {
        TightMMABuffer<T, ROWS + 8, COLS> res;
        uint32_t tid = threadIdx.x & 0x1f;
        uint32_t tidm4 = tid & 0x3;
        uint32_t tidr3 = tid >> 2;

        T2Tuple<T> remote_vals[ROW_UNITS][COL_UNITS];

        #pragma unroll
        for(int r = 0; r < ROW_UNITS; r++){
            static_for<COL_UNITS>([&](auto cType){
                constexpr int c = decltype(cType)::value;
                constexpr int x_shift = (7 - (c % 8));
                constexpr int y_shift = (7 - ((c + COL_UNITS) % 8));
                remote_vals[r][c] = packed_shfl_sync_pairs<-4*x_shift, -4*y_shift, T>(data[r][c]);
            });
        }


        #pragma unroll
        for(int c = 0; c < COL_UNITS; c++){
            uint32_t x_shift = (7 - (c % 8));
            uint32_t y_shift = (7 - ((c + COL_UNITS) % 8));
            #pragma unroll
            for(int r = 0; r < ROW_UNITS + 1; r++){
                res.data[r][c].x = (tidr3 < x_shift) ? (r > 0 ? remote_vals[r - 1][c].x : ZERO) : (r < ROW_UNITS ? remote_vals[r][c].x : ZERO);
                res.data[r][c].y = (tidr3 < y_shift) ? (r > 0 ? remote_vals[r - 1][c].y : ZERO) : (r < ROW_UNITS ? remote_vals[r][c].y : ZERO);
            }
        }

        return res;
    }

    __device__ __forceinline__ TightMMABuffer<T, ROWS, COLS> with_padding(T ZERO) const {
        TightMMABuffer<T, ROWS, COLS> res;

        uint32_t tid = threadIdx.x & 0x1f;
        uint32_t tidm4 = tid & 0x3;
        uint32_t tidr3 = tid >> 2;

        #pragma unroll
        for(int c = 0; c < COL_UNITS; c++){
            uint32_t x_shift = (7 - (c % 8));
            uint32_t y_shift = (7 - ((c + COL_UNITS) % 8));
            #pragma unroll
            for(int r = 0; r < ROW_UNITS; r++){
                res.data[r][c].x = (tidr3 < x_shift) ? (r > 0 ? data[r][c].x : ZERO) : (r < ROW_UNITS - 1 ? data[r][c].x : ZERO);
                res.data[r][c].y = (tidr3 < y_shift) ? (r > 0 ? data[r][c].y : ZERO) : (r < ROW_UNITS - 1 ? data[r][c].y : ZERO);
            }
        }
        return res;
    }

    __device__ __forceinline__ TightMMABuffer<T, ROWS - 8, COLS> fold_after_diag_scan() const {
        TightMMABuffer<T, ROWS - 8, COLS> res;
        uint32_t tid = threadIdx.x & 0x1f;
        uint32_t tidm4 = tid & 0x3;
        uint32_t tidr3 = tid >> 2;

        #pragma unroll
        for(int r = 0; r < ROW_UNITS; r++){
            static_for<COL_UNITS>([&](auto cType){
                constexpr int c = decltype(cType)::value;
                constexpr int x_shift = (7 - (c % 8));
                constexpr int y_shift = (7 - ((c + COL_UNITS) % 8));
                T2Tuple<T> local_val = {
                    (tidr3 < x_shift) ? (data[r + 1][c].x) : data[r][c].x,
                    (tidr3 < y_shift) ? (data[r + 1][c].y) : data[r][c].y
                };
                res.data[r][c] = packed_shfl_sync_pairs<4*x_shift, 4*y_shift, T>(local_val);
            });
        }
        return res;
    }

    __device__ __forceinline__ kittens::rt<T, ROWS - 8, COLS> fold_to_rt() const {
        return fold_after_diag_scan().to_rt();
    }
};


template<int ROWS, typename... Ts>
struct LeftRightVec{
    static constexpr int ROW_UNITS = ROWS / 8;
    std::tuple<Ts...> data[ROW_UNITS];

    __device__ LeftRightVec(){}
    __device__ LeftRightVec(Ts... val){
        #pragma unroll
        for(int i = 0; i < ROW_UNITS; i++) data[i] = {val...};
    }
    __device__ LeftRightVec(const std::tuple<Ts...>& val){
        #pragma unroll
        for(int i = 0; i < ROW_UNITS; i++) data[i] = val;
    }
    __forceinline__ __device__ LeftRightVec& operator=(const std::tuple<Ts...>& val){
        #pragma unroll
        for(int i = 0; i < ROW_UNITS; i++) data[i] = val;
        return *this;
    }
    template<int PANEL>
    __device__ __forceinline__ void set(const std::tuple_element_t<PANEL, std::tuple<Ts...>>& value){
        #pragma unroll
        for(int i = 0; i < ROW_UNITS; i++) {
            std::get<PANEL>(data[i]) = value;
        }
    }
    template<int PANEL, size_t VLEN>
    __device__ __forceinline__ void load_right(kittens::sv_fl<VLEN>& src, int offset){
        uint32_t tid = threadIdx.x & 0x1f;
        uint32_t tidm4 = tid & 0x3;
        uint32_t tidr3 = tid >> 2;
        uint32_t shmem_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&src));
        #pragma unroll
        for(int i = 0; i < ROW_UNITS; i++) {
            if(tidm4 == 0) kittens::move<float>::lds(std::get<PANEL>(data[i]), shmem_ptr + sizeof(float) * (offset + tidr3 + i * 8));
        }
    }

    template<int PANEL, bool INV = false, size_t VLEN>
    __device__ __forceinline__ void load_left(kittens::sv_fl<VLEN>& src, int offset){
        uint32_t tid = threadIdx.x & 0x1f;
        uint32_t tidm4 = tid & 0x3;
        uint32_t tidr3 = (tid >> 2) ^ (INV ? 0x7 : 0x0);
        uint32_t shmem_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&src));
        #pragma unroll
        for(int i = 0; i < ROW_UNITS; i++) {
            int ii = INV ? (ROW_UNITS - 1 - i) : i;
            if(tidm4 == 3) kittens::move<float>::lds(std::get<PANEL>(data[i]), shmem_ptr + sizeof(float) * (offset + tidr3 + ii * 8));
        }
    }

    template<int PANEL, bool RTL_OUT = true>
    __device__ void print(){
        using common::print;
        using T = std::tuple_element_t<PANEL, std::tuple<Ts...>>;
        // Print data rows
        #pragma unroll
        for (int r = 0; r < ROWS; r++) {
            if(r % 8 == 0) print("    +---------+\n");
            
            print("%3d |", r); // Row index

            T val = std::get<PANEL>(data[r / 8]);

            T print_val = pack_shfl_sync(~0u, val, (RTL_OUT ? 3 : 0) + 4 * (r % 8));
            if constexpr(std::is_same_v<T, float>) print("%8.4f |", print_val);
            if constexpr(std::is_same_v<T, float2>) print("%8.4f |", print_val.y);
            print("\n");
        }
        print("\n");
    }
};

template<int ROWS, typename T>
struct SharedLeftRightVec{
    static_assert(ROWS % 8 == 0, "LeftRightVec rows must be a multiple of 8");
    static constexpr int ROW_UNITS = ROWS / 8;
    static_assert(ROW_UNITS % 2 == 0, "LeftRightVec row units must be even for vectorized shared load/store");

    T data[ROW_UNITS * 8];

    __device__ __forceinline__ static int lane_slot(){
        return threadIdx.x & 0x1f;
    }

    __device__ __forceinline__ static int row_slot(){
        return lane_slot() >> 2;
    }

    __device__ __forceinline__ static bool is_forward_lane(){
        return (lane_slot() & 0x3) == 3;
    }

    __device__ __forceinline__ static bool is_backward_lane(){
        return (lane_slot() & 0x3) == 0;
    }

    __device__ __forceinline__ static int smem_index(int logical_row){
        int unit = logical_row / 8;
        int row = logical_row % 8;
        return (unit / 2) * 16 + row * 2 + (unit & 1);
    }

    __device__ __forceinline__ T& at_logical_row(int row){
        return data[smem_index(row)];
    }

    __device__ __forceinline__ const T& at_logical_row(int row) const {
        return data[smem_index(row)];
    }

    __device__ __forceinline__ void store(const LeftRightVec<ROWS, T>& src, bool reverse){
        int r = row_slot();
        uint32_t dst = static_cast<uint32_t>(__cvta_generic_to_shared(&data[0]));
        if(reverse ? is_backward_lane() : is_forward_lane()){
            #pragma unroll
            for(int i = 0; i < ROW_UNITS / 2; i++){
                float2 vals{std::get<0>(src.data[2 * i]), std::get<0>(src.data[2 * i + 1])};
                kittens::move<float2>::sts(dst + sizeof(T) * (2 * r + i * 16), vals);
            }
        }
    }

    __device__ __forceinline__ void store_forward(const LeftRightVec<ROWS, T>& src){ store(src, false); }
    __device__ __forceinline__ void store_backward(const LeftRightVec<ROWS, T>& src){ store(src, true); }

    __device__ __forceinline__ void load(LeftRightVec<ROWS, T>& dst, bool reverse) const {
        int r = row_slot();
        uint32_t src = static_cast<uint32_t>(__cvta_generic_to_shared(&data[0]));
        if(reverse ? is_backward_lane() : is_forward_lane()){
            #pragma unroll
            for(int i = 0; i < ROW_UNITS / 2; i++){
                float2 vals;
                kittens::move<float2>::lds(vals, src + sizeof(T) * (2 * r + i * 16));
                std::get<0>(dst.data[2 * i]) = vals.x;
                std::get<0>(dst.data[2 * i + 1]) = vals.y;
            }
        }
    }

    __device__ __forceinline__ void load_forward(LeftRightVec<ROWS, T>& dst) const { load(dst, false); }
    __device__ __forceinline__ void load_backward(LeftRightVec<ROWS, T>& dst) const { load(dst, true); }

    __device__ __forceinline__ void load(const T *src){
        int lane = lane_slot();
        #pragma unroll
        for(int r = 0; r < ROWS; r += 32){
            if(r + lane < ROWS) at_logical_row(r + lane) = src[r + lane];
        }
    }

    __device__ __forceinline__ void store(T *dst) const {
        int lane = lane_slot();
        #pragma unroll
        for(int r = 0; r < ROWS; r += 32){
            if(r + lane < ROWS) dst[r + lane] = at_logical_row(r + lane);
        }
    }

    template<kittens::ducks::gl::all GL, kittens::ducks::coord::vec COORD=kittens::coord<kittens::sv<T, ROWS>>>
    __device__ __forceinline__ void load(const GL& src, const COORD& idx={}){
        static_assert(std::is_same_v<typename GL::dtype, T>, "SharedLeftRightVec and global layout dtype must match");
        T *src_ptr = (T*)&src[(idx.template unit_coord<-1, 3>())];
        load(src_ptr);
    }

    template<kittens::ducks::gl::all GL, kittens::ducks::coord::vec COORD=kittens::coord<kittens::sv<T, ROWS>>>
    __device__ __forceinline__ void store(GL& dst, const COORD& idx={}) const {
        static_assert(std::is_same_v<typename GL::dtype, T>, "SharedLeftRightVec and global layout dtype must match");
        T *dst_ptr = (T*)&dst[(idx.template unit_coord<-1, 3>())];
        store(dst_ptr);
    }
};

template<int COLS, typename T>
struct SharedTopBottomVec{
    static_assert(COLS % 32 == 0, "TopBottomVec columns must be a multiple of 32");
    static constexpr int TB_UNITS = COLS / 32;

    T data[TB_UNITS * 32];

    __device__ __forceinline__ static int lane_slot(){
        return threadIdx.x & 0x1f;
    }

    __device__ __forceinline__ static int smem_index(int unit, int lane){
        constexpr int UNIT_SWIZZLE = COLS == 64 ? 2 : 1;
        return unit * 32 + ((lane + unit * UNIT_SWIZZLE) & 0x1f);
    }

    __device__ __forceinline__ void store(int unit, const T& val){
        data[smem_index(unit, lane_slot())] = val;
    }

    __device__ __forceinline__ void load(int unit, T& val) const {
        val = data[smem_index(unit, lane_slot())];
    }

    __device__ __forceinline__ void load(const T *src){
        int lane = lane_slot();
        #pragma unroll
        for(int c = 0; c < COLS; c += 32){
            at_logical_col(c + lane) = src[c + lane];
        }
    }

    __device__ __forceinline__ void store(T *dst) const {
        int lane = lane_slot();
        #pragma unroll
        for(int c = 0; c < COLS; c += 32){
            dst[c + lane] = at_logical_col(c + lane);
        }
    }

    template<kittens::ducks::gl::all GL, kittens::ducks::coord::vec COORD=kittens::coord<kittens::sv<T, COLS>>>
    __device__ __forceinline__ void load(const GL& src, const COORD& idx={}){
        static_assert(std::is_same_v<typename GL::dtype, T>, "SharedTopBottomVec and global layout dtype must match");
        T *src_ptr = (T*)&src[(idx.template unit_coord<-1, 3>())];
        load(src_ptr);
    }

    template<kittens::ducks::gl::all GL, kittens::ducks::coord::vec COORD=kittens::coord<kittens::sv<T, COLS>>>
    __device__ __forceinline__ void store(GL& dst, const COORD& idx={}) const {
        static_assert(std::is_same_v<typename GL::dtype, T>, "SharedTopBottomVec and global layout dtype must match");
        T *dst_ptr = (T*)&dst[(idx.template unit_coord<-1, 3>())];
        store(dst_ptr);
    }

    __device__ __forceinline__ auto& at_logical_col(int col){
        int thread_col_idx = col % (COLS / 4);
        int unit = thread_col_idx / 8;
        int lane = ((thread_col_idx & 7) ^ 7) * 4 + (col / (COLS / 4));
        return data[smem_index(unit, lane)];
    }

    __device__ __forceinline__ const auto& at_logical_col(int col) const {
        int thread_col_idx = col % (COLS / 4);
        int unit = thread_col_idx / 8;
        int lane = ((thread_col_idx & 7) ^ 7) * 4 + (col / (COLS / 4));
        return data[smem_index(unit, lane)];
    }
};

template<int COLS, typename... Ts>
struct TopBottomVec{
    static constexpr int TB_UNITS = COLS / 32;
    std::tuple<Ts...> data[TB_UNITS];
    __device__ TopBottomVec(){}
    __device__ TopBottomVec(Ts... val){
        #pragma unroll
        for(int i = 0; i < TB_UNITS; i++) data[i] = {val...};
    }

    __forceinline__ __device__ TopBottomVec& operator=(const std::tuple<Ts...>& val){
        #pragma unroll
        for(int i = 0; i < TB_UNITS; i++) data[i] = val;
        return *this;
    }

    template<int PANEL>
    __device__ __forceinline__ void store(SharedTopBottomVec<COLS, std::tuple_element_t<PANEL, std::tuple<Ts...>>>& dst) const {
        #pragma unroll
        for(int i = 0; i < TB_UNITS; i++) dst.store(i, std::get<PANEL>(data[i]));
    }

    template<int PANEL>
    __device__ __forceinline__ void set(const std::tuple_element_t<PANEL, std::tuple<Ts...>>& value){
        #pragma unroll
        for(int i = 0; i < TB_UNITS; i++) {
            std::get<PANEL>(data[i]) = value;
        }
    }

    template<int PANEL>
    __device__ __forceinline__ void load(const SharedTopBottomVec<COLS, std::tuple_element_t<PANEL, std::tuple<Ts...>>>& src) {
        using T = std::tuple_element_t<PANEL, std::tuple<Ts...>>;
        uint32_t smem_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&src.data[0]));
        #pragma unroll
        for(int i = 0; i < TB_UNITS; i++) {
            int sid = src.smem_index(i, threadIdx.x & 0x1f);
            int offset = (sid) * sizeof(T);
            kittens::move<T>::lds(std::get<PANEL>(data[i]), smem_ptr + offset);
        }
    }

    template<int PANEL>
    __device__ __forceinline__ void print(){
        using common::print;
        using T = std::tuple_element_t<PANEL, std::tuple<Ts...>>;

        constexpr int COL_REG_BITS = __builtin_ffs(COLS/8) - 1;
        // Print column headers
        print("   "); // Padding for row indices
        for (int c = 0; c < COLS; c++) print("%10d", c);
        print("\n");
        print("    +");
        for (int c = 0; c < COLS; c++) {
            print("---------+");
        }
        print("\n");
        // Print data rows
        #pragma unroll
        for (int r = 0; r < 1; r++) {
            print("%3d |", r); // Row index

            constexpr uint32_t hmask = 1 ^ ((1u << (COL_REG_BITS)) - 1);

            #pragma unroll
            for (int c = 0; c < COLS; c++) {
                int thread_col_idx = c % (COLS / 4);
                T val = std::get<PANEL>(data[(thread_col_idx/8)]);
                T print_val = pack_shfl_sync(~0u, val, ((thread_col_idx%8)^7)*4 + (c/(COLS / 4)));
                //T print_val = pack_shfl_sync(~0u, ((c>>COL_REG_BITS) & 1 ? val.y : val.x), (c >> (COL_REG_BITS + 1)) + (r % 8) * 4);
                if constexpr(std::is_same_v<T, float>) print("%8.4f |", print_val);
                if constexpr(std::is_same_v<T, float2>) print("%8.4f |", print_val.y);
            }
            print("\n");
        }
        print("    +");
        for (int c = 0; c < COLS; c++) {
            print("---------+");
        }
        print("\n");
    }
};

template<int ROWS, int COLS, typename... Ts>
struct StatePair{
    LeftRightVec<ROWS, Ts...> lr;
    TopBottomVec<COLS, Ts...> tb;

    template<typename TOp>
    static __device__ __forceinline__ StatePair zero(){
        std::tuple<Ts...> z = TOp::template get_zero<std::tuple<Ts...>>();
        return from_tuple(z, z);
    }

    template<typename TOp>
    static __device__ __forceinline__ StatePair from_lr(const LeftRightVec<ROWS, Ts...>& lr_){
        std::tuple<Ts...> z = TOp::template get_zero<std::tuple<Ts...>>();
        StatePair res;
        res.lr = lr_;
        fill_tb(res.tb, z);
        return res;
    }

    template<typename TOp>
    static __device__ __forceinline__ StatePair from_tb(const TopBottomVec<COLS, Ts...>& tb_){
        std::tuple<Ts...> z = TOp::template get_zero<std::tuple<Ts...>>();
        StatePair res;
        fill_lr(res.lr, z);
        res.tb = tb_;
        return res;
    }

    static __device__ __forceinline__ StatePair from(const LeftRightVec<ROWS, Ts...>& lr_, const TopBottomVec<COLS, Ts...>& tb_){
        return {lr_, tb_};
    }

private:
    static __device__ __forceinline__ StatePair from_tuple(const std::tuple<Ts...>& lr_zero, const std::tuple<Ts...>& tb_zero){
        StatePair res;
        fill_lr(res.lr, lr_zero);
        fill_tb(res.tb, tb_zero);
        return res;
    }

    static __device__ __forceinline__ void fill_lr(LeftRightVec<ROWS, Ts...>& lr_, const std::tuple<Ts...>& val){
        #pragma unroll
        for(int i = 0; i < LeftRightVec<ROWS, Ts...>::ROW_UNITS; i++) lr_.data[i] = val;
    }

    static __device__ __forceinline__ void fill_tb(TopBottomVec<COLS, Ts...>& tb_, const std::tuple<Ts...>& val){
        #pragma unroll
        for(int i = 0; i < TopBottomVec<COLS, Ts...>::TB_UNITS; i++) tb_.data[i] = val;
    }
};


template<typename TOp>
struct ScanPairHelpers {
private:
    template<size_t Start, size_t Stride, size_t N, typename F, size_t... Is>
    __forceinline__ __device__ static void for_indices_impl(F&& f, std::index_sequence<Is...>) {
        (f(std::integral_constant<size_t, Start + Is * Stride>{}), ...);
    }

    template<size_t Start, size_t Stride, size_t N, typename F>
    __forceinline__ __device__ static void for_indices(F&& f) {
        if constexpr (Start < N) {
            constexpr size_t count = ((N - 1 - Start) / Stride) + 1;
            for_indices_impl<Start, Stride, N>(std::forward<F>(f),
                                               std::make_index_sequence<count>{});
        }
    }

    template<size_t STEP, typename... Ts, size_t N>
    __forceinline__ __device__ static void upsweep(const std::array<std::tuple<Ts&...>, N>& a) {
        if constexpr (STEP < N) {
            for_indices<2 * STEP - 1, 2 * STEP, N>([&](auto i) {
                constexpr size_t idx = i;
                assign(a[idx], TOp::op(deref(a[idx - STEP]), deref(a[idx])));
            });
            upsweep<STEP * 2>(a);
        }
    }

    template<size_t STEP, typename... Ts, size_t N>
    __forceinline__ __device__ static void upsweep_refs(std::array<std::tuple<Ts...>, N>& a) {
        if constexpr (STEP < N) {
            for_indices<2 * STEP - 1, 2 * STEP, N>([&](auto i) {
                constexpr size_t idx = i;
                a[idx] = TOp::op(a[idx - STEP], a[idx]);
            });
            upsweep_refs<STEP * 2>(a);
        }
    }

    template<size_t STEP, typename... Ts, size_t N>
    __forceinline__ __device__ static void downsweep(const std::array<std::tuple<Ts&...>, N>& a) {
        if constexpr (STEP >= 1) {
            for_indices<2 * STEP - 1, 2 * STEP, N>([&](auto i) {
                constexpr size_t idx = i;
                auto lhs = deref(a[idx - STEP]);
                auto rhs = deref(a[idx]);
                assign(a[idx - STEP], rhs);
                assign(a[idx], TOp::op(rhs, lhs)); // NOTE: high slots contains initial values
            });
            if constexpr (STEP > 1) downsweep<STEP / 2>(a);
        }
    }

    template<typename T>
    __forceinline__ __device__ static T deref_val(T& ref) { return ref; }

public:
    template<typename... Ts, size_t N>
    __forceinline__ __device__ static auto upsweep_only(const std::array<std::tuple<Ts&...>, N>& a) {
        static_assert(N > 0, "Sequence empty");
        static_assert((N & (N - 1)) == 0, "vals count must be power of two");

        upsweep<1>(a);

        auto x_vals = std::apply([](auto&... vs){ return std::make_tuple(deref_val(vs.x)...); }, a[N-1]);
        auto y_vals = std::apply([](auto&... vs){ return std::make_tuple(deref_val(vs.y)...); }, a[N-1]);
        auto res = TOp::op(x_vals, y_vals);

        ([&]<size_t... IDX>(std::index_sequence<IDX...>){
            ((std::get<IDX>(a[N-1]).y = std::get<IDX>(res)), ...);
        })(std::make_index_sequence<sizeof...(Ts)>{});
        
        return res;
    }

    template<typename... Ts, size_t N>
    __forceinline__ __device__ static auto upsweep_only_v2(const std::array<std::tuple<Ts&...>, N>& a) {
        static_assert(N > 0, "Sequence empty");
        static_assert((N & (N - 1)) == 0, "vals count must be power of two");
        upsweep<1>(a);
        return a[N-1];
    }

    template<typename... Ts, size_t N>
    __forceinline__ __device__ static void downsweep_only_v2(const std::array<std::tuple<Ts&...>, N>& a) {
        static_assert((N & (N - 1)) == 0, "vals count must be power of two");
        downsweep<N / 2>(a);
    }

    template<typename... Ts, size_t N>
    __forceinline__ __device__ static auto reduce_strict_assoc(const std::array<std::tuple<Ts&...>, N>& a_) {
        static_assert((N & (N - 1)) == 0, "vals count must be power of two");
        std::array<std::tuple<Ts...>, N> a;
        #pragma unroll
        for(size_t i = 0; i < N; i++) a[i] = deref(a_[i]);
        upsweep_refs<1>(a);
        return a[N-1];
    }
};


template<int ROWS, int COLS, typename... Ts>
struct ScanTile{
    static constexpr int ROW_UNITS = ROWS / 8;
    static constexpr int COL_UNITS = COLS / 8;
    static constexpr int TB_UNITS = COLS / 32;

    std::tuple<TightMMABuffer<Ts, ROWS + 8, COLS>...> buffers;

    template<typename... TRT>
    __device__ ScanTile(const TightMMABuffer<TRT, ROWS + 8, COLS>&... src): buffers(src...){
    }

    template<int R, int C>
    __forceinline__ __device__ std::tuple<T2Tuple<Ts>&...> extract_buffer(){
        return std::apply([&](auto&... xs){ return std::tuple<T2Tuple<Ts>&...>{ xs.data[R][C]... }; }, buffers);
    }
    template<int R, int C>
    __forceinline__ __device__ std::tuple<Ts&...> extract_buffer_x(){
        return std::apply([&](auto&... xs){ return std::tuple<Ts&...>{ xs.data[R][C].x... }; }, buffers);
    }
    template<int R, int C>
    __forceinline__ __device__ std::tuple<Ts&...> extract_buffer_y(){
        return std::apply([&](auto&... xs){ return std::tuple<Ts&...>{ xs.data[R][C].y... }; }, buffers);
    }

    template<bool REVERSE, bool MUTATE, typename TOp>
    __forceinline__ __device__ StatePair<ROWS, COLS, Ts...> diagScanOrReduce(const StatePair<ROWS, COLS, Ts...>& init){
        uint32_t tid = threadIdx.x & 0x1f;
        uint32_t tidm4 = tid & 0x3;
        uint32_t target_x1 = (tid & 0b11100) | ((tidm4 + (REVERSE ? 1 : -1)) & 0x3);
        uint32_t target_x2 = (tid & 0b11100) | ((tidm4 + (REVERSE ? 2 : -2)) & 0x3);

        std::tuple<Ts...> reduce_tmp[ROW_UNITS + TB_UNITS];

        static constexpr int HALF_PHASE_UNITS = ((TB_UNITS+1)/2);
        static constexpr int HALF_PHASE_UNITS_FLOOR = ((TB_UNITS)/2);
        static constexpr int HALF_PHASE_SLOTS = TB_UNITS * 4;

        static_for<ROW_UNITS + HALF_PHASE_UNITS>([&](auto diagC){
            static constexpr int DIAG = decltype(diagC)::value;
            static constexpr int ROW_START = REVERSE
                ? ROW_UNITS - ct_max_v<0, DIAG - (HALF_PHASE_UNITS - 1)>
                : ct_max_v<0, DIAG - (HALF_PHASE_UNITS - 1)>;
            static constexpr int COL_START = REVERSE
                ? ct_min_v<DIAG, HALF_PHASE_UNITS - 1>
                : ct_max_v<0, (HALF_PHASE_UNITS - 1) - DIAG>;
            static constexpr int N_SLOTS = REVERSE
                ? ct_min_v<8 * (ROW_START + 1), HALF_PHASE_SLOTS - COL_START * 8>
                : ct_min_v<8 * (ROW_UNITS + 1 - ROW_START), HALF_PHASE_SLOTS - COL_START * 8>;

            using ArrayType = std::array<std::tuple<T2Tuple<Ts>&...>, N_SLOTS>;
            auto res = ([&]<size_t... Is>(std::index_sequence<Is...>){
                auto refs = [&]() {
                    if constexpr(REVERSE) {
                        return ArrayType{ extract_buffer<ROW_START - ((N_SLOTS - 1 - Is) / 8), COL_START * 8 + (N_SLOTS - 1 - Is)>()... };
                    } else {
                        return ArrayType{ extract_buffer<ROW_START + (Is / 8), COL_START * 8 + Is>()... };
                    }
                }();
                if constexpr(MUTATE) return ScanPairHelpers<TOp>::upsweep_only_v2(refs);
                else return ScanPairHelpers<TOp>::reduce_strict_assoc(refs);
            })(std::make_index_sequence<N_SLOTS>{});

            if constexpr(REVERSE) {
                auto x_part = std::apply([](auto... xs){ return std::tuple<Ts...>{xs.x...}; }, res);
                reduce_tmp[DIAG + HALF_PHASE_UNITS_FLOOR] = std::apply([](auto... xs){ return std::tuple<Ts...>{xs.y...}; }, res);
                if(DIAG >= HALF_PHASE_UNITS_FLOOR) {
                    reduce_tmp[DIAG] = TOp::op(reduce_tmp[DIAG], x_part);
                } else {
                    reduce_tmp[DIAG] = x_part;
                }
            } else {
                reduce_tmp[DIAG + HALF_PHASE_UNITS_FLOOR] = std::apply([](auto... xs){ return std::tuple<Ts...>{xs.x...}; }, res);
                auto y_part = std::apply([](auto... xs){ return std::tuple<Ts...>{xs.y...}; }, res);
                if(DIAG >= HALF_PHASE_UNITS_FLOOR) {
                    reduce_tmp[DIAG] = TOp::op(reduce_tmp[DIAG], y_part);
                } else {
                    reduce_tmp[DIAG] = y_part;
                }
            }
        });

        std::tuple<Ts...> ZERO = TOp::template get_zero<std::tuple<Ts...>>();
        std::tuple<Ts...> remote_vals[ROW_UNITS];
        std::tuple<Ts...> local_backup[ROW_UNITS + TB_UNITS];
        TopBottomVec<COLS, Ts...> final_tb;
        LeftRightVec<ROWS, Ts...> final_lr;

        do{
            #pragma unroll
            for(int i = 0; i < ROW_UNITS; i++){
                bool takes_init_lr = REVERSE ? (tidm4 == 0) : (tidm4 == 3);
                std::tuple<Ts...> local_val = takes_init_lr ? init.lr.data[REVERSE ? (ROW_UNITS - 1 - i) : i] : reduce_tmp[i];
                remote_vals[i] = std::apply([=](auto&... xs){ return std::tuple<Ts...>{ pack_shfl_sync(~0u, xs, target_x1)... }; }, local_val);
            }
            #pragma unroll
            for(int i = 0; i < TB_UNITS; i++){
                local_backup[i] = reduce_tmp[i];
                reduce_tmp[i] = REVERSE ? init.tb.data[i] : init.tb.data[TB_UNITS - 1 - i];
            }
            #pragma unroll
            for(int i = 0; i < ROW_UNITS; i++){
                local_backup[i + TB_UNITS] = reduce_tmp[i + TB_UNITS];
                reduce_tmp[i + TB_UNITS] = remote_vals[i];
            }
        } while(0);

        do{
            #pragma unroll
            for(int i = 0; i < ROW_UNITS; i++){
                bool outside = REVERSE ? (tidm4 <= 0) : (tidm4 >= 3);
                std::tuple<Ts...> local_val = outside ? ZERO : reduce_tmp[i];
                remote_vals[i] = std::apply([=](auto&... xs){ return std::tuple<Ts...>{ pack_shfl_sync(~0u, xs, target_x1)... }; }, local_val);
            }
            #pragma unroll
            for(int i = 0; i < ROW_UNITS; i++){
                reduce_tmp[i + TB_UNITS] = TOp::op(remote_vals[i], reduce_tmp[i + TB_UNITS]);
            }
        } while(0);

        do{
            #pragma unroll
            for(int i = 0; i < ROW_UNITS - TB_UNITS; i++){
                bool outside = REVERSE ? (tidm4 <= 1) : (tidm4 >= 2);
                std::tuple<Ts...> local_val = outside ? ZERO : reduce_tmp[i];
                remote_vals[i] = std::apply([=](auto&... xs){ return std::tuple<Ts...>{ pack_shfl_sync(~0u, xs, target_x2)... }; }, local_val);
            }
            #pragma unroll
            for(int i = 0; i < ROW_UNITS - TB_UNITS; i++){
                reduce_tmp[i + 2 * TB_UNITS] = TOp::op(remote_vals[i], reduce_tmp[i + 2 * TB_UNITS]);
            }
        } while(0);

        #pragma unroll
        for(int i = 0; i < ROW_UNITS; i++) final_lr.data[REVERSE ? (ROW_UNITS - 1 - i) : i] = TOp::op(reduce_tmp[i], local_backup[i]);
        #pragma unroll
        for(int i = 0; i < TB_UNITS; i++){
            if constexpr(REVERSE) final_tb.data[i] = TOp::op(reduce_tmp[i + ROW_UNITS], local_backup[i + ROW_UNITS]);
            else final_tb.data[TB_UNITS - 1 - i] = TOp::op(reduce_tmp[i + ROW_UNITS], local_backup[i + ROW_UNITS]);
        }

        if constexpr(MUTATE) {
            static_for<ROW_UNITS + HALF_PHASE_UNITS>([&](auto diagC){
                static constexpr int DIAG = decltype(diagC)::value;
                static constexpr int ROW_START = REVERSE
                    ? ROW_UNITS - ct_max_v<0, DIAG - (HALF_PHASE_UNITS - 1)>
                    : ct_max_v<0, DIAG - (HALF_PHASE_UNITS - 1)>;
                static constexpr int COL_START = REVERSE
                    ? ct_min_v<DIAG, HALF_PHASE_UNITS - 1>
                    : ct_max_v<0, (HALF_PHASE_UNITS - 1) - DIAG>;
                static constexpr int N_SLOTS = REVERSE
                    ? ct_min_v<8 * (ROW_START + 1), HALF_PHASE_SLOTS - COL_START * 8>
                    : ct_min_v<8 * (ROW_UNITS + 1 - ROW_START), HALF_PHASE_SLOTS - COL_START * 8>;

                if constexpr(REVERSE) {
                    auto y_refs = extract_buffer_y<ROW_START, COL_START * 8>();
                    auto y_inits = reduce_tmp[DIAG + HALF_PHASE_UNITS_FLOOR];
                    if constexpr(DIAG + HALF_PHASE_UNITS_FLOOR < ROW_UNITS + HALF_PHASE_UNITS){
                        reduce_tmp[DIAG + HALF_PHASE_UNITS_FLOOR] = TOp::op(y_inits, deref(y_refs));
                    }
                    assign(y_refs, y_inits);

                    auto x_refs = extract_buffer_x<ROW_START, COL_START * 8>();
                    assign(x_refs, reduce_tmp[DIAG]);

                    using ArrayType = std::array<std::tuple<T2Tuple<Ts>&...>, N_SLOTS>;
                    ([&]<size_t... Is>(std::index_sequence<Is...>){
                        ScanPairHelpers<TOp>::downsweep_only_v2(
                            ArrayType{ extract_buffer<ROW_START - ((N_SLOTS - 1 - Is) / 8), COL_START * 8 + (N_SLOTS - 1 - Is)>()... }
                        );
                    })(std::make_index_sequence<N_SLOTS>{});
                } else {
                    auto x_refs = extract_buffer_x<ROW_START + ((N_SLOTS-1) / 8), COL_START * 8 + (N_SLOTS-1)>();
                    auto x_inits = reduce_tmp[DIAG + HALF_PHASE_UNITS_FLOOR];
                    if constexpr(DIAG + HALF_PHASE_UNITS_FLOOR < ROW_UNITS + HALF_PHASE_UNITS){
                        reduce_tmp[DIAG + HALF_PHASE_UNITS_FLOOR] = TOp::op(x_inits, deref(x_refs));
                    }
                    assign(x_refs, x_inits);
                    auto y_refs = extract_buffer_y<ROW_START + ((N_SLOTS-1) / 8), COL_START * 8 + (N_SLOTS-1)>();
                    assign(y_refs, reduce_tmp[DIAG]); 

                    using ArrayType = std::array<std::tuple<T2Tuple<Ts>&...>, N_SLOTS>;
                    ([&]<size_t... Is>(std::index_sequence<Is...>){
                        ScanPairHelpers<TOp>::downsweep_only_v2(
                            ArrayType{ extract_buffer<ROW_START + (Is / 8), COL_START * 8 + Is>()... }
                        );
                    })(std::make_index_sequence<N_SLOTS>{});
                }
            });
        }

        return {final_lr, final_tb};
    }

    template<Direction DIR, typename TOp>
    __forceinline__ __device__ StatePair<ROWS, COLS, Ts...> scan(const StatePair<ROWS, COLS, Ts...>& init){
        return diagScanOrReduce<DIR == Direction::Backward, true, TOp>(init);
    }

    template<Direction DIR, typename TOp>
    __forceinline__ __device__ StatePair<ROWS, COLS, Ts...> reduce(const StatePair<ROWS, COLS, Ts...>& init){
        return diagScanOrReduce<DIR == Direction::Backward, false, TOp>(init);
    }

    template<int PANEL>
    __forceinline__ __device__ auto fold_panel() const {
        return std::get<PANEL>(buffers).fold_after_diag_scan();
    }

    template<int PANEL>
    __forceinline__ __device__ auto fold_panel_to_rt() const {
        return fold_panel<PANEL>().to_rt();
    }


};

template<int ROWS, int COLS, typename... Ts>
__forceinline__ __device__ auto make_scan_tile(const TightMMABuffer<Ts, ROWS + 8, COLS>&... buffers){
    return ScanTile<ROWS, COLS, Ts...>(buffers...);
}

} // namespace sxdiag


} // namespace