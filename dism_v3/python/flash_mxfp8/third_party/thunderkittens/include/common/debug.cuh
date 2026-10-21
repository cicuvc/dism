#pragma once

#include <kittens.cuh>

// Reset
#define TK_RESET "\033[0m"

// Foreground colors
#define TK_FG_BLACK "\033[30m"
#define TK_FG_RED "\033[31m"
#define TK_FG_GREEN "\033[32m"
#define TK_FG_YELLOW "\033[33m"
#define TK_FG_BLUE "\033[34m"
#define TK_FG_MAGENTA "\033[35m"
#define TK_FG_CYAN "\033[36m"
#define TK_FG_WHITE "\033[37m"

// Background colors
#define TK_BG_BLACK "\033[40m"
#define TK_BG_RED "\033[41m"
#define TK_BG_GREEN "\033[42m"
#define TK_BG_YELLOW "\033[43m"
#define TK_BG_BLUE "\033[44m"
#define TK_BG_MAGENTA "\033[45m"
#define TK_BG_CYAN "\033[46m"
#define TK_BG_WHITE "\033[47m"

// Bright foreground colors
#define TK_FG_BRIGHT_BLACK "\033[90m"
#define TK_FG_BRIGHT_RED "\033[91m"
#define TK_FG_BRIGHT_GREEN "\033[92m"
#define TK_FG_BRIGHT_YELLOW "\033[93m"
#define TK_FG_BRIGHT_BLUE "\033[94m"
#define TK_FG_BRIGHT_MAGENTA "\033[95m"
#define TK_FG_BRIGHT_CYAN "\033[96m"
#define TK_FG_BRIGHT_WHITE "\033[97m"

// Bright background colors
#define TK_BG_BRIGHT_BLACK "\033[100m"
#define TK_BG_BRIGHT_RED "\033[101m"
#define TK_BG_BRIGHT_GREEN "\033[102m"
#define TK_BG_BRIGHT_YELLOW "\033[103m"
#define TK_BG_BRIGHT_BLUE "\033[104m"
#define TK_BG_BRIGHT_MAGENTA "\033[105m"
#define TK_BG_BRIGHT_CYAN "\033[106m"
#define TK_BG_BRIGHT_WHITE "\033[107m"

// Text styles
#define TK_BOLD "\033[1m"
#define TK_DIM "\033[2m"
#define TK_ITALIC "\033[3m"
#define TK_UNDERLINE "\033[4m"
#define TK_BLINK "\033[5m"
#define TK_REVERSE "\033[7m"
#define TK_HIDDEN "\033[8m"

// Macro to combine styles
#define TK_STYLE(...) "\033[" #__VA_ARGS__ "m"

namespace kittens {

template<int N_WARPS = 1>
struct print_utils_group {
using group = kittens::group<N_WARPS>;

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wformat-security"
    template <typename... Ts>
    static __device__ void print(const char *fmt, Ts &&...args) {
        if (group::laneid() == 0)
            printf(fmt, std::forward<Ts>(args)...);
    }
#pragma clang diagnostic pop

    template <int RS, int CS>
    static __forceinline__ __device__ void print(const float (&src)[RS][CS]) {
        print("   "); // Padding for row indices
        for (int c = 0; c < 8 * CS; c++)
            print("%10d", c);
        print("\n");
#pragma unroll
        for (int r = 0; r < 8 * RS; r++) {
            if (r % 8 == 0) {
                print("    +");
                for (int c = 0; c < 8 * CS; c++) {
                    print("---------+");
                }
                print("\n");
            }

            print("%3d |", r); // Row index
#pragma unroll
            for (int c = 0; c < 8 * CS; c++) {
                uint32_t target_tid = ((r % 8) << 2) + ((c / 2) % 4);
                float local_val = c & 1 ? (src[r / 8][c / 8]) : 0.f;
                float remote_val = __shfl_sync(~0u, local_val, target_tid, 32);
                print("%8.4f |", remote_val);
            }
            print("\n");
        }
        print("\n");
    }

    template <int RS, int CS>
    static __forceinline__ __device__ void print(const float2 (&src)[RS][CS]) {
        // Print column headers
        print("   "); // Padding for row indices
        for (int c = 0; c < 8 * CS; c++)
            print("%10d", c);
        print("\n");

// Print data rows
#pragma unroll
        for (int r = 0; r < 8 * RS; r++) {
            if (r % 8 == 0) {
                print("    +");
                for (int c = 0; c < 8 * CS; c++) {
                    print("---------+");
                }
                print("\n");
            }

            print("%3d |", r); // Row index
#pragma unroll
            for (int c = 0; c < 8 * CS; c++) {
                uint32_t target_tid = ((r % 8) << 2) + ((c / 2) % 4);
                float local_val = c & 1 ? (src[r / 8][c / 8].y) : (src[r / 8][c / 8].x);
                float remote_val = __shfl_sync(~0u, local_val, target_tid, 32);
                print("%8.4f |", remote_val);
            }
            print("\n");
        }
        print("\n");
    }

    template <int colsep_interval = 1, typename T, int RV, int CV, ducks::rt_layout::all L>
    static __forceinline__ __device__ void print(const kittens::rt<T, RV, CV, L> &src) {
        static_assert(colsep_interval > 0, "Column separator interval must be positive.");
        using print_warp = print_utils_group<1>;

        #pragma unroll 1
        for(int warp = 0; warp < N_WARPS; warp++){
            if(group::warpid() == warp){
                print_warp::print("Data from warp %u\n", warp);

                print_warp::print("     ");
                for (int c = 0; c < CV; c++) print_warp::print(" %9d", c);
                print_warp::print("\n      ");
                for (int c = 0; c < CV; c++)
                    print_warp::print("%s─────────",
                                      c % colsep_interval == 0 ? (c ? "┬" : "┌") : "─");
                print_warp::print("┐\n");

                #pragma unroll
                for (int r = 0; r < RV; r++) {
                    print_warp::print("%4d ", r);
                    #pragma unroll
                    for (int c = 0; c < CV; c++) {
                        int tile_r = r / 16, tile_c = c / 16;
                        int local_r = r % 16, local_c = c % 16;
                        int d;
                        uint32_t target_tid;
                        T local_val;
                        if constexpr (std::is_same_v<L, ducks::rt_layout::row>) {
                            d = (local_c / 8) * 2 + local_r / 8;
                            target_tid = (local_r % 8) * 4 + (local_c / 2) % 4;
                            local_val = local_c & 1 ? src.tiles[tile_r][tile_c].data[d].y
                                                  : src.tiles[tile_r][tile_c].data[d].x;
                        } else {
                            d = (local_r / 8) * 2 + local_c / 8;
                            target_tid = (local_c % 8) * 4 + (local_r / 2) % 4;
                            local_val = local_r & 1 ? src.tiles[tile_r][tile_c].data[d].y
                                                  : src.tiles[tile_r][tile_c].data[d].x;
                        }
                        T remote_val = __shfl_sync(~0u, local_val, target_tid, 32);
                        print_warp::print(c % colsep_interval == 0 ? " │" : "  ");
                        if constexpr (std::is_same_v<T, float>)
                            print_warp::print("%8.3f", remote_val);
                        if constexpr (std::is_same_v<T, half>)
                            print_warp::print("%8.3f", __half2float(remote_val));
                        if constexpr (std::is_same_v<T, kittens::bf16>)
                            print_warp::print("%8.3f", __bfloat162float(remote_val));
                        if constexpr (std::is_same_v<T, int>)
                            print_warp::print("%8d", remote_val);
                    }
                    print_warp::print(" │\n");
                    if (r % 8 == 7 || r == RV - 1) {
                        print_warp::print("      ");
                        for (int c = 0; c < CV; c++) {
                            const bool boundary = c % colsep_interval == 0;
                            if (r == RV - 1)
                                print_warp::print("%s─────────", boundary ? (c ? "┴" : "└") : "─");
                            else
                                print_warp::print("%s─────────", boundary ? (c ? "┼" : "├") : "─");
                        }
                        print_warp::print("%s\n", r == RV - 1 ? "┘" : "┤");
                    }
                }
            }
            group::sync(group::groupid() + 2);
        }
    }

    template <int colsep_interval = 1, typename T, size_t CV, ducks::rv_layout::all L>
    static __forceinline__ __device__ void print(const kittens::rv<T, CV, L> &src) {
        static_assert(colsep_interval > 0, "Column separator interval must be positive.");
        using print_warp = print_utils_group<1>;

        #pragma unroll 1
        for (int warp = 0; warp < N_WARPS; warp++) {
            if (group::warpid() == warp) {
                print_warp::print("Data from warp %u\n", warp);
                print_warp::print("     ");
                for (int c = 0; c < int(CV); c++) print_warp::print(" %9d", c);
                print_warp::print("\n      ");
                for (int c = 0; c < int(CV); c++)
                    print_warp::print("%s─────────",
                                      c % colsep_interval == 0 ? (c ? "┬" : "┌") : "─");
                print_warp::print("┐\n%4d ", 0);

                #pragma unroll
                for (int c = 0; c < int(CV); c++) {
                    T local_val;
                    uint32_t target_tid;
                    if constexpr (std::is_same_v<L, ducks::rv_layout::naive>) {
                        local_val = src.data[c / 32][0];
                        target_tid = c % 32;
                    } else if constexpr (std::is_same_v<L, ducks::rv_layout::align>) {
                        const int local_c = c % 16;
                        const auto packed = src.data[c / 16][local_c / 8];
                        local_val = local_c & 1 ? packed.y : packed.x;
                        target_tid = (local_c / 2) % 4 + 4 * (local_c % 2);
                    } else {
                        const int local_c = c % 16;
                        const auto packed = src.data[c / 16][0];
                        local_val = local_c / 8 ? packed.y : packed.x;
                        target_tid = (local_c * 4) % 32;
                    }
                    T remote_val = __shfl_sync(~0u, local_val, target_tid, 32);
                    print_warp::print(c % colsep_interval == 0 ? " │" : "  ");
                    if constexpr (std::is_same_v<T, float>)
                        print_warp::print("%8.3f", remote_val);
                    if constexpr (std::is_same_v<T, half>)
                        print_warp::print("%8.3f", __half2float(remote_val));
                    if constexpr (std::is_same_v<T, kittens::bf16>)
                        print_warp::print("%8.3f", __bfloat162float(remote_val));
                    if constexpr (std::is_same_v<T, int>)
                        print_warp::print("%8d", remote_val);
                }
                print_warp::print(" │\n      ");
                for (int c = 0; c < int(CV); c++)
                    print_warp::print("%s─────────",
                                      c % colsep_interval == 0 ? (c ? "┴" : "└") : "─");
                print_warp::print("┘\n");
            }
            group::sync(group::groupid() + 2);
        }
    }

    template <int colsep_interval = 1, typename T, size_t CV>
    static __forceinline__ __device__ void print(const kittens::sv<T, CV> &src, const char *prompt = nullptr) {
        static_assert(colsep_interval > 0, "Column separator interval must be positive.");
        if (prompt != nullptr)
            print("%s", prompt);
        if (group::laneid() == 0) ::kittens::print<colsep_interval>(src);
    }

    template <int colsep_interval = 1, typename T, int RV, int CV>
    static __forceinline__ __device__ void print(const kittens::st<T, RV, CV> &src, const char *prompt = nullptr) {
        static_assert(colsep_interval > 0, "Column separator interval must be positive.");
        if (prompt != nullptr)
            print("%s", prompt);
        if (group::laneid() == 0) ::kittens::print<colsep_interval>(src);
    }

    template <typename T, int RV, int CV>
    static __forceinline__ __device__ void print_raw(const kittens::st<T, RV, CV> &src, const char *prompt = nullptr) {
        if (prompt != nullptr)
            print("%s", prompt);
        constexpr int RS = RV / 8, CS = CV / 8;
        // Print column headers
        print("   "); // Padding for row indices
        for (int c = 0; c < 8 * CS; c++)
            print("%10d", c);
        print("\n");

        // Print data rows
        uint32_t ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&src.data[0]));
#pragma unroll
        for (int r = 0; r < 8 * RS; r++) {
            if (r % 8 == 0) {
                print("    +");
                for (int c = 0; c < 8 * CS; c++) {
                    print("---------+");
                }
                print("\n");
            }

            print("%3d |", r); // Row index
#pragma unroll
            for (int c = 0; c < 8 * CS; c++) {
                T remote_val = src.data[r * 8 * CS + c];
                if constexpr (std::is_same_v<T, float>)
                    print("%8.4f |", remote_val);
                if constexpr (std::is_same_v<T, half>)
                    print("%8.4f |", __half2float(remote_val));
                if constexpr (std::is_same_v<T, kittens::bf16>)
                    print("%8.4f |", __bfloat162float(remote_val));
                if constexpr (std::is_same_v<T, int>)
                    print("%8d |", remote_val);
            }
            print("\n");
        }
        print("\n");
    }
};

using print_utils = print_utils_group<1>;

} // namespace kittens
