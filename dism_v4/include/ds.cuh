// Diagonal / anti-diagonal scans with column-aligned horizontal boundaries.
// Self-contained CUDA/C++20 header. No shared-memory transfer API is provided.
// Contract, validation, and communication audit: docs/EXPERIMENTAL_BOUNDARY_ALIGNMENT.md
#pragma once

#include <cstdio>
#include <concepts>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <type_traits>
#include <utility>

namespace glx {

// These declarations also appear in diagonal_scan.cuh so each header remains
// independently usable. Keep this guarded block compatible between the headers.
#ifndef GLX_DIAGONAL_SCAN_VALUE_TYPES_DEFINED
#define GLX_DIAGONAL_SCAN_VALUE_TYPES_DEFINED

constexpr int THREAD_ROWS = 8;
constexpr int THREAD_COLS = 4;

template<typename TMaybePacked>
struct UnaryElement;
template<typename TMaybePacked>
struct BinaryElement;

struct PackedF2;

struct AddOp {
    __forceinline__ __device__ static void combine_either(
        bool predicate, const UnaryElement<PackedF2>& lhs,
        UnaryElement<PackedF2>& yes, UnaryElement<PackedF2>& no);
    template<typename T>
    __forceinline__ __host__ __device__ static UnaryElement<T> identity(){
        return {T{0.f}};
    }
    template<typename T>
    __forceinline__ __host__ __device__ static UnaryElement<T> apply(
        const UnaryElement<T>& lhs, const UnaryElement<T>& rhs){
        return {lhs.value + rhs.value};
    }
};

struct MulOp {
    template<typename T>
    __forceinline__ __host__ __device__ static UnaryElement<T> identity(){
        return {T{1.f}};
    }
    template<typename T>
    __forceinline__ __host__ __device__ static UnaryElement<T> apply(
        const UnaryElement<T>& lhs, const UnaryElement<T>& rhs){
        return {lhs.value * rhs.value};
    }
};

// (a, b) represents f(x) = a*x + b.  lhs followed by rhs is rhs(lhs(x)).
// This operation is associative but intentionally non-commutative.
struct AffineComposeOp {
    __forceinline__ __device__ static void combine_either(
        bool predicate, const BinaryElement<PackedF2>& lhs,
        BinaryElement<PackedF2>& yes, BinaryElement<PackedF2>& no);
    template<typename T>
    __forceinline__ __host__ __device__ static BinaryElement<T> identity(){
        return {T{1.f}, T{0.f}};
    }
    template<typename T>
    __forceinline__ __host__ __device__ static BinaryElement<T> apply(
        const BinaryElement<T>& lhs, const BinaryElement<T>& rhs){
        return {lhs.first * rhs.first,
                lhs.second * rhs.first + rhs.second};
    }
};

struct UnpackedF1{
    float u0;
    __forceinline__ __host__ __device__ UnpackedF1() {}
    __forceinline__ __host__ __device__ UnpackedF1(float _u0): u0{_u0}{}
    __forceinline__ __host__ __device__ UnpackedF1 operator+(const UnpackedF1& rhs) const { return {u0 + rhs.u0}; }
    __forceinline__ __host__ __device__ UnpackedF1 operator*(const UnpackedF1& rhs) const { return {u0 * rhs.u0}; }
    __forceinline__ __device__ UnpackedF1 shuffle(int src_lane) const {
        return {
            __shfl_sync(0xffffffff, u0, src_lane, 32),
        };
    }
    __forceinline__ __device__ UnpackedF1 shuffle_xor(int lane_mask) const {
        return {
            __shfl_xor_sync(0xffffffff, u0, lane_mask, 32),
        };
    }
    __forceinline__ __device__ auto extract(bool) const {
        return UnpackedF1{u0};
    }
    __forceinline__ __device__ void insert(bool, UnpackedF1 val){
        u0 = val.u0;
    }
    __forceinline__ __device__ void print() const {
        printf("%8.3f", u0);
    }
    __forceinline__ __device__ static UnpackedF1 select(unsigned int idx, UnpackedF1 x0, UnpackedF1 x1){
        float out;
        asm volatile(
            "{\n"
            "   .reg .pred p0;\n"
            "   setp.ne.u32 p0, %1, 0x0;\n"
            "   selp.b32 %0, %3, %2, p0;\n"
            "}\n":"=f"(out):"r"(idx),"f"(x0.u0),"f"(x1.u0)
        );
        return {out};
    }
    __forceinline__ __device__ static UnpackedF1 select(unsigned int idx, UnpackedF1 x0, UnpackedF1 x1, UnpackedF1 x2, UnpackedF1 x3){
        float out;
        asm volatile(
            "{\n"
            "   .reg .pred p0, p1;\n"
            "   .reg .b32 x;\n"
            "   setp.ge.u32 p1, %1, 0x2;\n"
            "   and.b32 x, %1, 0x1;\n"
            "   setp.ne.u32 p0, x, 0x0;\n"
            "   @!p1 selp.b32 %0, %3, %2, p0;\n"
            "   @p1 selp.b32 %0, %5, %4, p0;\n"
            "}\n":"=f"(out):"r"(idx),"f"(x0.u0),"f"(x1.u0),"f"(x2.u0),"f"(x3.u0)
        );
        return {out};
    }

    __forceinline__ __device__ static UnpackedF1 select(unsigned int idx, UnpackedF1 x0, UnpackedF1 x1, UnpackedF1 x2, UnpackedF1 x3, UnpackedF1 x4, UnpackedF1 x5, UnpackedF1 x6, UnpackedF1 x7){
        auto lo = select(idx & 3, x0, x1, x2, x3);
        auto hi = select(idx & 3, x4, x5, x6, x7);
        return select(idx >> 2, lo, hi);
    }
};

struct PackedF2 {
    float u0, u1;
    __forceinline__ __host__ __device__ PackedF2() {}
    __forceinline__ __host__ __device__ PackedF2(float value): u0{value}, u1{value}{}
    __forceinline__ __host__ __device__ PackedF2(float _u0, float _u1): u0{_u0}, u1{_u1}{}
    __forceinline__ __host__ __device__ PackedF2 operator+(const PackedF2& rhs) const { return {u0 + rhs.u0, u1 + rhs.u1}; }
    __forceinline__ __host__ __device__ PackedF2 operator*(const PackedF2& rhs) const { return {u0 * rhs.u0, u1 * rhs.u1}; }
    __forceinline__ __device__ PackedF2 shuffle(int src_lane) const {
        return {
            __shfl_sync(0xffffffff, u0, src_lane, 32),
            __shfl_sync(0xffffffff, u1, src_lane, 32),
        };
    }
    __forceinline__ __device__ PackedF2 shuffle_xor(int lane_mask) const {
        return {
            __shfl_xor_sync(0xffffffff, u0, lane_mask, 32),
            __shfl_xor_sync(0xffffffff, u1, lane_mask, 32),
        };
    }
    __forceinline__ __device__ auto extract(bool eid) const {
        return UnpackedF1::select(eid, UnpackedF1{u0}, UnpackedF1{u1});
    }
    __forceinline__ __device__ void insert(bool eid, UnpackedF1 val){
        if(eid) u1 = val.u0;
        else u0 = val.u0;
    }
    __forceinline__ __device__ void print() const {
        printf("(%8.3f, %8.3f)", u0, u1);
    }
    __forceinline__ __device__ static PackedF2 select(unsigned int idx, PackedF2 x0, PackedF2 x1, PackedF2 x2, PackedF2 x3){
        return { UnpackedF1::select(idx,{x0.u0},{x1.u0},{x2.u0},{x3.u0}).u0, UnpackedF1::select(idx,{x0.u1},{x1.u1},{x2.u1},{x3.u1}).u0 };
    }
    __forceinline__ __device__ static PackedF2 select(unsigned int idx, PackedF2 x0, PackedF2 x1){
        return { UnpackedF1::select(idx,{x0.u0},{x1.u0}).u0, UnpackedF1::select(idx,{x0.u1},{x1.u1}).u0 };
    }
    __forceinline__ __device__ static PackedF2 select(unsigned int idx, PackedF2 x0, PackedF2 x1, PackedF2 x2, PackedF2 x3, PackedF2 x4, PackedF2 x5, PackedF2 x6, PackedF2 x7){
        return { UnpackedF1::select(idx,{x0.u0},{x1.u0},{x2.u0},{x3.u0},{x4.u0},{x5.u0},{x6.u0},{x7.u0}).u0,
                 UnpackedF1::select(idx,{x0.u1},{x1.u1},{x2.u1},{x3.u1},{x4.u1},{x5.u1},{x6.u1},{x7.u1}).u0 };
    }
};

struct BF16x1 {
    __nv_bfloat16 value;

    __forceinline__ __host__ __device__ BF16x1() {}
    __forceinline__ __host__ __device__ BF16x1(float x)
        : value{__float2bfloat16(x)} {}
    __forceinline__ __host__ __device__ explicit BF16x1(__nv_bfloat16 x)
        : value{x} {}
    __forceinline__ __host__ __device__ BF16x1(const BF16x1& x)
        : value{x.value} {}
    __forceinline__ __host__ __device__ BF16x1& operator=(const BF16x1& x){
        value = x.value;
        return *this;
    }

    __forceinline__ __host__ __device__ BF16x1 operator+(BF16x1 rhs) const {
        return BF16x1{__hadd(value, rhs.value)};
    }
    __forceinline__ __host__ __device__ BF16x1 operator*(BF16x1 rhs) const {
        return BF16x1{__hmul(value, rhs.value)};
    }
    __forceinline__ __host__ __device__ uint32_t raw() const {
        return __bfloat16_as_ushort(value);
    }
    __forceinline__ __host__ __device__ static BF16x1 from_raw(uint32_t x) {
        return BF16x1{__ushort_as_bfloat16(
            static_cast<unsigned short>(x))};
    }
    __forceinline__ __device__ BF16x1 shuffle(int src_lane) const {
        return from_raw(__shfl_sync(0xffffffff, raw(), src_lane, 32));
    }
    __forceinline__ __device__ BF16x1 shuffle_xor(int lane_mask) const {
        return from_raw(__shfl_xor_sync(0xffffffff, raw(), lane_mask, 32));
    }
    __forceinline__ __device__ BF16x1 extract(bool) const {
        return *this;
    }
    __forceinline__ __device__ void insert(bool, BF16x1 x) {
        value = x.value;
    }
    __forceinline__ __device__ void print() const {
        printf("%8.3f", __bfloat162float(value));
    }
    __forceinline__ __device__ UnpackedF1 select_word() const {
        return {__uint_as_float(raw())};
    }
    template<typename... Ts>
    __forceinline__ __device__ static BF16x1 select(
            unsigned int idx, const Ts&... args){
        static_assert((std::is_same_v<BF16x1, Ts> && ...));
        auto selected = UnpackedF1::select(idx, args.select_word()...);
        return from_raw(__float_as_uint(selected.u0));
    }
};

struct BF16x2 {
    __nv_bfloat162 value;

    __forceinline__ __host__ __device__ BF16x2() {}
    __forceinline__ __host__ __device__ BF16x2(float x)
        : value{__float2bfloat162_rn(x)} {}
    __forceinline__ __host__ __device__ BF16x2(float x, float y)
        : value{__floats2bfloat162_rn(x, y)} {}
    __forceinline__ __host__ __device__ explicit BF16x2(__nv_bfloat162 x)
        : value{x} {}
    __forceinline__ __host__ __device__ BF16x2(const BF16x2& x)
        : value{x.value} {}
    __forceinline__ __host__ __device__ BF16x2& operator=(const BF16x2& x){
        value = x.value;
        return *this;
    }

    __forceinline__ __host__ __device__ BF16x2 operator+(BF16x2 rhs) const {
        return BF16x2{__hadd2(value, rhs.value)};
    }
    __forceinline__ __host__ __device__ BF16x2 operator*(BF16x2 rhs) const {
        return BF16x2{__hmul2(value, rhs.value)};
    }
    __forceinline__ __host__ __device__ uint32_t raw() const {
        __nv_bfloat162_raw bits = value;
        return uint32_t(bits.x) | (uint32_t(bits.y) << 16);
    }
    __forceinline__ __host__ __device__ static BF16x2 from_raw(uint32_t x) {
        __nv_bfloat162_raw bits{
            static_cast<unsigned short>(x),
            static_cast<unsigned short>(x >> 16)};
        return BF16x2{__nv_bfloat162{bits}};
    }
    __forceinline__ __device__ BF16x2 shuffle(int src_lane) const {
        return from_raw(__shfl_sync(0xffffffff, raw(), src_lane, 32));
    }
    __forceinline__ __device__ BF16x2 shuffle_xor(int lane_mask) const {
        return from_raw(__shfl_xor_sync(0xffffffff, raw(), lane_mask, 32));
    }
    __forceinline__ __device__ BF16x1 extract(bool eid) const {
        return BF16x1::from_raw(eid ? raw() >> 16 : raw());
    }
    __forceinline__ __device__ void insert(bool eid, BF16x1 x) {
        uint32_t bits = raw();
        bits = eid ? (bits & 0xffffu) | (x.raw() << 16)
                   : (bits & 0xffff0000u) | x.raw();
        value = from_raw(bits).value;
    }
    __forceinline__ __device__ void print() const {
        printf("(%8.3f, %8.3f)",
               __low2float(value), __high2float(value));
    }
    __forceinline__ __device__ UnpackedF1 select_word() const {
        return {__uint_as_float(raw())};
    }
    template<typename... Ts>
    __forceinline__ __device__ static BF16x2 select(
            unsigned int idx, const Ts&... args){
        static_assert((std::is_same_v<BF16x2, Ts> && ...));
        auto selected = UnpackedF1::select(idx, args.select_word()...);
        return from_raw(__float_as_uint(selected.u0));
    }
};

using F32x1 = UnpackedF1;
using F32x2 = PackedF2;

template<typename PackedValue>
struct PackTraits;

template<>
struct PackTraits<F32x2> {
    using unpacked_type = F32x1;
    using scalar_type = float;
    static constexpr int lanes = 2;
};

template<>
struct PackTraits<BF16x2> {
    using unpacked_type = BF16x1;
    using scalar_type = BF16x1;
    static constexpr int lanes = 2;
};

// Conversion at the memory-facing edge is deliberately separate from the
// register wrapper.  GEMM integrations normally fill MMABuffer::data
// directly; these helpers keep the standalone tests and examples generic.
template<typename PackedValue>
struct PackedValueIO;

template<>
struct PackedValueIO<F32x2> {
    __forceinline__ __device__ static F32x2 pack(float low, float high){
        return {low, high};
    }
    __forceinline__ __device__ static float low(F32x2 value){ return value.u0; }
    __forceinline__ __device__ static float high(F32x2 value){ return value.u1; }
};

template<>
struct PackedValueIO<BF16x2> {
    __forceinline__ __device__ static BF16x2 pack(float low, float high){
        return {low, high};
    }
    __forceinline__ __device__ static float low(BF16x2 value){
        return __low2float(value.value);
    }
    __forceinline__ __device__ static float high(BF16x2 value){
        return __high2float(value.value);
    }
};

template<typename TMaybePacked>
struct UnaryElement {
    using value_type = TMaybePacked;
    static constexpr int size = 1;
    TMaybePacked value;

    template<int I>
    __host__ __device__ TMaybePacked& get(){
        static_assert(I == 0);
        return value;
    }
    template<int I>
    __host__ __device__ const TMaybePacked& get() const {
        static_assert(I == 0);
        return value;
    }

    __forceinline__ __device__ UnaryElement shuffle(int src_lane) const {
        return {value.shuffle(src_lane)};
    }
    __forceinline__ __device__ UnaryElement shuffle_xor(int lane_mask) const {
        return {value.shuffle_xor(lane_mask)};
    }
    __forceinline__ __device__ auto extract(bool eid) const {
        using R = decltype(value.extract(eid));
        return UnaryElement<R>{value.extract(eid)};
    }
    template<typename T>
    __forceinline__ __device__ void insert(bool eid, const UnaryElement<T>& val){
        value.insert(eid, val.value);
    }
    template<typename T>
    __forceinline__ __device__ void insert_if(
        bool eid, bool condition, const UnaryElement<T>& val){
        auto current = extract(eid);
        insert(eid, UnaryElement<T>::select(condition, current, val));
    }
    __forceinline__ __device__ void print() const {
        value.print();
    }

    template<typename... Ts>
    __forceinline__ __device__ static UnaryElement select(unsigned int idx, const Ts&... args){
        static_assert((std::is_same_v<UnaryElement, Ts> && ...));
        return {TMaybePacked::select(idx, args.value...)};
    }
};

template<typename TMaybePacked>
struct BinaryElement {
    using value_type = TMaybePacked;
    static constexpr int size = 2;
    TMaybePacked first;
    TMaybePacked second;

    template<int I>
    __host__ __device__ TMaybePacked& get(){
        static_assert(I == 0 || I == 1);
        if constexpr(I == 0) return first;
        else return second;
    }
    template<int I>
    __host__ __device__ const TMaybePacked& get() const {
        static_assert(I == 0 || I == 1);
        if constexpr(I == 0) return first;
        else return second;
    }

    __forceinline__ __device__ BinaryElement shuffle(int src_lane) const {
        return {first.shuffle(src_lane), second.shuffle(src_lane)};
    }
    __forceinline__ __device__ BinaryElement shuffle_xor(int lane_mask) const {
        return {first.shuffle_xor(lane_mask), second.shuffle_xor(lane_mask)};
    }
    __forceinline__ __device__ auto extract(bool eid) const {
        using R = decltype(first.extract(eid));
        return BinaryElement<R>{first.extract(eid), second.extract(eid)};
    }
    template<typename T>
    __forceinline__ __device__ void insert(bool eid, const BinaryElement<T>& val){
        first.insert(eid, val.first);
        second.insert(eid, val.second);
    }
    template<typename T>
    __forceinline__ __device__ void insert_if(
        bool eid, bool condition, const BinaryElement<T>& val){
        auto current = extract(eid);
        insert(eid, BinaryElement<T>::select(condition, current, val));
    }
    __forceinline__ __device__ void print() const {
        printf("{"); first.print(); printf(", "); second.print(); printf("}");
    }
    template<typename... Ts>
    __forceinline__ __device__ static BinaryElement select(unsigned int idx, const Ts&... args){
        static_assert((std::is_same_v<BinaryElement, Ts> && ...));
        return {TMaybePacked::select(idx, args.first...),
                TMaybePacked::select(idx, args.second...)};
    }
};

// Optional hook: update only the selected destination with apply(lhs, dst).
// lhs may alias either destination; preserve its original value for the combine.
// Exact function-pointer matching rejects non-static and convertible signatures,
// while also supporting overloads and static function templates.
template<typename Op, typename E>
concept EitherBinaryOp = requires {
    static_cast<void (*)(bool, const E&, E&, E&)>(&Op::combine_either);
};

__forceinline__ __device__ void AddOp::combine_either(
    bool predicate, const UnaryElement<PackedF2>& lhs,
    UnaryElement<PackedF2>& yes, UnaryElement<PackedF2>& no){
    asm volatile(
        "{ .reg .pred p; .reg .f32 l0, l1;\n"
        "  setp.ne.u32 p, %6, 0;\n"
        "  mov.f32 l0, %4; mov.f32 l1, %5;\n"
        "  @p add.f32 %0, l0, %0; @p add.f32 %1, l1, %1;\n"
        "  @!p add.f32 %2, l0, %2; @!p add.f32 %3, l1, %3;\n"
        "}"
        : "+f"(yes.value.u0), "+f"(yes.value.u1), "+f"(no.value.u0), "+f"(no.value.u1)
        : "f"(lhs.value.u0), "f"(lhs.value.u1), "r"(static_cast<unsigned>(predicate)));
}

__forceinline__ __device__ void AffineComposeOp::combine_either(
    bool predicate, const BinaryElement<PackedF2>& lhs,
    BinaryElement<PackedF2>& yes, BinaryElement<PackedF2>& no){
    // Save all lhs inputs before updating any potentially aliased register.
    // Update offsets before scales, since offsets use the OLD rhs scales.
    asm volatile(
        "{ .reg .pred p; .reg .f32 a0, a1, b0, b1;\n"
        "  setp.ne.u32 p, %12, 0;\n"
        "  mov.f32 b0, %8; mov.f32 b1, %9;\n"
        "  mov.f32 a0, %10; mov.f32 a1, %11;\n"
        "  @p fma.rn.f32 %0, b0, %2, %0;\n"
        "  @p fma.rn.f32 %1, b1, %3, %1;\n"
        "  @p mul.rn.f32 %2, a0, %2;\n"
        "  @p mul.rn.f32 %3, a1, %3;\n"
        "  @!p fma.rn.f32 %4, b0, %6, %4;\n"
        "  @!p fma.rn.f32 %5, b1, %7, %5;\n"
        "  @!p mul.rn.f32 %6, a0, %6;\n"
        "  @!p mul.rn.f32 %7, a1, %7;\n"
        "}"
        : "+f"(yes.second.u0), "+f"(yes.second.u1), "+f"(yes.first.u0), "+f"(yes.first.u1),
          "+f"(no.second.u0), "+f"(no.second.u1), "+f"(no.first.u0), "+f"(no.first.u1)
        : "f"(lhs.second.u0), "f"(lhs.second.u1), "f"(lhs.first.u0), "f"(lhs.first.u1),
          "r"(static_cast<unsigned>(predicate)));
}

template<typename Op, typename E>
concept ClosedBinaryOp = requires(const E& lhs, const E& rhs) {
    { Op::template identity<typename E::value_type>() } -> std::same_as<E>;
    { Op::apply(lhs, rhs) } -> std::same_as<E>;
};

// Optional fast path for a conditional combine.  The operation must leave
// *rhs unchanged when predicate is false and otherwise perform
// *rhs = Op::apply(lhs, *rhs).  Keeping the predicate inside Op lets an
// expensive operation predicate only its final writes.
template<typename Op, typename E>
concept PredicatedBinaryOp = requires(
    const E& lhs, E* rhs, bool predicate) {
    { Op::apply(lhs, rhs, predicate) } -> std::same_as<void>;
};

static_assert(ClosedBinaryOp<AddOp, UnaryElement<float>>);
static_assert(ClosedBinaryOp<MulOp, UnaryElement<float>>);
static_assert(ClosedBinaryOp<AffineComposeOp, BinaryElement<float>>);
static_assert(!ClosedBinaryOp<AffineComposeOp, UnaryElement<float>>);
static_assert(ClosedBinaryOp<AddOp, UnaryElement<BF16x2>>);
static_assert(ClosedBinaryOp<AddOp, UnaryElement<BF16x1>>);
static_assert(ClosedBinaryOp<MulOp, UnaryElement<BF16x2>>);
static_assert(ClosedBinaryOp<MulOp, UnaryElement<BF16x1>>);
static_assert(ClosedBinaryOp<AffineComposeOp, BinaryElement<BF16x2>>);
static_assert(ClosedBinaryOp<AffineComposeOp, BinaryElement<BF16x1>>);

#endif // GLX_DIAGONAL_SCAN_VALUE_TYPES_DEFINED

// Diagonal and anti-diagonal scans. All 32 lanes participate.
// Forward HState owns columns [0,COLS-1], VState rows [-1,ROWS-2].
// Reverse HState owns columns [0,COLS-1], VState rows [1,ROWS].
// The signed outside row is a corner from incoming HState, not wrapped data.
// Prescan returns final boundaries; postscan
// consumes the original incoming HState and the saved intermediate state.
// Anti-diagonal entry points use the same aligned H / shifted V convention.
template<int ROWS_, int COLS_,
         template<typename> class ElementType = UnaryElement,
         typename Op = glx::AddOp, typename PackedType = glx::F32x2>
struct SplitScanBuffer{
    static constexpr int ROWS = ROWS_;
    static constexpr int COLS = COLS_;
    static_assert(ROWS >= 16 && (ROWS & (ROWS - 1)) == 0,
                  "ROWS must be a power of two >= 16");
    static_assert(COLS >= 16 && (COLS & (COLS - 1)) == 0,
                  "COLS must be a power of two >= 16");
    using PElement = ElementType<PackedType>;
    using UElement = ElementType<typename glx::PackTraits<PackedType>::unpacked_type>;
    using packed_type = PackedType;
    using operation_type = Op;
    static_assert(glx::PackTraits<PackedType>::lanes == 2);
    static_assert(glx::ClosedBinaryOp<Op, PElement>);
    static_assert(glx::ClosedBinaryOp<Op, UElement>);

    template<typename E>
    __forceinline__ __host__ __device__ static E identity(){
        return Op::template identity<typename E::value_type>();
    }
    template<typename E>
    __forceinline__ __device__ static E combine(const E& lhs, const E& rhs){
        return Op::apply(lhs, rhs);
    }

    __forceinline__ __device__ static void combine_either(
            bool predicate, const PElement& lhs, PElement& yes, PElement& no){
        if constexpr(EitherBinaryOp<Op, PElement>){
            Op::combine_either(predicate, lhs, yes, no);
        }else{
            // Generic expensive operators are evaluated ONCE, never once per
            // destination. All selections are between two fixed registers.
            auto rhs = choose(predicate, yes, no);
            auto result = combine(lhs, rhs);
            yes = choose(predicate, result, yes);
            no = choose(predicate, no, result);
        }
    }

    // Layout: example for COL_BLOCK = 4, ROW_BLOCK = 1. ROWS = 8, COLS = 32
    // +--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+
    // | T0: (x0,x1,x2,x3)  | T1: (x0,x1,x2,x3)  | T2: (x0,x1,x2,x3)  | T3: (x0,x1,x2,x3)  | T0: (y0,y1,y2,y3)  | T1: (y0,y1,y2,y3)  | T2: (y0,y1,y2,y3)  | T3: (y0,y1,y2,y3)  |
    // +--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+
    // | T4: (x0,x1,x2,x3)  | T5: (x0,x1,x2,x3)  | T6: (x0,x1,x2,x3)  | T7: (x0,x1,x2,x3)  | T4: (y0,y1,y2,y3)  | T5: (y0,y1,y2,y3)  | T6: (y0,y1,y2,y3)  | T7: (y0,y1,y2,y3)  |
    // +--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+
    // | T8: (x0,x1,x2,x3)  | T9: (x0,x1,x2,x3)  | T10: (x0,x1,x2,x3) | T11: (x0,x1,x2,x3) | T8: (y0,y1,y2,y3)  | T9: (y0,y1,y2,y3)  | T10: (y0,y1,y2,y3  | T11: (y0,y1,y2,y3) |
    // +--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+
    // | T12: (x0,x1,x2,x3) | T13: (x0,x1,x2,x3) | T14: (x0,x1,x2,x3) | T15: (x0,x1,x2,x3) | T12: (y0,y1,y2,y3) | T13: (y0,y1,y2,y3) | T14: (y0,y1,y2,y3) | T15: (y0,y1,y2,y3) |
    // +--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+
    // | T16: (x0,x1,x2,x3) | T17: (x0,x1,x2,x3) | T18: (x0,x1,x2,x3) | T19: (x0,x1,x2,x3) | T16: (y0,y1,y2,y3) | T17: (y0,y1,y2,y3) | T18: (y0,y1,y2,y3) | T19: (y0,y1,y2,y3) |
    // +--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+
    // | T20: (x0,x1,x2,x3) | T21: (x0,x1,x2,x3) | T22: (x0,x1,x2,x3) | T23: (x0,x1,x2,x3) | T20: (y0,y1,y2,y3) | T21: (y0,y1,y2,y3) | T22: (y0,y1,y2,y3) | T23: (y0,y1,y2,y3) |
    // +--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+
    // | T24: (x0,x1,x2,x3) | T25: (x0,x1,x2,x3) | T26: (x0,x1,x2,x3) | T27: (x0,x1,x2,x3) | T24: (y0,y1,y2,y3) | T25: (y0,y1,y2,y3) | T26: (y0,y1,y2,y3) | T27: (y0,y1,y2,y3) |
    // +--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+
    // | T28: (x0,x1,x2,x3) | T29: (x0,x1,x2,x3) | T30: (x0,x1,x2,x3) | T31: (x0,x1,x2,x3) | T28: (y0,y1,y2,y3) | T29: (y0,y1,y2,y3) | T30: (y0,y1,y2,y3) | T31: (y0,y1,y2,y3) |
    // +--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+
    // lam4 = lane & 0x3, lar3 = lane >> 2
    // example for c:
    // T0: (x0,x1,x2,x3)
    //   ------ c ----> for c = 0, 1, 2, 3
    // r: r increase by 1 every 8 rows
    // eid: bool value, 1 for y's region and 0 for x's region


    //
    static constexpr int ROW_BLOCKS = ROWS / THREAD_ROWS;
    static constexpr int COL_BLOCKS = COLS / THREAD_COLS / 2;
    PElement data[ROW_BLOCKS][COL_BLOCKS];

    struct VState {
        static constexpr int length = ROW_BLOCKS;
        UElement init[ROW_BLOCKS];
        __host__ __device__ VState(){
            #pragma unroll
            for(int r = 0; r < ROW_BLOCKS; ++r) init[r] = identity<UElement>();
        }
    };

    struct HState {
        static constexpr int length = (COL_BLOCKS - 1) / THREAD_ROWS + 1;
        PElement init[length];
        __host__ __device__ HState(): HState(identity<PElement>()){}
        __host__ __device__ HState(const PElement& zero) {
            #pragma unroll
            for(int i = 0; i < length; i++) init[i] = zero;
        }

        __device__ void print(int colsep_interval = 1) const {
            bool leader = (threadIdx.x & 0x1f) == 0;
            if(leader)printf("     ");
            for(int i = 0; i < COLS; i++) if(leader) printf(" %9d", i);
            if(leader)printf("\n      ");
            for(int i = 0; i < COLS; i++) if(leader) printf("%s─────────", (i % colsep_interval == 0) ? (i ? "┬" : "┌") : "─");
            if(leader)printf("┐\n");

            constexpr int hstate_stride = (COL_BLOCKS < THREAD_ROWS ? COL_BLOCKS : THREAD_ROWS);
            for(int i = 0; i < 8; i++){
                if(leader) printf("%4d ", i);
                for(int j = 0; j < COLS; j++){
                    if((j + 1) % hstate_stride == 0){
                        int src_laneid = (i % THREAD_ROWS) * THREAD_COLS + (j / COL_BLOCKS) % THREAD_COLS;
                        auto curr = init[(j % COL_BLOCKS) / hstate_stride].extract((j / (COL_BLOCKS * 4)) & 0x1).shuffle(src_laneid);
                        if(leader) printf(j % colsep_interval == 0 ? " │" : "  "), curr.print();
                    }else{
                        if(leader) printf(j % colsep_interval == 0 ? " │" : "  "), identity<decltype(init[0].extract(0))>().print();
                    }

                }
                if(leader) printf(" │\n");
                if(i == 7){
                    if(leader)printf("      ");
                    for(int j = 0; j < COLS; j++) if(leader) printf("%s─────────",(j % colsep_interval == 0) ? (i != 7 ? (j ? "┼" : "├") : (j ? "┴" : "└")) : "─");
                    if(leader) printf("%s\n", i != 7 ? "┤" : "┘");
                }
            }
        }
    };

    struct ImmState {
        static constexpr int length = ROW_BLOCKS;
        PElement imm[ROW_BLOCKS];
        __host__ __device__ ImmState(){
            #pragma unroll
            for(int r = 0; r < ROW_BLOCKS; ++r) imm[r] = identity<PElement>();
        }
        __device__ PElement& operator[](int idx) { return imm[idx]; }
        __device__ const PElement& operator[](int idx) const { return imm[idx]; }
    };

    struct Coordinate { int first, second; };
    __forceinline__ __device__ static Coordinate layout(int r, int c, int e){
        int lane = threadIdx.x & 0x1f;
        int lam4 = lane & 0x3, lar3 = lane >> 2;
        return { r * THREAD_ROWS + lar3, c + e * (COL_BLOCKS * 4) + lam4 * COL_BLOCKS };
    }

    template<bool FWD = true, bool ANTI = false>
    __device__ void roll(){
        int lane = threadIdx.x & 31;
        int lam4 = lane & 3, lar3 = oriented_lar<ANTI>(lane >> 2);
        PElement tmp[ROW_BLOCKS][COL_BLOCKS];
        static_for<ROW_BLOCKS>([&]<int r>(){
            static_for<COL_BLOCKS>([&]<int j>(){
                tmp[r][j] = data[oriented_row<ANTI>(r)][COL_BLOCKS - 1 - j].shuffle(
                    source_lane<ANTI>(lam4, FWD ? lar3 - ((j & 7) + 1) : lar3 + ((j & 7) + 1)));
            });
        });
        static_for<ROW_BLOCKS>([&]<int r>(){
            static_for<COL_BLOCKS>([&]<int j>(){
                if constexpr(FWD)
                    data[oriented_row<ANTI>(r)][COL_BLOCKS - 1 - j] = choose(lar3 < ((j & 7) + 1), tmp[wrap_row(r - 1)][j], tmp[r][j]);
                else
                    data[oriented_row<ANTI>(r)][COL_BLOCKS - 1 - j] = choose(lar3 > 7 - ((j & 7) + 1), tmp[wrap_row(r + 1)][j], tmp[r][j]);
            });
        });
    }

    template<int colsep_interval = 1>
    __device__ void print() const {
        bool leader = (threadIdx.x & 0x1f) == 0;
        if(leader)printf("     ");
        for(int i = 0; i < COLS; i++) if(leader) printf(" %9d", i);
        if(leader)printf("\n      ");
        for(int i = 0; i < COLS; i++) if(leader) printf("%s─────────", (i % colsep_interval == 0) ? (i ? "┬" : "┌") : "─");
        if(leader)printf("┐\n");
        for(int i = 0; i < ROWS; i++){
            if(leader) printf("%4d ", i);
            for(int j = 0; j < COLS; j++){
                int src_laneid = (i % THREAD_ROWS) * THREAD_COLS + (j / COL_BLOCKS) % THREAD_COLS;
                auto curr = data[i / THREAD_ROWS][j % COL_BLOCKS].extract((j / (COL_BLOCKS * 4)) & 0x1).shuffle(src_laneid);
                if(leader) printf(j % colsep_interval == 0 ? " │" : "  "), curr.print();
            }
            if(leader) printf(" │\n");
            if(i % 8 == 7){
                if(leader)printf("      ");
                for(int j = 0; j < COLS; j++) if(leader) printf("%s─────────",(j % colsep_interval == 0) ? (i != ROWS - 1 ? (j ? "┼" : "├") : (j ? "┴" : "└")) : "─");
                if(leader) printf("%s\n", i != ROWS - 1 ? "┤" : "┘");
            }
        }
    }

    template<int N, typename Function>
    __forceinline__ __device__ static void static_for(Function&& function){
        [&]<int... I>(std::integer_sequence<int, I...>){
            (function.template operator()<I>(), ...);
        }(std::make_integer_sequence<int, N>{});
    }

    template<typename T>
    __forceinline__ __device__ static T choose(bool predicate, const T& yes, const T& no){
        return T::select(static_cast<unsigned>(predicate), no, yes);
    }

    // Anti-diagonal traversal uses the opposite row axis in the same tree.
    // Register indices are resolved at compile time. No tile reflection,
    // packed-half swap, or additional shuffle is performed.
    template<bool ANTI>
    __forceinline__ __host__ __device__ static constexpr int oriented_row(int r){
        return ANTI ? ROW_BLOCKS - 1 - r : r;
    }
    template<bool ANTI>
    __forceinline__ __host__ __device__ static constexpr int oriented_lar(int lar){
        return ANTI ? 7 - lar : lar;
    }

    // A rolled register (r,c) represents this physical row in both halves.
    __forceinline__ __device__ static int physical_row(int r, int c, int lar3){
        return (r * 8 + lar3 - ((COL_BLOCKS - 1 - c) & 7) - 1 + ROWS) % ROWS;
    }

    __forceinline__ __device__ static constexpr int wrap_row(int r){
        return r & (ROW_BLOCKS - 1);
    }

    template<bool ANTI = false>
    __forceinline__ __device__ static int source_lane(int lam4, int lar3){
        return (lam4 & 3) + (oriented_lar<ANTI>(lar3 & 7) * 4);
    }

    __forceinline__ __device__ static int forward_root_row(int r, int lar){
        return (8*r + lar - 1 + ROWS) & (ROWS-1);
    }
    __forceinline__ __device__ static int backward_root_row(int r, int lar){
        return (8*r + lar + 1) & (ROWS-1);
    }

    // ceil((DISTANCE-lar3)/8) has at most two values. Never use the
    // runtime displacement as an array index: select two fixed registers.
    template<int R, int DISTANCE>
    __forceinline__ __device__ static PElement shifted_root(const ImmState& tmp, int lar3){
        constexpr int base = DISTANCE / 8;
        if constexpr(DISTANCE % 8 == 0)
            return tmp[wrap_row(R - base)];
        else
            return choose(lar3 < DISTANCE % 8,
                          tmp[wrap_row(R - base - 1)], tmp[wrap_row(R - base)]);
    }

    // Exactly the prototype's three packed exchanges: distances 1, 2, 1.
    // The last exchange is saved for postscan; postscan has no communication.
    template<int STRIPS, bool ANTI = false>
    __forceinline__ __device__ ImmState exchange_roots(const VState& vstate, const HState& hstate) const {
        int lane = threadIdx.x & 31;
        int lam4 = lane & 3, lar3 = oriented_lar<ANTI>(lane >> 2);
        ImmState tmp;
        static_for<ROW_BLOCKS>([&]<int r>(){
            auto send = data[oriented_row<ANTI>(r)][COL_BLOCKS - 1];
            // This wrapped tile root is physically on the bottom edge, but
            // its use as a predecessor above the tile must read the top input.
            if constexpr(r == 0) send = choose(lar3 == 0, hstate.init[HState::length-1], send);
            auto shifted = send;
            shifted.insert(1, send.extract(0));
            shifted.insert(0, vstate.init[oriented_row<ANTI>(r)]);
            send = choose(lam4 >= 4 - STRIPS, shifted, send);
            tmp[r] = send.shuffle(source_lane<ANTI>(lam4 - STRIPS, lar3 - STRIPS * COL_BLOCKS));
        });
        return tmp;
    }

    // Fine downsweep has at most two live sources per register. In forward
    // coordinates, row block 0 can consume this group's HState; row block 1
    // can consume the previous group's HState only at lar==0. These cases
    // never coexist for the same compile-time (r,c). Mirror only coordinates
    // for reverse traversal, without moving any tile registers.
    template<int R, int C, int STRIDE, bool REVERSE = false>
    __forceinline__ __device__ static PElement fine_boundary(
            const PElement& predecessor, const HState& hstate, int lar){
        constexpr int group = COL_BLOCKS < 8 ? COL_BLOCKS : 8;
        constexpr int r = REVERSE ? ROW_BLOCKS - 1 - R : R;
        constexpr int c = REVERSE ? COL_BLOCKS - 1 - C : C;
        int a = REVERSE ? 7 - lar : lar;
        if constexpr(r == 0){
            constexpr int first = group - c % group;
            constexpr int end = first + STRIDE < group ? first + STRIDE : group;
            if constexpr(first < end)
                return choose(static_cast<unsigned>(a - first) < end - first,
                              hstate.init[C/group], predecessor);
        }else if constexpr(group == 8 && r == 1 && c % group == STRIDE - 1 && c/group > 0){
            return choose(a == 0, hstate.init[C/group + (REVERSE ? 1 : -1)], predecessor);
        }
        return predecessor;
    }

    // Root nodes at or above an eight-column group are completed in prescan.
    // Only the smaller within-group nodes remain for postscan.
    template<bool COARSE, bool ANTI = false>
    __forceinline__ __device__ void downsweep(const HState& hstate, const ImmState& tmp){
        constexpr int group = COL_BLOCKS < 8 ? COL_BLOCKS : 8;
        constexpr int bits = __builtin_ffs(COL_BLOCKS) - 1;
        int lar3 = oriented_lar<ANTI>((threadIdx.x & 31) >> 2);
        static_for<bits>([&]<int sweep>(){
            constexpr int stride = 1 << (bits - 1 - sweep);
            if constexpr(stride < ROWS && ((stride >= group) == COARSE)){
                static_for<COL_BLOCKS / (2 * stride)>([&]<int node>(){
                    constexpr int c = stride - 1 + node * 2 * stride;
                    static_for<ROW_BLOCKS>([&]<int r>(){
                        PElement lhs;
                        if constexpr(c >= stride){
                            constexpr int displacement = c / 8 - (c - stride) / 8;
                            lhs = data[oriented_row<ANTI>(wrap_row(r - displacement))][c - stride];
                        }else if constexpr(COL_BLOCKS >= 8){
                            lhs = tmp[wrap_row(r - (stride + 7) / 8)];
                        }else{
                            lhs = shifted_root<r, COL_BLOCKS>(tmp, lar3);
                        }
                        if constexpr(COARSE){
                            // Roots reaching the top already include HState.
                            int row = physical_row(r, c, lar3);
                            bool pred = c < stride ? row + 1 >= stride : row >= stride;
                            data[oriented_row<ANTI>(r)][c] = choose(pred, combine(lhs, data[oriented_row<ANTI>(r)][c]), data[oriented_row<ANTI>(r)][c]);
                        }else{
                            lhs = fine_boundary<r, c, stride>(lhs, hstate, lar3);
                            data[oriented_row<ANTI>(r)][c] = combine(lhs, data[oriented_row<ANTI>(r)][c]);
                        }
                    });
                });
            }
        });
    }

    struct PrescanResult {
        VState vertical;
        HState horizontal;
        ImmState intermediate;
    };

    // Input is rolled. Returns final boundaries while leaving the within-group
    // downsweep unfinished. Keep the ORIGINAL incoming HState for postscan.
    // VState: only lam4==3 owns row r*8+lar3-1 of the right edge (or left input).
    // HState: only lar3<min(COL_BLOCKS,8) is active. Slot s, half e stores
    // column (lam4+4*e)*COL_BLOCKS+s*group+group-1-lar3 of the bottom edge
    // (or top input). The top-right corner is carried in VState row -1.
    template<bool ANTI = false>
    __device__ PrescanResult inclusive_prescan(const VState& vstate, const HState& hstate){
        // A root spans at most the tile height, even in a wider strip.
        constexpr int root_span = COL_BLOCKS < ROWS ? COL_BLOCKS : ROWS;
        constexpr int bits = __builtin_ffs(COL_BLOCKS) - 1;
        constexpr int group = COL_BLOCKS < 8 ? COL_BLOCKS : 8;
        int lane = threadIdx.x & 31;
        int lam4 = lane & 3, lar3 = oriented_lar<ANTI>(lane >> 2);
        HState out_hstate;
        VState out_vstate;

        static_for<bits>([&]<int iter>(){
            constexpr int stride = 1 << iter;
            if constexpr(stride < ROWS){
                static_for<COL_BLOCKS / (2 * stride)>([&]<int node>(){
                    constexpr int c = 2 * stride - 1 + node * 2 * stride;
                    static_for<ROW_BLOCKS>([&]<int r>(){
                        auto lhs = data[oriented_row<ANTI>(wrap_row(r - stride / 8))][c - stride];
                        bool pred = physical_row(r, c, lar3) >= stride;
                        if constexpr(r == 0 && stride < group){
                            // The two destinations are updated under opposite
                            // predicates; each lane evaluates only one op.
                            combine_either(
                                pred, lhs, data[oriented_row<ANTI>(r)][c], out_hstate.init[c / group]);
                        }else{
                            data[oriented_row<ANTI>(r)][c] = choose(pred, combine(lhs, data[oriented_row<ANTI>(r)][c]), data[oriented_row<ANTI>(r)][c]);
                        }
                    });
                });
                if constexpr(stride == group / 2){
                    static_for<HState::length>([&]<int s>(){
                        auto& root = data[oriented_row<ANTI>(0)][(s + 1) * group - 1];
                        if constexpr(group == 8 && s > 0){
                            auto incoming = choose(lar3 == 0, hstate.init[s-1], hstate.init[s]);
                            combine_either(lar3 != 0, incoming, root, data[oriented_row<ANTI>(1)][(s+1)*group-1]);
                        }else root = choose(lar3 > 0 && lar3 < group, combine(hstate.init[s], root), root);
                    });
                }
            }
        });

        // Propagate roots using the signed frontier row 8*r+lar3-1.
        // At r==0, lar3==0 this is the incoming top boundary, not the
        // wrapped bottom tile root. Defer that tile root to the final H
        // combine so intermediate exchanges do not become live for it.
        auto tmp = exchange_roots<1, ANTI>(vstate, hstate);
        static_for<ROW_BLOCKS>([&]<int r>(){
            if constexpr(r * 8 + 7 >= COL_BLOCKS){
                auto& root = data[oriented_row<ANTI>(r)][COL_BLOCKS - 1];
                auto lhs = shifted_root<r, COL_BLOCKS>(tmp, lar3);
                root = choose(r * 8 + lar3 >= COL_BLOCKS, combine(lhs, root), root);
            }
        });
        tmp = exchange_roots<2, ANTI>(vstate, hstate);
        static_for<ROW_BLOCKS>([&]<int r>(){
            if constexpr(r * 8 + 7 >= 2 * COL_BLOCKS){
                auto& root = data[oriented_row<ANTI>(r)][COL_BLOCKS - 1];
                auto lhs = shifted_root<r, 2 * COL_BLOCKS>(tmp, lar3);
                auto combined = combine(lhs, root);
                bool pred = r * 8 + lar3 >= 2 * COL_BLOCKS;
                // Low-half strip 0 already includes its left initial value.
                root.insert(0, choose(pred && lam4 != 0, combined.extract(0), root.extract(0)));
                root.insert(1, choose(pred, combined.extract(1), root.extract(1)));
            }
        });

        // Complete low-half strip 3 BEFORE high-half strip 7 consumes it.
        // This replaces the incorrect phase5, including for ROWS > COLS.
        static_for<ROW_BLOCKS>([&]<int r>(){
            if constexpr(r * 8 >= 4 * COL_BLOCKS){
                auto& root = data[oriented_row<ANTI>(r)][COL_BLOCKS - 1];
                auto lhs = vstate.init[oriented_row<ANTI>(wrap_row(r - 4 * COL_BLOCKS / 8))];
                auto old = root.extract(0);
                root.insert(0, choose(lam4 == 3, combine(lhs, old), old));
            }
        });
        static_for<ROW_BLOCKS>([&]<int r>(){
            // 4*COL_BLOCKS is a multiple of eight for all supported shapes.
            if constexpr(r * 8 >= 4 * COL_BLOCKS){
                auto lhs = data[oriented_row<ANTI>(wrap_row(r - 4 * COL_BLOCKS / 8))][COL_BLOCKS - 1].extract(0);
                lhs = choose(forward_root_row(r, lar3) + 1 == 4 * COL_BLOCKS,
                             hstate.init[HState::length-1].extract(0), lhs);
                auto& root = data[oriented_row<ANTI>(r)][COL_BLOCKS - 1];
                root.insert(1, combine(lhs, root.extract(1)));
            }
        });

        tmp = exchange_roots<1, ANTI>(vstate, hstate);
        downsweep<true, ANTI>(hstate, tmp);
        static_for<ROW_BLOCKS>([&]<int r>(){
            auto value = data[oriented_row<ANTI>(r)][COL_BLOCKS - 1].extract(1);
            if constexpr(r == 0) value = choose(lar3 == 0, hstate.init[HState::length-1].extract(1), value);
            out_vstate.init[oriented_row<ANTI>(r)] = choose(lam4 == 3, value, out_vstate.init[oriented_row<ANTI>(r)]);
        });
        static_for<HState::length>([&]<int s>(){
            auto preceding = tmp[ROW_BLOCKS - 1];
            if constexpr(s > 0) preceding = data[oriented_row<ANTI>(ROW_BLOCKS - 1)][s * group - 1];
            auto& root = out_hstate.init[s];
            if constexpr((s+1)*group == root_span){
                // The wrapped bottom endpoint is not a predecessor inside
                // the tile. Finish it using the FINAL exchange, and share
                // this combine with the horizontal-boundary output.
                auto prefix = shifted_root<0, root_span>(tmp, lar3);
                preceding = choose(lar3 == 0, prefix, preceding);
                auto tail = choose(lar3 == 0, data[oriented_row<ANTI>(0)][(s+1)*group-1], root);
                root = choose(lar3 < group, combine(preceding, tail), root);
                data[oriented_row<ANTI>(0)][(s+1)*group-1] = choose(lar3 == 0, root, data[oriented_row<ANTI>(0)][(s+1)*group-1]);
            }else{
                root = choose(lar3 < group, combine(preceding, root), root);
                root = choose(lar3 == 0, data[oriented_row<ANTI>(0)][(s+1)*group-1], root);
            }
        });
        return {out_vstate, out_hstate, tmp};
    }

    template<bool ANTI = false>
    __device__ void inclusive_postscan(const HState& incoming_hstate, const ImmState& tmp){
        downsweep<false, ANTI>(incoming_hstate, tmp);
    }

    struct ScanResult { VState first; HState second; };
    using StatePair = ScanResult;
    // Same rolled input and boundary ownership as inclusive_prescan. Only the
    // boundary outputs escape; no postscan is needed. Keep the caller's tile
    // intact while reusing the prescan dependency graph (no extra exchanges).
    __device__ ScanResult reduce_forward(const VState& vstate, const HState& hstate) const {
        SplitScanBuffer work = *this;
        auto result = work.inclusive_prescan(vstate, hstate);
        return {result.vertical, result.horizontal};
    }

    template<bool ANTI = false>
    __device__ ScanResult inclusive_scan(const VState& vstate, const HState& hstate){
        auto [vs, hs, tmp] = inclusive_prescan<ANTI>(vstate, hstate);
        inclusive_postscan<ANTI>(hstate, tmp);
        return {vs, hs};
    }

    // Native reverse layout: physical row = (8*r + lar3 + (c&7) + 1) % ROWS.
    // This is the reverse scan's roll/unroll, not a tile reversal: columns,
    // packed halves and register rows retain their original identities.
    template<bool TO_ROLLED = true, bool ANTI = false>
    __device__ void reverse_roll(){
        int lane = threadIdx.x & 31;
        int lam4 = lane & 3, lar3 = oriented_lar<ANTI>(lane >> 2);
        PElement tmp[ROW_BLOCKS][COL_BLOCKS];
        static_for<ROW_BLOCKS>([&]<int r>(){
            static_for<COL_BLOCKS>([&]<int c>(){
                tmp[r][c] = data[oriented_row<ANTI>(r)][c].shuffle(
                    source_lane<ANTI>(lam4, TO_ROLLED ? lar3 + ((c & 7) + 1) : lar3 - ((c & 7) + 1)));
            });
        });
        static_for<ROW_BLOCKS>([&]<int r>(){
            static_for<COL_BLOCKS>([&]<int c>(){
                if constexpr(TO_ROLLED)
                    data[oriented_row<ANTI>(r)][c] = choose(lar3 > 7 - ((c & 7) + 1), tmp[wrap_row(r + 1)][c], tmp[r][c]);
                else
                    data[oriented_row<ANTI>(r)][c] = choose(lar3 < ((c & 7) + 1), tmp[wrap_row(r - 1)][c], tmp[r][c]);
            });
        });
    }

    template<int R, int DISTANCE>
    __forceinline__ __device__ static PElement reverse_shifted_root(const ImmState& tmp, int lar3){
        constexpr int base = DISTANCE / 8;
        if constexpr(DISTANCE % 8 == 0)
            return tmp[wrap_row(R + base)];
        else
            return choose(lar3 >= 8 - DISTANCE % 8,
                          tmp[wrap_row(R + base + 1)], tmp[wrap_row(R + base)]);
    }

    template<int STRIPS, bool ANTI = false>
    __forceinline__ __device__ ImmState reverse_exchange_roots(const VState& vstate, const HState& hstate) const {
        int lane = threadIdx.x & 31;
        int lam4 = lane & 3, lar3 = oriented_lar<ANTI>(lane >> 2);
        ImmState tmp;
        static_for<ROW_BLOCKS>([&]<int r>(){
            auto send = data[oriented_row<ANTI>(r)][0];
            if constexpr(r == ROW_BLOCKS-1) send = choose(lar3 == 7, hstate.init[0], send);
            auto shifted = send;
            shifted.insert(0, send.extract(1));
            shifted.insert(1, vstate.init[oriented_row<ANTI>(r)]);
            send = choose(lam4 < STRIPS, shifted, send);
            tmp[r] = send.shuffle(source_lane<ANTI>(lam4 + STRIPS, lar3 + STRIPS * COL_BLOCKS));
        });
        return tmp;
    }

    template<bool COARSE, bool ANTI = false>
    __forceinline__ __device__ void reverse_downsweep(const HState& hstate, const ImmState& tmp){
        constexpr int group = COL_BLOCKS < 8 ? COL_BLOCKS : 8;
        constexpr int bits = __builtin_ffs(COL_BLOCKS) - 1;
        int lar3 = oriented_lar<ANTI>((threadIdx.x & 31) >> 2);
        static_for<bits>([&]<int sweep>(){
            constexpr int stride = 1 << (bits - 1 - sweep);
            if constexpr(stride < ROWS && ((stride >= group) == COARSE)){
                static_for<COL_BLOCKS / (2 * stride)>([&]<int node>(){
                    constexpr int c = COL_BLOCKS - stride - node * 2 * stride;
                    static_for<ROW_BLOCKS>([&]<int index>(){
                        constexpr int r = ROW_BLOCKS - 1 - index;
                        PElement successor;
                        if constexpr(c + stride < COL_BLOCKS){
                            constexpr int displacement = (c + stride) / 8 - c / 8;
                            successor = data[oriented_row<ANTI>(wrap_row(r + displacement))][c + stride];
                        }else if constexpr(COL_BLOCKS >= 8){
                            successor = tmp[wrap_row(r + (stride + 7) / 8)];
                        }else{
                            successor = reverse_shifted_root<r, COL_BLOCKS>(tmp, lar3);
                        }
                        if constexpr(COARSE){
                            int row = (r * 8 + lar3 + (c & 7) + 1) & (ROWS - 1);
                            bool pred = c + stride >= COL_BLOCKS ? row + stride <= ROWS : row + stride < ROWS;
                            data[oriented_row<ANTI>(r)][c] = choose(pred, combine(successor, data[oriented_row<ANTI>(r)][c]), data[oriented_row<ANTI>(r)][c]);
                        }else{
                            successor = fine_boundary<r, c, stride, true>(successor, hstate, lar3);
                            data[oriented_row<ANTI>(r)][c] = combine(successor, data[oriented_row<ANTI>(r)][c]);
                        }
                    });
                });
            }
        });
    }

    // Native reverse successor tree. Input must be in reverse_roll() layout.
    // Composition follows traversal order: result[i,j] = result[i+1,j+1].op(input[i,j]).
    // VState: lam4==0 owns row 8*r+lar3+1 (right input / left output).
    // HState: lar3>=8-group is active; slot s, half e owns column
    // (lam4+4*e)*COL_BLOCKS+s*group+7-lar3 (bottom input / top output).
    // The bottom-left corner is carried in VState row ROWS.
    template<bool ANTI = false>
    __device__ PrescanResult reverse_inclusive_prescan(const VState& vstate, const HState& hstate){
        // A root spans at most the tile height, even in a wider strip.
        constexpr int root_span = COL_BLOCKS < ROWS ? COL_BLOCKS : ROWS;
        constexpr int bits = __builtin_ffs(COL_BLOCKS) - 1;
        constexpr int group = COL_BLOCKS < 8 ? COL_BLOCKS : 8;
        int lane = threadIdx.x & 31;
        int lam4 = lane & 3, lar3 = oriented_lar<ANTI>(lane >> 2);
        HState out_hstate;
        VState out_vstate;
        static_for<bits>([&]<int iter>(){
            constexpr int stride = 1 << iter;
            if constexpr(stride < ROWS){
                static_for<COL_BLOCKS / (2 * stride)>([&]<int node>(){
                    constexpr int c = COL_BLOCKS - 2 * stride - node * 2 * stride;
                    static_for<ROW_BLOCKS>([&]<int index>(){
                        constexpr int r = ROW_BLOCKS - 1 - index;
                        auto successor = data[oriented_row<ANTI>(wrap_row(r + stride / 8))][c + stride];
                        int row = (r * 8 + lar3 + (c & 7) + 1) & (ROWS - 1);
                        bool pred = row + stride < ROWS;
                        if constexpr(r == ROW_BLOCKS - 1 && stride < group){
                            combine_either(
                                pred, successor, data[oriented_row<ANTI>(r)][c], out_hstate.init[c / group]);
                        }else{
                            data[oriented_row<ANTI>(r)][c] = choose(pred, combine(successor, data[oriented_row<ANTI>(r)][c]), data[oriented_row<ANTI>(r)][c]);
                        }
                    });
                });
                if constexpr(stride == group / 2){
                    static_for<HState::length>([&]<int s>(){
                        auto& root = data[oriented_row<ANTI>(ROW_BLOCKS - 1)][s * group];
                        if constexpr(group == 8 && s+1 < HState::length){
                            auto incoming = choose(lar3 == 7, hstate.init[s+1], hstate.init[s]);
                            combine_either(lar3 != 7, incoming, root, data[oriented_row<ANTI>(ROW_BLOCKS-2)][s*group]);
                        }else root = choose(lar3 >= 8 - group && lar3 < 7, combine(hstate.init[s], root), root);
                    });
                }
            }
        });
        // Signed frontier row 8*r+lar3+1: ROWS denotes incoming bottom
        // state. Its wrapped top tile root is completed after the final
        // exchange, sharing the H-state combine rather than another hop.
        auto tmp = reverse_exchange_roots<1, ANTI>(vstate, hstate);
        static_for<ROW_BLOCKS>([&]<int index>(){
            constexpr int r = ROW_BLOCKS - 1 - index;
            if constexpr(r * 8 + COL_BLOCKS < ROWS){
                auto& root = data[oriented_row<ANTI>(r)][0];
                auto successor = reverse_shifted_root<r, COL_BLOCKS>(tmp, lar3);
                root = choose(r * 8 + lar3 + COL_BLOCKS < ROWS, combine(successor, root), root);
            }
        });
        tmp = reverse_exchange_roots<2, ANTI>(vstate, hstate);
        static_for<ROW_BLOCKS>([&]<int index>(){
            constexpr int r = ROW_BLOCKS - 1 - index;
            if constexpr(r * 8 + 2 * COL_BLOCKS < ROWS){
                auto& root = data[oriented_row<ANTI>(r)][0];
                auto successor = reverse_shifted_root<r, 2 * COL_BLOCKS>(tmp, lar3);
                auto combined = combine(successor, root);
                bool pred = r * 8 + lar3 + 2 * COL_BLOCKS < ROWS;
                // High-half strip 7 already includes its right initial value.
                root.insert(1, choose(pred && lam4 != 3, combined.extract(1), root.extract(1)));
                root.insert(0, choose(pred, combined.extract(0), root.extract(0)));
            }
        });
        // Finish high-half strip 4 before low-half strip 0 consumes it.
        static_for<ROW_BLOCKS>([&]<int index>(){
            constexpr int r = ROW_BLOCKS - 1 - index;
            if constexpr((r + 1) * 8 + 4 * COL_BLOCKS <= ROWS){
                auto& root = data[oriented_row<ANTI>(r)][0];
                auto successor = vstate.init[oriented_row<ANTI>(wrap_row(r + 4 * COL_BLOCKS / 8))];
                auto old = root.extract(1);
                root.insert(1, choose(lam4 == 0, combine(successor, old), old));
            }
        });
        static_for<ROW_BLOCKS>([&]<int index>(){
            constexpr int r = ROW_BLOCKS - 1 - index;
            if constexpr((r + 1) * 8 + 4 * COL_BLOCKS <= ROWS){
                auto successor = data[oriented_row<ANTI>(wrap_row(r + 4 * COL_BLOCKS / 8))][0].extract(1);
                successor = choose(backward_root_row(r, lar3) + 4 * COL_BLOCKS == ROWS,
                                   hstate.init[0].extract(1), successor);
                auto& root = data[oriented_row<ANTI>(r)][0];
                root.insert(0, combine(successor, root.extract(0)));
            }
        });
        tmp = reverse_exchange_roots<1, ANTI>(vstate, hstate);
        reverse_downsweep<true, ANTI>(hstate, tmp);
        static_for<ROW_BLOCKS>([&]<int r>(){
            auto value = data[oriented_row<ANTI>(r)][0].extract(0);
            if constexpr(r == ROW_BLOCKS-1) value = choose(lar3 == 7, hstate.init[0].extract(0), value);
            out_vstate.init[oriented_row<ANTI>(r)] = choose(lam4 == 0, value, out_vstate.init[oriented_row<ANTI>(r)]);
        });
        static_for<HState::length>([&]<int s>(){
            auto successor = tmp[0];
            if constexpr(s + 1 < HState::length) successor = data[oriented_row<ANTI>(0)][(s + 1) * group];
            auto& root = out_hstate.init[s];
            if constexpr(s*group == COL_BLOCKS-root_span){
                // Symmetric deferred top endpoint; no intermediate root
                // exchange needs this wrapped tile value.
                auto prefix = reverse_shifted_root<ROW_BLOCKS-1, root_span>(tmp, lar3);
                successor = choose(lar3 == 7, prefix, successor);
                auto tail = choose(lar3 == 7, data[oriented_row<ANTI>(ROW_BLOCKS-1)][s*group], root);
                root = choose(lar3 >= 8-group, combine(successor, tail), root);
                data[oriented_row<ANTI>(ROW_BLOCKS-1)][s*group] = choose(lar3 == 7, root, data[oriented_row<ANTI>(ROW_BLOCKS-1)][s*group]);
            }else{
                root = choose(lar3 >= 8 - group, combine(successor, root), root);
                root = choose(lar3 == 7, data[oriented_row<ANTI>(ROW_BLOCKS-1)][s*group], root);
            }
        });
        return {out_vstate, out_hstate, tmp};
    }

    template<bool ANTI = false>
    __device__ void reverse_inclusive_postscan(const HState& incoming_hstate, const ImmState& tmp){
        reverse_downsweep<false, ANTI>(incoming_hstate, tmp);
    }

    // Boundary-only reverse traversal, preserving the reverse-rolled tile.
    // Reuses the native successor tree without postscan or extra exchanges.
    __device__ ScanResult reduce_backward(const VState& vstate, const HState& hstate) const {
        SplitScanBuffer work = *this;
        auto result = work.reverse_inclusive_prescan(vstate, hstate);
        return {result.vertical, result.horizontal};
    }

    template<bool ANTI = false>
    __device__ ScanResult reverse_inclusive_scan(const VState& vstate, const HState& hstate){
        auto result = reverse_inclusive_prescan<ANTI>(vstate, hstate);
        reverse_inclusive_postscan<ANTI>(hstate, result.intermediate);
        return {result.vertical, result.horizontal};
    }



    // Anti forward: top-right -> bottom-left; physical rolled row is
    // (8*r + lar3 - (c&7) - 1) mod ROWS. Columns and packed halves stay in place.
    template<bool TO_ROLLED = true>
    __device__ void anti_roll(){ reverse_roll<TO_ROLLED, true>(); }

    // Anti reverse: bottom-left -> top-right; physical rolled row is
    // (8*r + lar3 + ((COL_BLOCKS-1-c)&7) + 1) mod ROWS.
    template<bool TO_ROLLED = true>
    __device__ void anti_reverse_roll(){ roll<TO_ROLLED, true>(); }

    // VState: lam4==0 owns physical row 8*r+lar3-1 (right input / left output).
    // HState: lar3<group; slot s, half e owns column
    // (lam4+4*e)*COL_BLOCKS+s*group+lar3 (top input / bottom output).
    // VState row -1 carries the incoming top-left corner. group=min(COL_BLOCKS,8).
    __device__ PrescanResult anti_inclusive_prescan(const VState& right, const HState& top){
        return reverse_inclusive_prescan<true>(right, top);
    }
    __device__ void anti_inclusive_postscan(const HState& top, const ImmState& tmp){
        reverse_inclusive_postscan<true>(top, tmp);
    }
    __device__ ScanResult anti_inclusive_scan(const VState& right, const HState& top){
        auto result = anti_inclusive_prescan(right, top);
        anti_inclusive_postscan(top, result.intermediate);
        return {result.vertical, result.horizontal};
    }

    // VState: lam4==3 owns physical row 8*r+lar3+1 (left input / right output).
    // HState: lar3>=8-group; slot s, half e owns column
    // (lam4+4*e)*COL_BLOCKS+s*group+group-8+lar3 (bottom input / top output).
    // VState row ROWS carries the incoming bottom-right corner.
    __device__ PrescanResult anti_reverse_inclusive_prescan(const VState& left, const HState& bottom){
        return inclusive_prescan<true>(left, bottom);
    }
    __device__ void anti_reverse_inclusive_postscan(const HState& bottom, const ImmState& tmp){
        inclusive_postscan<true>(bottom, tmp);
    }
    __device__ ScanResult anti_reverse_inclusive_scan(const VState& left, const HState& bottom){
        auto result = anti_reverse_inclusive_prescan(left, bottom);
        anti_reverse_inclusive_postscan(bottom, result.intermediate);
        return {result.vertical, result.horizontal};
    }

    // Boundary-only anti scans preserve the respective rolled tile.
    __device__ ScanResult anti_reduce_forward(const VState& right, const HState& top) const {
        SplitScanBuffer work = *this;
        auto result = work.anti_inclusive_prescan(right, top);
        return {result.vertical, result.horizontal};
    }
    __device__ ScanResult anti_reduce_backward(const VState& left, const HState& bottom) const {
        SplitScanBuffer work = *this;
        auto result = work.anti_reverse_inclusive_prescan(left, bottom);
        return {result.vertical, result.horizontal};
    }

};

} // namespace glx
