// Standalone primary-diagonal split scan/reduce with adjacent packed strips.
// Based on the self-shuffle-free SplitScanBuffer; no layout conversion wrapper.
// Native forward/reverse scans with column-aligned horizontal boundaries.
// Self-contained CUDA/C++20 header. No shared-memory transfer API is provided.
// Adjacent-strip ownership and API: experiments/alt_layout/README.md
// Each warp owns a tile; every lane must participate in scan, roll and print.
#pragma once

#include <cstdio>
#include <concepts>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <type_traits>
#include <utility>

namespace pscore {

// These declarations also appear in diagonal_scan.cuh so each header remains
// independently usable. Keep this guarded block compatible between the headers.
#ifndef pscore_DIAGONAL_SCAN_VALUE_TYPES_DEFINED
#define pscore_DIAGONAL_SCAN_VALUE_TYPES_DEFINED

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

#endif // pscore_DIAGONAL_SCAN_VALUE_TYPES_DEFINED

// Primary-diagonal forward scan only. All 32 lanes participate.
// Forward HState owns columns [0,COLS-1], VState rows [-1,ROWS-2].
// The signed outside row is a corner from incoming HState, not wrapped data.
// Prescan returns final boundaries; postscan
// consumes the original incoming HState and the saved intermediate state.
template<int ROWS_, int COLS_,
         template<typename> class ElementType = UnaryElement,
         typename Op = pscore::AddOp, typename PackedType = pscore::F32x2>
struct AltLayoutSplitScanBuffer{
    static constexpr int ROWS = ROWS_;
    static constexpr int COLS = COLS_;
    static_assert(ROWS >= 16 && (ROWS & (ROWS - 1)) == 0,
                  "ROWS must be a power of two >= 16");
    static_assert(COLS >= 16 && (COLS & (COLS - 1)) == 0,
                  "COLS must be a power of two >= 16");
    using PElement = ElementType<PackedType>;
    using UElement = ElementType<typename pscore::PackTraits<PackedType>::unpacked_type>;
    using packed_type = PackedType;
    using operation_type = Op;
    static_assert(pscore::PackTraits<PackedType>::lanes == 2);
    static_assert(pscore::ClosedBinaryOp<Op, PElement>);
    static_assert(pscore::ClosedBinaryOp<Op, UElement>);

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

    // lane = 4*row_lane + thread_column. Adjacent strips share a pack:
    // t0.u0, t0.u1, t1.u0, t1.u1, t2.u0, t2.u1, t3.u0, t3.u1.
    // Both halves retain the SAME physical row and roll displacement.
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

        // Print physical columns, not the sparse register/lane storage view.
        // Forward top/bottom: print(); reverse bottom/top: print<true>().
        // All 32 lanes must participate, including inactive state owners.
        template<bool REVERSE = false>
        __device__ __forceinline__ void print(int colsep_interval = 1) const {
            bool leader = (threadIdx.x & 31) == 0;
            constexpr int group = COL_BLOCKS < 8 ? COL_BLOCKS : 8;
            if(colsep_interval < 1) colsep_interval = 1;
            if(leader) printf("     ");
            for(int col = 0; col < COLS; ++col)
                if(leader) printf(" %9d", col);
            if(leader) printf("\nH    ");
            for(int col = 0; col < COLS; ++col){
                int strip = col / COL_BLOCKS;
                int row_lane = (REVERSE ? 7 : group - 1) - col % group;
                int src_lane = 4 * row_lane + strip / 2;
                auto value = init[(col % COL_BLOCKS) / group]
                    .extract(strip & 1).shuffle(src_lane);
                if(leader){
                    printf(col % colsep_interval == 0 ? " │" : "  ");
                    value.print();
                }
            }
            if(leader) printf(" │\n");
        }
    };

    struct ImmState {
        static constexpr int length = ROW_BLOCKS;
        PElement imm[ROW_BLOCKS];
        __host__ __device__ ImmState(){
            #pragma unroll
            for(int r = 0; r < ROW_BLOCKS; ++r) imm[r] = identity<PElement>();
        }
        __device__ __forceinline__ PElement& operator[](int idx) { return imm[idx]; }
        __device__ __forceinline__ const PElement& operator[](int idx) const { return imm[idx]; }
    };

    struct Coordinate { int first, second; };
    __forceinline__ __device__ static Coordinate layout(int r, int c, int e){
        int lane = threadIdx.x & 0x1f;
        int lam4 = lane & 0x3, lar3 = lane >> 2;
        return { r * THREAD_ROWS + lar3, c + (2 * lam4 + e) * COL_BLOCKS };
    }

    template<bool FWD = true>
    __device__ __forceinline__ void roll(){
        int lane = threadIdx.x & 31;
        int lam4 = lane & 3, lar3 = lane >> 2;
        PElement tmp[ROW_BLOCKS][COL_BLOCKS];
        static_for<ROW_BLOCKS>([&]<int r>() __attribute__((always_inline)) {
            static_for<COL_BLOCKS>([&]<int j>() __attribute__((always_inline)) {
                if constexpr((j & 7) == 7){
                    // A full eight-row rotation keeps the lane; the next
                    // loop handles the register row-block displacement.
                    tmp[r][j] = data[r][COL_BLOCKS - 1 - j];
                }else{
                    tmp[r][j] = data[r][COL_BLOCKS - 1 - j].shuffle(
                        source_lane(lam4, FWD ? lar3 - ((j & 7) + 1) : lar3 + ((j & 7) + 1)));
                }
            });
        });
        static_for<ROW_BLOCKS>([&]<int r>() __attribute__((always_inline)) {
            static_for<COL_BLOCKS>([&]<int j>() __attribute__((always_inline)) {
                if constexpr(FWD)
                    data[r][COL_BLOCKS - 1 - j] = choose(lar3 < ((j & 7) + 1), tmp[wrap_row(r - 1)][j], tmp[r][j]);
                else
                    data[r][COL_BLOCKS - 1 - j] = choose(lar3 > 7 - ((j & 7) + 1), tmp[wrap_row(r + 1)][j], tmp[r][j]);
            });
        });
    }

    // Print an unrolled tile in physical row/column order. Both packed halves
    // belong to the same thread column: strip=2*(lane&3)+half.
    __device__ __forceinline__ void print(int colsep_interval = 1) const {
        if(colsep_interval < 1) colsep_interval = 1;
        bool leader = (threadIdx.x & 0x1f) == 0;
        if(leader)printf("     ");
        for(int i = 0; i < COLS; i++) if(leader) printf(" %9d", i);
        if(leader)printf("\n      ");
        for(int i = 0; i < COLS; i++) if(leader) printf("%s─────────", (i % colsep_interval == 0) ? (i ? "┬" : "┌") : "─");
        if(leader)printf("┐\n");
        for(int i = 0; i < ROWS; i++){
            if(leader) printf("%4d ", i);
            for(int j = 0; j < COLS; j++){
                int src_laneid = (i % THREAD_ROWS) * THREAD_COLS + (j / (2 * COL_BLOCKS)) % THREAD_COLS;
                auto curr = data[i / THREAD_ROWS][j % COL_BLOCKS].extract((j / COL_BLOCKS) & 0x1).shuffle(src_laneid);
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
        [&]<int... I>(std::integer_sequence<int, I...>) __attribute__((always_inline)) {
            (function.template operator()<I>(), ...);
        }(std::make_integer_sequence<int, N>{});
    }

    template<typename T>
    __forceinline__ __device__ static T choose(bool predicate, const T& yes, const T& no){
        return T::select(static_cast<unsigned>(predicate), no, yes);
    }

    // A rolled register (r,c) represents this physical row in both halves.
    __forceinline__ __device__ static int physical_row(int r, int c, int lar3){
        return (r * 8 + lar3 - ((COL_BLOCKS - 1 - c) & 7) - 1 + ROWS) % ROWS;
    }

    __forceinline__ __device__ static constexpr int wrap_row(int r){
        return r & (ROW_BLOCKS - 1);
    }

    __forceinline__ __device__ static int source_lane(int lam4, int lar3){
        return (lam4 & 3) + ((lar3 & 7) * 4);
    }

    __forceinline__ __device__ static int forward_root_row(int r, int lar){
        return (8*r + lar - 1 + ROWS) & (ROWS-1);
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

    struct RootState { UElement root[ROW_BLOCKS]; };

    // The row-zero/lane-zero register is the wrapped bottom endpoint in the
    // tile, but represents the incoming top frontier when sent as a prefix.
    template<int R, int HALF>
    __forceinline__ __device__ UElement root_frontier(const HState& hstate, int lar) const {
        auto value = data[R][COL_BLOCKS-1].extract(HALF);
        if constexpr(R == 0)
            value = choose(lar == 0, hstate.init[HState::length-1].extract(HALF), value);
        return value;
    }

    // Only strip roots move independently. The tile's packed roll is unchanged.
    // THREADS==0 reads the other half of this thread column; for K divisible
    // by eight this is entirely register-local. Crossing the left edge reads
    // VState, whose owner remains thread column 3.
    template<int DISTANCE, int THREADS, int HALF>
    __forceinline__ __device__ RootState exchange_half_roots(
            const VState& vstate, const HState& hstate) const {
        int t = threadIdx.x & 3, lar = (threadIdx.x & 31) >> 2;
        RootState result;
        auto send_root = [&]<int r>() __attribute__((always_inline)) {
            auto send = root_frontier<r, HALF>(hstate, lar);
            if constexpr(THREADS > 0)
                send = choose(t >= 4-THREADS, vstate.init[r], send);
            return send;
        };
        if constexpr(THREADS == 0 && DISTANCE % 8 == 0){
            static_for<ROW_BLOCKS>([&]<int r>() __attribute__((always_inline)) {
                result.root[r] = send_root.template operator()<r>();
            });
        }else{
            // Two register row blocks have the same source lane. Pack their
            // roots together so BF16 transport also uses both word halves;
            // FP32 has the same word count as separate root shuffles.
            static_for<ROW_BLOCKS/2>([&]<int pair>() __attribute__((always_inline)) {
                PElement packet;
                packet.insert(0, send_root.template operator()<2*pair>());
                packet.insert(1, send_root.template operator()<2*pair+1>());
                auto received = packet.shuffle(source_lane(t-THREADS, lar-DISTANCE));
                result.root[2*pair] = received.extract(0);
                result.root[2*pair+1] = received.extract(1);
            });
        }
        return result;
    }

    template<int R, int DISTANCE>
    __forceinline__ __device__ static UElement shifted_half_root(const RootState& roots, int lar){
        constexpr int base = DISTANCE / 8;
        if constexpr(DISTANCE % 8 == 0)
            return roots.root[wrap_row(R-base)];
        else return choose(lar < DISTANCE % 8, roots.root[wrap_row(R-base-1)],
                                                  roots.root[wrap_row(R-base)]);
    }

    // Fine downsweep has at most two live sources per register. In forward
    // coordinates, row block 0 can consume this group's HState; row block 1
    // can consume the previous group's HState only at lar==0. These cases
    // never coexist for the same compile-time (r,c).
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
    template<bool COARSE>
    __forceinline__ __device__ void downsweep(const HState& hstate, const ImmState& tmp){
        constexpr int group = COL_BLOCKS < 8 ? COL_BLOCKS : 8;
        constexpr int bits = __builtin_ffs(COL_BLOCKS) - 1;
        int lar3 = (threadIdx.x & 31) >> 2;
        static_for<bits>([&]<int sweep>() __attribute__((always_inline)) {
            constexpr int stride = 1 << (bits - 1 - sweep);
            if constexpr(stride < ROWS && ((stride >= group) == COARSE)){
                static_for<COL_BLOCKS / (2 * stride)>([&]<int node>() __attribute__((always_inline)) {
                    constexpr int c = stride - 1 + node * 2 * stride;
                    static_for<ROW_BLOCKS>([&]<int r>() __attribute__((always_inline)) {
                        PElement lhs;
                        if constexpr(c >= stride){
                            constexpr int displacement = c / 8 - (c - stride) / 8;
                            lhs = data[wrap_row(r - displacement)][c - stride];
                        }else if constexpr(COL_BLOCKS >= 8){
                            lhs = tmp[wrap_row(r - (stride + 7) / 8)];
                        }else{
                            lhs = shifted_root<r, COL_BLOCKS>(tmp, lar3);
                        }
                        if constexpr(COARSE){
                            // Roots reaching the top already include HState.
                            int row = physical_row(r, c, lar3);
                            bool pred = c < stride ? row + 1 >= stride : row >= stride;
                            data[r][c] = choose(pred, combine(lhs, data[r][c]), data[r][c]);
                        }else{
                            lhs = fine_boundary<r, c, stride>(lhs, hstate, lar3);
                            data[r][c] = combine(lhs, data[r][c]);
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
    // column (2*lam4+e)*COL_BLOCKS+s*group+group-1-lar3 of the bottom edge
    // (or top input). The top-right corner is carried in VState row -1.
    __device__ __forceinline__ PrescanResult inclusive_prescan(const VState& vstate, const HState& hstate){
        // A root spans at most the tile height, even in a wider strip.
        constexpr int root_span = COL_BLOCKS < ROWS ? COL_BLOCKS : ROWS;
        constexpr int bits = __builtin_ffs(COL_BLOCKS) - 1;
        constexpr int group = COL_BLOCKS < 8 ? COL_BLOCKS : 8;
        int lane = threadIdx.x & 31;
        int lam4 = lane & 3, lar3 = lane >> 2;
        HState out_hstate;
        VState out_vstate;

        static_for<bits>([&]<int iter>() __attribute__((always_inline)) {
            constexpr int stride = 1 << iter;
            if constexpr(stride < ROWS){
                static_for<COL_BLOCKS / (2 * stride)>([&]<int node>() __attribute__((always_inline)) {
                    constexpr int c = 2 * stride - 1 + node * 2 * stride;
                    static_for<ROW_BLOCKS>([&]<int r>() __attribute__((always_inline)) {
                        auto lhs = data[wrap_row(r - stride / 8)][c - stride];
                        bool pred = physical_row(r, c, lar3) >= stride;
                        if constexpr(r == 0 && stride < group){
                            // The two destinations are updated under opposite
                            // predicates; each lane evaluates only one op.
                            combine_either(
                                pred, lhs, data[r][c], out_hstate.init[c / group]);
                        }else{
                            data[r][c] = choose(pred, combine(lhs, data[r][c]), data[r][c]);
                        }
                    });
                });
                if constexpr(stride == group / 2){
                    static_for<HState::length>([&]<int s>() __attribute__((always_inline)) {
                        auto& root = data[0][(s + 1) * group - 1];
                        if constexpr(group == 8 && s > 0){
                            auto incoming = choose(lar3 == 0, hstate.init[s-1], hstate.init[s]);
                            combine_either(lar3 != 0, incoming, root, data[1][(s+1)*group-1]);
                        }else root = choose(lar3 > 0 && lar3 < group, combine(hstate.init[s], root), root);
                    });
                }
            }
        });

        // Join the two adjacent strips locally before the four-thread scan.
        // In every stage the signed root row is 8*r+lar3-1. Wrapped bottom
        // endpoints are deferred to the existing H-state completion below.
        if constexpr(COL_BLOCKS < ROWS){
            auto low = exchange_half_roots<COL_BLOCKS, 0, 0>(vstate, hstate);
            static_for<ROW_BLOCKS>([&]<int r>() __attribute__((always_inline)) {
                if constexpr(r*8+7 >= COL_BLOCKS){
                    auto lhs = shifted_half_root<r, COL_BLOCKS>(low, lar3);
                    auto& root = data[r][COL_BLOCKS-1];
                    auto old = root.extract(1);
                    root.insert(1, choose(r*8+lar3 >= COL_BLOCKS, combine(lhs, old), old));
                }
            });
        }

        // Four pair-endpoints, distances one and two THREAD columns. Insert
        // left input at the wrap, just as in a four-element inclusive tree.
        static_for<2>([&]<int step>() __attribute__((always_inline)) {
            constexpr int threads = 1 << step;
            constexpr int distance = 2*threads*COL_BLOCKS;
            if constexpr(distance < ROWS){
                auto preceding = exchange_half_roots<distance, threads, 1>(vstate, hstate);
                static_for<ROW_BLOCKS>([&]<int r>() __attribute__((always_inline)) {
                    if constexpr(r*8+7 >= distance){
                        auto lhs = shifted_half_root<r, distance>(preceding, lar3);
                        auto& root = data[r][COL_BLOCKS-1];
                        auto old = root.extract(1);
                        bool pred = r*8+lar3 >= distance;
                        if constexpr(step == 1) pred = pred && lam4 != 0;
                        root.insert(1, choose(pred, combine(lhs, old), old));
                    }
                });
            }
        });
        // The last pair endpoint has accumulated the entire tile width but
        // has not received V yet. Its owner is already VState's owner (t3).
        static_for<ROW_BLOCKS>([&]<int r>() __attribute__((always_inline)) {
            if constexpr(r*8 >= COLS){
                auto& root = data[r][COL_BLOCKS-1];
                auto old = root.extract(1);
                auto lhs = vstate.init[wrap_row(r-COLS/8)];
                root.insert(1, choose(lam4 == 3, combine(lhs, old), old));
            }
        });

        // Complete the low roots from the preceding pair's final high root.
        // Keep this same exchange as the low-half postscan entry.
        auto preceding = exchange_half_roots<COL_BLOCKS, 1, 1>(vstate, hstate);
        static_for<ROW_BLOCKS>([&]<int r>() __attribute__((always_inline)) {
            if constexpr(r*8+7 >= COL_BLOCKS){
                auto lhs = shifted_half_root<r, COL_BLOCKS>(preceding, lar3);
                auto& root = data[r][COL_BLOCKS-1];
                auto old = root.extract(0);
                root.insert(0, choose(r*8+lar3 >= COL_BLOCKS, combine(lhs, old), old));
            }
        });
        // High-half postscan consumes this thread's completed low root.
        auto low_prefix = exchange_half_roots<COL_BLOCKS, 0, 0>(vstate, hstate);
        ImmState tmp;
        static_for<ROW_BLOCKS>([&]<int r>() __attribute__((always_inline)) {
            tmp[r].insert(0, preceding.root[r]);
            tmp[r].insert(1, low_prefix.root[r]);
        });
        downsweep<true>(hstate, tmp);
        static_for<ROW_BLOCKS>([&]<int r>() __attribute__((always_inline)) {
            auto value = data[r][COL_BLOCKS - 1].extract(1);
            if constexpr(r == 0) value = choose(lar3 == 0, hstate.init[HState::length-1].extract(1), value);
            out_vstate.init[r] = choose(lam4 == 3, value, out_vstate.init[r]);
        });
        static_for<HState::length>([&]<int s>() __attribute__((always_inline)) {
            auto preceding = tmp[ROW_BLOCKS - 1];
            if constexpr(s > 0) preceding = data[ROW_BLOCKS - 1][s * group - 1];
            auto& root = out_hstate.init[s];
            if constexpr((s+1)*group == root_span){
                // The wrapped bottom endpoint is not a predecessor inside
                // the tile. Finish it using the FINAL exchange, and share
                // this combine with the horizontal-boundary output.
                auto prefix = shifted_root<0, root_span>(tmp, lar3);
                preceding = choose(lar3 == 0, prefix, preceding);
                auto tail = choose(lar3 == 0, data[0][(s+1)*group-1], root);
                root = choose(lar3 < group, combine(preceding, tail), root);
                data[0][(s+1)*group-1] = choose(lar3 == 0, root, data[0][(s+1)*group-1]);
            }else{
                root = choose(lar3 < group, combine(preceding, root), root);
                root = choose(lar3 == 0, data[0][(s+1)*group-1], root);
            }
        });
        return {out_vstate, out_hstate, tmp};
    }

    __device__ __forceinline__ void inclusive_postscan(const HState& incoming_hstate, const ImmState& tmp){
        downsweep<false>(incoming_hstate, tmp);
    }

    struct ScanResult { VState first; HState second; };
    using StatePair = ScanResult;
    // Same rolled input and boundary ownership as inclusive_prescan. Only the
    // boundary outputs escape; no postscan is needed. Keep the caller's tile
    // intact while reusing the prescan dependency graph (no extra exchanges).
    __device__ __forceinline__ ScanResult reduce_forward(const VState& vstate, const HState& hstate) const {
        AltLayoutSplitScanBuffer work = *this;
        auto result = work.inclusive_prescan(vstate, hstate);
        return {result.vertical, result.horizontal};
    }

    __device__ __forceinline__ ScanResult inclusive_scan(const VState& vstate, const HState& hstate){
        auto [vs, hs, tmp] = inclusive_prescan(vstate, hstate);
        inclusive_postscan(hstate, tmp);
        return {vs, hs};
    }

    // Native reverse layout: physical row = (8*r + lar3 + (c&7) + 1) % ROWS.
    // This is the reverse scan's roll/unroll, not a tile reversal: columns,
    // packed halves and register rows retain their original identities.
    template<bool TO_ROLLED = true>
    __device__ __forceinline__ void reverse_roll(){
        int lane = threadIdx.x & 31;
        int lam4 = lane & 3, lar3 = (lane >> 2);
        PElement tmp[ROW_BLOCKS][COL_BLOCKS];
        static_for<ROW_BLOCKS>([&]<int r>() __attribute__((always_inline)) {
            static_for<COL_BLOCKS>([&]<int c>() __attribute__((always_inline)) {
                if constexpr((c & 7) == 7){
                    // Same-lane rotation: only the register row block moves.
                    tmp[r][c] = data[r][c];
                }else{
                    tmp[r][c] = data[r][c].shuffle(
                        source_lane(lam4, TO_ROLLED ? lar3 + ((c & 7) + 1) : lar3 - ((c & 7) + 1)));
                }
            });
        });
        static_for<ROW_BLOCKS>([&]<int r>() __attribute__((always_inline)) {
            static_for<COL_BLOCKS>([&]<int c>() __attribute__((always_inline)) {
                if constexpr(TO_ROLLED)
                    data[r][c] = choose(lar3 > 7 - ((c & 7) + 1), tmp[wrap_row(r + 1)][c], tmp[r][c]);
                else
                    data[r][c] = choose(lar3 < ((c & 7) + 1), tmp[wrap_row(r - 1)][c], tmp[r][c]);
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

    // Reverse roots live at c==0, with signed row 8*r+lar+1.
    // The wrapped row-zero tile value is deferred; send bottom input instead.
    template<int R, int HALF>
    __forceinline__ __device__ UElement reverse_root_frontier(const HState& hstate, int lar) const {
        auto value = data[R][0].extract(HALF);
        if constexpr(R == ROW_BLOCKS-1)
            value = choose(lar == 7, hstate.init[0].extract(HALF), value);
        return value;
    }

    template<int DISTANCE, int THREADS, int HALF>
    __forceinline__ __device__ RootState reverse_exchange_half_roots(
            const VState& vstate, const HState& hstate) const {
        int t = threadIdx.x & 3, lar = (threadIdx.x & 31) >> 2;
        RootState result;
        auto send_root = [&]<int r>() __attribute__((always_inline)) {
            auto send = reverse_root_frontier<r, HALF>(hstate, lar);
            if constexpr(THREADS > 0)
                send = choose(t < THREADS, vstate.init[r], send);
            return send;
        };
        if constexpr(THREADS == 0 && DISTANCE % 8 == 0){
            static_for<ROW_BLOCKS>([&]<int r>() __attribute__((always_inline)) {
                result.root[r] = send_root.template operator()<r>();
            });
        }else{
            // Keep BF16 roots packed, as in the forward exchange.
            static_for<ROW_BLOCKS/2>([&]<int pair>() __attribute__((always_inline)) {
                PElement packet;
                packet.insert(0, send_root.template operator()<2*pair>());
                packet.insert(1, send_root.template operator()<2*pair+1>());
                auto received = packet.shuffle(source_lane(t+THREADS, lar+DISTANCE));
                result.root[2*pair] = received.extract(0);
                result.root[2*pair+1] = received.extract(1);
            });
        }
        return result;
    }

    template<int R, int DISTANCE>
    __forceinline__ __device__ static UElement reverse_shifted_half_root(const RootState& roots, int lar){
        constexpr int base = DISTANCE / 8;
        if constexpr(DISTANCE % 8 == 0)
            return roots.root[wrap_row(R+base)];
        else return choose(lar >= 8-DISTANCE%8, roots.root[wrap_row(R+base+1)],
                                                roots.root[wrap_row(R+base)]);
    }

    template<bool COARSE>
    __forceinline__ __device__ void reverse_downsweep(const HState& hstate, const ImmState& tmp){
        constexpr int group = COL_BLOCKS < 8 ? COL_BLOCKS : 8;
        constexpr int bits = __builtin_ffs(COL_BLOCKS) - 1;
        int lar3 = ((threadIdx.x & 31) >> 2);
        static_for<bits>([&]<int sweep>() __attribute__((always_inline)) {
            constexpr int stride = 1 << (bits - 1 - sweep);
            if constexpr(stride < ROWS && ((stride >= group) == COARSE)){
                static_for<COL_BLOCKS / (2 * stride)>([&]<int node>() __attribute__((always_inline)) {
                    constexpr int c = COL_BLOCKS - stride - node * 2 * stride;
                    static_for<ROW_BLOCKS>([&]<int index>() __attribute__((always_inline)) {
                        constexpr int r = ROW_BLOCKS - 1 - index;
                        PElement successor;
                        if constexpr(c + stride < COL_BLOCKS){
                            constexpr int displacement = (c + stride) / 8 - c / 8;
                            successor = data[wrap_row(r + displacement)][c + stride];
                        }else if constexpr(COL_BLOCKS >= 8){
                            successor = tmp[wrap_row(r + (stride + 7) / 8)];
                        }else{
                            successor = reverse_shifted_root<r, COL_BLOCKS>(tmp, lar3);
                        }
                        if constexpr(COARSE){
                            int row = (r * 8 + lar3 + (c & 7) + 1) & (ROWS - 1);
                            bool pred = c + stride >= COL_BLOCKS ? row + stride <= ROWS : row + stride < ROWS;
                            data[r][c] = choose(pred, combine(successor, data[r][c]), data[r][c]);
                        }else{
                            successor = fine_boundary<r, c, stride, true>(successor, hstate, lar3);
                            data[r][c] = combine(successor, data[r][c]);
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
    // (2*lam4+e)*COL_BLOCKS+s*group+7-lar3 (bottom input / top output).
    // The bottom-left corner is carried in VState row ROWS.
    __device__ __forceinline__ PrescanResult reverse_inclusive_prescan(const VState& vstate, const HState& hstate){
        // A root spans at most the tile height, even in a wider strip.
        constexpr int root_span = COL_BLOCKS < ROWS ? COL_BLOCKS : ROWS;
        constexpr int bits = __builtin_ffs(COL_BLOCKS) - 1;
        constexpr int group = COL_BLOCKS < 8 ? COL_BLOCKS : 8;
        int lane = threadIdx.x & 31;
        int lam4 = lane & 3, lar3 = (lane >> 2);
        HState out_hstate;
        VState out_vstate;
        static_for<bits>([&]<int iter>() __attribute__((always_inline)) {
            constexpr int stride = 1 << iter;
            if constexpr(stride < ROWS){
                static_for<COL_BLOCKS / (2 * stride)>([&]<int node>() __attribute__((always_inline)) {
                    constexpr int c = COL_BLOCKS - 2 * stride - node * 2 * stride;
                    static_for<ROW_BLOCKS>([&]<int index>() __attribute__((always_inline)) {
                        constexpr int r = ROW_BLOCKS - 1 - index;
                        auto successor = data[wrap_row(r + stride / 8)][c + stride];
                        int row = (r * 8 + lar3 + (c & 7) + 1) & (ROWS - 1);
                        bool pred = row + stride < ROWS;
                        if constexpr(r == ROW_BLOCKS - 1 && stride < group){
                            combine_either(
                                pred, successor, data[r][c], out_hstate.init[c / group]);
                        }else{
                            data[r][c] = choose(pred, combine(successor, data[r][c]), data[r][c]);
                        }
                    });
                });
                if constexpr(stride == group / 2){
                    static_for<HState::length>([&]<int s>() __attribute__((always_inline)) {
                        auto& root = data[ROW_BLOCKS - 1][s * group];
                        if constexpr(group == 8 && s+1 < HState::length){
                            auto incoming = choose(lar3 == 7, hstate.init[s+1], hstate.init[s]);
                            combine_either(lar3 != 7, incoming, root, data[ROW_BLOCKS-2][s*group]);
                        }else root = choose(lar3 >= 8 - group && lar3 < 7, combine(hstate.init[s], root), root);
                    });
                }
            }
        });
        // Join the high strip into the low strip, then scan four low
        // pair-endpoints toward decreasing thread columns. No tile reversal.
        if constexpr(COL_BLOCKS < ROWS){
            auto high = reverse_exchange_half_roots<COL_BLOCKS, 0, 1>(vstate, hstate);
            static_for<ROW_BLOCKS>([&]<int index>() __attribute__((always_inline)) {
                constexpr int r = ROW_BLOCKS-1-index;
                if constexpr(r*8+COL_BLOCKS < ROWS){
                    auto successor = reverse_shifted_half_root<r, COL_BLOCKS>(high, lar3);
                    auto& root = data[r][0];
                    auto old = root.extract(0);
                    root.insert(0, choose(r*8+lar3+COL_BLOCKS < ROWS, combine(successor, old), old));
                }
            });
        }
        static_for<2>([&]<int step>() __attribute__((always_inline)) {
            constexpr int threads = 1 << step;
            constexpr int distance = 2*threads*COL_BLOCKS;
            if constexpr(distance < ROWS){
                auto following = reverse_exchange_half_roots<distance, threads, 0>(vstate, hstate);
                static_for<ROW_BLOCKS>([&]<int index>() __attribute__((always_inline)) {
                    constexpr int r = ROW_BLOCKS-1-index;
                    if constexpr(r*8+distance < ROWS){
                        auto successor = reverse_shifted_half_root<r, distance>(following, lar3);
                        auto& root = data[r][0];
                        auto old = root.extract(0);
                        bool pred = r*8+lar3+distance < ROWS;
                        if constexpr(step == 1) pred = pred && lam4 != 3;
                        root.insert(0, choose(pred, combine(successor, old), old));
                    }
                });
            }
        });
        // The leftmost pair's owner is also the right-input VState owner.
        static_for<ROW_BLOCKS>([&]<int index>() __attribute__((always_inline)) {
            constexpr int r = ROW_BLOCKS-1-index;
            if constexpr((r+1)*8+COLS <= ROWS){
                auto& root = data[r][0];
                auto old = root.extract(0);
                auto successor = vstate.init[wrap_row(r+COLS/8)];
                root.insert(0, choose(lam4 == 0, combine(successor, old), old));
            }
        });
        // Complete high roots from the following pair's final low root;
        // retain that exchange for the high-half postscan entry.
        auto following = reverse_exchange_half_roots<COL_BLOCKS, 1, 0>(vstate, hstate);
        static_for<ROW_BLOCKS>([&]<int index>() __attribute__((always_inline)) {
            constexpr int r = ROW_BLOCKS-1-index;
            if constexpr(r*8+COL_BLOCKS < ROWS){
                auto successor = reverse_shifted_half_root<r, COL_BLOCKS>(following, lar3);
                auto& root = data[r][0];
                auto old = root.extract(1);
                root.insert(1, choose(r*8+lar3+COL_BLOCKS < ROWS, combine(successor, old), old));
            }
        });
        auto high_prefix = reverse_exchange_half_roots<COL_BLOCKS, 0, 1>(vstate, hstate);
        ImmState tmp;
        static_for<ROW_BLOCKS>([&]<int r>() __attribute__((always_inline)) {
            tmp[r].insert(0, high_prefix.root[r]);
            tmp[r].insert(1, following.root[r]);
        });
        reverse_downsweep<true>(hstate, tmp);
        static_for<ROW_BLOCKS>([&]<int r>() __attribute__((always_inline)) {
            auto value = data[r][0].extract(0);
            if constexpr(r == ROW_BLOCKS-1) value = choose(lar3 == 7, hstate.init[0].extract(0), value);
            out_vstate.init[r] = choose(lam4 == 0, value, out_vstate.init[r]);
        });
        static_for<HState::length>([&]<int s>() __attribute__((always_inline)) {
            auto successor = tmp[0];
            if constexpr(s + 1 < HState::length) successor = data[0][(s + 1) * group];
            auto& root = out_hstate.init[s];
            if constexpr(s*group == COL_BLOCKS-root_span){
                // Symmetric deferred top endpoint; no intermediate root
                // exchange needs this wrapped tile value.
                auto prefix = reverse_shifted_root<ROW_BLOCKS-1, root_span>(tmp, lar3);
                successor = choose(lar3 == 7, prefix, successor);
                auto tail = choose(lar3 == 7, data[ROW_BLOCKS-1][s*group], root);
                root = choose(lar3 >= 8-group, combine(successor, tail), root);
                data[ROW_BLOCKS-1][s*group] = choose(lar3 == 7, root, data[ROW_BLOCKS-1][s*group]);
            }else{
                root = choose(lar3 >= 8 - group, combine(successor, root), root);
                root = choose(lar3 == 7, data[ROW_BLOCKS-1][s*group], root);
            }
        });
        return {out_vstate, out_hstate, tmp};
    }

    __device__ __forceinline__ void reverse_inclusive_postscan(const HState& incoming_hstate, const ImmState& tmp){
        reverse_downsweep<false>(incoming_hstate, tmp);
    }

    // Boundary-only reverse traversal, preserving the reverse-rolled tile.
    // Reuses the native successor tree without postscan or extra exchanges.
    __device__ __forceinline__ ScanResult reduce_backward(const VState& vstate, const HState& hstate) const {
        AltLayoutSplitScanBuffer work = *this;
        auto result = work.reverse_inclusive_prescan(vstate, hstate);
        return {result.vertical, result.horizontal};
    }

    __device__ __forceinline__ ScanResult reverse_inclusive_scan(const VState& vstate, const HState& hstate){
        auto result = reverse_inclusive_prescan(vstate, hstate);
        reverse_inclusive_postscan(hstate, result.intermediate);
        return {result.vertical, result.horizontal};
    }

};

} // namespace pscore
