#ifndef METADATA_LIB_CUH
#define METADATA_LIB_CUH

#include <array>
#include <cstdarg>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <memory>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "types/types.cuh"

enum struct RelocationType { REL_RAW_PTR = 0, REL_TMA_TENSOR_MAP_PTR = 1, REL_INHERIT = 2 };

struct IBuilder {
    void (*addRelocation)(RelocationType type, size_t offset, void *symbol);
};

// Helper: map T -> type actually passed through '...'
template <typename T>
struct va_promoted {
    // enums: use underlying type for promotion rules
    using base_t = T;
    using type = std::conditional_t<
        std::is_same_v<base_t, float>,
        double, // float -> double
        std::conditional_t<(std::is_integral_v<base_t> && sizeof(base_t) < sizeof(int)), int, base_t>>;
};

template <typename T>
using va_promoted_t = typename va_promoted<T>::type;

// Helper to compute max sizeof/align of a pack
template <typename... Ts>
struct max_size_align;

template <typename T>
struct max_size_align<T> {
    static constexpr std::size_t size = sizeof(T);
    static constexpr std::size_t align = alignof(T);
};

template <typename T, typename U, typename... Rest>
struct max_size_align<T, U, Rest...> {
    static constexpr std::size_t size =
        (sizeof(T) > max_size_align<U, Rest...>::size) ? sizeof(T) : max_size_align<U, Rest...>::size;
    static constexpr std::size_t align =
        (alignof(T) > max_size_align<U, Rest...>::align) ? alignof(T) : max_size_align<U, Rest...>::align;
};

template <typename TOp, typename... Ts>
struct VariadicConstructorInvoker {
    static_assert((!std::is_reference_v<Ts> && ...), "argument types must be value types (no references)");
    static_assert((std::is_trivially_copyable_v<Ts> && ...),
                  "argument types must be trivially copyable to be passed through '...'");
    // We intentionally allow pointer, integer, float, enum, trivially-copyable
    // structs.

    static constexpr size_t n_args = sizeof...(Ts);
    using storage_unit = std::aligned_storage_t<max_size_align<char, Ts...>::size, max_size_align<char, Ts...>::align>;

    va_list &va;
    std::vector<storage_unit> storage; // one slot per argument

    VariadicConstructorInvoker(va_list &va_) : va(va_), storage(n_args) {
        // Fill storage with placement-new constructed copies from va_arg (with
        // promotions handled)
        ([this]<size_t... Is>(std::index_sequence<Is...>) {
            // Use comma operator expansion to do each index
            (void)std::initializer_list<int>{(this->construct_one<Is>(), 0)...};
        })(std::make_index_sequence<n_args>{});
    }

    template <size_t I>
    using nth_type = typename std::tuple_element<I, std::tuple<Ts...>>::type;

    template <size_t I>
    void construct_one() {
        using T = nth_type<I>;
        using prom_t = va_promoted_t<T>;
        prom_t v = va_arg(va, prom_t);
        // placement new: construct T from promoted value (works for trivially
        // copyable)
        void *slot = &storage[I];
        std::construct_at((T *)slot, v);
    }

    template <typename... TOthers>
    void call(TOthers &&...others) {
        ([this, ... others = std::forward<TOthers>(others)]<size_t... Is>(std::index_sequence<Is...>) mutable {
            TOp::constructor(std::forward<TOthers>(others)..., *reinterpret_cast<nth_type<Is> *>(&storage[Is])...);
        })(std::make_index_sequence<n_args>{});
    }

    ~VariadicConstructorInvoker() {
        // If types were non-trivially-destructible we'd call destructors. We
        // asserted trivially_copyable so destructor trivial, but for
        // completeness:
        ([this]<size_t... Is>(std::index_sequence<Is...>) {
            ((void)std::destroy_at(reinterpret_cast<nth_type<Is> *>(&storage[Is])), ...);
        })(std::make_index_sequence<n_args>{});
    }
};

template <typename T>
struct ArgConstructor {
    static size_t size() { return sizeof(T); }
    static void constructor(void *dst, IBuilder *builder, T value) {
        *reinterpret_cast<T *>(dst) = value;
        builder->addRelocation(RelocationType::REL_TMA_TENSOR_MAP_PTR, 12, nullptr);
    }
};

struct UntypedArgConstructor;

using ConstructorFn = void(void *, IBuilder *, ...);
using SizeFn = size_t();
using ArgNumFn = size_t();
using GetSubConstructorFn = const UntypedArgConstructor *(int idx);
using GetTypeFn = const char *();

template <typename T>
struct TypedArgConstructor;

struct UntypedArgConstructor {
    ConstructorFn *construct;
    SizeFn *size;
    ArgNumFn *nargs;
    GetSubConstructorFn *sub;
    GetTypeFn *type;
};

template <typename T>
struct Box {
    T *ptr;
};

template <typename T>
struct remove_box {
    using type = T;
};
template <typename T>
struct remove_box<Box<T>> {
    using type = T;
};

template <typename T>
using remove_box_t = typename remove_box<T>::type;

template <typename... TPrevs>
struct FunctionArgTypeExtractor {
    template <typename T>
    struct ArgTypeExtractor;

    template <typename... Ts>
    struct ArgTypeExtractor<void(TPrevs..., Ts...)> {
        template <typename TOp>
        using Invoker = VariadicConstructorInvoker<TOp, std::remove_cvref_t<Ts>...>;

        static constexpr size_t n_args = sizeof...(Ts);

        template <int idx>
        static UntypedArgConstructor getConstructor() {
            using Tc = std::tuple_element<idx, std::tuple<Ts...>>::type;
            return TypedArgConstructor<remove_box_t<Tc>>::build();
        }
    };
};

template <typename T>
struct TypedArgConstructor {
    using Extractor =
        FunctionArgTypeExtractor<void *, IBuilder *>::ArgTypeExtractor<decltype(ArgConstructor<T>::constructor)>;
    using Invoker = typename Extractor::template Invoker<ArgConstructor<T>>;

    inline static constexpr UntypedArgConstructor build() { return {&construct, &size, &nargs, &sub, &type}; }

    static void construct(void *dst, IBuilder *builder, ...) {
        va_list va;
        va_start(va, builder);
        Invoker(va).call(dst, builder);
        va_end(va);
    }
    static size_t size() { return sizeof(T); }
    static size_t nargs() { return Extractor::n_args; }
    static const UntypedArgConstructor *sub(int idx) {
        static auto ctors = ([]<size_t... Idx>(std::index_sequence<Idx...>) {
            return std::array<UntypedArgConstructor, Extractor::n_args>{Extractor::template getConstructor<Idx>()...};
        })(std::make_index_sequence<Extractor::n_args>{});
        return &ctors[idx];
    }
    static const char *type() {
        if constexpr (std::is_same_v<T, int>)
            return "int";
        if constexpr (std::is_same_v<T, size_t>)
            return "size_t";
        if constexpr (std::is_same_v<T, float>)
            return "float";

        if constexpr (std::is_pointer_v<T>)
            return "ptr";

        return __PRETTY_FUNCTION__;
    }
};

template <typename T, int b, int d, int r, int c>
struct ArgConstructor<kittens::gl<T, b, d, r, c>> {
    using GL = kittens::gl<T, b, d, r, c>;
    static size_t size() { return sizeof(GL); }
    static void constructor(void *dst, IBuilder *builder, T *ptr, size_t batch, size_t depth, size_t row, size_t col) {
        std::construct_at(reinterpret_cast<GL *>(dst), ptr, batch, depth, row, col);
        builder->addRelocation(RelocationType::REL_RAW_PTR, offsetof(GL, raw_ptr), ptr);
    }
};

template <typename T, int b, int d, int r, int c, typename... Tsms>
struct ArgConstructor<kittens::gl<T, b, d, r, c, Tsms...>> {
    using GL = kittens::gl<T, b, d, r, c, Tsms...>;
    static size_t size() { return sizeof(GL); }
    static void constructor(void *dst, IBuilder *builder, T *ptr, size_t batch, size_t depth, size_t row, size_t col) {
        auto *pgl = reinterpret_cast<GL *>(dst);
        std::construct_at(pgl, ptr, batch, depth, row, col);
        builder->addRelocation(RelocationType::REL_RAW_PTR, offsetof(GL, raw_ptr), ptr);

        std::vector<CUtensorMap *> tmaps;
        pgl->collect_tmaps(tmaps);

        for (auto i : tmaps) {
            builder->addRelocation(RelocationType::REL_TMA_TENSOR_MAP_PTR, size_t(((char *)i) - (char *)pgl), ptr);
        }
    }
};

struct ExportedKernel {
    template <typename TFn>
    inline static const UntypedArgConstructor *helpers(int idx) {
        using Extractor = FunctionArgTypeExtractor<>::ArgTypeExtractor<TFn>;
        static auto ctors = ([]<size_t... Idx>(std::index_sequence<Idx...>) {
            return std::array<UntypedArgConstructor, Extractor::n_args>{Extractor::template getConstructor<Idx>()...};
        })(std::make_index_sequence<Extractor::n_args>{});
        return idx < Extractor::n_args ? &ctors[idx] : nullptr;
    }
    ExportedKernel *next;
    const char *name;
    void const *stubAddress;
    const UntypedArgConstructor *(*kernel_helpers)(int idx);

    std::string nameString;

    template <typename TFn>
    inline ExportedKernel(std::string name, const TFn &func);
};

inline ExportedKernel *exported_kernels = nullptr;

template <typename TFn>
inline ExportedKernel::ExportedKernel(std::string name_, const TFn &func)
    : next{exported_kernels}, name{nullptr}, stubAddress{reinterpret_cast<const void *>(&func)},
      kernel_helpers{&ExportedKernel::template helpers<TFn>}, nameString{std::move(name_)} {
    name = nameString.c_str();
    auto old = exported_kernels;
    exported_kernels = this;
}

#define CONCAT(x, y) x##y
#define CONCAT2(x, y) CONCAT(x, y)
#define EXPORT_KERNEL(NAME, FUNC) static ExportedKernel CONCAT2(exported, __COUNTER__)(NAME, FUNC)

#endif // METADATA_LIB_CUH
