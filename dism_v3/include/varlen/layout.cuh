#pragma once

#include <climits>
#include <cstring>
#include <type_traits>
#include <vector>
#if DISM_HOST_API
#include "host/metadata_upload.h"
#endif

#include "variant.cuh"
namespace DISM_VARIANT {

namespace dism_varlen {
// Offsets for checkpoint families are per head. Each sequence owns a contiguous
// [H,checkpoints,padded] region, starting at family_offset * H.
enum Field : int {
    Begin, Length, PaddedBegin, PaddedLength,
    ForwardOffset, VerticalOffset, BackwardOffset, Fields
};

#if DISM_HOST_API
struct Layout {
    at::Tensor cpu;
    int64_t sequences, tokens=0, padded=0, forward=0, vertical=0, backward=0;
    int64_t max_padded=0;

    explicit Layout(const at::Tensor& table) : cpu(table), sequences(table.size(0)) {
        TORCH_CHECK(table.device().is_cpu() && table.scalar_type()==at::kLong &&
                    table.is_contiguous() && table.dim()==2 && table.size(1)==Fields,
                    "layout must be CPU int64 [sequences,7]");
        auto data=table.accessor<int64_t,2>();
        for (int64_t s=0;s<sequences;++s) {
            int64_t n=data[s][Length], p=data[s][PaddedLength];
            TORCH_CHECK(n>=0 && n<=INT_MAX-255 && n%256==0 && p==n,
                        "varlen boundaries including T must be256-token aligned; no input padding");
            TORCH_CHECK(data[s][Begin]==tokens && data[s][PaddedBegin]==padded &&
                        data[s][ForwardOffset]==forward && data[s][VerticalOffset]==vertical &&
                        data[s][BackwardOffset]==backward,"inconsistent varlen offsets");
            tokens+=n;
            padded+=p;
            forward+=(n ? (n-1)/32 : 0)*p;
            vertical+=(n ? (n-1)/16 : 0)*p;
            backward+=(n+31)/32*p;
            max_padded=std::max(max_padded,p);
        }
    }
    int64_t get(int64_t sequence,Field field) const {
        return cpu.data_ptr<int64_t>()[sequence*Fields+field];
    }
};

// Tensor maps are embedded by value in TK globals. Preserve their native
// alignment and object stride when copying an array to device memory.
template<class T>
inline at::Tensor upload_records(const std::vector<T>& records,at::Device device) {
    // TK gl has a user-defined memberwise copy constructor, so the enclosing
    // record is not trivially_copyable. Its audited contents are inline tensor
    // maps, dimensions/strides and device pointers, with no owning host state.
    static_assert(std::is_trivially_destructible_v<T>);
    static_assert(alignof(T)<=256 && sizeof(T)%alignof(T)==0);
    return dism_metadata::upload(records.data(),records.size()*sizeof(T),device);
}
#endif // DISM_HOST_API
} // namespace dism_varlen

} // namespace DISM_VARIANT
