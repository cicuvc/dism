#define DISM_HOST_API 1
#include "host/torch.cuh"
#include "summary/kernel_common.cuh"
#include "summary/launch.cuh"

#include "variant.cuh"

namespace DISM_VARIANT {

const void* summary_address();
std::tuple<at::Tensor, at::Tensor> summarization_cuda(at::Tensor query, at::Tensor key, at::Tensor q_lse, at::Tensor k_lse,
                              at::Tensor q_labels, at::Tensor k_labels, at::Tensor hard,
                              at::Tensor direction, at::Tensor tau, int64_t seqlen,
                              int64_t requested_ctas) {
    TORCH_CHECK(query.dim()==4 && seqlen>0 && seqlen%256==0 && query.size(1)==seqlen,
                "sequence length must be256-token aligned; padding is not accepted");
    return launch_summary(summary_address(), query, key, q_lse, k_lse,
                          q_labels, k_labels, hard, direction, tau, seqlen, requested_ctas);
}

int summary_variant_cuda() { return 0; }
int summary_key_dim_cuda() { return CONFIG.KcKeyDim; }

} // namespace DISM_VARIANT
