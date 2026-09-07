#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include "../../dism_v2/csrc/core_api.h"
namespace dism_v2 { void launch_recompute_probe(const Args&,int,float*,cudaStream_t); }

// Private test binding: callers supply the already validated forward operands.
torch::Tensor recompute(torch::Tensor a,torch::Tensor b,torch::Tensor lse,torch::Tensor tau,
        torch::Tensor ql,torch::Tensor kl,torch::Tensor vertical,torch::Tensor horizontal,
        double scale,bool column_lse,double probability,uint64_t seed,uint64_t offset) {
    const c10::cuda::CUDAGuard guard(a.device());
    int batch=a.size(0),heads=a.size(1),n=a.size(2),d=a.size(3),np=(n+63)/64*64;
    auto output=torch::empty({batch,heads,np,np},lse.options());
    dism_v2::Args p{};
    p.a=a.data_ptr(); p.b=b.data_ptr(); p.lse=lse.data_ptr<float>(); p.tau=tau.data_ptr<float>();
    p.q_label=ql.data_ptr<int64_t>(); p.k_label=kl.data_ptr<int64_t>();
    p.vertical=vertical.data_ptr<float>(); p.horizontal=horizontal.data_ptr<float>();
    p.n=n; p.padded_n=np; p.batch_heads=batch*heads; p.heads=heads;
    p.scale=scale; p.column_lse=column_lse; p.hard_prob=probability; p.seed=seed; p.offset=offset;
    dism_v2::launch_recompute_probe(p,d,output.data_ptr<float>(),c10::cuda::getCurrentCUDAStream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) { m.def("recompute",&recompute); }
