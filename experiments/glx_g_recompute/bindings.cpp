#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include "../../dism_v2/csrc/core_api.h"
namespace dism_v2 {void launch_g_probe(const Args&,int,int,const void*,const float*,const float*,float*,cudaStream_t);}
// Private diagnostic API: receives validated production forward/backward operands.
torch::Tensor run(torch::Tensor a,torch::Tensor b,torch::Tensor v,torch::Tensor dout,
                  torch::Tensor lse,torch::Tensor tau,torch::Tensor ql,torch::Tensor kl,
                  torch::Tensor norm,torch::Tensor delta,torch::Tensor vertical,torch::Tensor horizontal,
                  torch::Tensor boundary,double scale,bool column_lse,double probability,uint64_t seed,uint64_t offset) {
    c10::cuda::CUDAGuard guard(a.device());
    int batch=a.size(0),heads=a.size(1),n=a.size(2),d=a.size(3),dv=v.size(3),np=(n+63)/64*64;
    TORCH_CHECK((d==32 || d==64 || d==128) && (dv==32 || dv==64 || dv==128),"invalid dimensions");
    auto out=torch::empty({batch,heads,np,np},norm.options());
    dism_v2::Args p{};
    p.a=a.data_ptr();p.b=b.data_ptr();p.v=v.data_ptr();p.lse=lse.data_ptr<float>();p.tau=tau.data_ptr<float>();
    p.q_label=ql.data_ptr<int64_t>();p.k_label=kl.data_ptr<int64_t>();p.normalizer=norm.data_ptr<float>();
    p.vertical=vertical.data_ptr<float>();p.horizontal=horizontal.data_ptr<float>();
    p.n=n;p.padded_n=np;p.batch_heads=batch*heads;p.heads=heads;p.scale=scale;p.column_lse=column_lse;
    p.hard_prob=probability;p.seed=seed;p.offset=offset;
    dism_v2::launch_g_probe(p,d,dv,dout.data_ptr(),delta.data_ptr<float>(),boundary.data_ptr<float>(),out.data_ptr<float>(),c10::cuda::getCurrentCUDAStream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {m.def("run",&run);}
