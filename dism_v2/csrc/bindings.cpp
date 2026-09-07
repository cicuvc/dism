#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include "core_api.h"
#include <climits>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <cmath>
#include <optional>

std::tuple<std::vector<torch::Tensor>,uint64_t,uint64_t> core_forward(
        torch::Tensor a,torch::Tensor b,torch::Tensor v,torch::Tensor lse,torch::Tensor tau,
        torch::Tensor q_label,torch::Tensor k_label,double scale,bool column_lse,double hard_prob,
        std::optional<at::Generator> generator,std::optional<std::pair<uint64_t,uint64_t>> replay) {
    TORCH_CHECK(a.dim()==4 && a.is_cuda(),"A must be CUDA [B,H,N,D]");
    for(const auto& x:{a,b,v,lse,tau,q_label,k_label})
        TORCH_CHECK(x.device()==a.device() && x.is_contiguous(),"inputs must be contiguous on the same CUDA device");
    int64_t batch=a.size(0),heads=a.size(1),n=a.size(2),d=a.size(3);
    TORCH_CHECK(batch>0 && heads>0 && n>0 && batch*heads<=65535 && batch*heads*n<=INT_MAX/2,"unsupported launch dimensions");
    TORCH_CHECK(b.sizes()==a.sizes() && v.dim()==4 && v.size(0)==batch && v.size(1)==heads && v.size(2)==n,"A/B/V shape mismatch");
    int64_t dv=v.size(3);
    TORCH_CHECK((d==32||d==64||d==128) && (dv==32||dv==64||dv==128),"D/DV must be 32,64,128");
    TORCH_CHECK(a.scalar_type()==at::kBFloat16 && b.scalar_type()==at::kBFloat16 && v.scalar_type()==at::kBFloat16,"A/B/V must be BF16");
    TORCH_CHECK(lse.scalar_type()==at::kFloat && tau.scalar_type()==at::kFloat,"LSE/tau must be FP32");
    TORCH_CHECK(q_label.scalar_type()==at::kLong && k_label.scalar_type()==at::kLong,"labels must be int64");
    for(const auto& x:{lse,q_label,k_label}) TORCH_CHECK(x.dim()==3 && x.size(0)==batch && x.size(1)==heads && x.size(2)==n,"metadata must be [B,H,N]");
    TORCH_CHECK(tau.dim()==1 && tau.numel()==heads,"tau must be [H]");
    const c10::cuda::CUDAGuard guard(a.device());
    cudaDeviceProp prop; C10_CUDA_CHECK(cudaGetDeviceProperties(&prop,a.get_device()));
    TORCH_CHECK(prop.major==12 && prop.minor==0,"initial core supports sm120 only");
    TORCH_CHECK(std::isfinite(hard_prob) && hard_prob>=0 && hard_prob<=1,"hard_prob must be in [0,1]");
    cudaStreamCaptureStatus capture;
    C10_CUDA_CHECK(cudaStreamIsCapturing(c10::cuda::getCurrentCUDAStream(),&capture));
    TORCH_CHECK(capture==cudaStreamCaptureStatusNone,"CUDA graph capture is not supported yet");
    TORCH_CHECK(!(generator && replay),"generator and replay are mutually exclusive");
    if(generator) TORCH_CHECK(generator->device().is_cuda() &&
        (!generator->device().has_index() || generator->device()==a.device()),
        "generator must match the input CUDA device");
    uint64_t seed=0,offset=0;
    if(replay) {
        seed=replay->first; offset=replay->second;
        TORCH_CHECK(offset%4==0,"replay offset must be a multiple of four");
    } else if(hard_prob>0 && hard_prob<1) {
        auto g=generator.value_or(at::cuda::detail::getDefaultCUDAGenerator(a.get_device()));
        auto* impl=g.get<at::CUDAGeneratorImpl>();
        std::lock_guard<std::mutex> lock(impl->mutex_);
        TORCH_CHECK(impl->get_offset()<=UINT64_MAX-4,"RNG offset overflow");
        auto state=impl->philox_engine_inputs(4);
        seed=state.first; offset=state.second;
    }
    int cp=(n+31)/32,np=(n+63)/64*64;
    auto summary=torch::empty({batch,heads,cp,np,2},lse.options());
    auto boundary=torch::empty({batch,heads,cp,np},lse.options());
    auto out=torch::empty_like(v), norm=torch::empty({batch,heads,n},lse.options());
    dism_v2::Args p{a.data_ptr(),b.data_ptr(),v.data_ptr(),lse.data_ptr<float>(),tau.data_ptr<float>(),
        q_label.data_ptr<int64_t>(),k_label.data_ptr<int64_t>(),summary.data_ptr<float>(),boundary.data_ptr<float>(),
        norm.data_ptr<float>(),out.data_ptr(),int(batch*heads),int(heads),int(n),np,cp,float(scale),column_lse,float(hard_prob),seed,offset};
    auto stream=c10::cuda::getCurrentCUDAStream();
    dism_v2::launch_summary(p,d,stream); C10_CUDA_KERNEL_LAUNCH_CHECK();
    dism_v2::launch_passing(p,stream); C10_CUDA_KERNEL_LAUNCH_CHECK();
    dism_v2::launch_output(p,d,dv,stream); C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {{out,norm,summary,boundary},seed,offset};
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) { m.def("forward",&core_forward); }
