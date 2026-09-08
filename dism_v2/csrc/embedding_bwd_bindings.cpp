#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <climits>
#include <cmath>
#include "embedding_bwd_api.h"
std::vector<torch::Tensor> backward(torch::Tensor x,torch::Tensor y,torch::Tensor key,
        torch::Tensor value,torch::Tensor out,torch::Tensor u,torch::Tensor lx,
        torch::Tensor ly,torch::Tensor lambda,double scale,bool ws) {
    TORCH_CHECK(x.is_cuda() && x.dim()==4,"x must be CUDA [B,H,N,D]");
    auto b=x.size(0),h=x.size(1),n=x.size(2),d=x.size(3);
    TORCH_CHECK(b>0 && h>0 && n>0 && b*h<=65535 && b*h*n<=INT_MAX-4,"unsupported dimensions");
    TORCH_CHECK(d==32 || d==64 || d==128,"D must be 32/64/128");
    for(const auto& t:{x,y,key,value,out,u,lx,ly,lambda})
        TORCH_CHECK(t.device()==x.device() && t.is_contiguous(),"same-device contiguous inputs required");
    for(const auto& t:{x,y,key,value,out}) TORCH_CHECK(t.scalar_type()==at::kBFloat16,"BF16 operands required");
    for(const auto& t:{u,lx,ly,lambda}) TORCH_CHECK(t.scalar_type()==at::kFloat,"U/LSE/lambda must be FP32");
    TORCH_CHECK(y.sizes()==x.sizes() && out.sizes()==x.sizes() && u.sizes()==x.sizes(),"token shape mismatch");
    TORCH_CHECK(key.dim()==3 && key.size(0)==h && key.size(2)==d && value.sizes()==key.sizes(),"vocab shape mismatch");
    auto v=key.size(1);
    TORCH_CHECK(v>0 && v<=INT_MAX-64,"invalid vocabulary size");
    for(const auto& t:{lx,ly,lambda})
        TORCH_CHECK(t.dim()==3 && t.size(0)==b && t.size(1)==h && t.size(2)==n,"row metadata shape mismatch");
    TORCH_CHECK(std::isfinite(scale) && std::isfinite(float(scale)),"scale must be finite FP32");
    const c10::cuda::CUDAGuard guard(x.device());
    cudaDeviceProp prop;
    C10_CUDA_CHECK(cudaGetDeviceProperties(&prop,x.get_device()));
    TORCH_CHECK(prop.major==12 && prop.minor==0,"initial embedding backward supports sm120 only");
    auto opt=x.options().dtype(at::kFloat);
    auto dx=torch::empty(x.sizes(),opt),dy=torch::empty_like(dx);
    auto dk=torch::empty(key.sizes(),opt),dv=torch::empty_like(dk);
    auto packed=torch::empty_like(x),delta=torch::empty_like(lx);
    auto lx2=torch::empty_like(lx),ly2=torch::empty_like(ly);
    dism_v2::embedding_bwd::Args a{x.data_ptr(),y.data_ptr(),key.data_ptr(),value.data_ptr(),out.data_ptr(),
        u.data_ptr<float>(),lx.data_ptr<float>(),ly.data_ptr<float>(),lambda.data_ptr<float>(),
        packed.data_ptr(),delta.data_ptr<float>(),dx.data_ptr<float>(),dy.data_ptr<float>(),
        dk.data_ptr<float>(),dv.data_ptr<float>(),lx2.data_ptr<float>(),ly2.data_ptr<float>(),
        int(b),int(h),int(n),int(v),float(scale),float(scale)*1.4426950408889634f};
    dism_v2::embedding_bwd::launch(a,d,ws,c10::cuda::getCurrentCUDAStream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {dx,dy,dk,dv,delta,packed};
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {m.def("backward",&backward);}
