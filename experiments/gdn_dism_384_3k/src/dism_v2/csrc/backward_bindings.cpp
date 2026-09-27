#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <climits>
#include <cmath>
#include "core_api.h"
namespace dism_v2 { void launch_backward_delta(const void*,const void*,float*,int,int,cudaStream_t); }
namespace dism_v2 {
void launch_operand_backward(const Args&,int,int,const void*,const float*,const float*,float*,float*,float*,float*,cudaStream_t);
void launch_operand_backward_ws(const Args&,int,int,const void*,const float*,const float*,float*,float*,float*,float*,cudaStream_t);
void launch_tau_reduce(const float*,float*,int,int,int,cudaStream_t);
void launch_value_backward(const Args&,int,int,const void*,float*,const float*,float2*,cudaStream_t);
void launch_backward_passing(const float2*,float2*,float*,int,int,cudaStream_t);
void launch_value_backward_ws(const Args&,int,int,const void*,const float*,float*,float2*,float*,cudaStream_t);
}

std::vector<torch::Tensor> value_gradient(torch::Tensor a,torch::Tensor b,torch::Tensor dout,
        torch::Tensor lse,torch::Tensor tau,torch::Tensor ql,torch::Tensor kl,
        torch::Tensor norm,torch::Tensor vertical,torch::Tensor horizontal,
        double scale,bool column_lse,double probability,uint64_t seed,uint64_t offset,
        std::optional<torch::Tensor> value,std::optional<torch::Tensor> delta,bool warp_specialized,
        std::optional<torch::Tensor> hard_bits) {
    TORCH_CHECK(a.is_cuda() && a.dim()==4,"A must be CUDA [B,H,N,D]");
    for(const auto& x:{a,b,dout,lse,tau,ql,kl,norm,vertical,horizontal})
        TORCH_CHECK(x.device()==a.device() && x.is_contiguous(),"inputs must be contiguous on same CUDA device");
    int64_t batch=a.size(0),heads=a.size(1),n=a.size(2),d=a.size(3);
    TORCH_CHECK(batch>0 && heads>0 && n>0 && batch*heads<=65535 && batch*heads*n<=INT_MAX/2,"unsupported dimensions");
    TORCH_CHECK(b.sizes()==a.sizes() && dout.dim()==4 && dout.size(0)==batch && dout.size(1)==heads && dout.size(2)==n,"A/B/dO shape mismatch");
    int dv=dout.size(3),np=(n+63)/64*64;
    TORCH_CHECK((d==32||d==64||d==128) && (dv==32||dv==64||dv==128),"D/DV must be 32,64,128");
    for(const auto& x:{a,b,dout}) TORCH_CHECK(x.scalar_type()==at::kBFloat16,"A/B/dO must be BF16");
    for(const auto& x:{lse,tau,norm,vertical,horizontal}) TORCH_CHECK(x.scalar_type()==at::kFloat,"states must be FP32");
    TORCH_CHECK((ql.scalar_type()==at::kLong || ql.scalar_type()==at::kInt) && kl.scalar_type()==ql.scalar_type(),"labels must have matching int32/int64 dtype");
    for(const auto& x:{lse,ql,kl,norm})
        TORCH_CHECK(x.dim()==3 && x.size(0)==batch && x.size(1)==heads && x.size(2)==n,"row metadata shape mismatch");
    TORCH_CHECK(tau.dim()==1 && tau.size(0)==heads,"tau must be [H]");
    for(const auto& x:{vertical,horizontal}) TORCH_CHECK(x.dim()==4 && x.size(0)==batch && x.size(1)==heads && x.size(3)==np,"boundary shape mismatch");
    TORCH_CHECK(vertical.size(2)==np/16 && horizontal.size(2)==np/64,"boundary granularity mismatch");
    TORCH_CHECK(std::isfinite(scale) && std::isfinite(probability) && probability>=0 && probability<=1 && offset%4==0,"invalid scale/RNG metadata");
    TORCH_CHECK(bool(value)==bool(delta),"value and delta must be supplied together");
    TORCH_CHECK(!warp_specialized || value.has_value(),"warp-specialized path requires V and delta");
    if(value) {
        TORCH_CHECK(value->device()==a.device() && value->is_contiguous() && value->sizes()==dout.sizes() && value->scalar_type()==at::kBFloat16,"V must match BF16 dO shape/device");
        TORCH_CHECK(delta->device()==a.device() && delta->is_contiguous() && delta->sizes()==norm.sizes() && delta->scalar_type()==at::kFloat,"delta must match FP32 normalizer shape/device");
    }
    const c10::cuda::CUDAGuard guard(a.device());
    cudaDeviceProp prop; C10_CUDA_CHECK(cudaGetDeviceProperties(&prop,a.get_device()));
    TORCH_CHECK(prop.major==12 && prop.minor==0,"initial backward supports sm120 only");
    auto output=torch::empty(dout.sizes(),dout.options().dtype(at::kFloat));
    dism_v2::Args p{};
    p.a=a.data_ptr(); p.b=b.data_ptr(); p.lse=lse.data_ptr<float>(); p.tau=tau.data_ptr<float>();
    p.q_label=reinterpret_cast<const int64_t*>(ql.data_ptr()); p.k_label=reinterpret_cast<const int64_t*>(kl.data_ptr());
    p.normalizer=norm.data_ptr<float>(); p.vertical=vertical.data_ptr<float>(); p.horizontal=horizontal.data_ptr<float>();
    p.n=n; p.padded_n=np; p.batch_heads=batch*heads; p.heads=heads;
    p.scale=scale; p.column_lse=column_lse; p.hard_prob=probability; p.seed=seed; p.offset=offset;
    p.label32=ql.scalar_type()==at::kInt;
    if(hard_bits) {
        TORCH_CHECK(hard_bits->device()==a.device() && hard_bits->scalar_type()==at::kInt && hard_bits->is_contiguous() &&
            hard_bits->sizes()==torch::IntArrayRef({batch,heads,(n+31)/32}),"invalid hard bitset");
        p.hard_bits=reinterpret_cast<const uint32_t*>(hard_bits->data_ptr<int>());
    }
    auto stream=c10::cuda::getCurrentCUDAStream();
    if(value) {
        auto summary=torch::empty({batch,heads,np/32,np,2},norm.options());
        auto boundary=torch::empty({batch,heads,np/32,np},norm.options());
        p.v=value->data_ptr();
        if(warp_specialized) {
            dism_v2::launch_value_backward_ws(p,d,dv,dout.data_ptr(),delta->data_ptr<float>(),output.data_ptr<float>(),
                reinterpret_cast<float2*>(summary.data_ptr<float>()),boundary.data_ptr<float>(),stream);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            return {output,summary,boundary};
        }
        auto local=torch::empty({batch,heads,np/16,np,2},norm.options());
        dism_v2::launch_value_backward(p,d,dv,dout.data_ptr(),output.data_ptr<float>(),delta->data_ptr<float>(),reinterpret_cast<float2*>(local.data_ptr<float>()),stream);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        dism_v2::launch_backward_passing(reinterpret_cast<float2*>(local.data_ptr<float>()),reinterpret_cast<float2*>(summary.data_ptr<float>()),boundary.data_ptr<float>(),np,batch*heads,stream);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return {output,summary,boundary};
    }
    dism_v2::launch_value_backward(p,d,dv,dout.data_ptr(),output.data_ptr<float>(),nullptr,nullptr,stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {output};
}

torch::Tensor delta(torch::Tensor dout,torch::Tensor out) {
    TORCH_CHECK(out.is_cuda() && out.dim()==4,"O must be CUDA [B,H,N,DV]");
    TORCH_CHECK(dout.device()==out.device() && dout.sizes()==out.sizes(),"dO/O shape or device mismatch");
    TORCH_CHECK(out.is_contiguous() && dout.is_contiguous(),"dO/O must be contiguous");
    TORCH_CHECK(out.scalar_type()==at::kBFloat16 && dout.scalar_type()==at::kBFloat16,"dO/O must be BF16");
    int64_t dv=out.size(3),rows=out.size(0)*out.size(1)*out.size(2);
    TORCH_CHECK((dv==32 || dv==64 || dv==128) && rows>0 && rows<=INT_MAX-7,"unsupported dimensions");
    const c10::cuda::CUDAGuard guard(out.device());
    cudaDeviceProp prop; C10_CUDA_CHECK(cudaGetDeviceProperties(&prop,out.get_device()));
    TORCH_CHECK(prop.major==12 && prop.minor==0,"initial backward supports sm120 only");
    auto result=torch::empty({out.size(0),out.size(1),out.size(2)},out.options().dtype(at::kFloat));
    dism_v2::launch_backward_delta(dout.data_ptr(),out.data_ptr(),result.data_ptr<float>(),rows,dv,
        c10::cuda::getCurrentCUDAStream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return result;
}
std::vector<torch::Tensor> operand_gradient(torch::Tensor a,torch::Tensor b,torch::Tensor v,torch::Tensor dout,
        torch::Tensor lse,torch::Tensor tau,torch::Tensor ql,torch::Tensor kl,torch::Tensor norm,
        torch::Tensor dd,torch::Tensor vertical,torch::Tensor horizontal,torch::Tensor boundary,
        double scale,bool column_lse,double probability,uint64_t seed,uint64_t offset,bool warp_specialized,
        std::optional<torch::Tensor> hard_bits) {
    TORCH_CHECK(a.is_cuda() && a.dim()==4,"A must be CUDA [B,H,N,D]");
    for(const auto& x:{a,b,v,dout,lse,tau,ql,kl,norm,vertical,horizontal,dd,boundary})
        TORCH_CHECK(x.device()==a.device() && x.is_contiguous(),"inputs must be contiguous on same CUDA device");
    int64_t batch=a.size(0),heads=a.size(1),n=a.size(2),d=a.size(3);
    TORCH_CHECK(batch>0 && heads>0 && n>0 && batch*heads<=65535 && batch*heads*n<=INT_MAX/2,"unsupported dimensions");
    TORCH_CHECK(b.sizes()==a.sizes() && dout.dim()==4 && dout.size(0)==batch && dout.size(1)==heads && dout.size(2)==n,"A/B/dO shape mismatch");
    int dv=dout.size(3),np=(n+63)/64*64;
    TORCH_CHECK((d==32||d==64||d==128) && (dv==32||dv==64||dv==128),"D/DV must be 32,64,128");
    for(const auto& x:{a,b,dout}) TORCH_CHECK(x.scalar_type()==at::kBFloat16,"A/B/dO must be BF16");
    for(const auto& x:{lse,tau,norm,vertical,horizontal}) TORCH_CHECK(x.scalar_type()==at::kFloat,"states must be FP32");
    TORCH_CHECK((ql.scalar_type()==at::kLong || ql.scalar_type()==at::kInt) && kl.scalar_type()==ql.scalar_type(),"labels must have matching int32/int64 dtype");
    for(const auto& x:{lse,ql,kl,norm})
        TORCH_CHECK(x.dim()==3 && x.size(0)==batch && x.size(1)==heads && x.size(2)==n,"row metadata shape mismatch");
    TORCH_CHECK(tau.dim()==1 && tau.size(0)==heads,"tau must be [H]");
    for(const auto& x:{vertical,horizontal}) TORCH_CHECK(x.dim()==4 && x.size(0)==batch && x.size(1)==heads && x.size(3)==np,"boundary shape mismatch");
    TORCH_CHECK(vertical.size(2)==np/16 && horizontal.size(2)==np/64,"boundary granularity mismatch");
    TORCH_CHECK(std::isfinite(scale) && std::isfinite(probability) && probability>=0 && probability<=1 && offset%4==0,"invalid scale/RNG metadata");

    TORCH_CHECK(v.sizes()==dout.sizes() && v.scalar_type()==at::kBFloat16,"V must match BF16 dO");
    TORCH_CHECK(dd.sizes()==norm.sizes() && dd.scalar_type()==at::kFloat,"delta must match FP32 normalizer");
    TORCH_CHECK(boundary.dim()==4 && boundary.size(0)==batch && boundary.size(1)==heads &&
        boundary.size(2)==np/32 && boundary.size(3)==np && boundary.scalar_type()==at::kFloat,"G32 boundary shape/dtype mismatch");
    c10::cuda::CUDAGuard guard(a.device());
    cudaDeviceProp prop;C10_CUDA_CHECK(cudaGetDeviceProperties(&prop,a.get_device()));
    TORCH_CHECK(prop.major==12 && prop.minor==0,"initial backward supports sm120 only");
    auto da=torch::zeros(a.sizes(),a.options().dtype(at::kFloat));
    auto db=torch::empty_like(da);
    auto dlse=torch::zeros_like(lse),dtau=torch::empty_like(tau);
    int partial_count=warp_specialized?((np+127)/128)*8:np/32;
    auto tau_partial=torch::empty({batch,heads,partial_count},tau.options());
    dism_v2::Args p{};
    p.a=a.data_ptr();p.b=b.data_ptr();p.v=v.data_ptr();p.lse=lse.data_ptr<float>();p.tau=tau.data_ptr<float>();
    p.q_label=reinterpret_cast<const int64_t*>(ql.data_ptr());p.k_label=reinterpret_cast<const int64_t*>(kl.data_ptr());p.normalizer=norm.data_ptr<float>();
    p.vertical=vertical.data_ptr<float>();p.horizontal=horizontal.data_ptr<float>();
    p.n=n;p.padded_n=np;p.batch_heads=batch*heads;p.heads=heads;p.scale=scale;p.column_lse=column_lse;
    p.hard_prob=probability;p.seed=seed;p.offset=offset;
    p.label32=ql.scalar_type()==at::kInt;
    if(hard_bits) {
        TORCH_CHECK(hard_bits->device()==a.device() && hard_bits->scalar_type()==at::kInt && hard_bits->is_contiguous() &&
            hard_bits->sizes()==torch::IntArrayRef({batch,heads,(n+31)/32}),"invalid hard bitset");
        p.hard_bits=reinterpret_cast<const uint32_t*>(hard_bits->data_ptr<int>());
    }
    auto launch=warp_specialized?dism_v2::launch_operand_backward_ws:dism_v2::launch_operand_backward;
    launch(p,d,dv,dout.data_ptr(),dd.data_ptr<float>(),boundary.data_ptr<float>(),
        da.data_ptr<float>(),db.data_ptr<float>(),dlse.data_ptr<float>(),tau_partial.data_ptr<float>(),
        c10::cuda::getCurrentCUDAStream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    dism_v2::launch_tau_reduce(tau_partial.data_ptr<float>(),dtau.data_ptr<float>(),
        batch,heads,partial_count,c10::cuda::getCurrentCUDAStream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();return {da,db,dlse,dtau};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) { m.def("delta",&delta); m.def("value_gradient",&value_gradient); m.def("operand_gradient",&operand_gradient); }
