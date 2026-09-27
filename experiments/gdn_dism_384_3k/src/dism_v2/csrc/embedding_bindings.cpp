#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cmath>
#include <climits>

void launch_embedding(const void*, const void*, const void*, void*, float*, float*, int*,
                      int, int, int, int, int, float, cudaStream_t);
void launch_embedding_fused(const void*,const void*,const void*,const void*,void*,void*,
                           float*,float*,float*,float*,int*,int*,int,int,int,int,int,int,float,cudaStream_t,
                           uint32_t*,uint64_t,uint64_t,float);

std::vector<torch::Tensor> embedding_forward(torch::Tensor q, torch::Tensor k,
        torch::Tensor eq, torch::Tensor ek, double scale, bool ws, int block_v,
        std::optional<std::tuple<uint64_t,uint64_t,float>> row_rng) {
    TORCH_CHECK(q.is_cuda() && q.dim()==4, "q must be CUDA [B,H,N,D]");
    for (const auto& x : {q,k,eq,ek})
        TORCH_CHECK(x.device()==q.device() && x.is_contiguous() && x.scalar_type()==at::kBFloat16,
                    "inputs must be contiguous BF16 on the same CUDA device");
    auto b=q.size(0),h=q.size(1),n=q.size(2),d=q.size(3);
    TORCH_CHECK(k.sizes()==q.sizes(), "q/k shape mismatch");
    TORCH_CHECK(eq.dim()==3 && ek.sizes()==eq.sizes() && eq.size(0)==h && eq.size(2)==d,
                "vocab must be [H,V,D]");
    auto v=eq.size(1);
    TORCH_CHECK(b>0 && h>0 && n>0 && v>0 && b*h<=65535 && n<=INT_MAX-64 && v<=INT_MAX-64,
                "unsupported dimensions");
    TORCH_CHECK(d==32 || d==64 || d==128, "D must be 32,64,128");
    TORCH_CHECK(block_v==64 || (block_v==128 && ws && d!=128),
                "block_v=128 requires WS and D=32/64");
    TORCH_CHECK(std::isfinite(scale) && std::isfinite(float(scale)), "scale must be finite FP32");
    const c10::cuda::CUDAGuard guard(q.device());
    cudaDeviceProp prop;
    C10_CUDA_CHECK(cudaGetDeviceProperties(&prop,q.get_device()));
    TORCH_CHECK(prop.major==12 && prop.minor==0, "initial embedding supports sm120 only");
    auto oq=torch::empty_like(q),ok=torch::empty_like(k);
    auto opt=q.options().dtype(at::kFloat);
    auto lk=torch::empty({b,h,n},opt),lq=torch::empty_like(lk);
    auto pk=torch::empty_like(lk),pq=torch::empty_like(lk);
    auto ik=torch::empty({b,h,n},q.options().dtype(at::kInt)),iq=torch::empty_like(ik);
    auto stream=c10::cuda::getCurrentCUDAStream();
    torch::Tensor bits;
    uint64_t seed=0,offset=0;float probability=0;
    if(row_rng) {
        std::tie(seed,offset,probability)=*row_rng;
        TORCH_CHECK(ws && probability>0 && probability<1 && offset%4==0,"bitset requires WS and mixed probability");
        bits=torch::empty({b,h,(n+31)/32},q.options().dtype(at::kInt));
    }
    if(ws) {
        launch_embedding_fused(q.data_ptr(),k.data_ptr(),eq.data_ptr(),ek.data_ptr(),oq.data_ptr(),ok.data_ptr(),
            lk.data_ptr<float>(),lq.data_ptr<float>(),pk.data_ptr<float>(),pq.data_ptr<float>(),
            ik.data_ptr<int>(),iq.data_ptr<int>(),b*h,h,n,v,d,block_v,float(scale),stream,
            row_rng?reinterpret_cast<uint32_t*>(bits.data_ptr<int>()):nullptr,seed,offset,probability);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else {
    launch_embedding(k.data_ptr(),ek.data_ptr(),eq.data_ptr(),oq.data_ptr(),lk.data_ptr<float>(),
                     pk.data_ptr<float>(),ik.data_ptr<int>(),b*h,h,n,v,d,float(scale),stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    launch_embedding(q.data_ptr(),eq.data_ptr(),ek.data_ptr(),ok.data_ptr(),lq.data_ptr<float>(),
                     pq.data_ptr<float>(),iq.data_ptr<int>(),b*h,h,n,v,d,float(scale),stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    std::vector<torch::Tensor> result{oq,ok,lk,lq,pk,pq,ik,iq};
    if(row_rng) result.push_back(bits);
    return result;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) { m.def("forward", &embedding_forward); }
