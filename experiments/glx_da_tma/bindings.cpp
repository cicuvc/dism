#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
void launch_da_probe(const float*,const void*,float*,int,int,int,int,float,cudaStream_t);
void run(torch::Tensor g,torch::Tensor b,torch::Tensor out,int warps,double scale) {
    TORCH_CHECK(g.is_cuda() && g.dim()==3 && g.size(1)==16 && g.scalar_type()==at::kFloat && g.is_contiguous(),"G must be FP32 [groups,16,N]");
    TORCH_CHECK(b.device()==g.device() && b.dim()==3 && b.size(0)==g.size(0) && b.size(1)==16 && b.scalar_type()==at::kBFloat16 && b.is_contiguous(),"B must be BF16 [groups,16,D]");
    int d=b.size(2),n=g.size(2);
    TORCH_CHECK((d==32 || d==64 || d==128) && n>0 && g.size(0)>0 && (warps==1 || warps==8),"invalid dimensions");
    TORCH_CHECK(out.device()==g.device() && out.scalar_type()==at::kFloat && out.is_contiguous() && out.dim()==2 && out.size(0)==n && out.size(1)==d,"output must be FP32 [N,D]");
    c10::cuda::CUDAGuard guard(g.device());
    launch_da_probe(g.data_ptr<float>(),b.data_ptr(),out.data_ptr<float>(),n,d,g.size(0),warps,scale,c10::cuda::getCurrentCUDAStream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void launch_db_probe(const float*,const void*,float*,int,int,int,float,cudaStream_t);
torch::Tensor run_db(torch::Tensor g,torch::Tensor a,double scale) {
    TORCH_CHECK(g.is_cuda() && g.dim()==3 && g.size(1)==16 && g.scalar_type()==at::kFloat && g.is_contiguous(),"invalid G");
    TORCH_CHECK(a.device()==g.device() && a.dim()==2 && a.size(0)==g.size(2) && a.scalar_type()==at::kBFloat16 && a.is_contiguous(),"invalid A");
    int d=a.size(1);
    TORCH_CHECK((d==32 || d==64 || d==128) && g.size(0)>0 && g.size(2)>0,"invalid dimensions");
    c10::cuda::CUDAGuard guard(g.device());
    auto out=torch::empty({g.size(0),16,d},g.options());
    launch_db_probe(g.data_ptr<float>(),a.data_ptr(),out.data_ptr<float>(),g.size(2),d,g.size(0),scale,c10::cuda::getCurrentCUDAStream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) { m.def("run",&run);m.def("run_db",&run_db); }
