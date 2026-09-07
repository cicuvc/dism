#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <climits>
namespace dism_v2 { void launch_backward_delta(const void*,const void*,float*,int,int,cudaStream_t); }

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
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) { m.def("delta",&delta); }
