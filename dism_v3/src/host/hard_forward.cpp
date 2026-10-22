#define DISM_HOST_API 1
#include "host/torch.cuh"
#include "host/graph_route.h"
#include "forward/hard.cuh"
#include "varlen/layout.cuh"

namespace DISM_VARIANT {
const void* hard_forward_address();
const void* hard_log_weights_address();

at::Tensor hard_log_weights_cuda(at::Tensor tau,int64_t n, std::optional<at::Tensor> choice) {
    TORCH_CHECK(tau.is_cuda() && tau.scalar_type()==at::kFloat && tau.dim()==1 &&
                tau.is_contiguous() && tau.numel()>0 && tau.numel()<=65535,
                "tau must be contiguous CUDA FP32 [H]");
    TORCH_CHECK(n>0 && n<=INT_MAX-256 && n%256==0,"LUT maximum length must be positive and 256-aligned");
    c10::cuda::CUDAGuard guard(tau.device());
    const auto* route=dism_graph_route::validate(choice,tau.device());
    auto result=at::empty({tau.numel(),n+1},tau.options());
    launch_kernel(hard_log_weights_address(),dim3((n+256)/256,tau.numel()),256,0,
        at::cuda::getCurrentCUDAStream(),tau.data_ptr<float>(),result.data_ptr<float>(),int(n+1),route);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return result;
}

std::tuple<at::Tensor,at::Tensor> hard_readout_cuda(std::vector<at::Tensor> x,
        at::Tensor boundary,at::Tensor lut,std::optional<at::Tensor> table,int64_t requested_ctas, std::optional<at::Tensor> choice) {
    using namespace dism_forward;
    TORCH_CHECK(x.size()==6,"expected SQ,SK,V,query labels,key labels,hard flags");
    const auto& sq=x[0];
    TORCH_CHECK(sq.is_cuda() && sq.dim()==4,"SQ must be CUDA [B,N,H,R]");
    c10::cuda::CUDAGuard guard(sq.device());
    const auto* route=dism_graph_route::validate(choice,sq.device());
    int64_t b=sq.size(0),n=sq.size(1),h=sq.size(2);
    TORCH_CHECK(b>0 && h>0 && b*h<=INT_MAX && n>0 && n<=INT_MAX-256 && n%256==0,
                "positive dimensions and 256-aligned length required");
    auto check=[&](const at::Tensor& tensor,at::ScalarType type,at::IntArrayRef shape,const char* name) {
        TORCH_CHECK(tensor.device()==sq.device() && tensor.scalar_type()==type &&
                    tensor.is_contiguous() && tensor.sizes()==shape,
                    name,": invalid device, dtype, shape or stride");
    };
    check(sq,at::kBFloat16,{b,n,h,READOUT_DIM},"SQ");
    check(x[1],at::kBFloat16,sq.sizes(),"SK");
    check(x[2],at::kBFloat16,{b,n,h,DIM},"V");
    check(x[3],at::kInt,{b,h,n},"query labels");
    check(x[4],at::kInt,{b,h,n},"key labels");
    const auto& hard=x[5];
    TORCH_CHECK(hard.device()==sq.device() && hard.scalar_type()==at::kBool && hard.dim()==2 &&
                hard.size(0)>0 && hard.size(1)==h && hard.is_contiguous(),
                "hard must be contiguous bool [training_rows,H]; token-level masks are unsupported");
    TORCH_CHECK(lut.device()==sq.device() && lut.scalar_type()==at::kFloat && lut.dim()==2 &&
                lut.size(0)==h && lut.size(1)<=INT_MAX && lut.is_contiguous(),"invalid hard log-weight table");
    TORCH_CHECK(requested_ctas>=0 && requested_ctas<=INT_MAX,"invalid CTA count");
    int64_t tasks=b*h*(n/QROWS),max_length=n;
    at::Tensor task_storage;
    if (table) {
        using namespace dism_varlen;
        Layout layout(*table);
        TORCH_CHECK(b==1 && layout.tokens==n,"packed labels/layout length mismatch");
        TORCH_CHECK(n%hard.size(0)==0 && (n/hard.size(0))%256==0,
                    "packed hard rows must evenly divide tokens into 256-aligned training rows");
        int64_t row_tokens=n/hard.size(0);
        check(boundary,at::kInt,{layout.forward*h},"packed hard boundary");
        max_length=layout.max_padded;
        auto build=[h,n,row_tokens](const at::Tensor& table) {
        Layout layout(table);
        std::vector<dism_hard_forward::Task> records;
        for (int s=0;s<layout.sequences;++s) {
            int64_t begin=layout.get(s,Begin),length=layout.get(s,Length);
            if (!length) continue;
            TORCH_CHECK(begin/row_tokens==(begin+length-1)/row_tokens,
                        "document cannot cross a hard training-row boundary");
            for (int head=0;head<h;++head)
                for (int q=0;q<length;q+=QROWS)
                    records.push_back({0,head,int(begin),int(length),q,0,head*n+begin,
                        layout.get(s,ForwardOffset)*h+head*((length-1)/32)*length,
                        (begin/row_tokens)*h+head});
        }
        return records;
        };
        auto records=build(*table);
        tasks=records.size();
        task_storage=dism_graph::state(*table) ? dism_graph::records(*table,
            "hard_forward:"+std::to_string(h),sq.device(),
            [build](const at::Tensor& table) { return dism_graph::bytes(build(table)); })
            : upload_records(records,sq.device());
    } else {
        TORCH_CHECK(hard.size(0)==b,"fixed hard must be [B,H]");
        check(boundary,at::kInt,{b,h,(n-1)/32,n},"hard boundary");
    }
    TORCH_CHECK(tasks>0 && tasks<=INT_MAX,"invalid readout task count");
    TORCH_CHECK(lut.size(1)>max_length,"hard log-weight table is too short");
    auto output=at::zeros({b,n,h,DIM},sq.options());
    auto normalizer=at::zeros({b,h,n},sq.options().dtype(at::kFloat));
    using Args=dism_hard_forward::Args;
    using SQG=decltype(Args::sq); using SKG=decltype(Args::sk); using VG=decltype(Args::v);
    Args args{int(b),int(h),int(n),int(tasks),int(lut.size(1)),
        SQG(reinterpret_cast<kt::bf16*>(sq.data_ptr()),b,h,n,READOUT_DIM,
            kt::gl_strides{size_t(n*h*READOUT_DIM),READOUT_DIM,size_t(h*READOUT_DIM)}),
        SKG(reinterpret_cast<kt::bf16*>(x[1].data_ptr()),b,n,h,READOUT_DIM),
        VG(reinterpret_cast<kt::bf16*>(x[2].data_ptr()),b,n,h,DIM),
        x[3].data_ptr<int>(),x[4].data_ptr<int>(),boundary.data_ptr<int>(),lut.data_ptr<float>(),
        hard.data_ptr<bool>(),table?reinterpret_cast<const dism_hard_forward::Task*>(task_storage.data_ptr()):nullptr,
        normalizer.data_ptr<float>(),OutputGlobal<false>(reinterpret_cast<kt::bf16*>(output.data_ptr()),b,h,n,DIM,
            kt::gl_strides{size_t(n*h*DIM),DIM,size_t(h*DIM)})};
    int sms=at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    int ctas=std::min(int(tasks),requested_ctas?int(requested_ctas):sms);
    C10_CUDA_CHECK(cudaFuncSetAttribute(hard_forward_address(),cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      dism_hard_forward::SharedBytes));
    launch_kernel(hard_forward_address(),ctas,384,dism_hard_forward::SharedBytes,
                  at::cuda::getCurrentCUDAStream(),args,route);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {output,normalizer};
}
} // namespace DISM_VARIANT
