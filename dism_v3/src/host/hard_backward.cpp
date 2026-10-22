#define DISM_HOST_API 1
#include "host/torch.cuh"
#include "backward/hard.cuh"
#include "backward/launch.cuh"
#include "varlen/layout.cuh"

namespace DISM_VARIANT {
const void* hard_backward_address();
const void* hard_dlog_weights_address();
const void* hard_tau_reduce_address();

std::vector<at::Tensor> hard_backward_cuda(std::vector<at::Tensor> x,at::Tensor boundary,
        at::Tensor logw,at::Tensor normalizer,int64_t requested_ctas,
        std::optional<at::Tensor> table) {
    namespace hb=dism_hard_backward;
    using namespace dism_backward;
    TORCH_CHECK(x.size()==7,"expected SQ,SK,V,dO,query labels,key labels,hard");
    const auto& sq=x[0];
    TORCH_CHECK(sq.is_cuda() && sq.dim()==4,"SQ must be CUDA BNHR");
    c10::cuda::CUDAGuard guard(sq.device());
    int64_t b=sq.size(0),n=sq.size(1),h=sq.size(2);
    TORCH_CHECK(b>0 && h>0 && b*h<=INT_MAX && n>0 && n<=INT_MAX-256 && n%256==0,
                "positive dimensions and 256-aligned length required");
    TORCH_CHECK(b*h*(n/128)<=INT_MAX && b*n<=INT_MAX,"hard backward grid is too large");
    TORCH_CHECK(requested_ctas>=0 && requested_ctas<=INT_MAX,"invalid CTA count");
    auto check=[&](const at::Tensor& t,at::ScalarType dtype,at::IntArrayRef shape,const char* name) {
        TORCH_CHECK(t.device()==sq.device() && t.scalar_type()==dtype && t.is_contiguous() && t.sizes()==shape,
                    name,": invalid device, dtype, shape or stride");
    };
    check(sq,at::kBFloat16,{b,n,h,R},"SQ");
    check(x[1],at::kBFloat16,{b,n,h,R},"SK");
    check(x[2],at::kBFloat16,{b,n,h,DV},"V");
    check(x[3],at::kBFloat16,{b,n,h,DV},"dO");
    check(x[4],at::kInt,{b,h,n},"query labels");
    check(x[5],at::kInt,{b,h,n},"key labels");
    check(normalizer,at::kFloat,{b,h,n},"log2 normalizer");
    int64_t task_count=b*h*(n/128);
    int64_t lut_stride=n+1;
    at::Tensor task_storage, tau_task_storage;
    int64_t tau_task_count=0;
    if(table) {
        using namespace dism_varlen;
        Layout layout(*table);
        lut_stride=layout.max_padded+1;
        TORCH_CHECK(b==1 && layout.tokens==n,"packed labels/layout length mismatch");
        TORCH_CHECK(n%x[6].size(0)==0 && (n/x[6].size(0))%256==0,
                    "packed hard rows must evenly divide tokens into 256-aligned training rows");
        TORCH_CHECK(x[6].device()==sq.device() && x[6].scalar_type()==at::kBool &&
                    x[6].dim()==2 && x[6].size(1)==h && x[6].is_contiguous(),
                    "packed hard flags must be contiguous bool [training_rows,H]");
        int64_t row_tokens=n/x[6].size(0);
        check(boundary,at::kInt,{layout.forward*h},"packed integer boundary");
        check(logw,at::kFloat,{h,layout.max_padded+1},"packed log weight LUT");
        auto build=[h,n,row_tokens](const at::Tensor& table) {
        Layout layout(table);
        std::vector<dism_hard_backward::Task> records;
        std::vector<dism_hard_backward::Task> tau_records;
        for(int s=0;s<layout.sequences;++s) {
            int64_t begin=layout.get(s,Begin),length=layout.get(s,Length);
            if(!length) continue;
            TORCH_CHECK(begin/row_tokens==(begin+length-1)/row_tokens,
                        "document cannot cross a hard training-row boundary");
            for(int head=0;head<h;++head) {
                dism_hard_backward::Task first_task{
                        0,head,int(begin),int(length),head,0,
                        int64_t(head)*n+begin,
                        layout.get(s,ForwardOffset)*h
                            +int64_t(head)*((length-1)/32)*length,
                        int(begin/row_tokens)*int(h)+head};
                tau_records.push_back(first_task);
                for(int k0=0;k0<length;k0+=128) {
                    auto task=first_task;
                    task.k0=k0;
                    records.push_back(task);
                }
            }
        }
        if (dism_graph::state(table)) {
            dism_hard_backward::Task empty{};
            empty.head=-1; // tau reducer skips before touching the hard mask.
            tau_records.resize(layout.sequences*h,empty);
        }
        return std::make_pair(records,tau_records);
        };
        auto [records,tau_records]=build(*table);
        task_count=records.size();
        tau_task_count=tau_records.size();
        if (dism_graph::state(*table)) {
            task_storage=dism_graph::records(*table,"hard_backward:"+std::to_string(h),sq.device(),
                [build](const at::Tensor& table) { return dism_graph::bytes(build(table).first); });
            tau_task_storage=dism_graph::records(*table,"hard_tau:"+std::to_string(h),sq.device(),
                [build](const at::Tensor& table) { return dism_graph::bytes(build(table).second); });
        } else {
            task_storage=upload_records(records,sq.device());
            tau_task_storage=upload_records(tau_records,sq.device());
        }
    } else {
        check(x[6],at::kBool,{b,h},"sequence hard flags");
        check(boundary,at::kInt,{b,h,(n-1)/32,n},"integer boundary");
        check(logw,at::kFloat,{h,n+1},"log weight LUT");
    }
    TORCH_CHECK(task_count>0 && task_count<=INT_MAX,"invalid hard backward task count");
    auto options=sq.options().dtype(at::kFloat);
    auto dsq=at::zeros({b,n,h,R},options),dsk=at::zeros_like(x[1]),dv=at::zeros_like(x[2]);
    auto moments=at::zeros({4,b,h,n},options),dlogw=at::empty_like(logw),dtau=at::empty({h},options);
    hb::Args args{{int(b),int(h),int(n)},int(task_count),int(lut_stride),
        SoftQueryGlobal(reinterpret_cast<kt::bf16*>(sq.data_ptr()),b,n,h,R),
        DerivativeGlobal(reinterpret_cast<kt::bf16*>(x[3].data_ptr()),b,n,h,DV),
        HeldSoftGlobal(reinterpret_cast<kt::bf16*>(x[1].data_ptr()),b,h,n,R,
            kt::gl_strides{size_t(n*h*R),R,size_t(h*R)}),
        HeldValueGlobal(reinterpret_cast<kt::bf16*>(x[2].data_ptr()),b,h,n,DV,
            kt::gl_strides{size_t(n*h*DV),DV,size_t(h*DV)}),
        x[4].data_ptr<int>(),x[5].data_ptr<int>(),boundary.data_ptr<int>(),x[6].data_ptr<bool>(),
        table?reinterpret_cast<const dism_hard_backward::Task*>(task_storage.data_ptr()):nullptr,
        normalizer.data_ptr<float>(),logw.data_ptr<float>(),dlogw.data_ptr<float>(),
        moments.data_ptr<float>(),dv.data_ptr(),dsk.data_ptr()};
    args.key_output[0]=KeyOutputGlobal(reinterpret_cast<kt::bf16*>(dv.data_ptr()),b,h,n,DV,
        kt::gl_strides{size_t(n*h*DV),DV,size_t(h*DV)});
    args.key_output[1]=KeyOutputGlobal(reinterpret_cast<kt::bf16*>(dsk.data_ptr()),b,h,n,R,
        kt::gl_strides{size_t(n*h*R),R,size_t(h*R)});
    kt::gl<float,-1,-1,-1,-1,GradientTile> query_map(dsq.data_ptr<float>(),b,h,n,R,
        kt::gl_strides{size_t(n*h*R),R,size_t(h*R)});
    copy_output_map(args.query_output,query_map);
    auto stream=at::cuda::getCurrentCUDAStream();
    launch_kernel(hard_dlog_weights_address(),dim3((n+256)/256,h),256,0,stream,
                  logw.data_ptr<float>(),dlogw.data_ptr<float>(),int(lut_stride),int(h));
    int sms=at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    int ctas=std::min(int(task_count),requested_ctas?int(requested_ctas):sms);
    C10_CUDA_CHECK(cudaFuncSetAttribute(hard_backward_address(),cudaFuncAttributeMaxDynamicSharedMemorySize,hb::SharedBytes));
    launch_kernel(hard_backward_address(),ctas,384,hb::SharedBytes,stream,args);
    launch_kernel(hard_tau_reduce_address(),h,256,0,stream,moments.data_ptr<float>(),normalizer.data_ptr<float>(),
                  dlogw.data_ptr<float>(),dtau.data_ptr<float>(),int(b),int(h),int(n),int(lut_stride),
                  x[6].data_ptr<bool>(),
                  table?reinterpret_cast<const dism_hard_backward::Task*>(tau_task_storage.data_ptr()):nullptr,
                  int(tau_task_count));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {dsq,dsk,dv,dtau};
}
} // namespace DISM_VARIANT
