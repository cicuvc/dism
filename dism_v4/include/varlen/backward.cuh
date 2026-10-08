#pragma once
#include "backward/types.cuh"
#if DISM_HOST_API
#include "backward/launch.cuh"
#endif
#include "varlen/operands.cuh"

#include "variant.cuh"
namespace DISM_VARIANT {

namespace dism_varlen {
#if DISM_HOST_API
struct BackwardInputs {
    Layout layout;
    const std::vector<at::Tensor>& operands;
    at::Tensor vertical,dout,normalizer,delta;
    int heads;

    BackwardInputs(const std::vector<at::Tensor>& x,at::Tensor w,at::Tensor d,
                   at::Tensor l,at::Tensor z,at::Tensor table)
        : layout(table),operands(x),vertical(w),dout(d),normalizer(l),delta(z),
          heads(validate_operands(x,layout)) {
        auto check=[&](const at::Tensor& tensor,at::ScalarType dtype,at::IntArrayRef shape) {
            TORCH_CHECK(tensor.device()==x[0].device() && tensor.scalar_type()==dtype &&
                        tensor.is_contiguous() && tensor.sizes()==shape,"invalid varlen backward input");
        };
        TORCH_CHECK((x.size()==13 || x.size()==14),"missing saved horizontal checkpoints");
        check(x[12],at::kFloat,{layout.forward*heads});
        check(w,at::kFloat,{layout.vertical*heads});
        check(d,at::kBFloat16,{layout.padded,heads,dism_backward::DV});
        check(l,at::kFloat,{layout.padded*heads});
        check(z,at::kFloat,{layout.padded*heads});
    }

    dism_backward::Args common() const {
        using namespace dism_backward;
        TORCH_CHECK(layout.tokens*heads<=INT_MAX,"packed scalar TMA coordinate exceeds int32");
        int t=layout.tokens,h=heads;
        const auto& x=operands;
        auto ptr=[&](int i) { return reinterpret_cast<kt::bf16*>(x[i].data_ptr()); };
        RecomputeArgs score{1,h,t,t,
            QueryGlobal(ptr(0),1,t,h,D),
            KeyGlobal(ptr(1),1,h,t,D,kt::gl_strides{size_t(t)*h*D,D,size_t(h*D)}),
            x[5].data_ptr<float>(),x[6].data_ptr<float>(),x[11].data_ptr<float>(),
            vertical.data_ptr<float>(),x[12].data_ptr<float>(),
            x[7].data_ptr<int>(),x[8].data_ptr<int>(),x[9].data_ptr<uint8_t>(),x[10].data_ptr<uint8_t>(),x.size()==14 ? x[13].data_ptr<float>() : nullptr};
        return {score,
            SoftQueryGlobal(ptr(2),1,t,h,R),
            DerivativeGlobal(reinterpret_cast<kt::bf16*>(dout.data_ptr()),1,t,h,DV),
            HeldKeyGlobal(ptr(1),1,h,t,D,kt::gl_strides{size_t(t)*h*D,D,size_t(h*D)}),
            HeldSoftGlobal(ptr(3),1,h,t,R,kt::gl_strides{size_t(t)*h*R,R,size_t(h*R)}),
            HeldValueGlobal(ptr(4),1,h,t,DV,kt::gl_strides{size_t(t)*h*DV,DV,size_t(h*DV)}),
            normalizer.data_ptr<float>(),delta.data_ptr<float>(),nullptr,
            nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr};
    }
};

#endif

// Only one common argument object, including all tensor maps, per launch.
// The global arrays below contain small integer metadata, never Args records.
struct BackwardArgs {
    dism_backward::Args common;
    const int64_t* documents;
    const int2* tasks;
};
static_assert(sizeof(BackwardArgs)<=4096);

struct BackwardTask {
    int begin,n,head,bh,k0;
    int64_t metadata,forward,vertical,reverse;
};

__device__ __forceinline__ BackwardTask backward_task(const BackwardArgs& a) {
    int2 work=a.tasks[blockIdx.x];
    const int64_t* row=a.documents+int64_t(work.x)*Fields;
    int n=row[Length],begin=row[Begin],h=a.common.score.heads;
    int blocks=n/128,head=work.y/blocks;
    return {begin,n,head,head,(work.y%blocks)*128,
        int64_t(head)*a.common.score.n+begin,row[ForwardOffset]*h,
        row[VerticalOffset]*h,row[BackwardOffset]*h};
}

// A small register view, with no tensor maps or owning state. All base pointers
// originate in the common kernel parameter, adjusted once for this document.
struct LocalScore {
    int heads,n,padded;
    const float *q_lse,*k_lse,*tau,*vertical,*horizontal;
    const int *q_label,*k_label;
    const uint8_t *hard,*direction;
    const float *gate_delta=nullptr;
};
__device__ __forceinline__ LocalScore local_score(
        const dism_backward::Args& a,const BackwardTask& task) {
    // Recompute helpers add head*n to scalar pointers, while checkpoint
    // strides must remain document-local. Compensate only the scalar base.
    int64_t start=task.metadata-int64_t(task.head)*task.n;
    return {a.score.heads,task.n,task.n,
        a.score.q_lse+start,a.score.k_lse+start,a.score.tau,
        a.score.vertical+task.vertical,a.score.horizontal+task.forward,
        a.score.q_label+start,a.score.k_label+start,a.score.hard+start,a.score.direction,a.score.gate_delta};
}

template<class HeldPipe,class InputPipe>
__device__ __forceinline__ void produce_backward(const dism_backward::Args& a,
        dism_backward::Shared& shared,HeldPipe& held,InputPipe& input,const BackwardTask& task) {
    using namespace dism_backward;
    kt::warpgroup::decrease_registers<40>();
    if (kt::warpgroup::warpid()!=0) return;
    bool leader=kt::warp::elect_leader();
    auto first=held.waitBuffer(0,shared.held);
    if (leader) {
        auto& data=first.template get<0>();
        kt::tma::expect_bytes(first.getBarrier(),sizeof(Shared::Held));
        kt::tma::load_async(data.k,a.k,{0,task.head,task.begin+task.k0,0},first.getBarrier());
        kt::tma::load_async(data.sk,a.sk,{0,task.head,task.begin+task.k0,0},first.getBarrier());
        kt::tma::load_async(data.v,a.v,{0,task.head,task.begin+task.k0,0},first.getBarrier());
    }
    first.submitToNextAndTrigger();
    held.moveNext();
#pragma unroll 1
    for (int q0=task.n-Q;q0>=task.k0;q0-=Q) {
        auto packet=input.waitBuffer(0,shared.input);
        if (leader) {
            auto& data=packet.template get<0>();
            kt::tma::expect_bytes(packet.getBarrier(),sizeof(QueryTile)+sizeof(SoftQueryTile)+sizeof(DerivativeTile));
            kt::tma::load_async(data.q,a.score.q,{0,task.begin+q0,task.head,0},packet.getBarrier());
            kt::tma::load_async(data.sq,a.sq,{0,task.begin+q0,task.head,0},packet.getBarrier());
            kt::tma::load_async(data.dout,a.dout,{0,task.begin+q0,task.head,0},packet.getBarrier());
        }
        load_row_gate(packet.template get<0>().row_gate,
            a.score.gate_delta ? a.score.gate_delta+task.metadata+q0 : nullptr,min(Q+1,task.n-q0));
        packet.submitToNextAndTrigger();
        input.moveNext();
    }
}

#if DISM_HOST_API
template<class Kernel>
inline void launch_backward(Kernel kernel,const dism_backward::Args& common,
                            const Layout& layout,at::Device device) {
    std::vector<int2> tasks;
    for (int s=0;s<layout.sequences;++s)
        append_tasks(tasks,s,common.score.heads*(layout.get(s,Length)/128));
    if (tasks.empty()) return;
    auto documents=dism_metadata::documents(layout.cpu,device);
    auto work=upload_records(tasks,device);
    BackwardArgs args{common,documents.data_ptr<int64_t>(),reinterpret_cast<const int2*>(work.data_ptr())};
    C10_CUDA_CHECK(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      dism_backward::SharedBytes));
    launch_kernel(kernel,tasks.size(),384,dism_backward::SharedBytes,at::cuda::getCurrentCUDAStream(),args);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
#endif
} // namespace dism_varlen

} // namespace DISM_VARIANT
