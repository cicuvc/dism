#define DISM_HOST_API 1
#include "host/torch.cuh"
#include "forward/producer.cuh"
#include "forward/readout.cuh"
#include "varlen/forward.cuh"

#include "variant.cuh"

namespace DISM_VARIANT {
using namespace dism_forward;
template<bool FP32> const void* varlen_forward_address();
std::tuple<at::Tensor,at::Tensor,at::Tensor> varlen_forward_cuda(
        std::vector<at::Tensor> operands,at::Tensor table,bool save_boundaries,bool fp32_output) {
    auto run = [&]<bool FP32Output>() -> std::tuple<at::Tensor,at::Tensor,at::Tensor> {
        using Args = ArgsT<FP32Output>;
        using DType = OutputDType<FP32Output>;
        using OutputMap = OutputGlobal<FP32Output>;
        using namespace dism_varlen;
        Layout layout(table);
        int h=validate_operands(operands,layout);
        TORCH_CHECK((operands.size()==13 || operands.size()==14) && operands[12].device()==operands[0].device() &&
                    operands[12].scalar_type()==at::kFloat && operands[12].dim()==1 &&
                    operands[12].is_contiguous() && operands[12].numel()==layout.forward*h,
                    "expected ragged horizontal boundary buffer");
        c10::cuda::CUDAGuard guard(operands[0].device());
        auto options=operands[0].options();
        auto output=at::empty({1,layout.tokens,h,DIM},
                             options.dtype(FP32Output?at::kFloat:at::kBFloat16));
        // Global [H,T]; aligned document starts preserve scalar TMA alignment.
        auto normalizer=at::empty({layout.padded*h},options.dtype(at::kFloat));
        auto vertical=at::full({save_boundaries?layout.vertical*h:0},LOG_ZERO,options.dtype(at::kFloat));
        if (!layout.tokens) return {output,normalizer,vertical};
        TORCH_CHECK(layout.tokens*h<=INT_MAX,"packed metadata coordinate exceeds int32");
        int t=layout.tokens;
        const auto& x=operands;
        using QG=decltype(Args::q); using SQG=decltype(Args::sq);
        using KG=decltype(Args::k); using SKG=decltype(Args::sk); using VG=decltype(Args::v);
        using FG=decltype(Args::q_lse); using IG=decltype(Args::q_label);
        auto* out=reinterpret_cast<DType*>(output.data_ptr());
        Args args{1,h,t,t,0,
            QG(reinterpret_cast<kt::bf16*>(x[0].data_ptr()),1,h,t,CONFIG.KcKeyDim,
               kt::gl_strides{size_t(t)*h*CONFIG.KcKeyDim,CONFIG.KcKeyDim,size_t(h*CONFIG.KcKeyDim)}),
            SQG(reinterpret_cast<kt::bf16*>(x[2].data_ptr()),1,h,t,READOUT_DIM,
                kt::gl_strides{size_t(t)*h*READOUT_DIM,READOUT_DIM,size_t(h*READOUT_DIM)}),
            KG(reinterpret_cast<kt::bf16*>(x[1].data_ptr()),1,t,h,CONFIG.KcKeyDim),
            SKG(reinterpret_cast<kt::bf16*>(x[3].data_ptr()),1,t,h,READOUT_DIM),
            VG(reinterpret_cast<kt::bf16*>(x[4].data_ptr()),1,t,h,DIM),
            FG(x[5].data_ptr<float>(),1,1,1,t*h),FG(x[6].data_ptr<float>(),1,1,1,t*h),
            IG(x[7].data_ptr<int>(),1,1,1,t*h),IG(x[8].data_ptr<int>(),1,1,1,t*h),
            x[9].data_ptr<uint8_t>(),x[10].data_ptr<uint8_t>(),x[11].data_ptr<float>(),
            x[12].data_ptr<float>(),out,normalizer.data_ptr<float>(),
            save_boundaries?vertical.data_ptr<float>():nullptr,
            OutputMap(out,1,h,t,DIM,kt::gl_strides{size_t(t)*h*DIM,DIM,size_t(h*DIM)})};
        args.gate_delta=x.size()==14 ? x[13].data_ptr<float>() : nullptr;
        std::vector<int2> tasks;
        for (int s=0;s<layout.sequences;++s)
            append_tasks(tasks,s,h*(layout.get(s,Length)/QROWS));
        launch_common(varlen_forward_address<FP32Output>(),args,layout,tasks,shared_bytes<FP32Output>,operands[0].device());
        return {output,normalizer,vertical};
    };
    #if DISM_ENABLE_FP32
    return fp32_output ? run.template operator()<true>() : run.template operator()<false>();
#else
    TORCH_CHECK(!fp32_output,"FP32 output instances are disabled in this build");
    return run.template operator()<false>();
#endif
}

} // namespace DISM_VARIANT
