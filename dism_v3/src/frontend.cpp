// Compiled once: ATen preparation and native configuration dispatch, no TK/CUDA syntax.
#include <ATen/ATen.h>
#include <ATen/core/grad_mode.h>
#include <torch/csrc/utils/pybind.h>
#include <pybind11/stl.h>
#include <climits>
#include "config_registry.h"

namespace py=pybind11;
using T=at::Tensor;
using V=std::vector<T>;
using Pair=std::tuple<T,T>;
using Triple=std::tuple<T,T,T>;

#define DECLARE(R,D,DV) namespace dism_r##R##_d##D##_v##DV { \
Pair summarization_cuda(T,T,T,T,T,T,T,T,T,int64_t,int64_t); \
T chunk_scan_cuda(T,T,int64_t); \
Pair forward_cuda(T,T,T,T,T,T,T,T,T,T,T,T,T,int64_t,int64_t,std::optional<T>,bool); \
T backward_delta_cuda(T,T); \
V backward_summary_cuda(V,T,T,T,T,int64_t,int64_t,bool); \
T backward_chunk_cuda(T,T); \
V backward_qk_cuda(V,T,T,T,T,T,int64_t,int64_t,bool); \
Pair varlen_summary_cuda(V,T); T varlen_chunk_cuda(T,T,T,int64_t); \
Triple varlen_forward_cuda(V,T,bool,bool); T varlen_delta_cuda(T,T,T); \
V varlen_backward_summary_cuda(V,T,T,T,T,T,bool); \
T varlen_backward_chunk_cuda(T,T,T,int64_t); \
V varlen_backward_qk_cuda(V,T,T,T,T,T,T,bool); }
DISM_CONFIGS(DECLARE)
#undef DECLARE

struct Backend {
    int r,d,dv;
    Pair (*summary)(T,T,T,T,T,T,T,T,T,int64_t,int64_t);
    T (*chunk)(T,T,int64_t);
    Pair (*forward)(T,T,T,T,T,T,T,T,T,T,T,T,T,int64_t,int64_t,std::optional<T>,bool);
    T (*delta)(T,T);
    V (*b1)(V,T,T,T,T,int64_t,int64_t,bool);
    T (*b2)(T,T);
    V (*b3)(V,T,T,T,T,T,int64_t,int64_t,bool);
    Pair (*vs)(V,T); T (*vc)(T,T,T,int64_t);
    Triple (*vf)(V,T,bool,bool); T (*vd)(T,T,T);
    V (*vb1)(V,T,T,T,T,T,bool); T (*vb2)(T,T,T,int64_t);
    V (*vb3)(V,T,T,T,T,T,T,bool);
};
#define ENTRY(R,D,DV) {R,D,DV, \
    dism_r##R##_d##D##_v##DV::summarization_cuda, dism_r##R##_d##D##_v##DV::chunk_scan_cuda, \
    dism_r##R##_d##D##_v##DV::forward_cuda, dism_r##R##_d##D##_v##DV::backward_delta_cuda, \
    dism_r##R##_d##D##_v##DV::backward_summary_cuda, dism_r##R##_d##D##_v##DV::backward_chunk_cuda, \
    dism_r##R##_d##D##_v##DV::backward_qk_cuda, dism_r##R##_d##D##_v##DV::varlen_summary_cuda, \
    dism_r##R##_d##D##_v##DV::varlen_chunk_cuda, dism_r##R##_d##D##_v##DV::varlen_forward_cuda, \
    dism_r##R##_d##D##_v##DV::varlen_delta_cuda, dism_r##R##_d##D##_v##DV::varlen_backward_summary_cuda, \
    dism_r##R##_d##D##_v##DV::varlen_backward_chunk_cuda, dism_r##R##_d##D##_v##DV::varlen_backward_qk_cuda},
static const Backend backends[]={ DISM_CONFIGS(ENTRY) };
#undef ENTRY

static const Backend& select(int r,int d,int dv) {
    for (const auto& b:backends) if (b.r==r && b.d==d && b.dv==dv) return b;
    TORCH_CHECK(false,"unsupported DISM (R,D,DV)=",r,",",d,",",dv);
}

static void precision(bool fp32) {
    TORCH_CHECK(!fp32 || DISM_ENABLE_FP32,"FP32 validation instances are disabled");
}

static void check_tensor(const T& x,const T& anchor,at::ScalarType dtype,
                         at::IntArrayRef shape,const char* name) {
    TORCH_CHECK(x.device()==anchor.device() && x.scalar_type()==dtype && x.sizes()==shape,
                name,": invalid device, dtype or shape");
}

// Raw public argument order: Q,K,SQ,SK,V,LQ,LK,IQ,IK,direction,hard,tau.
static V prepare(const V& x,bool packed,bool absorbed=false) {
    TORCH_CHECK(x.size()==12,"expected twelve core operands");
    const auto& q=x[0];
    TORCH_CHECK(q.is_cuda() && q.dim()==4,"q must be CUDA [B,N,H,D]");
    auto b=q.size(0),n=q.size(1),h=q.size(2),d=q.size(3);
    TORCH_CHECK(b>0 && h>0 && n<=INT_MAX-255 && n%256==0 && (packed || n>0),
                "sequence length must be256-token aligned (no automatic padding)");
    TORCH_CHECK(!packed || b==1,"varlen requires batch1");
    TORCH_CHECK(x[2].dim()==4 && x[4].dim()==4,"sq/v must be rank4");
    select(x[2].size(3),d,x[4].size(3));
    for (int i=0;i<5;++i) {
        int64_t c=i<2?d:i<4?x[2].size(3):x[4].size(3);
        check_tensor(x[i],q,at::kBFloat16,{b,n,h,c},"vector");
    }
    for (int i:{5,6}) check_tensor(x[i],q,at::kFloat,{b,n,h},"LSE");
    for (int i:{7,8}) {
        TORCH_CHECK(x[i].scalar_type()==at::kInt || x[i].scalar_type()==at::kLong,"labels must be integer");
        check_tensor(x[i],q,x[i].scalar_type(),{b,h,n},"labels");
    }
    check_tensor(x[9],q,at::kBool,{b,h},"direction");
    check_tensor(x[10],q,at::kBool,{b,h,n},"hard");
    check_tensor(x[11],q,at::kFloat,{h},"tau");
    V result;
    for (int i=0;i<5;++i) {
        auto value=x[i].contiguous();
        result.push_back(packed?value.squeeze(0):value);
    }
    for (int i:{5,6}) {
        auto lse=absorbed?x[i]:x[i]-x[11].view({1,1,h});
        result.push_back(lse.transpose(1,2).contiguous());
    }
    for (int i:{7,8}) result.push_back(x[i].to(at::kInt).contiguous());
    result.push_back(x[10].to(at::kByte).contiguous());
    if (packed) for (int i=5;i<10;++i) result[i]=result[i].view({-1});
    result.push_back(x[9].to(at::kByte).contiguous());
    result.push_back(x[11].contiguous());
    return result;
}

static py::tuple core_forward(V raw,std::optional<T> table,int64_t ctas,bool save,bool fp32,bool absorbed) {
    at::NoGradGuard no_grad;
    precision(fp32);
    TORCH_CHECK(!table || ctas==0,"ctas is only supported for fixed execution");
    auto x=prepare(raw,table.has_value(),absorbed);
    const auto& backend=select(raw[2].size(3),raw[0].size(3),raw[4].size(3));
    int64_t b=raw[0].size(0),n=raw[0].size(1),h=raw[0].size(2);
    T a,s,boundary,out,norm,vertical;
    if (table) {
        std::tie(a,s)=backend.vs(x,*table);
        boundary=backend.vc(a,s,*table,h);
    } else {
        std::tie(a,s)=backend.summary(x[0],x[1],x[5],x[6],x[7],x[8],x[9],x[10],x[11],n,ctas);
        boundary=backend.chunk(a,s,n);
    }
    a=T(); s=T();
    x.push_back(boundary);
    if (table) std::tie(out,norm,vertical)=backend.vf(x,*table,save,fp32);
    else {
        if (save) vertical=at::full({b,h,(n-1)/16,n},-1e6,raw[0].options().dtype(at::kFloat));
        std::tie(out,norm)=backend.forward(x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8],
            x[9],x[10],x[11],x[12],n,ctas,save?std::optional<T>(vertical):std::nullopt,fp32);
    }
    auto lse2=norm.view({b,h,n});
    py::dict state;
    if (save) {
        state["operands"]=x; state["vertical"]=vertical; state["output"]=out;
        state["lse2"]=lse2; state["normalizer"]=norm; state["n"]=n;
    }
    return py::make_tuple(out,lse2,state);
}

static py::tuple core_backward(V x,T vertical,T output,T normalizer,T dout,
                              std::optional<T> table,int64_t ctas,bool fp32,bool diagnostics) {
    at::NoGradGuard no_grad;
    precision(fp32);
    TORCH_CHECK(x.size()==13 && x[0].dim()==(table?3:4),"invalid saved operands");
    TORCH_CHECK(!table || ctas==0,"ctas is only supported for fixed execution");
    check_tensor(dout,output,at::kBFloat16,output.sizes(),"dout");
    int64_t n=output.size(1),h=output.size(2),b=output.size(0);
    TORCH_CHECK(n%256==0 && (table || n>0),"sequence length must be256-token aligned");
    const auto& backend=select(x[2].size(-1),x[0].size(-1),x[4].size(-1));
    dout=dout.contiguous();
    T delta=table?backend.vd(output,dout,*table):backend.delta(output,dout);
    T input=table?dout.squeeze(0):dout;
    V first=table?backend.vb1(x,vertical,input,normalizer,delta,*table,fp32):
        backend.b1(x,vertical,input,normalizer,delta,n,ctas,fp32);
    T boundary=table?backend.vb2(first[3],first[4],*table,h):backend.b2(first[3],first[4]);
    py::dict details;
    if (diagnostics) {
        details["delta"]=delta; details["summary_a"]=first[3]; details["summary_b"]=first[4];
        details["boundary"]=boundary;
    }
    first[3]=T(); first[4]=T();
    V last=table?backend.vb3(x,vertical,input,normalizer,delta,boundary,*table,fp32):
        backend.b3(x,vertical,input,normalizer,delta,boundary,n,ctas,fp32);
    py::dict gradients;
    gradients["q_vec"]=last[0]; gradients["k_vec"]=last[1]; gradients["rtau"]=last[4];
    gradients["v"]=first[0]; gradients["sq_vec"]=first[1]; gradients["sk_vec"]=first[2];
    // Public gradients are BNH views of native BHN storage. Interpolation
    // transposes them back without copying; do not materialize this view.
    gradients["q_lse"]=last[2].view({b,h,n}).transpose(1,2);
    gradients["k_lse"]=last[3].view({b,h,n}).transpose(1,2);
    return py::make_tuple(gradients,details);
}

static py::tuple make_layout(T cu,int64_t total) {
    TORCH_CHECK(cu.dim()==1 && cu.scalar_type()==at::kInt && cu.is_contiguous() && cu.numel()>0,
                "cu_seqlens must be contiguous int32 [sequences+1]");
    auto cpu=cu.cpu(); const int* offsets=cpu.data_ptr<int>();
    TORCH_CHECK(total>=0 && offsets[0]==0 && offsets[cu.numel()-1]==total,"invalid cu_seqlens endpoints");
    auto table=at::empty({cu.numel()-1,7},at::TensorOptions().dtype(at::kLong).device(at::kCPU));
    auto rows=table.accessor<int64_t,2>();
    int64_t f=0,v=0,r=0; std::vector<int64_t> lengths;
    for (int64_t i=0;i<cu.numel();++i) {
        TORCH_CHECK(offsets[i]>=0 && offsets[i]%256==0,"all varlen boundaries must be256-token aligned");
        if (i==cu.numel()-1) break;
        int64_t n=int64_t(offsets[i+1])-offsets[i];
        TORCH_CHECK(n>=0 && n<=INT_MAX-255,"cu_seqlens must be monotonic with supported lengths");
        rows[i][0]=rows[i][2]=offsets[i]; rows[i][1]=rows[i][3]=n;
        rows[i][4]=f; rows[i][5]=v; rows[i][6]=r;
        f+=(n?(n-1)/32:0)*n; v+=(n?(n-1)/16:0)*n; r+=(n+31)/32*n;
        lengths.push_back(n);
    }
    return py::make_tuple(table,lengths,total,total,f,v,r);
}

void bind_frontend(py::module_& m) {
    m.def("validate_interpolation",[](T q,T k,T eq,T ek,std::optional<T> tau) {
        TORCH_CHECK(q.is_cuda() && q.dim()==4 && eq.dim()==3,"invalid interpolation ranks/device");
        TORCH_CHECK(q.size(1)>0 && q.size(1)%256==0,"sequence length must be256-token aligned");
        check_tensor(q,q,at::kBFloat16,q.sizes(),"q");
        check_tensor(k,q,at::kBFloat16,q.sizes(),"k");
        check_tensor(eq,q,at::kBFloat16,{q.size(2),eq.size(1),q.size(3)},"Q vocabulary");
        check_tensor(ek,q,at::kBFloat16,eq.sizes(),"K vocabulary");
        TORCH_CHECK(eq.size(1)>0 && (q.size(3)==32 || q.size(3)==64),"invalid vocabulary/channel size");
        for (const auto& x:{q,k,eq,ek}) TORCH_CHECK(x.stride(-1)==1,"last dimension must be contiguous");
        if (tau) {
            check_tensor(*tau,q,at::kFloat,{q.size(2)},"tau");
            TORCH_CHECK(tau->is_contiguous() && !tau->requires_grad(),
                        "interpolation tau must be contiguous and detached; core owns its gradient");
        }
    });
    m.def("validate_interpolation_grads",[](T oq,T ok,T lq,T lk,T dq,T dk,T dlq,T dlk) {
        check_tensor(dq,oq,oq.scalar_type(),oq.sizes(),"dOq");
        check_tensor(dk,ok,ok.scalar_type(),ok.sizes(),"dOk");
        check_tensor(dlq,lq,at::kFloat,lq.sizes(),"dLq");
        check_tensor(dlk,lk,at::kFloat,lk.sizes(),"dLk");
    });
    m.def("summary_chunk",[](T a,T b,int64_t n){return backends[0].chunk(a,b,n);});
    m.def("core_forward",&core_forward);
    m.def("core_backward",&core_backward);
    m.def("make_varlen_layout",&make_layout);
    m.def("prepare_operands",[](V x,bool packed,bool absorbed){
        at::NoGradGuard guard;return prepare(x,packed,absorbed);
    },py::arg("operands"),py::arg("packed"),py::arg("absorbed")=false);
    m.def("summary_forward",[](T q,T k,T lq,T lk,T iq,T ik,T direction,T hard,T tau,int64_t ctas) {
        at::NoGradGuard guard;
        TORCH_CHECK(q.dim()==4 && q.is_cuda(),"q must be CUDA [B,N,H,D]");
        auto b=q.size(0),n=q.size(1),h=q.size(2);
        TORCH_CHECK(n>0 && n%256==0,"sequence length must be256-token aligned");
        const auto& backend=select(32,q.size(3),64);
        check_tensor(lq,q,at::kFloat,{b,n,h},"LSE");
        check_tensor(lk,q,at::kFloat,{b,n,h},"LSE");
        check_tensor(direction,q,at::kBool,{b,h},"direction");
        check_tensor(hard,q,at::kBool,{b,h,n},"hard");
        for (const auto& label:{iq,ik}) {
            TORCH_CHECK(label.scalar_type()==at::kInt || label.scalar_type()==at::kLong,"labels must be integer");
            check_tensor(label,q,label.scalar_type(),{b,h,n},"labels");
        }
        return backend.summary(q.contiguous(),k.contiguous(),lq.transpose(1,2).contiguous(),
            lk.transpose(1,2).contiguous(),iq.to(at::kInt).contiguous(),ik.to(at::kInt).contiguous(),
            hard.to(at::kByte).contiguous(),direction.to(at::kByte).contiguous(),tau.contiguous(),n,ctas);
    });
    m.def("prepare_embedding",[](T q,T k,T sq,T sk,T v,T eq,T ek,T tau,T direction,T hard,bool materialize) {
        // ATen autograd must stay enabled here: vocabulary casts/expansion and
        // later operand selection are part of the differentiable embedding path.
        TORCH_CHECK(q.dim()==4 && q.is_cuda(),"q must be CUDA [B,N,H,D]");
        auto b=q.size(0),n=q.size(1),h=q.size(2),d=q.size(3);
        TORCH_CHECK(n>=0 && n%256==0,"sequence length must be256-token aligned");
        TORCH_CHECK(sq.dim()==4 && v.dim()==4,"sq/v must be rank4");
        select(sq.size(3),d,v.size(3));
        check_tensor(q,q,at::kBFloat16,{b,n,h,d},"q");
        check_tensor(k,q,at::kBFloat16,q.sizes(),"k");
        check_tensor(sq,q,at::kBFloat16,{b,n,h,sq.size(3)},"sq");
        check_tensor(sk,q,at::kBFloat16,sq.sizes(),"sk");
        check_tensor(v,q,at::kBFloat16,{b,n,h,v.size(3)},"v");
        check_tensor(direction,q,at::kBool,{b,h},"direction");
        check_tensor(hard,q,at::kBool,{b,h,n},"hard");
        check_tensor(tau,q,at::kFloat,{h},"tau");
        auto table=[&](T value) {
            TORCH_CHECK(value.device()==q.device() && value.is_floating_point(),"invalid vocabulary device/dtype");
            if (value.dim()==2) value=value.unsqueeze(0).expand({h,-1,-1});
            TORCH_CHECK(value.dim()==3 && value.size(0)==h && value.size(1)>0 && value.size(2)==d,
                        "vocabulary must be [V,D] or [H,V,D]");
            return materialize?value.to(at::kBFloat16).contiguous():value;
        };
        eq=table(eq); ek=table(ek);
        TORCH_CHECK(eq.sizes()==ek.sizes(),"Q/K vocabularies must have equal shapes");
        if (!materialize) return V{};
        return V{q,k,eq,ek};
    },py::arg("q"),py::arg("k"),py::arg("sq"),py::arg("sk"),py::arg("v"),
      py::arg("eq"),py::arg("ek"),py::arg("tau"),py::arg("direction"),py::arg("hard"),
      py::arg("materialize")=true);
    m.def("select_embedding",[](T q,T k,T qfk,T kfq,T direction) {
        auto choose=direction.unsqueeze(1).unsqueeze(-1);
        return Pair{at::where(choose,q,kfq).contiguous(),
                    at::where(choose,qfk,k).contiguous()};
    });
}
