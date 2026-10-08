from pathlib import Path
R=Path('/root/dism125m-v4/dism_v3')
def edit(n,a,b,count=1):
 p=R/n;s=p.read_text();assert s.count(a)==count,(n,a,s.count(a));p.write_text(s.replace(a,b))
for n in ('include/summary/primitives.cuh','include/forward/scan.cuh'):
 p=R/n;s=p.read_text().replace(' - pscore::F32x2{row_gate[row]}',' + pscore::F32x2{-row_gate[row]}');p.write_text(s)
for n in ('src/host/wmma_tma_preprocess.cpp','include/summary/launch.cuh'):
 edit(n,'int64_t requested_ctas) {','int64_t requested_ctas, std::optional<at::Tensor> gate_delta = std::nullopt) {')
edit('src/host/wmma_tma_preprocess.cpp','tau, seqlen, requested_ctas);','tau, seqlen, requested_ctas, gate_delta);')
edit('include/summary/launch.cuh','    // Omit only the32-row','    if(gate_delta) check(*gate_delta,at::kFloat,{batch,heads,padded},"gate_delta");\n    // Omit only the32-row')
edit('include/summary/launch.cuh','    // Two pipe allocations','    args.gate_delta=gate_delta ? gate_delta->data_ptr<float>() : nullptr;\n    // Two pipe allocations')
edit('src/host/forward.cpp','std::optional<at::Tensor> vertical, bool fp32_output)', 'std::optional<at::Tensor> vertical, bool fp32_output, std::optional<at::Tensor> gate_delta)')
edit('src/host/forward.cpp','        int checkpoints =','        if(gate_delta) check(*gate_delta,at::kFloat,{batch,heads,padded},"gate_delta");\n        int checkpoints =')
edit('src/host/forward.cpp','        int tasks =','        args.gate_delta=gate_delta ? gate_delta->data_ptr<float>() : nullptr;\n        int tasks =')
edit('src/interface.cpp','                              int64_t);','                              int64_t, std::optional<at::Tensor>);')
edit('src/interface.cpp','    std::optional<at::Tensor>, bool);','    std::optional<at::Tensor>, bool, std::optional<at::Tensor>);')
edit('src/interface.cpp','    m.def("summarization", &summarization_cuda);','''    m.def("summarization", &summarization_cuda,
        pybind11::arg("q"),pybind11::arg("k"),pybind11::arg("lq"),pybind11::arg("lk"),
        pybind11::arg("iq"),pybind11::arg("ik"),pybind11::arg("hard"),pybind11::arg("direction"),
        pybind11::arg("tau"),pybind11::arg("n"),pybind11::arg("ctas"),pybind11::arg("gate_delta")=pybind11::none());''')
edit('src/interface.cpp','          pybind11::arg("fp32_output") = false);','          pybind11::arg("fp32_output") = false, pybind11::arg("gate_delta") = pybind11::none());',2) if False else None
# Only forward_output binding (other probe functions retain their signatures).
p=R/'src/interface.cpp';s=p.read_text();s=s.replace('pybind11::arg("fp32_output") = false);','pybind11::arg("fp32_output") = false, pybind11::arg("gate_delta") = pybind11::none());',1);p.write_text(s)
# Appended optional gate lives at staged index13; index12 stays the checkpoint.
edit('src/frontend.cpp','Pair summarization_cuda(T,T,T,T,T,T,T,T,T,int64_t,int64_t);','Pair summarization_cuda(T,T,T,T,T,T,T,T,T,int64_t,int64_t,std::optional<T>);')
edit('src/frontend.cpp','Pair (*summary)(T,T,T,T,T,T,T,T,T,int64_t,int64_t);','Pair (*summary)(T,T,T,T,T,T,T,T,T,int64_t,int64_t,std::optional<T>);')
edit('src/frontend.cpp','int64_t,int64_t,std::optional<T>,bool);','int64_t,int64_t,std::optional<T>,bool,std::optional<T>);',2)
edit('src/frontend.cpp','TORCH_CHECK(x.size()==12,"expected twelve core operands");','TORCH_CHECK(x.size()==12 || x.size()==13,"expected twelve core operands plus optional gate_delta");')
edit('src/frontend.cpp','    return result;\n}', '''    if(x.size()==13) {
        check_tensor(x[12],q,at::kFloat,{b,h,n},"gate_delta");
        result.push_back(T()); // horizontal checkpoint is filled after summary
        result.push_back(x[12].contiguous().view(packed ? std::vector<int64_t>{b*h*n} : std::vector<int64_t>{b,h,n}));
    }
    return result;
}''')
edit('src/frontend.cpp','x[11],n,ctas);','x[11],n,ctas,x.size()==14?std::optional<T>(x[13]):std::nullopt);')
edit('src/frontend.cpp','    x.push_back(boundary);','    if(x.size()==14) x[12]=boundary; else x.push_back(boundary);')
edit('src/frontend.cpp','save?std::optional<T>(vertical):std::nullopt,fp32);','save?std::optional<T>(vertical):std::nullopt,fp32,x.size()==14?std::optional<T>(x[13]):std::nullopt);')
edit('src/frontend.cpp','x.size()==13 && x[0].dim()','(x.size()==13 || x.size()==14) && x[0].dim()')
edit('src/frontend.cpp','    gradients["q_vec"]=last[0];','    if(x.size()==14) gradients["gate_delta"]=last[5].view({b,h,n});\n    gradients["q_vec"]=last[0];')
edit('src/frontend.cpp','tau.contiguous(),n,ctas);','tau.contiguous(),n,ctas,std::nullopt);')
# Packed validation/host dispatch.
edit('include/varlen/operands.cuh','    return h;','    if(operands.size()==14) check(13,at::kFloat,{layout.padded*h});\n    return h;')
edit('src/host/varlen_summary.cpp','    std::vector<int2> tasks;','    args.gate_delta=x.size()==14 ? x[13].data_ptr<float>() : nullptr;\n    std::vector<int2> tasks;')
edit('src/host/varlen_forward.cpp','operands.size()==13 &&','(operands.size()==13 || operands.size()==14) &&')
edit('src/host/varlen_forward.cpp','        std::vector<int2> tasks;','        args.gate_delta=x.size()==14 ? x[13].data_ptr<float>() : nullptr;\n        std::vector<int2> tasks;')
edit('include/varlen/backward.cuh','x.size()==13,','(x.size()==13 || x.size()==14),')
edit('include/varlen/backward.cuh','x[9].data_ptr<uint8_t>(),x[10].data_ptr<uint8_t>()};','x[9].data_ptr<uint8_t>(),x[10].data_ptr<uint8_t>(),x.size()==14 ? x[13].data_ptr<float>() : nullptr};')
# QK produces an additional FP32 [B,H,N] gradient only when requested.
edit('src/host/backward_qk.cpp','    args.dq=dq.data_ptr<float>();','    auto dgate=operands.size()==14 ? at::zeros({b,h,n},options) : at::Tensor();\n    args.dgate=dgate.defined()?dgate.data_ptr<float>():nullptr;\n    args.dq=dq.data_ptr<float>();')
edit('src/host/backward_qk.cpp','    return {dq,dk,dlq,dlk,dtau};','    if(dgate.defined()) return {dq,dk,dlq,dlk,dtau,dgate};\n    return {dq,dk,dlq,dlk,dtau};')
edit('src/host/varlen_backward_qk.cpp','    if (!layout.tokens) return {dq,dk,dlq,dlk,dtau};','    auto dgate=operands.size()==14 ? at::zeros({layout.padded*h},options) : at::Tensor();\n    if (!layout.tokens) {\n        if(dgate.defined()) return {dq,dk,dlq,dlk,dtau,dgate};\n        return {dq,dk,dlq,dlk,dtau};\n    }')
edit('src/host/varlen_backward_qk.cpp','    auto args=input.common();','    auto args=input.common();\n    args.dgate=dgate.defined()?dgate.data_ptr<float>():nullptr;')
edit('src/host/varlen_backward_qk.cpp','    return {dq,dk,dlq,dlk,dtau};\n}', '    if(dgate.defined()) return {dq,dk,dlq,dlk,dtau,dgate};\n    return {dq,dk,dlq,dlk,dtau};\n}')
print('Host changes applied')
