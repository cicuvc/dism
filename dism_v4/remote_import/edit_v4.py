from pathlib import Path
R=Path('/root/dism125m-v4/dism_v3')
def edit(n,a,b,count=1):
 p=R/n;s=p.read_text();assert s.count(a)==count,(n,a,s.count(a));p.write_text(s.replace(a,b))
# Device argument additions preserve existing aggregate initialization.
edit('include/summary/primitives.cuh','float *SummaryB; // log2 affine second component; retained after passing','float *SummaryB; // log2 affine second component; retained after passing\n    const float *gate_delta = nullptr; // natural-log [B,H,N] attenuation')
edit('include/summary/primitives.cuh','    bool direction;\n    float tau2;', '    float row_gate[CONFIG.getQBlockSize()]; // task lifetime; never aliases the K ring\n    bool direction;\n    float tau2;')
edit('include/forward/types.cuh','    bool query_lse;','    float row_gate[QROWS]; // task lifetime; never aliases metadata.kv\n    bool query_lse;')
edit('include/forward/types.cuh','    OutputGlobal<FP32Output> output_map;','    OutputGlobal<FP32Output> output_map;\n    const float *gate_delta = nullptr;')
# Shared loads use natural -> log2 conversion once per producer row. Sync warp
# before packet publication (which supplies producer/consumer memory ordering).
helper='''// A producer warp stages one query-row attenuation vector, including optional
// halo. The caller publishes its normal pipe packet only after this completes.
template<int Count>
__device__ __forceinline__ void load_row_gate(float (&dst)[Count], const float* src,
                                             int valid) {
    for (int i=kt::warp::laneid(); i<Count; i+=32)
        dst[i]=(src && i<valid) ? src[i]*1.4426950408889634f : 0.f;
    __syncwarp();
}

'''
edit('include/summary/primitives.cuh','struct LogAffineOp {',helper+'struct LogAffineOp {')
edit('include/summary/primitives.cuh','auto make_scan(const kt::rt_fl<ROWS, COLS> &buffer)', 'auto make_scan(const kt::rt_fl<ROWS, COLS> &buffer, const float* row_gate = nullptr)')
edit('include/summary/primitives.cuh','            res.data[i][j].first = res.data[i][j].second = single.data[i][j].value;', '''            res.data[i][j].first = res.data[i][j].second = single.data[i][j].value;
            if (row_gate) {
                int row=decltype(res)::physical_row(i,j,kt::warp::laneid()/4);
                res.data[i][j].first = res.data[i][j].first - pscore::F32x2{row_gate[row]};
            }''')
edit('include/forward/scan.cuh','make_forward_scan(const kt::rt_fl<16,WarpKSize>& score)', 'make_forward_scan(const kt::rt_fl<16,WarpKSize>& score, const float* row_gate = nullptr)')
edit('include/forward/scan.cuh','''        for (int c = 0; c < Scalar::COL_BLOCKS; ++c)
            scan.data[r][c].first = scan.data[r][c].second = scalar.data[r][c].value;''','''        for (int c = 0; c < Scalar::COL_BLOCKS; ++c) {
            scan.data[r][c].first = scan.data[r][c].second = scalar.data[r][c].value;
            if (row_gate) {
                int row=Scan::physical_row(r,c,kt::warp::laneid()/4);
                scan.data[r][c].first = scan.data[r][c].first - pscore::F32x2{row_gate[row]};
            }
        }''')
for name,offset in [('include/summary/kernel_common.cuh','(int64_t(task.Batch)*args.Head+task.Head)*args.PaddedSeqlen+task.QStart'),('include/varlen/summary.cuh','task.metadata+task.QStart')]:
 edit(name,'            kt::warp::load_async_commit_group(prefetch.getBarrier());','            load_row_gate(shared.row_gate,args.gate_delta ? args.gate_delta+'+offset+' : nullptr,CONFIG.getQBlockSize());\n            kt::warp::load_async_commit_group(prefetch.getBarrier());')
for name,offset in [('include/forward/producer.cuh','(int64_t(task.batch)*args.heads+task.head)*args.padded+task.q_start'),('include/varlen/forward.cuh','task.metadata+task.q_start')]:
 edit(name,'        kt::warp::load_async_commit_group(first.getBarrier());','        load_row_gate(shared.row_gate,args.gate_delta ? args.gate_delta+'+offset+' : nullptr,QROWS);\n        kt::warp::load_async_commit_group(first.getBarrier());')
for name in ('src/wmma_tma_preprocess.cu','src/varlen_summary.cu'):
 p=R/name;s=p.read_text();a=s.index('                // Every reader publishes');b=s.index('                auto result = scan.reduce_forward',a)
 old=s[a:b]
 new='''                finish_score(score, query_labels, key_labels, hard_rows, tau2);
                auto scan = make_scan(score, args.gate_delta ?
                    shared.row_gate+query_block*CONFIG.WarpQSize : nullptr);
                // All delta reads precede release; the producer may now reuse
                // the task-level shared row buffer after the final K slot.
                packet.submitToNextAndTrigger();
                load_pipe.moveNext();

'''
 s=s[:a]+new+s[b:];p.write_text(s)
for name in ('src/forward.cu','src/varlen_forward.cu'):
 edit(name,'auto scan = make_forward_scan(score);','auto scan = make_forward_scan(score, args.gate_delta ? shared.row_gate+16*query_block : nullptr);')
# Recompute uses transposed [key,query] layout: roll preserves query columns.
edit('include/backward/recompute.cuh','    const uint8_t *hard, *direction;','    const uint8_t *hard, *direction;\n    const float *gate_delta = nullptr;')
edit('include/backward/recompute.cuh','template<class ScoreArgs>\n__device__ __forceinline__ Scalar recompute(', 'template<bool GateGradient=false, class ScoreArgs>\n__device__ __forceinline__ Scalar recompute(')
edit('include/backward/recompute.cuh','const QueryTile& query_shared, int bh, int k0, int q0)', 'const QueryTile& query_shared, int bh, int k0, int q0,\n        const float* row_gate=nullptr, Scalar* gate_alpha=nullptr)')
edit('include/backward/recompute.cuh','''        for (int c = 0; c < 4; ++c)
            scan.data[r][c] = {values.data[r][c].value,values.data[r][c].value};''','''        for (int c = 0; c < 4; ++c) {
            scan.data[r][c] = {values.data[r][c].value,values.data[r][c].value};
            if (row_gate) {
                auto pos=Scalar::layout(r,c,0);
                scan.data[r][c].first.u0 -= row_gate[pos.second];
                scan.data[r][c].first.u1 -= row_gate[pos.second+4];
            }
        }''')
edit('include/backward/recompute.cuh','''        for (int c = 0; c < 4; ++c) values.data[r][c].value = scan.data[r][c].second;''','''        for (int c = 0; c < 4; ++c) {
            if constexpr (GateGradient) {
                // 1-exp(m-W) is the local incoming recurrence derivative.
                // Masks and the first row/column must never inherit history.
                auto pos=Scalar::layout(r,c,0);
                int key=k0+Forward::physical_row(r,c,kt::warp::laneid()/4);
                for (int e=0;e<2;++e) {
                    int query=q0+pos.second+4*e;
                    float m=e?values.data[r][c].value.u1:values.data[r][c].value.u0;
                    float w=e?scan.data[r][c].second.u1:scan.data[r][c].second.u0;
                    float a=0.f;
                    if (gate_alpha && row_gate && key>0 && query>0 && key<=query &&
                        key<args.n && query<args.n && m>LOG_ZERO*.5f &&
                        isfinite(row_gate[pos.second+4*e]))
                        a=-expm1f(fminf(0.f,(m-w)*0.6931471805599453f));
                    if (gate_alpha) {
                        if(e) gate_alpha->data[r][c].value.u1=a;
                        else gate_alpha->data[r][c].value.u0=a;
                    }
                }
            }
            values.data[r][c].value = scan.data[r][c].second;
        }''')
edit('include/backward/recompute.cuh','    values.roll<false>();\n    return values;', '    values.roll<false>();\n    if constexpr (GateGradient) { if(gate_alpha) gate_alpha->roll<false>(); }\n    return values;')
edit('include/backward/recompute.cuh','operands.size()==13','(operands.size()==13 || operands.size()==14)')
edit('include/backward/recompute.cuh','    return {batch,heads,n,padded,','    if(operands.size()==14) check(operands[13],at::kFloat,{batch,heads,padded});\n    return {batch,heads,n,padded,')
edit('include/backward/recompute.cuh','operands[9].data_ptr<uint8_t>(),operands[10].data_ptr<uint8_t>()};','operands[9].data_ptr<uint8_t>(),operands[10].data_ptr<uint8_t>(),\n        operands.size()==14 ? operands[13].data_ptr<float>() : nullptr};')
# Backward slots carry current rows and one successor halo, independent of TMA bytes.
edit('include/backward/types.cuh','    OutputMap lse_output;','    OutputMap lse_output;\n    float *dgate = nullptr;')
edit('include/backward/types.cuh','        DerivativeTile dout;','        DerivativeTile dout;\n        float row_gate[Q+1];')
for name in ('include/backward/types.cuh','include/varlen/backward.cuh'):
 edit(name,'sizeof(Shared::Input)', 'sizeof(QueryTile)+sizeof(SoftQueryTile)+sizeof(DerivativeTile)')
 offset='int64_t(task.bh)*a.score.padded+q0' if name.endswith('types.cuh') else 'task.metadata+q0'
 valid='min(Q+1,a.score.n-q0)' if name.endswith('types.cuh') else 'min(Q+1,task.n-q0)'
 needle='            packet.submitToNextAndTrigger();' if name.endswith('types.cuh') else '        packet.submitToNextAndTrigger();'
 indent=needle[:len(needle)-len(needle.lstrip())]
 edit(name,needle,indent+'load_row_gate(packet.template get<0>().row_gate,\n'+indent+'    a.score.gate_delta ? a.score.gate_delta+'+offset+' : nullptr,'+valid+');\n'+needle)
# Delta pointer is only a presence flag in local_score; row data comes from the slot.
edit('include/varlen/backward.cuh','    const uint8_t *hard,*direction;','    const uint8_t *hard,*direction;\n    const float *gate_delta=nullptr;')
edit('include/varlen/backward.cuh','a.score.hard+start,a.score.direction};','a.score.hard+start,a.score.direction,a.score.gate_delta};')
for stem in ('backward_summary','backward_qk','varlen_backward_summary','varlen_backward_qk'):
 name='src/'+stem+'.cu';p=R/name;s=p.read_text();scoreargs='score_args' if stem.startswith('varlen') else 'args.score'
 old='auto score=recompute('+scoreargs+',keys,data.q,task.bh,k0,q0);'
 if stem.endswith('qk'):
  new='Scalar gate_alpha;\n                auto score=recompute<true>('+scoreargs+',keys,data.q,task.bh,k0,q0,\n                    '+scoreargs+'.gate_delta ? data.row_gate : nullptr, args.dgate ? &gate_alpha : nullptr);'
 else:new='auto score=recompute('+scoreargs+',keys,data.q,task.bh,k0,q0,\n                    '+scoreargs+'.gate_delta ? data.row_gate : nullptr);'
 assert old in s,(name,old);s=s.replace(old,new)
 s=s.replace('alpha[e]=valid?sigmoid2(w):1.f;', 'alpha[e]=valid ? ((query+1<'+scoreargs+'.n) ?\n                                sigmoid2(w-('+scoreargs+'.gate_delta ? data.row_gate[pos.second+4*e+1] : 0.f)) : 0.f) : 1.f;')
 if stem.endswith('qk'):
  s=s.replace('float row_lse[4][2]{};','float row_lse[4][2]{};\n                float gate_gradient[4][2]{};')
  s=s.replace('                            tau_sum+=g;', '''                            tau_sum+=g;
                            if(args.dgate) gate_gradient[c][e]-=g*(e ? gate_alpha.data[r][c].value.u1 : gate_alpha.data[r][c].value.u0);''')
  off='task.metadata+q0+c+(2*(lane&3)+e)*4' if stem.startswith('varlen') else 'int64_t(task.bh)*args.score.n+q0+c+(2*(lane&3)+e)*4'
  needle='                if (query_lse) {'
  assert s.count(needle)==1
  s=s.replace(needle,'''                if(args.dgate) {
                    for(int c=0;c<4;++c) for(int e=0;e<2;++e) {
                        float sum=gate_gradient[c][e];
                        sum+=__shfl_xor_sync(0xffffffff,sum,4);
                        sum+=__shfl_xor_sync(0xffffffff,sum,8);
                        sum+=__shfl_xor_sync(0xffffffff,sum,16);
                        if(lane/4==0) atomicAdd(args.dgate+'''+off+''',sum);
                    }
                }
'''+needle)
 p.write_text(s)
print('Device changes applied')
