"""Cross-thread CUDA-launch attribution for the bounded model-training profiles.

NVTX's built-in same-thread projection misses autograd launches. Attribute GPU
activities by their correlated CPU launch timestamp within the outer blocking
forward/backward calls, regardless of launch thread. Never sum nested scopes.
"""
import argparse
import collections
import json
from pathlib import Path
import sqlite3


def union(intervals):
    result=[]
    for a,b in sorted(intervals):
        if result and a<=result[-1][1]:
            result[-1][1]=max(result[-1][1],b)
        else:
            result.append([a,b])
    return result


def category(name,shapes):
    if name.startswith(('Memcpy','Memset')):
        return 'transfers_and_memset'
    if name in ('fwd_short_conv_silu','bwd_short_conv_silu','bwd_mma_gemm'):
        return 'shortconv_projection'
    if 'causal_conv1d_' in name:
        return 'FLA_shortconv_without_linear'
    if name in ('_linear_cross_entropy_fwd','_cross_entropy_dlogits','_dhidden_gemm',
                '_dweight_gemm_accumulate','_mean_loss_reduce',
                '_chunk_partial_stats','_chunk_row_stats','_chunk_dlogits',
                '_accumulate_fp32','_valid_token_count','_token_scale',
                '_document_valid_counts','_nonzero_count','_document_scale',
                '_token_nll_sum','_document_nll_mean','_document_mean_sum',
                '_finalize_loss'):
        return 'linear_cross_entropy'
    if 'dism_r' in name:
        if any(s in name for s in ('hard_backward_kernel','hard_dlog_weights_kernel',
                                   'hard_tau_reduce_kernel')):
            return 'hard_cuda_backward'
        return 'hard_cuda_forward' if 'hard_' in name else 'soft_cuda_core'
    if name in ('_chunk_backward','_chunk_summaries','_summary_scan'):
        return 'hard_event_backward'
    if name in ('emb_fwd','_dq','_dkv','_interp_bwd_preprocess','_select','_merge',
                '_split','_prepare_vocab','_vocab_cast'):
        return 'vocabulary_interpolation_and_glue'
    if name.startswith(('chunk_','recompute_w_u','prepare_wy','merge_16x16','l2norm_')):
        return 'GDN_recurrence_and_preparation'
    if any(s in name for s in ('cutlass','gemm','splitKreduce')):
        # The measured 72M model's output vocabulary has 32000 entries.
        if any(32000 in dim for dim in shapes if isinstance(dim,list)):
            return 'linear_cross_entropy'
        # 2576 is the real 72M model's FFN intermediate width.
        if any(2576 in dim for dim in shapes if isinstance(dim,list)):
            return 'FFN_GEMMs'
        return 'other_GEMMs_or_shape_unavailable'
    if any(2576 in dim for dim in shapes if isinstance(dim,list)):
        return 'FFN_pointwise_and_copies'
    return 'other_pointwise_norm_reductions_optimizer'


def read_torch(path):
    events=json.loads(path.read_text())['traceEvents']
    scopes=[e for e in events if e.get('cat')=='user_annotation' and
            (e['name'].startswith(('microbatch_','step/')) or e['name']=='training_step')]
    phases=[dict(name=e['name'],start=e['ts'],end=e['ts']+e['dur']) for e in scopes if e['name']!='training_step']
    whole=next(e for e in scopes if e['name']=='training_step')
    cpu={e['args']['External id']:e for e in events if e.get('cat')=='cpu_op' and 'External id' in e.get('args',{})}
    api={e['args']['correlation']:e for e in events if e.get('cat') in ('cuda_runtime','cuda_driver') and 'correlation' in e.get('args',{})}
    gpu=[]
    for e in events:
        if e.get('cat') not in ('kernel','gpu_memcpy','gpu_memset'):
            continue
        launch=api.get(e['args'].get('correlation'))
        op=cpu.get(e['args'].get('External id'),{})
        gpu.append(dict(name=e['name'],start=e['ts'],end=e['ts']+e['dur'],
                        launch=launch['ts'] if launch else None,stream=e['args'].get('stream'),
                        kernel=e['cat']=='kernel',cpu_op=op.get('name'),
                        shapes=op.get('args',{}).get('Input Dims',[])))
    return phases,(whole['ts'],whole['ts']+whole['dur']),gpu


def read_nsys(path):
    c=sqlite3.connect(f'file:{path.resolve()}?mode=ro',uri=True)
    strings=dict(c.execute('select id,value from StringIds'))
    scopes=[(a/1000,b/1000,text or strings.get(textid,'')) for a,b,text,textid in
            c.execute('select start,end,text,textId from NVTX_EVENTS where end is not null')]
    phases=[dict(name=n,start=a,end=b) for a,b,n in scopes if n.startswith(('microbatch_','step/'))]
    whole=next((a,b) for a,b,n in scopes if n=='training_step')
    api={corr:start/1000 for start,corr in c.execute('select start,correlationId from CUPTI_ACTIVITY_KIND_RUNTIME')}
    gpu=[]
    for a,b,name,corr,stream in c.execute('select start,end,demangledName,correlationId,streamId from CUPTI_ACTIVITY_KIND_KERNEL'):
        gpu.append(dict(name=strings[name],start=a/1000,end=b/1000,launch=api.get(corr),stream=stream,kernel=True,shapes=[]))
    for a,b,kind,corr,stream in c.execute('select start,end,copyKind,correlationId,streamId from CUPTI_ACTIVITY_KIND_MEMCPY'):
        gpu.append(dict(name=f'Memcpy kind {kind}',start=a/1000,end=b/1000,launch=api.get(corr),stream=stream,kernel=False,shapes=[]))
    for a,b,corr,stream in c.execute('select start,end,correlationId,streamId from CUPTI_ACTIVITY_KIND_MEMSET'):
        gpu.append(dict(name='Memset',start=a/1000,end=b/1000,launch=api.get(corr),stream=stream,kernel=False,shapes=[]))
    return phases,whole,gpu


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('input',type=Path)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    if args.output.exists():
        ap.error('choose a fresh output')
    phases,whole,gpu=(read_nsys if args.input.suffix=='.sqlite' else read_torch)(args.input)
    overall=collections.defaultdict(float)
    phase_groups=collections.defaultdict(lambda:collections.defaultdict(float))
    phase_kernels=collections.defaultdict(list)
    names=collections.defaultdict(lambda:[0,0.])
    for e in gpu:
        group=category(e['name'],e['shapes'])
        duration=(e['end']-e['start'])/1000
        overall[group]+=duration
        name=next((p['name'] for p in phases if e['launch'] is not None and p['start']<=e['launch']<=p['end']),'outside_phases_or_uncorrelated')
        phase_groups[name][group]+=duration
        if e['kernel']:
            phase_kernels[name].append((e['start'],e['end']))
        names[e['name']][0]+=1
        names[e['name']][1]+=duration
    covered=union([(max(whole[0],e['start']),min(whole[1],e['end'])) for e in gpu
                   if e['end']>whole[0] and e['start']<whole[1]])
    gaps=[]
    cursor=whole[0]
    for a,b in covered:
        if a>cursor:gaps.append((cursor,a))
        cursor=b
    if cursor<whole[1]:gaps.append((cursor,whole[1]))
    phase_report=[]
    for p in sorted(phases,key=lambda p:p['start']):
        spans=phase_kernels[p['name']]
        phase_report.append(dict(name=p['name'],cpu_scope_ms=(p['end']-p['start'])/1000,
            kernel_sum_ms=sum(b-a for a,b in spans)/1000,
            kernel_span_ms=(max(b for a,b in spans)-min(a for a,b in spans))/1000 if spans else 0.,
            activity_groups_ms=dict(phase_groups[p['name']])))
    report=dict(source=str(args.input),method='kernel/copy duration sums, not additive critical-path budgets; '
                'launch timestamps map across threads into outer CPU phase ranges; GPU span includes internal gaps; '
                'all-engine idle uses interval union of kernels+copies+memsets, not SM occupancy',
                step_cpu_ms=(whole[1]-whole[0])/1000,
                gpu_activity_union_ms=sum(b-a for a,b in covered)/1000,
                gpu_no_activity_ms=sum(b-a for a,b in gaps)/1000,
                activity_groups_ms=dict(overall),phases=phase_report,
                outside_phases_ms=dict(phase_groups['outside_phases_or_uncorrelated']),
                largest_idle_gaps=[dict(start_from_step_ms=(a-whole[0])/1000,duration_ms=(b-a)/1000)
                                   for a,b in sorted(gaps,key=lambda x:x[0]-x[1])[:20]],
                kernels=sorted([dict(name=n,count=v[0],ms=v[1]) for n,v in names.items()],key=lambda x:-x['ms']))
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('kernels','largest_idle_gaps')},indent=2))


if __name__=='__main__':
    main()
