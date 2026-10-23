"""Separate uninstrumented decode latency and CPU/CUDA component profiles."""
import argparse
from contextlib import contextmanager
import json
import statistics
import sys
import time
from pathlib import Path
import torch
from torch.profiler import profile, record_function, ProfilerActivity


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--length',type=int,default=8191)
    p.add_argument('--intervals',type=int,nargs='+',default=[128,512])
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    torch.set_num_threads(1); torch.backends.cuda.matmul.allow_tf32=False
    sys.path.insert(0,'/home/cicuvc/downstream-eval')
    from adapter_layout_bounded import NanochatLM
    from flash_dism.inference import HardDismPrefill
    from flash_dism.inference.gpu_runtime import GpuPlannerCache
    from nanochat.models.dism.inference import SamDecodeSession
    length=args.length
    context=((length+max(args.intervals)+16+255)//256)*256
    lm=NanochatLM(args.checkpoint,'/home/cicuvc/downstream-eval/tokenizer',batch_size=1,context=context)
    # Real contiguous PG19 text. No repetitive synthetic prefix or downloads.
    import pyarrow.parquet as pq
    book=pq.read_table('/home/cicuvc/downstream-eval/data/pg19/test.parquet',columns=['text']).column('text')[10].as_py()
    ids=lm.tok_encode(book)[:length+max(args.intervals)+16]
    assert len(ids)==length+max(args.intervals)+16
    tokens=torch.tensor([ids],device='cuda')
    result=dict(checkpoint=args.checkpoint,prompt_length=length,source='PG19 test book 10',intervals=[])
    original=GpuPlannerCache.rebuild
    def rebuild(cache):
        with record_function('decode/rebuild'):
            return original(cache)
    def summarize(path):
        events=json.loads(path.read_text())['traceEvents']
        scopes=[e for e in events if e.get('cat')=='user_annotation' and e['name'].startswith('decode/')]
        cpu={e['args']['External id']: e for e in events if e.get('cat') in ('cpu_op','user_annotation') and 'External id' in e.get('args',{})}
        launches={e['args']['correlation']:e for e in events if e.get('cat') in ('cuda_runtime','cuda_driver') and 'correlation' in e.get('args',{})}
        out={}
        for e in scopes:
            row=out.setdefault(e['name'],dict(calls=0,cpu_ms=0.,gpu_ms=0.,kernels=0,copy_ms=0.,copies=0))
            row['calls']+=1; row['cpu_ms']+=e['dur']/1000
        kernels=[e for e in events if e.get('cat')=='kernel']
        for e in events:
            if e.get('cat') not in ('kernel','gpu_memcpy','gpu_memset'): continue
            launch=launches.get(e.get('args',{}).get('correlation')) or cpu.get(e.get('args',{}).get('External id'))
            owners=[] if launch is None else [s for s in scopes if s['tid']==launch['tid'] and s['ts']<=launch['ts']<s['ts']+s['dur']]
            name=min(owners,key=lambda x:x['dur'])['name'] if owners else 'unattributed'
            row=out.setdefault(name,dict(calls=0,cpu_ms=0.,gpu_ms=0.,kernels=0,copy_ms=0.,copies=0))
            if e['cat']=='kernel': row['gpu_ms']+=e['dur']/1000; row['kernels']+=1
            else: row['copy_ms']+=e['dur']/1000; row['copies']+=1
        return dict(components=out,kernel_count=len(kernels),gpu_kernel_ms=sum(e['dur'] for e in kernels)/1000)
    with torch.inference_mode(), HardDismPrefill(workers=16) as engine:
        for interval in args.intervals:
            session=SamDecodeSession(lm.model,engine,context,rebuild_interval=interval)
            session.features(tokens[:,:length]); torch.cuda.synchronize()
            times=[]
            for step in range(1,interval+4):
                torch.cuda.synchronize(); start=time.perf_counter()
                hidden=session.features(tokens[:,length+step-1:length+step])
                torch.cuda.synchronize()
                times.append(dict(step=step,ms=(time.perf_counter()-start)*1000,rebuild=step==interval+1))
            for state in session.states:
                if state['dism'] is not None: assert state['dism'].core.check_status()==session.position
            assert torch.isfinite(hidden).all()
            ordinary=[r['ms'] for r in times if r['step']>2 and not r['rebuild']]
            row=dict(interval=interval,ordinary_median_ms=statistics.median(ordinary),
                ordinary_first_quarter_ms=statistics.median(ordinary[:interval//4]),
                ordinary_last_quarter_ms=statistics.median(ordinary[-interval//4:]),
                rebuild_ms=times[interval]['ms'],amortized_ms=statistics.mean(r['ms'] for r in times[2:]),steps=times,
                allocated_gib=torch.cuda.memory_allocated()/2**30)
            # Lightweight host attribution, without CUPTI/profiler overhead.
            host_samples=[]
            for repeat in range(8):
                host={}
                @contextmanager
                def timed(name):
                    start=time.perf_counter()
                    try: yield
                    finally: host[name]=host.get(name,0.)+(time.perf_counter()-start)*1000
                session._scope=timed
                index=length+interval+3+repeat
                torch.cuda.synchronize(); start=time.perf_counter()
                hidden=session.features(tokens[:,index:index+1])
                with timed('decode/lm_head_greedy'): lm._greedy(hidden[:,-1]).tolist()
                torch.cuda.synchronize()
                host['total_wall_ms']=(time.perf_counter()-start)*1000
                host_samples.append(host)
            row['host_components_ms']={k:statistics.median(r[k] for r in host_samples[2:]) for k in host_samples[0]}
            session.close()
            print(json.dumps({k:v for k,v in row.items() if k!='steps'}),flush=True)
            # Fresh identical session, profile a late ordinary step then rebuild.
            session=SamDecodeSession(lm.model,engine,context,rebuild_interval=interval)
            session.features(tokens[:,:length])
            for step in range(1,interval):
                session.features(tokens[:,length+step-1:length+step])
            torch.cuda.synchronize()
            session._scope=record_function
            GpuPlannerCache.rebuild=rebuild
            try:
                for label,step in [('ordinary',interval),('rebuild',interval+1)]:
                    with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA]) as prof:
                        hidden=session.features(tokens[:,length+step-1:length+step])
                        with record_function('decode/lm_head_greedy'):
                            lm._greedy(hidden[:,-1]).tolist()
                        torch.cuda.synchronize()
                    trace=args.output.with_suffix(f'.r{interval}.{label}.trace.json')
                    prof.export_chrome_trace(str(trace))
                    row[label+'_profile']=summarize(trace)
            finally:
                GpuPlannerCache.rebuild=original
                session.close()
            result['intervals'].append(row)
            args.output.write_text(json.dumps(result,indent=2)+'\n')
    lm.close()

if __name__=='__main__': main()
