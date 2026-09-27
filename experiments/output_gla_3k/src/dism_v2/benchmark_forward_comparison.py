"""Three forward-stage GPU durations in ordinary streams, excluding embedding."""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys


STAGES = ('summary', 'passing', 'output')


def worker(variant):
    import torch
    import triton
    torch.manual_seed(0)
    with torch.no_grad():
        q,k,v = [torch.randn(64,4,1024,64,device='cuda',dtype=torch.bfloat16)
                 for _ in range(3)]
        tau = torch.full((4,),3.,device='cuda')
        if variant == 'triton':
            import tt_dism as old
            torch.set_default_device('cuda')
            beta = torch.randn(64,4,1024,device='cuda')
            q,k = [torch.softmax(x,-1).bfloat16().contiguous() for x in (q,k)]
            def run(): return old.parallel_attn_fwd(q,k,v,beta,tau)
            names = dict(zip(('perprocess_kernel_hh','chunk_passing_kernel',
                              'attn_fwd_kernel_hh'),STAGES))
            def stage(name): return names.get(name)
        else:
            from . import core
            from .embedding import forward as embedding
            qv,kv = [torch.randn(4,512,64,device='cuda',dtype=torch.bfloat16)
                     for _ in range(2)]
            emb = embedding(q,k,qv,kv,1.)
            direction = 'q_from_k' if variant.endswith('_q') else 'k_from_q'
            probability = 0. if 'soft' in variant else .5
            a,b,lse = (q,emb[0],emb[3]) if direction=='q_from_k' else (emb[1],k,emb[2])
            state = None
            torch.cuda.manual_seed(777)
            def run():
                nonlocal state
                result = core.forward(a,b,v,lse,tau,emb[7],emb[6],sm_scale=1.,
                    direction=direction,hard_prob=probability,rng_state=state,
                    return_rng_state=True,save_boundaries=True)
                state = result[-1]
                return result
            def stage(name):
                if 'summary_persistent<' in name: return 'summary'
                if 'dism_v2::passing(' in name: return 'passing'
                if 'void dism_v2::core<' in name: return 'output'
        for _ in range(20): result = run()
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA]) as prof:
            for _ in range(30): result = run()
            torch.cuda.synchronize()
        samples = {name:[] for name in STAGES}
        sequence = []
        for event in sorted(prof.events(),key=lambda e:e.time_range.start):
            if event.device_type != torch.autograd.DeviceType.CUDA: continue
            name = stage(event.name)
            if name is not None:
                samples[name].append(event.device_time_total)
                sequence.append(name)
        assert sequence == list(STAGES)*30, sequence[:12]
        assert torch.isfinite(result[0]).all()
        samples['total'] = [sum(items) for items in zip(*(samples[s] for s in STAGES))]
    print(json.dumps(dict(variant=variant,gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,triton=triton.__version__,samples_us=samples,
        medians_us={k:statistics.median(v) for k,v in samples.items()})))


def main():
    variants = ['triton','cuda_soft_q','cuda_soft_k','cuda_mixed_q','cuda_mixed_k']
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker',choices=variants)
    parser.add_argument('--rounds',type=int,default=3)
    parser.add_argument('--output',type=Path,default=Path('/tmp/dism-forward-comparison.json'))
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
        return
    if args.rounds<1: parser.error('positive rounds required')
    rows = []
    for rep in range(args.rounds):
        for variant in (variants if rep%2==0 else list(reversed(variants))):
            p = subprocess.run([sys.executable,'-m','dism_v2.benchmark_forward_comparison',
                '--worker',variant],capture_output=True,text=True,timeout=180)
            if p.returncode: raise RuntimeError(p.stdout+'\n'+p.stderr)
            row = json.loads(p.stdout.strip().splitlines()[-1])
            row['round'] = rep
            rows.append(row)
            print(variant,rep,row['medians_us'],flush=True)
    medians = {v:{s:statistics.median(r['medians_us'][s] for r in rows if r['variant']==v)
                  for s in (*STAGES,'total')} for v in variants}
    means = {v:{s:statistics.mean(t for r in rows if r['variant']==v
                                 for t in r['samples_us'][s])
                for s in (*STAGES,'total')} for v in variants}
    data = dict(scope='CUPTI forward-stage durations in ordinary streams; total is sum '
        'of three GPU durations per invocation, excludes launch gaps/embedding/allocation; '
        '20 warmups/30 measured calls per worker, no per-launch sync/graphs',
        b=64,h=4,n=1024,d=64,dv=64,cuda_vocab=512,triton_n_vocab=64,
        tile_lse=os.environ.get('DISM_TILE_LSE','full'),
        lineinfo=os.environ.get('DISM_LINEINFO','0'),rounds=args.rounds,
        medians_us=medians,
        pooled_means_us=means,
        mean_total_relative_throughput={v:means['triton']['total']/means[v]['total']
                                        for v in variants},
        relative_throughput={v:{s:medians['triton'][s]/medians[v][s]
                                for s in (*STAGES,'total')} for v in variants},
        measurements=rows)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps({k:v for k,v in data.items() if k!='measurements'}),flush=True)


if __name__ == '__main__': main()
