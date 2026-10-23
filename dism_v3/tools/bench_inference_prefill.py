"""Current hard prefill baseline: public API plus separately timed prepare stages.

Synthetic full-hard single-document rows, GPU-resident labels/vectors/tau.
No model, checkpoint, dataset, decode-cache prime or production edits. Stage
timings mirror prepare() with explicit fences; end-to-end uses the public API
without instrumentation. Legacy planning includes list conversion and per-head
exports; native planning packs directly to pinned storage. Upload includes
dispatch metadata. --backend legacy preserves the pre-optimization pipeline.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from flash_dism.inference import HardDismPrefill
from flash_dism.inference.build import load_prefill
from flash_dism.inference.prefill_batch import _merge_chunks, PreparedHardPrefill
from flash_dism.inference.prefill_parallel import ParallelPrefillPlanner
from flash_dism.inference.prefill_triton import TritonPrefillPlan


class LegacyPrefill(HardDismPrefill):
    """825b5f2 preparation, retained only in this benchmark for paired A/B."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.planner.close()
        self.planner=ParallelPrefillPlanner(self.workers)

    def prepare(self, idx_q, idx_k, tau, *, reset=None):
        for x in (idx_q,idx_k):
            if not isinstance(x,torch.Tensor) or x.ndim!=3 or x.dtype!=torch.int32:
                raise ValueError('labels must be int32 tensors [B,H,N]')
        if idx_q.shape!=idx_k.shape or min(idx_q.shape)<=0 or idx_q.device!=idx_k.device:
            raise ValueError('nonempty matching label shapes/devices required')
        if idx_q.device.type!='cpu' and idx_q.device!=self.device:
            raise ValueError('labels must be CPU or on the engine CUDA device')
        b,h,n=idx_q.shape
        if b*h*n>np.iinfo(np.int32).max:raise ValueError('flattened row count exceeds int32')
        if reset is None:reset=torch.zeros_like(idx_q,dtype=torch.bool)
        if reset.shape!=idx_q.shape or reset.dtype!=torch.bool or reset.device!=idx_q.device:
            raise ValueError('reset must be bool [B,H,N] on the label device')
        host=torch.stack((idx_q,idx_k,reset.to(torch.int32))).cpu().numpy()
        if isinstance(tau,torch.Tensor):tau=tau.detach().double().cpu().numpy()
        results=self.planner.plan(host[0],host[1],tau,reset=host[2].astype(bool),chunk_size=self.chunk_size)
        merged=_merge_chunks(results,b,h,n)
        del results
        core=TritonPrefillPlan.from_packed(merged,b*h*n,device=self.device,
            chunk_size=self.chunk_size,mma_precision=self.mma_precision,dispatch_threshold=self.dispatch_threshold)
        return PreparedHardPrefill(core,b,h,n)


def summary(values):
    return dict(median_ms=float(np.median(values)), mean_ms=float(np.mean(values)),
                min_ms=min(values), max_ms=max(values), samples_ms=values)


def wall(fn):
    start = time.perf_counter()
    result = fn()
    return result, (time.perf_counter()-start)*1000


@torch.no_grad()
def case(args, b, n, pattern, workers):
    h, r, dv = args.heads, args.readout, args.value_dim
    rng = np.random.default_rng(args.seed)
    labels = rng.integers(0, args.alphabet, (2,b,h,n), dtype=np.int32)
    if pattern == 'repeat':
        labels.fill(0)
    q, k = torch.from_numpy(labels).to('cuda').unbind(0)
    torch.manual_seed(args.seed)
    sq = torch.randn(b,n,h,r,device='cuda',dtype=torch.bfloat16)
    sk = torch.randn_like(sq)
    v = torch.randn(b,n,h,dv,device='cuda',dtype=torch.bfloat16)
    tau = torch.full((h,), args.tau, device='cuda', dtype=torch.float32)
    vectors = (sq.view(-1,r), sk.view(-1,r), v.view(-1,dv))
    engine_class = LegacyPrefill if args.backend=='legacy' else HardDismPrefill
    with engine_class(workers=workers) as engine:
        torch.cuda.synchronize()
        cold_start = time.perf_counter()
        prepared = engine.prepare(q,k,tau)
        out = prepared.execute_dispatch(sq,sk,v)
        torch.cuda.synchronize()
        cold_ms = (time.perf_counter()-cold_start)*1000
        assert torch.isfinite(out).all().item(), 'nonfinite prefill output'
        del out
        # Warm both variants and all chunk-parallel phases before measuring.
        for _ in range(args.warmup):
            engine(q,k,sq,sk,v,tau)
            prepared.execute(sq,sk,v)
        torch.cuda.synchronize()
        stages = {name: [] for name in ('labels_d2h', 'plan_pack', 'merge_release',
                                       'upload_setup', 'execute_wall')}
        metadata = None
        native_samples = []
        for _ in range(args.repeats):
            def download():
                if args.backend=='legacy':
                    reset = torch.zeros_like(q,dtype=torch.bool)
                    planes = (q,k,reset.to(torch.int32))
                else:planes=(q,k)
                host = torch.stack(planes).cpu().numpy()
                host_tau = tau.detach().double().cpu().numpy()
                return host, host_tau
            (host, host_tau), elapsed = wall(download)
            stages['labels_d2h'].append(elapsed)
            if args.backend=='legacy':
                results, elapsed = wall(lambda: engine.planner.plan(
                    host[0],host[1],host_tau,reset=host[2].astype(bool),chunk_size=engine.chunk_size))
                start = time.perf_counter()
                merged = _merge_chunks(results,b,h,n)
                del results
                stages['merge_release'].append((time.perf_counter()-start)*1000)
            else:
                native, elapsed = wall(lambda: engine.planner.plan(
                    host,host_tau,chunk_size=engine.chunk_size,dispatch_threshold=engine.dispatch_threshold))
                native_samples.append(native.timings)
                merged = {name: tensor.numpy() for name,tensor in native.data.items()}
                stages['merge_release'].append(0.)
            stages['plan_pack'].append(elapsed)
            def upload():
                if args.backend=='legacy':
                    core = TritonPrefillPlan.from_packed(
                        merged,b*h*n,device=engine.device,chunk_size=engine.chunk_size,
                        mma_precision=engine.mma_precision,dispatch_threshold=engine.dispatch_threshold)
                else:core=native.upload(device=engine.device,mma_precision=engine.mma_precision)
                torch.cuda.synchronize()
                return core
            core, elapsed = wall(upload)
            stages['upload_setup'].append(elapsed)
            start = time.perf_counter()
            result = core.execute_dispatch(*vectors)
            torch.cuda.synchronize()
            stages['execute_wall'].append((time.perf_counter()-start)*1000)
            if metadata is None:
                counts = np.diff(merged['offsets'])
                metadata = dict(core.statistics(), events=int(merged['lengths'].sum()),
                                max_chunks_per_stream=int(counts.max(initial=0)),
                                all_device_metadata_bytes=sum(t.numel()*t.element_size()
                                    for t in vars(core).values() if isinstance(t,torch.Tensor) and t.is_cuda),
                                labels_d2h_bytes=host.nbytes+host_tau.nbytes)
            del core, result, merged
            if args.backend=='native':del native
        # Already-prepared vector execution: CUDA events exclude CPU planning
        # and upload, but include output zeroing and parallel-path workspace.
        gpu = {}
        for name, fn in [('dispatch', prepared.execute_dispatch), ('serial', prepared.execute)]:
            samples = []
            for _ in range(args.repeats):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(args.gpu_iterations):
                    result = fn(sq,sk,v)
                end.record(); end.synchronize()
                samples.append(start.elapsed_time(end)/args.gpu_iterations)
                del result
            gpu[name] = summary(samples)
        totals = []
        # Synchronization defines request completion, not just CPU launch time.
        for _ in range(args.repeats):
            torch.cuda.synchronize()
            start = time.perf_counter()
            result = engine(q,k,sq,sk,v,tau)
            torch.cuda.synchronize()
            totals.append((time.perf_counter()-start)*1000)
            del result
    return dict(batch=b,heads=h,tokens=n,readout=r,value_dim=dv,pattern=pattern,backend=args.backend,
                workers=workers,cold_call_ms=cold_ms,metadata=metadata,
                stages={k:summary(v) for k,v in stages.items()},gpu=gpu,
                native_stages={k:summary([s[k] for s in native_samples]) for k in native_samples[0]} if native_samples else {},
                end_to_end=summary(totals),tokens_per_second=b*n*1000/float(np.median(totals)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend',choices=['native','legacy'],default='native')
    parser.add_argument('--batches',type=int,nargs='+',default=[1])
    parser.add_argument('--tokens',type=int,nargs='+',default=[2048,8192,16384])
    parser.add_argument('--patterns',choices=['random','repeat'],nargs='+',default=['random','repeat'])
    parser.add_argument('--workers',type=int,nargs='+',default=[8])
    parser.add_argument('--heads',type=int,default=12)
    parser.add_argument('--readout',type=int,default=32)
    parser.add_argument('--value-dim',type=int,default=64)
    parser.add_argument('--alphabet',type=int,default=512)
    parser.add_argument('--tau',type=float,default=.5)
    parser.add_argument('--seed',type=int,default=120)
    parser.add_argument('--warmup',type=int,default=2)
    parser.add_argument('--repeats',type=int,default=7)
    parser.add_argument('--gpu-iterations',type=int,default=10)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.output.exists() or not args.output.resolve().is_relative_to(root/'build'):
        parser.error('fresh output under this worktree dism_v3/build required')
    if min(*args.batches,*args.tokens,*args.workers,args.repeats,args.gpu_iterations,args.heads,
           args.readout,args.value_dim,args.alphabet) <= 0 or args.warmup < 0:
        parser.error('positive dimensions/counts and nonnegative warmup required')
    torch.set_num_threads(1)
    sources = [Path(__file__), *sorted((root/'python/flash_dism/inference').glob('prefill*.py')),
               *sorted((root/'python/flash_dism/inference/csrc').glob('prefill*'))]
    report = dict(command=sys.argv,git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                  gpu=torch.cuda.get_device_name(),torch_version=torch.__version__,
                  native_module=load_prefill().__file__,cpu=subprocess.check_output(['lscpu'],text=True),
                  sources={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                  arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},cases=[])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    for workers in args.workers:
        for b in args.batches:
            for n in args.tokens:
                for pattern in args.patterns:
                    print(f'START W{workers} B{b} N{n} {pattern}',flush=True)
                    result = case(args,b,n,pattern,workers)
                    report['cases'].append(result)
                    args.output.write_text(json.dumps(report,indent=2)+'\n')
                    print(f'DONE E2E {result["end_to_end"]["median_ms"]:.3f} ms '
                          f'plan/pack {result["stages"]["plan_pack"]["median_ms"]:.3f} ms '
                          f'GPU {result["gpu"]["dispatch"]["median_ms"]:.3f} ms',flush=True)


if __name__ == '__main__':
    main()
