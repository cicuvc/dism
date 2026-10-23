"""Split real-model SAM prefill time with explicit diagnostic stage fences.

Uninstrumented forward timings are reported separately. A fence before label
D2H separates waiting for earlier model kernels from the actual transfer path.
This changes overlap and adds event overhead: stage sums are diagnostic, not
an exact attribution of an unfenced trace. No production code is modified.
"""
import argparse
import importlib.util
import json
from pathlib import Path
from statistics import median
import time

import pyarrow.parquet as pq
import torch
from flash_dism.inference.prefill_batch import PreparedHardPrefill


class ProfilePrefill:
    def __init__(self, engine):
        self.engine = engine
        self.rows = []

    def __call__(self, iq, ik, sq, sk, value, tau):
        engine = self.engine
        start = time.perf_counter()
        fence = torch.cuda.Event()
        fence.record(); fence.synchronize()
        waited = time.perf_counter()
        host = torch.stack((iq, ik)).cpu().numpy()
        host_tau = tau.detach().double().cpu().numpy()
        downloaded = time.perf_counter()
        plan = engine.planner.plan(host, host_tau, chunk_size=engine.chunk_size,
                                   dispatch_threshold=engine.dispatch_threshold)
        planned = time.perf_counter()
        upload_begin, upload_end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        upload_begin.record()
        core = plan.upload(device=engine.device, mma_precision=engine.mma_precision)
        upload_end.record(); upload_end.synchronize()
        uploaded = time.perf_counter()
        b,h,n = iq.shape
        prepared = PreparedHardPrefill(core,b,h,n)
        gpu_begin, gpu_end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        gpu_begin.record()
        result = prepared.execute_dispatch(sq,sk,value)
        gpu_end.record(); gpu_end.synchronize()
        executed = time.perf_counter()
        stats = dict(plan.statistics)
        row = dict(prior_gpu_wait_ms=(waited-start)*1000,
                   labels_d2h_wall_ms=(downloaded-waited)*1000,
                   planner_wall_ms=(planned-downloaded)*1000,
                   **plan.timings,
                   h2d_upload_wall_ms=(uploaded-planned)*1000,
                   h2d_stream_ms=upload_begin.elapsed_time(upload_end),
                   gpu_execute_wall_ms=(executed-uploaded)*1000,
                   gpu_execute_ms=gpu_begin.elapsed_time(gpu_end),
                   labels_bytes=host.nbytes+host_tau.nbytes,
                   metadata_bytes=plan.storage.nbytes,
                   statistics=stats)
        release = time.perf_counter()
        del prepared, core, plan, host, host_tau
        row['release_ms']=(time.perf_counter()-release)*1000
        self.rows.append(row)
        return result


def difference(actual, expected):
    delta=actual.float()-expected.float()
    return dict(bitwise_equal=bool(torch.equal(actual,expected)),
                mismatch_fraction=float((actual!=expected).float().mean()),
                relative_l2=float(delta.norm()/expected.float().norm().clamp_min(1e-30)),
                max_abs=float(delta.abs().max()))


def timed(fn):
    torch.cuda.synchronize(); start=time.perf_counter()
    out=fn()
    torch.cuda.synchronize()
    return (time.perf_counter()-start)*1000, out


@torch.no_grad()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--suite',type=Path,default=Path.home()/'downstream-eval')
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--lengths',type=int,nargs='+',default=[2048,8192])
    p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--precision',choices=['tf32x3','bf16'],default='tf32x3')
    p.add_argument('--workers',type=int,default=8)
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    spec=importlib.util.spec_from_file_location('eval_adapter',args.suite/'adapter_layout_bounded.py')
    adapter=importlib.util.module_from_spec(spec);spec.loader.exec_module(adapter)
    lm=adapter.NanochatLM(str(args.checkpoint),str(args.suite/'tokenizer'),args.batch_size,max(args.lengths),
                         hard_backend='sam',prefill_precision=args.precision,prefill_workers=args.workers)
    engine=lm.prefill
    result=dict(checkpoint=str(args.checkpoint),batch=args.batch_size,workers=args.workers,
                precision=args.precision,cases=[])
    try:
        batch=next(pq.ParquetFile(args.suite/'data/pg19/test.parquet').iter_batches(batch_size=args.batch_size,columns=['text']))
        books=[lm.tok_encode(text,add_special_tokens=False) for text in batch.column(0).to_pylist()]
        for length in args.lengths:
            assert min(map(len,books))>=length-1
            x,cu=lm._inputs([[lm.prefix_token_id]+book[:length-1] for book in books])
            wall={}; repeat_drift={}
            for backend in ('quadratic','sam'):
                lm.hard_backend=backend
                timed(lambda:lm._features(x,cu))
                samples=[]; previous=None; drift=[]
                for _ in range(args.repeats):
                    elapsed,out=timed(lambda:lm._features(x,cu));samples.append(elapsed)
                    if previous is not None:drift.append(difference(out,previous))
                    previous=out.clone()
                wall[backend]=samples;repeat_drift[backend]=drift
            expected=out.clone()
            profiled=[]
            for _ in range(args.repeats):
                probe=ProfilePrefill(engine);lm.prefill=probe
                elapsed,actual=timed(lambda:lm._features(x,cu))
                # Report, do not hide, floating-point/reduction variation and
                # downstream hard-route changes under different launch timing.
                drift=difference(actual,expected)
                lm.prefill=engine
                totals={k:sum(row[k] for row in probe.rows) for k in probe.rows[0] if k!='statistics'}
                profiled.append(dict(wall_ms=elapsed,totals=totals,layers=probe.rows,output_drift=drift))
            medians={k:median(r['totals'][k] for r in profiled) for k in profiled[0]['totals']}
            case=dict(length=length,forward_wall_ms=wall,uninstrumented_repeat_drift=repeat_drift,profiled=profiled,stage_medians=medians,
                      profiled_wall_median_ms=median(r['wall_ms'] for r in profiled))
            result['cases'].append(case)
            print(json.dumps(dict(length=length,forward_wall_ms=wall,stage_medians=medians,
                                  profiled_wall_median_ms=case['profiled_wall_median_ms'],uninstrumented_repeat_drift=repeat_drift,profiled_output_drift=[r['output_drift'] for r in profiled])),flush=True)
            args.output.parent.mkdir(parents=True,exist_ok=True)
            args.output.write_text(json.dumps(result,indent=2))
    finally:
        lm.prefill=engine
        lm.close()


if __name__=='__main__':main()
