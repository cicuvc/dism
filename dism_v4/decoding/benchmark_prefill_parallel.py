"""CPU-only head parallel scaling; verifies exact native/packed metadata."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import threading
import numpy as np
from prefill_parallel import ParallelPrefillPlanner


def fingerprint(results):
    digest=hashlib.sha256()
    for plan,packed in results:
        for group in (plan.arrays(),packed):
            for name in sorted(group):digest.update(np.ascontiguousarray(group[name]).tobytes())
    return digest.hexdigest()


def rss_bytes():
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):return int(line.split()[1])*1024
    return 0


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--lengths",type=int,nargs="+",default=[4096,16384,65536])
    parser.add_argument("--repeats",type=int,default=3)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    rng=np.random.default_rng(1673)
    records=[]
    for pattern in ("zipf512","repeat"):
        for n in args.lengths:
            # Mixed independent heads; B2/H8 also exercises batch indexing.
            shape=(2,8,n)
            prob=np.arange(1,513,dtype=np.float64)**-1.2;prob/=prob.sum()
            q,k=[rng.choice(512,size=shape,p=prob).astype(np.int32) for _ in range(2)]
            if pattern=="repeat":q.fill(0);k.fill(0)
            tau=np.linspace(.7,np.log(64.),8)
            reset=np.zeros(shape,dtype=bool)
            reset[1,:,::997]=True
            baseline=None;serial_ms=None
            for workers in (1,2,4,8):
                with ParallelPrefillPlanner(workers) as pool:
                    # One full warmup; reuse pool; compilation excluded.
                    result=pool.plan(q,k,tau,reset=reset)
                    checksum=fingerprint(result)
                    if baseline is None:baseline=checksum
                    assert checksum==baseline,(pattern,n,workers,"metadata mismatch")
                    retained=sum(p.statistics()["program_bytes"]+sum(x.nbytes for x in c.values()) for p,c in result)
                    del result
                    peak=[rss_bytes()];stop=threading.Event()
                    def sample():
                        while not stop.wait(.01):peak[0]=max(peak[0],rss_bytes())
                    monitor=threading.Thread(target=sample,daemon=True);monitor.start()
                    samples=[]
                    try:
                        for _ in range(args.repeats):
                            start=time.perf_counter();result=pool.plan(q,k,tau,reset=reset)
                            samples.append((time.perf_counter()-start)*1000)
                            del result
                    finally:stop.set();monitor.join()
                    ms=float(np.median(samples))
                    if serial_ms is None:serial_ms=ms
                    rec=dict(pattern=pattern,n=n,batch=2,heads=8,workers=workers,
                             median_ms=ms,speedup=serial_ms/ms,samples_ms=samples,
                             sequences_tokens_s=2*n/(ms*.001),head_tokens_s=16*n/(ms*.001),
                             retained_program_and_packed_bytes=retained,
                             process_rss_peak_bytes=peak[0],metadata_sha256=checksum)
                    records.append(rec);print(json.dumps(rec),flush=True)
    args.output.write_text(json.dumps(dict(status="PASS",results=records,
        note="CPU plan+chunk packing only, reusable pool, excludes GPU and fingerprint checks. RSS is process-wide with allocator retention, not isolated per config."),indent=2)+"\n")


if __name__=="__main__":main()
