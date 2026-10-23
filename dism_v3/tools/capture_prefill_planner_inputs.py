"""Capture immutable real SAM planner labels/tau for CPU-only paired diagnosis."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import torch


class Capture:
    def __init__(self, engine, output):
        self.engine,self.output=engine,output
        self.rows=[]
    def __call__(self,iq,ik,sq,sk,v,tau):
        labels=torch.stack((iq,ik)).cpu().numpy()
        taus=tau.detach().double().cpu().numpy()
        b,h,n=iq.shape
        docs=np.array([(i,0,n) for i in range(b)],dtype=np.int64)
        fields=(np.array([b,h,n,b],dtype=np.int32),labels,np.ones((b,h),dtype=np.uint8),taus,docs)
        blob=b''.join(a.tobytes() for a in fields)
        path=self.output/f'n{n}_layer{len(self.rows)}.bin'
        if path.exists():raise FileExistsError(path)
        path.write_bytes(blob)
        self.rows.append(dict(path=str(path),sha256=hashlib.sha256(blob).hexdigest(),shape=[b,h,n],tau=taus.tolist()))
        return self.engine(iq,ik,sq,sk,v,tau)


@torch.no_grad()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--suite',type=Path,default=Path.home()/'downstream-eval')
    p.add_argument('--lengths',type=int,nargs='+',default=[2048,8192])
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    spec=importlib.util.spec_from_file_location('eval_adapter',args.suite/'adapter_layout_bounded.py')
    adapter=importlib.util.module_from_spec(spec);spec.loader.exec_module(adapter)
    lm=adapter.NanochatLM(str(args.checkpoint),str(args.suite/'tokenizer'),4,max(args.lengths),hard_backend='sam')
    engine=lm.prefill
    rows=[]
    try:
        batch=next(pq.ParquetFile(args.suite/'data/pg19/test.parquet').iter_batches(batch_size=4,columns=['text']))
        books=[lm.tok_encode(t,add_special_tokens=False) for t in batch.column(0).to_pylist()]
        for n in args.lengths:
            assert min(map(len,books))>=n-1
            x,cu=lm._inputs([[lm.prefix_token_id]+t[:n-1] for t in books])
            capture=Capture(engine,args.output);lm.prefill=capture
            lm._features(x,cu);torch.cuda.synchronize()
            rows.extend(capture.rows)
            print('captured',n,len(capture.rows),flush=True)
    finally:
        lm.prefill=engine;lm.close()
    (args.output/'manifest.json').write_text(json.dumps(dict(checkpoint=str(args.checkpoint),rows=rows),indent=2))


if __name__=='__main__':main()
