"""Frozen real-model layer summaries: compiler costs, scratch, CTA sweep."""
import argparse,json,sys,time
from pathlib import Path
import torch

@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--length',type=int,default=8191)
    p.add_argument('--workers',nargs='+',type=int,default=[64,256,512]);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(1)
    sys.path.insert(0,'/home/cicuvc/downstream-eval')
    from adapter_layout_bounded import NanochatLM
    from flash_dism.inference import HardDismPrefill
    from flash_dism.inference.build import load_summary_runtime
    from flash_dism.inference.summary_runtime import SummaryWorkspace
    from nanochat.models.dism.inference import SamDecodeSession
    import pyarrow.parquet as pq
    ckpt=json.load(open('dism_v3/build/prefill_planner_compare/inputs/manifest.json'))['checkpoint']
    lm=NanochatLM(ckpt,'/home/cicuvc/downstream-eval/tokenizer',batch_size=1,context=((a.length+1023)//256)*256)
    book=pq.read_table('/home/cicuvc/downstream-eval/data/pg19/test.parquet',columns=['text']).column('text')[10].as_py()
    tokens=torch.tensor([lm.tok_encode(book)[:a.length]],device='cuda')
    rows=[];pool=SummaryWorkspace('cuda')
    with HardDismPrefill(workers=16) as engine:
        session=SamDecodeSession(lm.model,engine,lm.max_length)
        session.features(tokens);torch.cuda.synchronize()
        for i,state in enumerate(session.states):
            if state['dism'] is None:continue
            d=state['dism'];snap=d.core.snapshot;history=snap[7][:a.length].cpu();keys=history[:,0].T.contiguous()
            start=time.perf_counter();program=load_summary_runtime().SummaryProgram(keys,snap[9].double().cpu().tolist(),d.r,d.dv,256,128,16,True,True,False)
            compile_ms=(time.perf_counter()-start)*1000
            out=program.prepare(snap[0],False,pool.buffers);pool.buffers=program.workspace();torch.cuda.synchronize()
            row=dict(layer=i,compile_ms=compile_ms,stats=dict(program.stats()),timing={})
            for workers in a.workers:
                program.run(snap[0],snap[1],workers);torch.cuda.synchronize()
                times=[]
                for _ in range(3):
                    begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    begin.record();program.run(snap[0],snap[1],workers);end.record();end.synchronize();times.append(begin.elapsed_time(end))
                row['timing'][workers]=sorted(times)[1]
            rows.append(row);print(json.dumps(row),flush=True);a.output.write_text(json.dumps(rows,indent=2))
            program.release_compiler()
        session.close()

if __name__=='__main__':main()
