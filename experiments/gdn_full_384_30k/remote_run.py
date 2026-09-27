"""Task-scoped A100 preflight/training; GPU summaries only, no reservations."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    root=Path(__file__).resolve().parent
    status=root/'status.json'
    if status.exists():
        raise FileExistsError('Existing run: inspect manually rather than restart')
    def record(phase,**extra):
        status.write_text(json.dumps(dict(phase=phase,pid=os.getpid(),**extra),indent=2))
    previous=set()
    try:
        while True:
            s=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu',
                                       '--format=csv,noheader,nounits'],text=True)
            free={int(i) for i,m,u in (line.split(',') for line in s.splitlines()) if int(m)<256 and int(u)==0}
            eligible=free&previous
            if eligible:
                gpu=min(eligible);break
            record('waiting_idle',candidates=sorted(free))
            previous=free;time.sleep(30)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='8',WANDB_MODE='offline',
                 NO_PROXY='127.0.0.1,localhost',TRITON_CACHE_DIR=str(root/'triton-cache'))
        def run(args,phase,log):
            record(phase,gpu=gpu)
            with (root/log).open('a') as f:
                subprocess.run([sys.executable,'-u',*args],cwd=root,env=env,
                               stdout=f,stderr=subprocess.STDOUT,check=True)
        run(['-m','dism_v2.check_remote_lm','--data-only','--stream-url','http://127.0.0.1:18482',
             '--secret-file',str(root/'auth_token')],'stream_preflight','stream.log')
        run(['-m','pytest','test_matched.py','-q','-s','-x'],'gpu_preflight','preflight.log')
        run(['-m','dism_v2.train_lm','--output',str(root/'run'),'--width','384','--layers','12',
             '--ffn-hidden','1522','--gdn-full','--steps','30000','--warmup','1000',
             '--batch','64','--micro-batch','8','--lr','.001','--weight-decay','.01',
             '--softcap','30','--seed','777','--eval-every','1000','--eval-batches','100',
             '--save-every','1000','--log-every','10','--wandb-mode','offline',
             '--wandb-project','dism-finewebedu','--stream-url','http://127.0.0.1:18482',
             '--stream-secret-file',str(root/'auth_token')],'training','console.log')
        import torch
        saved=torch.load(root/'run/latest.pt',map_location='cpu',weights_only=False)
        assert saved['step']==30000 and saved['config']['parameters']==72_182_172
        record('complete',gpu=gpu)
    except BaseException as e:
        record('failed',error=str(e));raise


if __name__=='__main__': main()
