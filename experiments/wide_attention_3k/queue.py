"""Wait for idle GPU summaries; preflight and run two isolated A100 controls."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
ARMS = ('swa_only', 'full_attention')


def worker(arm, gpu):
    dest = ROOT/arm
    port = 18477 if arm == 'swa_only' else 18478
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='8',
               NO_PROXY='127.0.0.1,localhost', WANDB_MODE='offline',
               TRITON_CACHE_DIR=str(dest/'triton-cache'))
    status = dict(architecture=arm, gpu=gpu, pid=os.getpid())
    def run(command, phase, log):
        status['phase'] = phase
        (dest/'status.json').write_text(json.dumps(status, indent=2)+'\n')
        with (dest/log).open('a') as f:
            subprocess.run([sys.executable, '-u', *command], cwd=ROOT, env=env,
                           stdout=f, stderr=subprocess.STDOUT, check=True)
    try:
        run(['preflight.py', '--architecture', arm, '--port', str(port), '--secret', str(dest/'auth_token')],
            'preflight', 'preflight.log')
        run(['-m','dism_v2.train_lm','--architecture',arm,'--attention-heads','12','--ffn-hidden','1050',
             '--output',str(dest/'run'),'--steps','3000','--warmup','100','--batch','64','--micro-batch','8',
             '--lr','.001','--weight-decay','.01','--softcap','30','--seed','777',
             '--stream-url',f'http://127.0.0.1:{port}','--stream-secret-file',str(dest/'auth_token'),
             '--eval-every','1000','--eval-batches','100','--save-every','500','--log-every','10',
             '--wandb-mode','offline','--wandb-project','dism-activation-study'], 'training', 'console.log')
        import torch
        ckpt = torch.load(dest/'run/latest.pt',map_location='cpu',weights_only=False)
        assert ckpt['step'] == 3000 and ckpt['config']['parameters'] == 49_675_276
        status['phase'] = 'complete'
    except BaseException as error:
        status['phase'], status['error'] = 'failed', str(error)
        raise
    finally:
        (dest/'status.json').write_text(json.dumps(status,indent=2)+'\n')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--worker',choices=ARMS)
    p.add_argument('--gpu',type=int)
    a = p.parse_args()
    if a.worker:
        return worker(a.worker,a.gpu)
    assert not (ROOT/'queue-status.json').exists(), 'Existing queue requires manual inspection'
    pending, active, finished, previous_free = list(ARMS), {}, {}, set()
    while pending or active:
        for arm, (gpu, process, log) in list(active.items()):
            code = process.poll()
            if code is not None:
                log.close(); finished[arm] = dict(gpu=gpu,exit_code=code); del active[arm]
        result = subprocess.run(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu',
                                 '--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
        # No process inspection/reservation. Require two consecutive idle samples.
        free = {int(i) for i,m,u in (line.split(',') for line in result.stdout.splitlines())
                if int(m) < 256 and int(u) == 0}
        used = {gpu for gpu,_,_ in active.values()}
        for gpu in sorted((free & previous_free)-used):
            if not pending:
                break
            arm = pending.pop(0)
            log = (ROOT/arm/'worker.log').open('a')
            process = subprocess.Popen([sys.executable,'-u',str(ROOT/'queue.py'),'--worker',arm,'--gpu',str(gpu)],
                                       cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
            active[arm] = (gpu,process,log)
            print(f'Started {arm} on idle GPU{gpu}',flush=True)
        previous_free = free
        state = dict(pending=pending,active={k:dict(gpu=v[0],pid=v[1].pid) for k,v in active.items()},
                     finished=finished,idle_candidates=sorted(free),updated_unix=time.time())
        (ROOT/'queue-status.json').write_text(json.dumps(state,indent=2)+'\n')
        if pending or active:
            time.sleep(30)


if __name__ == '__main__':
    main()
