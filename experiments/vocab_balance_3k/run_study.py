"""Wait for preconv SiLU to finish, then test/train balanced vocabulary arm."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import argparse

parser=argparse.ArgumentParser()
parser.add_argument('--start-now',action='store_true',help='Explicitly bypass the preconv post-evaluation dependency')
args=parser.parse_args()

source = Path(__file__).resolve().parent/'src'
study = Path('/media/cicuvc/c63abdf1-0e56-4153-9228-95df5a2f239b/cicuvc/dism-lm-runs/activation-study-3k-20260910')
output = study/'vocab_balance_001'
output.mkdir(exist_ok=True)
if (output/'queue-status.json').exists():
    prior_status=json.loads((output/'queue-status.json').read_text())
    assert args.start_now and prior_status['phase']=='failed' and not (output/'config.json').exists(), 'Existing training requires manual inspection'
    archived=output/'queue-status-before-direct-start.json'
    assert not archived.exists(), 'Do not overwrite previous failure record'
    (output/'queue-status.json').rename(archived)
def status(phase, **extra):
    (output/'queue-status.json').write_text(json.dumps(dict(phase=phase, pid=os.getpid(), **extra),indent=2)+'\n')
status('direct_start_authorized' if args.start_now else 'waiting_for_preconv_silu_and_final_vocab_eval')
try:
    while not args.start_now:
        prior = json.loads((study/'preconv_silu/queue-status.json').read_text())
        if prior['phase'] == 'complete':
            break
        if prior['phase'] in ('failed','stopped'):
            raise RuntimeError('Prior study stopped/failed: not starting the next arm')
        time.sleep(30)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='8',
               DISM_TILE_LSE='tanh_finite',DISM_BWD_OPT='13',DISM_ROW_BITSET='0',
               DISM_OUTPUT_Q_ALIAS='kv',DISM_OUTPUT_TMA='0',MAX_JOBS='8',WANDB_MODE='offline',
               TORCH_EXTENSIONS_DIR=str(output/'torch-extensions'),TRITON_CACHE_DIR=str(output/'triton-cache'))
    def run(args, phase, log):
        status(phase)
        with (output/log).open('a') as f:
            subprocess.run([sys.executable,'-u',*args],cwd=source,env=env,
                           stdout=f,stderr=subprocess.STDOUT,check=True)
    run(['-m','pytest','tests/test_vocab_balance.py','-q','-x'],'gpu_preflight','preflight.log')
    run(['-m','dism_v2.train_lm','--output',str(output),'--vocab-balance-weight','.01',
         '--steps','3000','--warmup','100','--batch','64','--micro-batch','8','--lr','.001',
         '--weight-decay','.01','--softcap','30','--seed','777','--eval-every','1000',
         '--eval-batches','100','--save-every','500','--log-every','10',
         '--wandb-mode','offline','--wandb-project','dism-activation-study'],'training','console.log')
    import torch
    ckpt=torch.load(output/'latest.pt',map_location='cpu',weights_only=False)
    assert ckpt['step']==3000 and ckpt['config']['parameters']==49_678_876
    del ckpt
    run(['-m','dism_v2.eval_vocab_load','--checkpoint',str(output/'latest.pt'),
         '--output',str(output/'vocab-load-final'),'--sequences','256','--micro-batch','8'],
        'vocab_eval','vocab-eval.log')
    status('complete')
except BaseException as error:
    status('failed',error=str(error))
    raise
