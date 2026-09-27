"""Wait for the original three-arm queue to finish, then test/train preconv SiLU."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

source = Path(__file__).resolve().parent/'src'
study = Path('/media/cicuvc/c63abdf1-0e56-4153-9228-95df5a2f239b/cicuvc/dism-lm-runs/activation-study-3k-20260910')
output = study/'preconv_silu'
output.mkdir(exist_ok=True)
assert not (output/'queue-status.json').exists(), 'Existing job requires manual inspection'
def status(phase, **extra):
    (output/'queue-status.json').write_text(json.dumps(dict(phase=phase, pid=os.getpid(), **extra),indent=2)+'\n')
status('waiting_for_no_qk_silu_and_final_vocab_eval')
try:
    while True:
        prior = json.loads((study/'manifest.json').read_text())
        if prior['status'] == 'complete':
            assert 'no_qk_silu' in prior['completed']
            break
        if prior['status'] in ('failed','stopped'):
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
    run(['-m','pytest','tests/test_lm_activation.py','-q','-x'],'gpu_preflight','preflight.log')
    run(['-m','dism_v2.train_lm','--output',str(output),'--dism-activation','preconv_silu',
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
