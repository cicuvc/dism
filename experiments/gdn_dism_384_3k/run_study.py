"""Preflight and train384-wide [GDN,GDN,GDN,DISM] x3 without SWA."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    source = Path(__file__).resolve().parent/'src'
    study = Path('/media/cicuvc/c63abdf1-0e56-4153-9228-95df5a2f239b/cicuvc/dism-lm-runs/activation-study-3k-20260910')
    output = study/'gdn_dism_384'
    output.mkdir(exist_ok=True)
    state = output/'queue-status.json'
    with state.open('x') as f:
        json.dump(dict(phase='starting', pid=os.getpid()), f)

    def status(phase, **extra):
        temporary = output/'queue-status.tmp'
        temporary.write_text(json.dumps(dict(phase=phase, pid=os.getpid(), **extra), indent=2)+'\n')
        temporary.replace(state)

    try:
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='8',
                   DISM_TILE_LSE='tanh_finite', DISM_BWD_OPT='13', DISM_ROW_BITSET='0',
                   DISM_OUTPUT_Q_ALIAS='kv', DISM_OUTPUT_TMA='0', MAX_JOBS='8', WANDB_MODE='offline',
                   TORCH_EXTENSIONS_DIR=str(output/'torch-extensions'), TRITON_CACHE_DIR=str(output/'triton-cache'))

        def run(args, phase, log):
            status(phase)
            with (output/log).open('a') as f:
                subprocess.run([sys.executable, '-u', *args], cwd=source, env=env,
                               stdout=f, stderr=subprocess.STDOUT, check=True)

        run(['-m','pytest','tests/test_lm_gdn_dism.py','-q','-s','-x'], 'gpu_preflight','preflight.log')
        run(['-m','dism_v2.train_lm','--output',str(output),'--width','384','--layers','12',
             '--ffn-hidden','1024','--qk-vocab','512','--gdn-dism',
             '--steps','3000','--warmup','100','--batch','64','--micro-batch','8',
             '--lr','.001','--weight-decay','.01','--softcap','30','--seed','777',
             '--eval-every','1000','--eval-batches','100','--save-every','500','--log-every','10',
             '--wandb-mode','offline','--wandb-project','dism-activation-study'], 'training','console.log')
        import torch
        ckpt = torch.load(output/'latest.pt', map_location='cpu', weights_only=False)
        assert ckpt['step'] == 3000 and ckpt['config']['parameters'] == 66_593_550
        del ckpt
        status('complete')
    except BaseException as error:
        status('failed', error=str(error))
        raise


if __name__ == '__main__':
    main()
