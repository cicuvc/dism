"""Preflight and train384-wide [GDN,GDN,GDN,DISM] x3 without SWA."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    # Independent frozen source; no live-run code is modified.
    source = Path(__file__).resolve().parent/'src'
    tests = Path(__file__).resolve().parent/'test_matched.py'
    study = Path('/media/cicuvc/c63abdf1-0e56-4153-9228-95df5a2f239b/cicuvc/dism-lm-runs/activation-study-3k-20260910')
    output = study.parent/'gdn-dism-vocab-silu-384-72m-30k-20260911'
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
                   TORCH_EXTENSIONS_DIR=str(study/'gdn_dism_384/torch-extensions'),
                   TRITON_CACHE_DIR=str(output/'triton-cache'), PYTHONPATH=str(source))

        def run(args, phase, log):
            status(phase)
            with (output/log).open('a') as f:
                subprocess.run([sys.executable, '-u', *args], cwd=source, env=env,
                               stdout=f, stderr=subprocess.STDOUT, check=True)

        run(['-m','pytest',str(tests),'-q','-s','-x'], 'gpu_preflight','preflight.log')
        run(['-m','dism_v2.train_lm','--output',str(output),'--width','384','--layers','12',
             '--ffn-hidden','1428','--qk-vocab','512','--gdn-dism','--dism-activation','vocab_silu',
             '--steps','30000','--warmup','1000','--batch','64','--micro-batch','8',
             '--lr','.001','--weight-decay','.01','--softcap','30','--seed','777',
             '--eval-every','1000','--eval-batches','100','--save-every','1000','--log-every','10',
             '--wandb-mode','offline','--wandb-project','dism-finewebedu'], 'training','console.log')
        import torch
        ckpt = torch.load(output/'latest.pt', map_location='cpu', weights_only=False)
        assert ckpt['step'] == 30000 and ckpt['config']['parameters'] == 72_188_142
        del ckpt
        status('complete')
    except BaseException as error:
        status('failed', error=str(error))
        raise


if __name__ == '__main__':
    main()
