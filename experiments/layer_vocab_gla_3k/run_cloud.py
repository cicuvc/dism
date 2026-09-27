"""Isolated cloud preflight and 3000-step training; no local queue."""
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    root = Path('/root/autodl-tmp/dism-layer-vocab-gla-3k-20260910')
    previous = Path('/root/autodl-tmp/dism-qk-tied-3k-20260910')
    python = '/root/autodl-tmp/dism-shared-20260910/venv/bin/python'
    state = root / 'status.json'
    with state.open('x') as f:
        json.dump(dict(phase='starting', pid=os.getpid()), f)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', TORCH_CUDA_ARCH_LIST='12.0a',
               PATH=str(Path(python).parent)+':/usr/local/cuda/bin:'+os.environ.get('PATH',''),
               CUDA_HOME='/usr/local/cuda', GLX_ROOT=str(previous/'glx'),
               TORCH_EXTENSIONS_DIR='/root/autodl-tmp/dism-output-gla-3k-20260910/torch-extensions', TRITON_CACHE_DIR=str(root/'triton-cache'),
               MAX_JOBS='8', DISM_TILE_LSE='tanh_finite', DISM_BWD_OPT='13', DISM_ROW_BITSET='0',
               DISM_OUTPUT_Q_ALIAS='kv', DISM_OUTPUT_TMA='0', OMP_NUM_THREADS='8', WANDB_MODE='offline')

    def status(phase, **extra):
        state.write_text(json.dumps(dict(phase=phase, pid=os.getpid(), **extra), indent=2)+'\n')

    def run(args, phase, log):
        status(phase)
        with (root/log).open('a') as f:
            subprocess.run([python, '-u', *args], cwd=root, env=env,
                           stdout=f, stderr=subprocess.STDOUT, check=True)
    try:
        assert not (root/'run/config.json').exists()
        run(['-m','pytest','tests/test_lm_layer_vocab.py','-q','-x'], 'preflight', 'preflight.log')
        run(['-m','dism_v2.train_lm','--output',str(root/'run'),'--dism-output-gla',
             '--dism-share-vocab-layers',
             '--steps','3000','--warmup','100','--batch','64','--micro-batch','8',
             '--lr','.001','--weight-decay','.01','--softcap','30','--seed','777',
             '--stream-url','http://127.0.0.1:18481','--stream-secret-file',str(root/'auth_token'),
             '--eval-every','1000','--eval-batches','100','--save-every','500','--log-every','10',
             '--wandb-mode','offline','--wandb-project','dism-activation-study'], 'training','console.log')
        import torch
        ckpt = torch.load(root/'run/latest.pt', map_location='cpu', weights_only=False)
        assert ckpt['step'] == 3000 and ckpt['config']['parameters'] == 46_024_340
        status('complete')
    except BaseException as error:
        status('failed', error=str(error))
        raise


if __name__ == '__main__':
    main()
