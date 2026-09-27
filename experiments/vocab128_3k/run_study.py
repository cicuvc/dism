"""Wait for vocab normalization, then run the original model with128 codewords."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    source = Path(__file__).resolve().parent/'src'
    study = Path('/media/cicuvc/c63abdf1-0e56-4153-9228-95df5a2f239b/cicuvc/dism-lm-runs/activation-study-3k-20260910')
    output = study/'vocab128'
    output.mkdir(exist_ok=True)
    state = output/'queue-status.json'
    with state.open('x') as f:
        json.dump(dict(phase='waiting_for_vocab_norm', pid=os.getpid()), f)

    def status(phase, **extra):
        temporary = output/'queue-status.tmp'
        temporary.write_text(json.dumps(dict(phase=phase, pid=os.getpid(), **extra), indent=2)+'\n')
        temporary.replace(state)

    try:
        while True:
            # Tolerate an in-progress status write.
            try:
                prior = json.loads((study/'vocab_norm/queue-status.json').read_text())
            except json.JSONDecodeError:
                time.sleep(1)
                continue
            if prior['phase'] == 'complete':
                break
            if prior['phase'] in ('failed', 'stopped'):
                raise RuntimeError('Vocabulary normalization failed/stopped; manual inspection required')
            time.sleep(30)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='8',
                   DISM_TILE_LSE='tanh_finite', DISM_BWD_OPT='13', DISM_ROW_BITSET='0',
                   DISM_OUTPUT_Q_ALIAS='kv', DISM_OUTPUT_TMA='0', MAX_JOBS='8', WANDB_MODE='offline',
                   TORCH_EXTENSIONS_DIR=str(output/'torch-extensions'), TRITON_CACHE_DIR=str(output/'triton-cache'))

        def run(args, phase, log):
            status(phase)
            with (output/log).open('a') as f:
                subprocess.run([sys.executable, '-u', *args], cwd=source, env=env,
                               stdout=f, stderr=subprocess.STDOUT, check=True)

        run(['-m','pytest','tests/test_lm_vocab128.py','-q','-x'], 'gpu_preflight','preflight.log')
        run(['-m','dism_v2.train_lm','--output',str(output),'--qk-vocab','128',
             '--steps','3000','--warmup','100','--batch','64','--micro-batch','8',
             '--lr','.001','--weight-decay','.01','--softcap','30','--seed','777',
             '--eval-every','1000','--eval-batches','100','--save-every','500','--log-every','10',
             '--wandb-mode','offline','--wandb-project','dism-activation-study'], 'training','console.log')
        import torch
        ckpt = torch.load(output/'latest.pt', map_location='cpu', weights_only=False)
        assert ckpt['step'] == 3000 and ckpt['config']['parameters'] == 46_729_756
        del ckpt
        run(['-m','dism_v2.eval_vocab_load','--checkpoint',str(output/'latest.pt'),
             '--output',str(output/'vocab-load-final'),'--sequences','256','--micro-batch','8'],
            'vocab_eval','vocab-eval.log')
        status('complete')
    except BaseException as error:
        status('failed', error=str(error))
        raise


if __name__ == '__main__':
    main()
