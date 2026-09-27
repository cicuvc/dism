"""Scoped cloud5090 preflight and 3000-step tied-QK training launcher."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    root = Path(__file__).resolve().parent
    os.chdir(root)
    if (root/'job-status.json').exists():
        raise FileExistsError('Existing job: inspect checkpoints before manual recovery')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', TORCH_CUDA_ARCH_LIST='12.0a',
               PATH=str(Path(sys.executable).parent)+':/usr/local/cuda/bin:'+os.environ.get('PATH', ''),
               CUDA_HOME='/usr/local/cuda', GLX_ROOT=str(root/'glx'),
               TORCH_EXTENSIONS_DIR=str(root/'torch-extensions'),
               TRITON_CACHE_DIR=str(root/'triton-cache'), MAX_JOBS='8',
               DISM_TILE_LSE='tanh_finite', DISM_BWD_OPT='13', DISM_ROW_BITSET='0',
               DISM_OUTPUT_Q_ALIAS='kv', DISM_OUTPUT_TMA='0',
               OMP_NUM_THREADS='8', WANDB_MODE='offline')
    state = dict(status='starting', pid=os.getpid(), root=str(root))
    def write():
        (root/'job-status.json').write_text(json.dumps(state, indent=2)+'\n')
    def run(args, filename, phase):
        state['status'] = phase
        write()
        with (root/filename).open('a') as log:
            process = subprocess.Popen([sys.executable, '-u', *args], env=env, stdout=log, stderr=subprocess.STDOUT)
            state['child_pid'] = process.pid
            write()
            code = process.wait()
        if code:
            raise RuntimeError(f'{phase} failed ({code}); see {filename}')
    train = ['-m', 'dism_v2.train_lm', '--output', str(root/'run'),
             '--steps', '3000', '--warmup', '100', '--batch', '64', '--micro-batch', '8',
             '--lr', '.001', '--weight-decay', '.01', '--softcap', '30', '--seed', '777',
             '--dism-tie-qk-vocab', '--ffn-hidden', '1194', '--dism-activation', 'baseline',
             '--stream-url', 'http://127.0.0.1:18476', '--stream-secret-file', str(root/'auth_token'),
             '--eval-every', '1000', '--eval-batches', '100', '--save-every', '500', '--log-every', '10',
             '--wandb-mode', 'offline', '--wandb-project', 'dism-activation-study']
    try:
        run(['-m', 'dism_v2.check_remote_lm', '--data-only', '--stream-url', 'http://127.0.0.1:18476',
             '--secret-file', str(root/'auth_token')], 'data-check.log', 'data_check')
        run(['-m', 'pytest', 'tests/test_lm_vocab_sharing.py', '-q', '-x'],
            'gradient-check.log', 'gradient_check')
        run(train+['--stop-after', '3'], 'console.log', 'smoke')
        import torch
        checkpoint = torch.load(root/'run/latest.pt', map_location='cpu', weights_only=False)
        assert checkpoint['step'] == 3 and checkpoint['config']['parameters'] == 49_676_296
        assert checkpoint['config']['model']['dism_tie_qk_vocab']
        assert checkpoint['config']['model']['dism_vocab_groups'] is None
        del checkpoint
        run(train+['--resume', str(root/'run/latest.pt')], 'console.log', 'training')
        checkpoint = torch.load(root/'run/latest.pt', map_location='cpu', weights_only=False)
        assert checkpoint['step'] == 3000
        state['status'] = 'complete'
    except BaseException as error:
        state['status'], state['error'] = 'failed', str(error)
        raise
    finally:
        state['updated_unix'] = time.time()
        write()


if __name__ == '__main__':
    main()
