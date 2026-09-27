"""GDC 72M pretraining: 2-GPU DDP on fixed GPUs 6,7; fails fast if busy."""
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parent
    status = root / 'status.json'
    if status.exists():
        raise FileExistsError('Existing run: inspect manually rather than restart')

    def record(phase, **extra):
        status.write_text(json.dumps(dict(phase=phase, pid=os.getpid(), **extra), indent=2))

    gpus = [6, 7]
    try:
        s = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
             '--format=csv,noheader,nounits'], text=True)
        busy = {int(i) for i, m, u in (line.split(',') for line in s.splitlines())
                if int(m) >= 256 or int(u) > 10}
        if busy & set(gpus):
            raise RuntimeError(f'GPUs {sorted(busy & set(gpus))} busy; refusing to start')
        env = dict(os.environ,
                   CUDA_VISIBLE_DEVICES=','.join(map(str, gpus)),
                   OMP_NUM_THREADS='8', WANDB_MODE='offline',
                   NO_PROXY='127.0.0.1,localhost',
                   TRITON_CACHE_DIR=str(root / 'triton-cache'))

        def run(args, phase, log, check=True):
            record(phase, gpus=gpus)
            with (root / log).open('a') as f:
                subprocess.run([sys.executable, '-u', *args], cwd=root, env=env,
                               stdout=f, stderr=subprocess.STDOUT, check=check)

        run(['-m', 'dism_v2.check_remote_lm', '--data-only',
             '--stream-url', 'http://127.0.0.1:18488',
             '--secret-file', str(root / 'auth_token')], 'stream_preflight', 'stream.log')
        run(['test_gdc.py'], 'gpu_preflight', 'preflight.log')
        run(['-m', 'torch.distributed.run', '--nproc_per_node=2',
             '--master_port', '29617', '-m', 'dism_v2.train_lm',
             '--output', str(root / 'run'),
             '--gdc', '--width', '384', '--layers', '12', '--ffn-hidden', '1664',
             '--steps', '30000', '--warmup', '1000',
             '--batch', '64', '--micro-batch', '8',
             '--lr', '.001', '--weight-decay', '.01', '--softcap', '30',
             '--seed', '777', '--eval-every', '1000', '--eval-batches', '100',
             '--save-every', '1000', '--log-every', '10',
             '--wandb-mode', 'offline', '--wandb-project', 'dism-finewebedu',
             '--stream-url', 'http://127.0.0.1:18488',
             '--stream-secret-file', str(root / 'auth_token')], 'training', 'console.log')
        import torch
        saved = torch.load(root / 'run/latest.pt', map_location='cpu', weights_only=False)
        assert saved['step'] == 30000 and saved['config']['parameters'] == 72_420_312
        record('complete', gpus=gpus)
    except BaseException as e:
        record('failed', error=str(e))
        raise


if __name__ == '__main__':
    main()
