"""Sequential three-arm 3000-update study; stop the queue on any failed arm.

Launch as a user service for persistence. No automatic retries or GPU reservations.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

MODES = ('baseline', 'vocab_silu', 'no_qk_silu')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    root = a.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if (root/'manifest.json').exists():
        raise FileExistsError('Existing study: inspect checkpoints before explicit manual resume')
    repo = Path(__file__).resolve().parent.parent
    files = sorted((repo/'dism_v2').glob('*.py'))+sorted((repo/'dism_v2/csrc').glob('*'))
    files = [f for f in files if f.is_file()]
    hashes = {str(f.relative_to(repo)): hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
    snapshot = root/'sources'
    for name in hashes:
        target = snapshot/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repo/name, target)
    manifest = dict(modes=MODES, steps=3000, warmup=100, batch=64, micro_batch=8,
                    context=2048, parameters=49678876, tokens_per_arm=3000*64*2048,
                    tokens_per_parameter=3000*64*2048/49678876, seed=777,
                    lr=1e-3, weight_decay=.01, hard_schedule='0 to1 over3000 updates',
                    source_hashes=hashes, status='starting', completed=[])
    def write():
        (root/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    write()
    child = None
    stopping = False
    def stop(_signal, _frame):
        nonlocal stopping
        stopping = True
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    env = dict(os.environ, DISM_TILE_LSE='tanh_finite', DISM_BWD_OPT='13',
               DISM_ROW_BITSET='0', DISM_OUTPUT_Q_ALIAS='kv', DISM_OUTPUT_TMA='0',
               WANDB_MODE='offline', OMP_NUM_THREADS='8', TOKENIZERS_PARALLELISM='true')
    def run(command, log):
        nonlocal child
        if stopping:
            raise RuntimeError('Study interrupted; no subsequent arm will start')
        for name, digest in hashes.items():
            if hashlib.sha256((repo/name).read_bytes()).hexdigest() != digest:
                raise RuntimeError(f'Source changed during study: {name}; refusing mixed-code comparison')
        print(json.dumps({'event': 'launch', 'command': command, 'log': str(log)}), flush=True)
        with log.open('a') as output:
            child = subprocess.Popen(command, cwd=repo, env=env, stdout=output, stderr=subprocess.STDOUT)
            manifest['child_pid'] = child.pid
            write()
            code = child.wait()
        child = None
        if code or stopping:
            raise RuntimeError(f'Child failed/interrupted: {code}; see {log}')
    def train(mode, smoke=False):
        dest = root/mode
        dest.mkdir(exist_ok=True)
        command = [sys.executable, '-u', '-m', 'dism_v2.train_lm',
                   '--output', str(dest), '--steps', '3000', '--warmup', '100',
                   '--batch', '64', '--micro-batch', '8', '--lr', '.001',
                   '--weight-decay', '.01', '--softcap', '30', '--seed', '777',
                   '--dism-activation', mode, '--wandb-mode', 'offline',
                   '--wandb-project', 'dism-activation-study', '--eval-every', '1000',
                   '--eval-batches', '100', '--save-every', '500', '--log-every', '10']
        if smoke:
            command += ['--stop-after', '3']
        else:
            command += ['--resume', str(dest/'latest.pt')]
        manifest['status'] = ('smoke:' if smoke else 'training:')+mode
        write()
        run(command, dest/'console.log')
        import torch
        saved = torch.load(dest/'latest.pt', map_location='cpu', weights_only=False)
        assert saved['step'] == (3 if smoke else 3000), 'Incomplete training: refusing to advance queue'
        assert saved['config']['parameters'] == manifest['parameters']
        del saved
    try:
        # All arms must pass real full-size forward/backward/AdamW before long runs.
        for mode in MODES:
            train(mode, smoke=True)
        for mode in MODES:
            train(mode)
            dest = root/mode
            manifest['status'] = 'vocab_eval:'+mode
            write()
            run([sys.executable, '-u', '-m', 'dism_v2.eval_vocab_load',
                 '--checkpoint', str(dest/'latest.pt'), '--output', str(dest/'vocab-load-final'),
                 '--sequences', '256', '--micro-batch', '8'], dest/'vocab-eval.log')
            manifest['completed'].append(mode)
            write()
        manifest['status'] = 'complete'
    except BaseException as error:
        manifest['status'] = 'stopped' if stopping else 'failed'
        manifest['error'] = str(error)
        raise
    finally:
        manifest['updated_unix'] = time.time()
        write()


if __name__ == '__main__':
    main()
