"""Restartable local downstream sweep using a frozen explicit manifest.

Runs one bounded-memory evaluator at a time, independently of training.
The manifest records exact checkpoints, contexts, tasks and source hashes.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args(); root = args.output.resolve()
    lock = (root/'sweep.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = json.loads((root/'manifest.json').read_text())
    fingerprint = digest(root/'manifest.json')
    state = dict(pid=os.getpid(), status='starting', started=time.time(), completed=[],
                 smoke_completed=[], manifest_sha256=fingerprint, total=len(manifest['jobs']))

    def save():
        tmp = root/'state.tmp'; tmp.write_text(json.dumps(state, indent=2)+'\n')
        tmp.replace(root/'state.json')

    def sources_unchanged():
        for path, expected in manifest['source_sha256'].items():
            if digest(path) != expected:
                raise RuntimeError(f'Evaluation source changed: {path}')

    def collect():
        records = []
        for job in manifest['jobs']:
            result = root/job['key']/'results.json'
            if result.exists() and (result.parent/'provenance.json').exists():
                data = json.loads(result.read_text())
                records.append(dict(**job, results=data['results']))
        tmp = root/'summary.tmp'
        tmp.write_text(json.dumps(dict(status=state['status'], records=records), indent=2)+'\n')
        tmp.replace(root/'summary.json')

    def run(job, smoke=False):
        sources_unchanged()
        key = f"smoke/{job['model']}_ctx{job['context']}" if smoke else job['key']
        out = root/key; out.mkdir(parents=True, exist_ok=True)
        expected = dict(manifest_sha256=fingerprint, model=job['model'],
                        checkpoint=manifest['models'][job['model']]['checkpoint'],
                        context=job['context'], task=job['task'], smoke=smoke)
        prov = out/'provenance.json'
        artifact = out/('protocol.json' if smoke else 'results.json')
        if prov.exists() and artifact.exists():
            previous = json.loads(prov.read_text())
            if any(previous.get(k) != v for k, v in expected.items()):
                raise RuntimeError(f'Provenance mismatch: {out}')
        else:
            if artifact.exists():
                raise RuntimeError(f'Uncommitted result exists, inspect before retry: {artifact}')
            state.update(status='preflight' if smoke else 'running', current=key,
                         current_started=time.time()); save()
            cmd = [sys.executable, '-u', manifest['adapter'],
                   '--checkpoint', expected['checkpoint'], '--tokenizer', manifest['tokenizer'],
                   '--output', str(out), '--tasks', job['task'],
                   '--batch-size', str(manifest['batch_size']), '--context', str(job['context']),
                   '--rope', 'native', '--memory-fraction', str(manifest['memory_fraction'])]
            if smoke:
                cmd.append('--smoke-only')
            print('START', key, flush=True)
            with (out/'console.log').open('a') as log:
                child = subprocess.Popen(cmd, cwd=manifest['eval_root'],
                    env=dict(os.environ, **manifest['environment']), stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT)
                state['evaluation_pid'] = child.pid; save()
                code = child.wait()
            state['evaluation_pid'] = None
            if code or not artifact.exists():
                raise RuntimeError(f'{key}: exit {code}; see {out}/console.log')
            expected.update(finished=time.time(), elapsed_seconds=time.time()-state['current_started'])
            prov.write_text(json.dumps(expected, indent=2)+'\n')
            print('DONE', key, flush=True)
        state['smoke_completed' if smoke else 'completed'].append(key); save()
        if not smoke:
            collect()

    try:
        save(); sources_unchanged()
        for info in manifest['models'].values():
            if digest(info['checkpoint']) != info['checkpoint_sha256']:
                raise RuntimeError(f"Checkpoint changed: {info['checkpoint']}")
        for model in manifest['models']:
            run(dict(model=model, context=8192, task='pg19'), smoke=True)
        for job in manifest['jobs']:
            run(job)
        state.update(status='finished', current=None, evaluation_pid=None, finished=time.time())
        save(); collect()
    except BaseException as exc:
        state.update(status='failed', error=repr(exc)); save(); collect()
        raise


if __name__ == '__main__':main()
