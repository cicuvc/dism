"""Evaluate preserved checkpoint pairs, sequentially on the local GPU."""
import json
import subprocess
import sys
from pathlib import Path

BASE = Path('/media/cicuvc/c63abdf1-0e56-4153-9228-95df5a2f239b/cicuvc/dism-runs')
ARCHIVE = BASE / 'remote-week-20261008'
WORK = BASE / 'local-match-length-20261008'
OUT = WORK / 'trajectory'
OUT.mkdir(exist_ok=True)
SOURCES = [
    ('late_true_4000', ARCHIVE / 'external-checkpoints/dism-gradient-pair-analysis-step4000/model_004000.pt',
     ARCHIVE / 'reports/dism-gradient-pair-analysis/checkpoint/meta_004000.json',
     'c738dde6329dd9cc2dde9dd402191bbff7d99032a2d1327e3e210716c7417bd5'),
    ('late_true_6000', ARCHIVE / 'external-checkpoints/dism-randomness-20261006/dual_late_true/model_006000.pt',
     ARCHIVE / 'reports/dism-randomness/dual_late_true/base_checkpoints/random40m/meta_006000.json',
     '56158b3d16f87ebc4b364debc24312f0e151295173c670ea23b28e76caf7fd84'),
]
for step in (6000, 12000):
    directory = ARCHIVE / 'autodl-tmp/dism-gdn-smallbatch/gdn_all_shared_b32/base_checkpoints/gdnsmall40m'
    SOURCES.append((f'shared_{step}', directory / f'model_{step:06d}.pt', directory / f'meta_{step:06d}.json', None))

for name, checkpoint, metadata, digest in SOURCES:
    destination = OUT / name
    if (destination / 'results.json').exists():
        continue
    command = [sys.executable, '-u', str(Path(__file__).with_name('run.py')),
               '--workspace', str(WORK), '--output', str(destination),
               '--checkpoint', str(checkpoint), '--metadata', str(metadata),
               '--data', str(BASE.parent / 'fwedu-100B'), '--batch', '8', '--tokens', '4254470',
               '--modes', 'native,full,cap1,cap2,cap4,cap1_far,cap1_near,window128,no_dism']
    if digest:
        command += ['--expected-sha256', digest]
    (OUT / 'state.json').write_text(json.dumps(dict(status='running', current=name)))
    print('START', name, flush=True)
    with (OUT / f'{name}.log').open('w') as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        (OUT / 'state.json').write_text(json.dumps(dict(status='failed', current=name, code=result.returncode)))
        raise SystemExit(result.returncode)
    print('DONE', name, flush=True)
(OUT / 'state.json').write_text(json.dumps(dict(status='complete')))
