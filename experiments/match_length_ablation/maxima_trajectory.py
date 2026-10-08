"""Collect longest native matching chains for the already-evaluated models."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path('/media/cicuvc/c63abdf1-0e56-4153-9228-95df5a2f239b/cicuvc/dism-runs/local-match-length-20261008')
for name in ('late_true_4000', 'late_true_6000', 'shared_6000', 'shared_12000'):
    source = ROOT / 'trajectory' / name
    protocol = json.loads((source / 'protocol.json').read_text())
    destination = ROOT / 'trajectory' / (name + '_maxima')
    command = [sys.executable, '-u', str(Path(__file__).with_name('run.py')),
               '--workspace', str(ROOT), '--output', str(destination),
               '--checkpoint', protocol['checkpoint'], '--metadata', protocol['metadata'],
               '--expected-sha256', protocol['checkpoint_sha256'],
               '--data', str(ROOT.parent.parent / 'fwedu-100B'), '--batch', '8',
               '--tokens', '4254470', '--modes', 'native,full', '--collect-max-chains']
    print('START', name, flush=True)
    with (ROOT / 'trajectory' / (name + '_maxima.log')).open('w') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    old = json.loads((source / 'results.json').read_text())
    new = json.loads((destination / 'results.json').read_text())
    assert old['input_sha256'] == new['input_sha256']
    assert abs(old['results']['native']['nll']-new['results']['native']['nll']) < 1e-6
    print('DONE', name, flush=True)
