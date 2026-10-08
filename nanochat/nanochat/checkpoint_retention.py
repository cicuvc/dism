"""Single-rank rolling checkpoints: publish metadata last, then retire old sets."""
import json,os,shutil
from pathlib import Path
import torch

def complete_steps(directory):
    d=Path(directory);out=[]
    for p in d.glob('meta_*.json'):
        try:
            step=int(p.stem.split('_')[1]);m=json.loads(p.read_text())
            names=[f'model_{step:06d}.pt',f'optim_{step:06d}_rank0.pt',f'train_state_{step:06d}_rank0.pt']
            if m['step']==step and all((d/n).is_file() and (d/n).stat().st_size>0 for n in names):out.append(step)
        except (ValueError,KeyError,json.JSONDecodeError):continue
    return sorted(out)

def validate_step(directory,step):
    d=Path(directory)
    for name in [f'model_{step:06d}.pt',f'optim_{step:06d}_rank0.pt',f'train_state_{step:06d}_rank0.pt']:
        value=torch.load(d/name,map_location='cpu',weights_only=False)
        assert isinstance(value,dict),name
        del value
    assert json.loads((d/f'meta_{step:06d}.json').read_text())['step']==step

def prune(directory,keep=2):
    d=Path(directory);steps=complete_steps(d)
    if len(steps)<=keep:return
    validate_step(d,steps[-1])
    removed=[]
    for step in steps[:-keep]:
        for name in [f'model_{step:06d}.pt',f'optim_{step:06d}_rank0.pt',f'train_state_{step:06d}_rank0.pt',f'meta_{step:06d}.json']:
            p=d/name;removed.append(dict(path=str(p),bytes=p.stat().st_size));p.unlink()
    with (d/'retention.jsonl').open('a') as f:f.write(json.dumps(dict(kept=steps[-keep:],removed=removed))+'\n')

def check_space(directory):
    d=Path(directory);d.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(d).free < 1024**3:raise OSError('Less than1GiB free before checkpoint; latest completed checkpoint retained')
