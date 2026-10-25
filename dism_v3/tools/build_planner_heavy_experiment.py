"""Build isolated heavy-chain CPU planner prototypes and a current-source control.

No production headers are edited. --native builds private Python extensions via
this version's inference/build.py. Timed and instrumented binaries are separate.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--native',action='store_true')
    args=ap.parse_args();root=Path(__file__).resolve().parents[2]
    out=args.output.resolve();src=root/'dism_v3/python/flash_dism/inference/csrc'
    if not out.is_relative_to(root/'dism_v3/build') or out.exists():
        ap.error('use a fresh output directory under this worktree dism_v3/build')
    sources=[*sorted(src.glob('*.hpp')),src/'prefill_bind.cpp',src.parent/'build.py',
             Path(__file__),Path(__file__).with_name('planner_heavy_prototype.hpp'),
             Path(__file__).with_name('build_planner_cpu_profile.py'),
             Path(__file__).with_name('profile_planner_native.cpp')]
    snapshot={p:p.read_bytes() for p in sources}
    original=snapshot[src/'prefill.hpp'].decode()
    begin=original.index('    void decompose(');end=original.index('\n  public:',begin)
    extra=snapshot[Path(__file__).with_name('planner_heavy_prototype.hpp')].decode()
    out.mkdir(parents=True);commands=[]
    for name,flag in [('baseline',None),('weighted',1),('binary',0)]:
        folder=out/name/'inference';headers=folder/'csrc';headers.mkdir(parents=True)
        for p in [*src.glob('*.hpp'),src/'prefill_bind.cpp']:
            (headers/p.name).write_bytes(snapshot[p])
        (folder/'build.py').write_bytes(snapshot[src.parent/'build.py'])
        if flag is not None:
            s=f'#define HEAVY_WEIGHTED {flag}\n'+original[:begin]+extra+original[end:]
            (headers/'prefill.hpp').write_text(s)
        for h in ('planner.hpp','prefill_matcher.hpp'):
            p=headers/h;p.write_text(p.read_text()+f'\n// Isolated {name} heavy-chain experiment.\n')
        cmd=[sys.executable,str(Path(__file__).with_name('build_planner_cpu_profile.py')),
             '--output',str(out/name/'profile'),'--source-dir',str(headers)]
        subprocess.run(cmd,check=True);commands.append(cmd)
        if args.native:
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',MAX_JOBS='2',TORCH_EXTENSIONS_DIR=str(out/name/'cache'))
            cmd=[sys.executable,str(folder/'build.py'),'--component','prefill','--output',str(out/name/'_dism_prefill.so')]
            subprocess.run(cmd,env=env,check=True);commands.append(cmd)
        print('built',name,flush=True)
    manifest=dict(revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),commands=commands,
                  sources={str(p.relative_to(root)):hashlib.sha256(data).hexdigest() for p,data in snapshot.items()})
    (out/'build_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')

if __name__=='__main__':main()
