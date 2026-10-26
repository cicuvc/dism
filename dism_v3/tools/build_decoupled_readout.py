"""Build the isolated K-block/Q-tile CPU converter via the v3 build helper."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--topk-headers',type=Path,
                    help='optional frozen coefficient-cache topk headers; default patches worktree sources in isolation')
    args=ap.parse_args();root=Path(__file__).resolve().parents[1]
    out=args.output.resolve();src=out/'inference/csrc';src.mkdir(parents=True,exist_ok=True)
    headers=args.topk_headers or root/'python/flash_dism/inference/csrc'
    paths=[*headers.glob('*.hpp'),*[Path(__file__).with_name(name) for name in
           ('decoupled_readout_bind.cpp','decoupled_readout_plan.hpp','decoupled_readout_direct.hpp')]]
    for p in paths:shutil.copy2(p,src/p.name)
    if args.topk_headers is None:
        # Preserve the previously checked top-k CPU experiment as a source
        # delta, so rebuilding does not depend on an ignored artifact tree.
        patch=Path(__file__).with_name('decoupled_topk.patch').resolve()
        for name in ('prefill.hpp','prefill_training.hpp'):
            shutil.copy2(src/name,src/('topk_'+name))
        subprocess.run(['patch','--batch','--fuzz=0','-p0','-i',str(patch)],cwd=src,check=True)
        paths.append(patch)
    # The captured top-k prototype has an A/B switch for the already validated
    # coefficient caches. Both control and decoupled plans use the cached side.
    p=src/'topk_prefill_training.hpp';s=p.read_text()
    if 'inline bool cache_forward_switch=false;' not in s:raise ValueError('use the validated coefficient-cache topk header snapshot')
    p.write_text(s.replace('inline bool cache_forward_switch=false;','inline bool cache_forward_switch=true;'))
    helper=root/'python/flash_dism/inference/build.py';shutil.copy2(helper,src.parent/'build.py')
    os.environ['MAX_JOBS']='2';os.environ['TORCH_EXTENSIONS_DIR']=str(out/'cache')
    spec=importlib.util.spec_from_file_location('decoupled_build',src.parent/'build.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    binary=mod._load('_dism_decoupled',['decoupled_readout_bind.cpp'],force=True)
    shutil.copy2(binary.__file__,out/'_dism_decoupled.so')
    (out/'sources.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [*paths,helper,Path(__file__)]},indent=2)+'\n')
    print(out,flush=True)


if __name__=='__main__':main()
