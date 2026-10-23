"""Process-local allocator A/B for an isolated native planner profiler binary."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--directory',type=Path,required=True)
    args=ap.parse_args();directory=args.directory.resolve()
    root=Path(__file__).resolve().parents[2]
    if not directory.is_relative_to(root/'dism_v3/build'):
        ap.error('directory must be under dism_v3/build')
    output=directory/'allocator_ab.json'
    if output.exists():raise FileExistsError(output)
    binary=directory/'native';jemalloc='/lib/x86_64-linux-gnu/libjemalloc.so.2'
    variants={'glibc':{},'glibc_tcache64':{'GLIBC_TUNABLES':'glibc.malloc.tcache_count=64'},
              'jemalloc':{'LD_PRELOAD':jemalloc}}
    for symbols in (512,32):
        baseline=directory/f'baseline{symbols}.bin'
        if not baseline.exists():
            subprocess.run([str(binary),str(directory/f'iid{symbols}.bin'),'0','1',str(baseline)],
                           check=True,stdout=subprocess.DEVNULL)
    for name,settings in variants.items():
        for symbols in (512,32):
            target=directory/f'allocator_{name}_{symbols}.bin'
            cmd=[str(binary),str(directory/f'iid{symbols}.bin'),'0','1',str(target)]
            subprocess.run(cmd,env=dict(os.environ,**settings),check=True,stdout=subprocess.DEVNULL)
            expected=(directory/f'baseline{symbols}.bin').read_bytes()
            assert target.read_bytes()==expected,(name,symbols)
    jobs=[(symbols,workers,name,rep) for rep in range(3) for symbols in (512,32)
          for workers in (1,16) for name in variants]
    random.Random(20261023).shuffle(jobs)
    rows=[]
    for symbols,workers,name,rep in jobs:
        cmd=[str(binary),str(directory/f'iid{symbols}.bin'),'1.5',str(workers)]
        if workers==1:cmd=['/usr/bin/taskset','-c','0']+cmd
        text=subprocess.check_output(cmd,env=dict(os.environ,**variants[name]),text=True)
        row=dict(symbols=symbols,workers=workers,allocator=name,repeat=rep,**json.loads(text))
        rows.append(row)
        output.write_text(json.dumps(dict(binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
            byte_exact_output_checks=True,rows=rows),indent=2)+'\n')
        print(symbols,workers,name,rep,row['ms_per_iteration'],flush=True)

if __name__=='__main__':main()
