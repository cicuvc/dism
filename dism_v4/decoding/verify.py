"""Run and retain the concrete acceptance evidence for this standalone backend."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from build import load_cuda

root=Path(__file__).resolve().parent
results=root/'results'
results.mkdir(exist_ok=True)
cuda=Path(os.environ.get('CUDA_HOME','/usr/local/cuda'))/'bin'
module=load_cuda()
checks={}


def run(name,command):
    with (results/(name+'.log')).open('w') as log:
        p=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
    checks[name]=dict(returncode=p.returncode,command=list(map(str,command)))
    if p.returncode:
        print((results/(name+'.log')).read_text()[-5000:]);raise SystemExit(p.returncode)
    print(name,'PASS',flush=True)


for test in ('planner','native','long','api','launch'):
    run(test,[sys.executable,str(root/f'test_{test}.py')])
for tool in ('memcheck','racecheck','synccheck'):
    run(tool,[str(cuda/'compute-sanitizer'),'--tool',tool,'--error-exitcode','99',sys.executable,str(root/'sanitize_smoke.py')])
sass=subprocess.check_output([str(cuda/'cuobjdump'),'--dump-sass',module.__file__],text=True)
resources=subprocess.check_output([str(cuda/'cuobjdump'),'--dump-resource-usage',module.__file__],text=True)
(results/'resources.txt').write_text(resources)
assert not re.search(r'\b(?:CALL|LDL|STL)\b',sass),'device CALL/local-memory operation present'
assert all(int(x)==0 for x in re.findall(r'(?:STACK|LOCAL):(\d+)',resources)),'stack/local memory present'
checks['codegen']=dict(call_instructions=0,local_memory_instructions=0,stack_bytes=0,
                       binary_sha256=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest())
sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir()
         if p.is_file() and p.suffix in ('.py','.cpp','.cu','.cuh','.hpp')}
(results/'verification.json').write_text(json.dumps(dict(checks=checks,source_sha256=sources),indent=2)+'\n')
print('ALL_VERIFIED',flush=True)
