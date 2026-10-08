"""Reproducible sequential sweep; no concurrent GPU benchmark processes."""
import json
import os
from pathlib import Path
import subprocess
import sys

root=Path(__file__).parent
output=root/'results'/'sweep'
output.mkdir(parents=True,exist_ok=True)
cases=[
    ('2k_default',2048,128,128,160,128),
    ('2k_tail512',2048,512,128,160,128),
    ('2k_more_matrices',2048,512,32,64,128),
    ('2k_fewer_matrices',2048,512,256,320,128),
    ('2k_chunk512',2048,512,128,160,512),
    ('16k_default',16384,512,128,160,128),
    ('16k_chunk512',16384,512,128,160,512),
    ('16k_tail1024',16384,1024,128,160,512),
]
rows=[]
for name,n,b,k,t,c in cases:
    path=output/(name+'.json')
    with (output/(name+'.log')).open('w') as log:
        subprocess.run([sys.executable,str(root/'benchmark.py'),'--n',str(n),'--steps',str(b+1),
            '--pattern','zipf','--interval',str(b),'--sample',str(k),'--threshold',str(t),
            '--chunk',str(c),'--output',str(path)],stdout=log,stderr=subprocess.STDOUT,check=True)
    row=json.loads(path.read_text());row['name']=name;rows.append(row)
    (output/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')
    print(name,'us:',round(row['amortized_us'],2),'linear:',round(row['linear_mean_us'],2),
          'summary MiB:',round(row['live']['summary_bytes']/2**20,3),'peak MiB:',round(row['peak_allocated_bytes']/2**20,2),flush=True)
