"""Launch independent group2 run, reusing verified task-owned code/environment."""
import os
from pathlib import Path
import subprocess
import json

root = Path('/root/autodl-tmp/dism-qk-group2-3k-20260910')
code = Path('/root/autodl-tmp/dism-qk-tied-3k-20260910')
python = '/root/autodl-tmp/dism-shared-20260910/venv/bin/python'
assert not (root/'run/config.json').exists()
env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', TORCH_CUDA_ARCH_LIST='12.0a',
           PATH=str(Path(python).parent)+':/usr/local/cuda/bin:'+os.environ.get('PATH',''),
           CUDA_HOME='/usr/local/cuda', GLX_ROOT=str(code/'glx'),
           TORCH_EXTENSIONS_DIR=str(code/'torch-extensions'), TRITON_CACHE_DIR=str(root/'triton-cache'),
           MAX_JOBS='8', DISM_TILE_LSE='tanh_finite', DISM_BWD_OPT='13',
           DISM_ROW_BITSET='0', DISM_OUTPUT_Q_ALIAS='kv', DISM_OUTPUT_TMA='0',
           OMP_NUM_THREADS='8', WANDB_MODE='offline')
args = [python, '-u', '-m', 'dism_v2.train_lm', '--output', str(root/'run'),
        '--steps','3000','--warmup','100','--batch','64','--micro-batch','8',
        '--lr','.001','--weight-decay','.01','--softcap','30','--seed','777',
        '--dism-vocab-groups','2','--ffn-hidden','1194','--dism-activation','baseline',
        '--stream-url','http://127.0.0.1:18479','--stream-secret-file',str(root/'auth_token'),
        '--eval-every','1000','--eval-batches','100','--save-every','500','--log-every','10',
        '--wandb-mode','offline','--wandb-project','dism-activation-study']
with (root/'console.log').open('a') as log:
    child = subprocess.Popen(args,cwd=code,env=env,stdout=log,stderr=subprocess.STDOUT,
                             stdin=subprocess.DEVNULL,start_new_session=True)
(root/'launch.json').write_text(json.dumps(dict(pid=child.pid,command=args,code=str(code)),indent=2)+'\n')
print(child.pid)
