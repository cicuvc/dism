import os
from pathlib import Path
import subprocess
import sys

root=Path(__file__).resolve().parent
for task,result in [('copy','results-copy-bias0'),('abba','results-abba-bias0')]:
    env=dict(os.environ,DELTA_INIT_BIAS='0',DELTA_TASK=task,DELTA_RESULTS=result)
    subprocess.run([sys.executable,'-u',str(root/'run.py')],env=env,check=True)
