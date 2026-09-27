"""Throttled serial evaluation; does not stop the active training job."""
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    root=Path(__file__).resolve().parents[1]
    study=Path('/media/cicuvc/c63abdf1-0e56-4153-9228-95df5a2f239b/cicuvc/dism-lm-runs/activation-study-3k-20260910')
    out=study/'vocab-large-4096-20260911'
    out.mkdir(exist_ok=False)
    def status(phase,**kw):
        (out/'status.json').write_text(json.dumps(dict(phase=phase,pid=os.getpid(),**kw),indent=2)+'\n')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',MAX_JOBS='4',
             DISM_TILE_LSE='tanh_finite',DISM_BWD_OPT='13',DISM_ROW_BITSET='0',
             DISM_OUTPUT_Q_ALIAS='kv',DISM_OUTPUT_TMA='0')
    try:
        for arm in ('width384_72m','baseline'):
            status('evaluation',arm=arm)
            with (out/(arm+'.log')).open('a') as log:
                subprocess.run([sys.executable,'-u','-m','dism_v2.eval_vocab_large',
                    '--checkpoint',str(study/arm/'latest.pt'),'--output',str(out/arm),
                    '--sequences','4096','--micro-batch','2','--pause-seconds','.2'],
                    cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        reports=[json.loads((out/arm/'report.json').read_text()) for arm in ('width384_72m','baseline')]
        assert reports[0]['packed_input_sha256']==reports[1]['packed_input_sha256']
        status('complete')
    except BaseException as e:
        status('failed',error=str(e))
        raise


if __name__=='__main__':main()
