"""Serial local training queue; advance only after successful full completion."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('queue_dir',type=Path)
    args=parser.parse_args();root=args.queue_dir.resolve()
    lock=(root/'queue.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan=json.loads((root/'plan.json').read_text())
    state={'pid':os.getpid(),'status':'waiting','completed':[]}
    def save():
        tmp=root/'state.tmp';tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(root/'state.json')
    def wait(run):
        state.update(waiting_for=str(run));save()
        while not (run/'exit_code.txt').exists():time.sleep(30)
        code=int((run/'exit_code.txt').read_text())
        if code!=0:raise RuntimeError(f'{run}: exit {code}; queue will not advance')
        # Exit zero alone is insufficient: require a final checkpoint.
        spec=json.loads((run/'model_spec.json').read_text())
        checkpoints=list((run/'base_checkpoints').glob('*/model_022888.pt'))
        if len(checkpoints)!=1:raise RuntimeError(f'{run}: missing final step 22888 checkpoint')
    try:
        wait(Path(plan['wait_for']))
        for path in plan['jobs']:
            run=Path(path)
            if (run/'started_at.txt').exists():
                raise RuntimeError(f'{run}: already started; inspect before restarting queue')
            state.update(status='running',current=str(run));save()
            with (run/'logs/train.log').open('ab') as log:
                child=subprocess.Popen(['/bin/bash',str(run/'supervise.sh')],cwd=run,
                    stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            (run/'supervisor.pid').write_text(str(child.pid)+'\n')
            state['supervisor_pid']=child.pid;save()
            code=child.wait()
            if code:raise RuntimeError(f'{run}: supervisor exit {code}')
            wait(run);state['completed'].append(str(run));save()
        state.update(status='finished',current=None,waiting_for=None);save()
    except BaseException as exc:
        state.update(status='failed',error=repr(exc));save();raise


if __name__=='__main__':main()
