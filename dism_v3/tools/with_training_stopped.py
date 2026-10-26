"""Run a bounded GPU check with an explicitly authorized training PID stopped.

A separate pipe watchdog resumes that same process even if this controller is
killed. No checkpoint changes, restart, process-group stop, or training signals
other than SIGSTOP/SIGCONT. Refuse to take ownership of an already stopped job.
"""
import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def identity(pid):
    text=Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()
    return text[19],text[0]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pid',type=int,required=True)
    p.add_argument('--match',required=True,help='Required substring of the training command line')
    p.add_argument('--timeout',type=float,default=1800)
    p.add_argument('command',nargs=argparse.REMAINDER)
    a=p.parse_args();command=a.command
    if command and command[0]=='--':command=command[1:]
    if not command:p.error('missing command')
    start,state=identity(a.pid)
    cmdline=Path(f'/proc/{a.pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
    if a.match not in cmdline or state in ('T','t','Z','X'):
        raise RuntimeError('PID identity/state does not authorize this stop')
    read_fd,write_fd=os.pipe()
    watchdog_code='''
import os,signal,sys
from pathlib import Path
fd,pid,start=int(sys.argv[1]),int(sys.argv[2]),sys.argv[3]
try:
 while os.read(fd,1):pass
finally:
 try:
  fields=Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()
  if fields[19]==start:
   os.kill(pid,signal.SIGCONT)
   print(f'RESUMED training PID {pid}',flush=True)
 except ProcessLookupError:pass
 except FileNotFoundError:pass
'''
    watchdog=subprocess.Popen([sys.executable,'-c',watchdog_code,str(read_fd),str(a.pid),start],
                              pass_fds=(read_fd,),start_new_session=True)
    os.close(read_fd)
    child=None
    def interrupted(signum,frame):
        raise KeyboardInterrupt(f'controller signal {signum}')
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,interrupted)
    try:
        os.kill(a.pid,signal.SIGSTOP)
        deadline=time.monotonic()+10
        while identity(a.pid)[1] not in ('T','t'):
            if time.monotonic()>deadline:raise TimeoutError('training did not stop')
            time.sleep(.05)
        print(f'STOPPED training PID {a.pid}; running {command}',flush=True)
        child=subprocess.Popen(command,start_new_session=True)
        code=child.wait(timeout=a.timeout)
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGKILL);child.wait()
        os.close(write_fd)
        watchdog.wait(timeout=10)
        if identity(a.pid)[0]!=start:raise RuntimeError('training PID identity changed')
        deadline=time.monotonic()+10
        while identity(a.pid)[1] in ('T','t'):
            if time.monotonic()>deadline:raise RuntimeError('training is still stopped')
            time.sleep(.05)
    raise SystemExit(code)


if __name__=='__main__':main()
