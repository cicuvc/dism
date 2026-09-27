"""User-authorized idle-GPU memory reservation; SIGTERM/SIGINT releases it."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import threading


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--pid-file', type=Path, required=True)
    parser.add_argument('--gib', type=int, default=32)
    args = parser.parse_args()
    # Only GPU aggregate state; do not query any other user's processes.
    state = subprocess.check_output(['nvidia-smi', '-i', str(args.gpu),
        '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
    used, utilization = map(int, state.strip().split(','))
    if used > 64 or utilization != 0:
        raise RuntimeError(f'GPU{args.gpu} no longer idle; refusing reservation')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    import torch
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    reservation = torch.empty(args.gib * 2**30, device='cuda', dtype=torch.uint8)
    args.pid_file.write_text(str(os.getpid()) + '\n')
    print(json.dumps(dict(event='reserved', gpu=args.gpu, gib=args.gib, pid=os.getpid())), flush=True)
    try:
        stop.wait()
    finally:
        del reservation
        torch.cuda.empty_cache()
        args.pid_file.unlink(missing_ok=True)
        print(json.dumps(dict(event='released', gpu=args.gpu)), flush=True)


if __name__ == '__main__':
    main()
