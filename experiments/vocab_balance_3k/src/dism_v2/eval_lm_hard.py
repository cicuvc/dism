"""Independent pure-hard validation of a completed LM checkpoint, optionally queued."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import time


def wait_for_training(unit):
    if not unit:
        return
    print(json.dumps(dict(event='waiting_for_training', unit=unit)), flush=True)
    while True:
        result = subprocess.run(['systemctl', '--user', 'show', unit,
                                 '--property=ActiveState', '--value'],
                                text=True, capture_output=True)
        state = result.stdout.strip()
        if state in ('inactive', 'failed'):
            return
        if result.returncode:
            raise RuntimeError(f'Cannot inspect training service: {result.stderr.strip()}')
        if state not in ('active', 'activating', 'deactivating', 'reloading'):
            raise RuntimeError(f'Unexpected training service state: {state!r}')
        time.sleep(30)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--batches', type=int, default=100, help='Effective batches, not microbatches')
    parser.add_argument('--wait-unit')
    args = parser.parse_args()
    if args.batches <= 0 or args.output.exists():
        raise ValueError('Require positive batch count and a new report path')
    # Waiting uses no torch import, GPU context, model allocation or data read.
    wait_for_training(args.wait_unit)
    import torch
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    config = saved['config']
    step = saved['step']
    if step != config['steps']:
        raise RuntimeError(f'Training not complete: checkpoint step{step}/{config["steps"]}; refusing early evaluation')
    os.environ['DISM_TILE_LSE'] = config['tile_lse']
    os.environ['DISM_BWD_OPT'] = str(config['backward_opt'])
    from .train_lm import load_tokenizer
    from .lm_model import DecoderLM, LMConfig
    from .lm_data import PackedStream, split_files
    model_config = LMConfig(**config['model'])
    if model_config.architecture != 'hybrid':
        raise ValueError('Pure-hard check requires the hybrid DISM model')
    model = DecoderLM(model_config)
    model.load_state_dict(saved['model'], strict=True)
    del saved
    model = model.cuda().eval()
    tokenizer = load_tokenizer(config['tokenizer'])
    _, files = split_files(config['data'])
    stream = PackedStream(files, tokenizer, context=model_config.context,
                          batch_size=config['micro_batch'], seed=config['seed'], repeat=False)
    generator = torch.Generator(device='cuda').manual_seed(config['seed'] + 2)
    accum = config['batch'] // config['micro_batch']
    torch.cuda.synchronize()
    start = time.perf_counter()
    totals = []
    total = torch.zeros((), device='cuda')
    with torch.no_grad():
        for batch in range(args.batches):
            batch_losses = []
            for _ in range(accum):
                x, y = stream.next_batch()
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    loss = model(x.cuda(), y.cuda(), 1., generator)
                value = loss.item()
                if not math.isfinite(value):
                    raise FloatingPointError(f'Nonfinite pure-hard loss at effective batch{batch}')
                batch_losses.append(value)
                total += loss.float()
            totals.append(sum(batch_losses) / accum)
            if (batch + 1) % 10 == 0:
                print(json.dumps(dict(event='validation_progress', effective_batches=batch + 1,
                                      batch_loss=totals[-1])), flush=True)
    loss = (total / (args.batches * accum)).item()
    if not math.isfinite(loss):
        raise FloatingPointError('Nonfinite aggregate loss')
    torch.cuda.synchronize()
    report = dict(checkpoint=str(args.checkpoint), step=step, hard_prob=1.,
                  effective_batches=args.batches, batch_size=config['batch'],
                  micro_batch=config['micro_batch'], microbatches=args.batches * accum,
                  predicted_tokens=args.batches * config['batch'] * model_config.context,
                  loss=loss, perplexity=math.exp(loss), all_microbatch_losses_finite=True,
                  batch_losses=totals, seconds=time.perf_counter() - start,
                  eval_rng_seed=config['seed'] + 2, tile_lse=config['tile_lse'],
                  softcap=model_config.softcap, gpu=torch.cuda.get_device_name())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as file:
        json.dump(report, file, indent=2)
        file.write('\n')
    print(json.dumps(dict(event='pure_hard_validation_complete', **report)), flush=True)


if __name__ == '__main__':
    main()
