"""Paired prefill length extrapolation on identical held-out packed tokens."""
import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import time

import torch


def plot_results(output, losses, lengths, training_context):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False})
    longest = max(lengths)
    fig, axes = plt.subplots(3, 1, figsize=(11, 12), constrained_layout=True)
    def curve(ax, values, label, block=64):
        # Average within sequence first: don't treat correlated tokens as IID.
        binned = values.double().reshape(values.shape[0], -1, block).mean(-1)
        mean = binned.mean(0).numpy()
        se = (binned.std(0) / values.shape[0] ** .5).numpy()
        x = np.arange(len(mean)) * block + (block + 1) / 2
        line, = ax.plot(x, mean, label=label, linewidth=1.5)
        ax.fill_between(x, mean-1.96*se, mean+1.96*se, color=line.get_color(), alpha=.13)
    curve(axes[0], losses[longest], f'Full {longest} prefill')
    curve(axes[0], losses[training_context], f'Reset every {training_context}')
    axes[0].axvline(training_context, color='black', linestyle='--', label='Training length')
    axes[0].set(title='Held-out per-position NLL (64-position bins)', ylabel='NLL (nats)', xlabel='Target position')
    axes[0].legend()
    for length in lengths:
        # First block only retains original sequence as sampling unit.
        curve(axes[1], losses[length][:, :length], f'Prefill {length}')
    axes[1].axvline(training_context, color='black', linestyle='--')
    axes[1].set(title='Same-prefix length comparison (first block only)', ylabel='NLL (nats)', xlabel='Target position')
    axes[1].legend(ncol=3)
    curve(axes[2], losses[longest] - losses[training_context], 'Full minus reset2048')
    axes[2].axhline(0, color='black', linewidth=.8)
    axes[2].axvline(training_context, color='black', linestyle='--')
    axes[2].set(title='Paired context effect: negative means full context helps', ylabel='Delta NLL (nats)', xlabel='Target position')
    fig.savefig(output / 'per_position.png', dpi=180)
    fig.savefig(output / 'per_position.svg')
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    for suffix, start in [('All identical target tokens', 0), ('Positions > 2048 only', training_context)]:
        means, errors = [], []
        for length in lengths:
            per_sequence = losses[length][:, start:].double().mean(-1)
            means.append(per_sequence.mean().item())
            errors.append(1.96 * per_sequence.std().item() / per_sequence.numel() ** .5)
        ax.errorbar(lengths, means, yerr=errors, marker='o', capsize=3, label=suffix)
    ax.axvline(training_context, color='black', linestyle='--', label='Training length')
    ax.set_xscale('log', base=2)
    ax.set_xticks(lengths, [str(x) for x in lengths])
    ax.set(xlabel='Independent prefill block length (no carry between blocks)', ylabel='NLL (nats)',
           title='Length extrapolation on matched tokens')
    ax.legend()
    fig.savefig(output / 'length_generalization.png', dpi=180)
    fig.savefig(output / 'length_generalization.svg')
    plt.close(fig)


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--lengths', nargs='+', type=int, default=[512, 1024, 2048, 4096, 8192])
    p.add_argument('--sequences', type=int, default=256)
    p.add_argument('--micro-batch', type=int, default=4)
    a = p.parse_args()
    lengths = sorted(set(a.lengths))
    if a.output.exists() or min(lengths) < 64 or any(max(lengths) % n or n % 64 for n in lengths):
        p.error('Use a new output directory and multiples of64 dividing max length')
    if a.micro_batch <= 0 or a.sequences < 2 or a.sequences % a.micro_batch:
        p.error('Need >=2 sequences divisible by positive micro-batch')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    saved = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    config, step = saved['config'], saved['step']
    os.environ['DISM_TILE_LSE'] = config['tile_lse']
    os.environ['DISM_BWD_OPT'] = str(config['backward_opt'])
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    from .lm_model import DecoderLM, LMConfig
    from .train_lm import load_tokenizer
    from .lm_data import PackedStream, split_files
    from .eval_lm_positions import evaluate_context
    original = LMConfig(**config['model'])
    if original.context not in lengths or original.architecture != 'hybrid':
        raise ValueError('Requires original hybrid and training context in lengths')
    # Only table capacity/length guard changes. Same RoPE frequencies, no scaling.
    model = DecoderLM(replace(original, context=max(lengths)))
    model.load_state_dict(saved['model'], strict=True)
    del saved
    model.cuda().eval()
    tokenizer = load_tokenizer(config['tokenizer'])
    _, files = split_files(config['data'])
    stream = PackedStream(files, tokenizer, context=max(lengths), batch_size=a.micro_batch, repeat=False)
    a.output.mkdir(parents=True)
    losses = {n: [] for n in lengths}
    masks = []
    digest = hashlib.sha256()
    start = time.perf_counter()
    for batch in range(a.sequences // a.micro_batch):
        x, y = stream.next_batch()
        digest.update(x.numpy().tobytes()); digest.update(y.numpy().tobytes())
        masks.append(y != tokenizer.eos_token_id)
        gx, gy = x.cuda(), y.cuda()
        for n in lengths:
            value = evaluate_context(model, gx, gy, 1., 779, reset_context=n, head_chunk=256)
            if not torch.isfinite(value).all():
                raise FloatingPointError(f'Nonfinite NLL at batch{batch}, length{n}')
            losses[n].append(value)
        if (batch + 1) % 4 == 0 or batch == 0:
            print(json.dumps(dict(sequences=(batch+1)*a.micro_batch, total=a.sequences,
                                  seconds=time.perf_counter()-start)), flush=True)
    losses = {n: torch.cat(v) for n, v in losses.items()}
    mask = torch.cat(masks)
    def stats(v):
        per_sequence = v.double().mean(-1)
        return dict(mean=per_sequence.mean().item(), stderr=per_sequence.std().item()/a.sequences**.5)
    report = dict(checkpoint=str(a.checkpoint), step=step, hard_prob=1., sequences=a.sequences,
                  unique_target_tokens=a.sequences*max(lengths), lengths=lengths,
                  training_context=original.context, swa_window=original.window,
                  rope='unchanged frequencies; enlarged nonpersistent table only',
                  softcap=original.softcap, tile_lse=config['tile_lse'],
                  nll='BF16 model/head, FP32 torch softcap and cross_entropy; natural logs',
                  validation_files=files, packed_xy_sha256=digest.hexdigest(),
                  seconds=time.perf_counter()-start, results={},
                  caveats=['Packed documents with EOS; not a long-single-document retrieval benchmark.',
                           'Blocks reset all layers/states and RoPE positions; same targets for every length.',
                           '95% plot bands use sequences as units; packed sequences may still be correlated.',
                           'More stable NLL alone does not establish useful long-range dependency.'])
    for n, v in losses.items():
        delta = v - losses[original.context]
        positions = torch.arange(max(lengths))
        warm = (positions >= original.context) & (positions % original.context >= 512)
        report['results'][n] = dict(all_tokens=stats(v), beyond_training_length=stats(v[:, original.context:]),
            first_block=stats(v[:, :n]), non_eos_mean=v[mask].double().mean().item(),
            paired_delta_vs_reset2048=stats(delta),
            paired_delta_beyond2048=stats(delta[:, original.context:]),
            paired_delta_beyond2048_exclude_first512_after_reset=stats(delta[:, warm]),
            position_mean=v.double().mean(0).tolist(),
            prefix2048_nll=stats(v[:, :original.context]))
    torch.save(dict(losses=losses, non_eos_mask=mask), a.output / 'per_token.pt')
    (a.output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    plot_results(a.output, losses, lengths, original.context)
    print(json.dumps({n: {k:v for k,v in r.items() if k!='position_mean'} for n,r in report['results'].items()}), flush=True)


if __name__ == '__main__':
    main()
