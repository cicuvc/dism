"""Paired held-out per-position NLL; no optimizer updates or training mutations."""
import argparse
from dataclasses import asdict
import csv
import hashlib
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from .train_lm import load_tokenizer
from .lm_data import PackedStream, split_files
from .lm_model import DecoderLM, LMConfig


class PositionMoments:
    def __init__(self, length):
        self.total = torch.zeros(length, dtype=torch.float64)
        self.squares = torch.zeros_like(self.total)
        self.count = torch.zeros(length, dtype=torch.int64)

    def update(self, values, mask=None):
        values = values.detach().cpu().double()
        if values.ndim != 2 or values.shape[1] != self.total.numel() or not torch.isfinite(values).all():
            raise ValueError("Expected finite per-token values[batch,context]")
        mask = torch.ones_like(values, dtype=torch.bool) if mask is None else mask.cpu().bool()
        self.total += torch.where(mask, values, 0.).sum(0)
        self.squares += torch.where(mask, values.square(), 0.).sum(0)
        self.count += mask.sum(0)

    def result(self):
        n = self.count
        mean = self.total / n.clamp_min(1)
        var = ((self.squares - self.total.square() / n.clamp_min(1)) / (n - 1).clamp_min(1)).clamp_min(0)
        se = (var / n.clamp_min(1)).sqrt()
        def optional(x, valid):
            return [float(v) if ok else None for v, ok in zip(x, valid)]
        return {"mean": optional(mean, n > 0), "stderr": optional(se, n > 1), "count": n.tolist(),
                "overall_mean": float(self.total.sum() / n.sum()) if n.sum() else None}


@torch.no_grad()
def token_nll(model, x, y, probability, generator, head_chunk=256):
    """BF16 model/head, accurate FP32 softcap+CE for diagnostic NLL curves.

    Only head_chunk*vocab logits are materialized. Does not use the approximate
    unreduced training kernel, whose strict per-token precision failures are known.
    """
    with torch.autocast("cuda", dtype=torch.bfloat16):
        h = model.forward_features(x, probability, generator).reshape(-1, model.config.width)
    target = y.reshape(-1)
    losses = torch.empty(target.numel(), device=x.device, dtype=torch.float32)
    for offset in range(0, h.shape[0], head_chunk):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model.lm_head(h[offset:offset + head_chunk])
        logits = logits.float()
        if model.config.softcap is not None:
            logits = model.config.softcap * torch.tanh(logits / model.config.softcap)
        losses[offset:offset + head_chunk] = F.cross_entropy(
            logits, target[offset:offset + head_chunk], reduction="none")
    return losses.reshape_as(y).cpu()


@torch.no_grad()
def evaluate_context(model, x, y, probability, seed, reset_context=0, head_chunk=256):
    if reset_context and probability not in (0., 1.) and model.config.architecture == 'hybrid':
        raise ValueError("Reset-context RNG matching requires hard_prob0 or1, not mixed row RNG")
    if head_chunk <= 0 or reset_context < 0:
        raise ValueError("Invalid chunk size")
    generator = torch.Generator(device=x.device)
    chunk = reset_context or x.shape[1]
    parts = []
    for offset in range(0, x.shape[1], chunk):
        # At probability endpoints, each layer consumes only its direction RNG.
        # Resetting the seed preserves the same per-layer global directions for
        # full vs every truncated block. No cross-block hidden/conv/scan state.
        generator.manual_seed(seed)
        parts.append(token_nll(model, x[:, offset:offset + chunk].contiguous(),
                               y[:, offset:offset + chunk].contiguous(), probability,
                               generator, head_chunk))
    return torch.cat(parts, dim=1)


def load_checkpoint(path):
    saved = torch.load(path, map_location="cpu", weights_only=False)  # Own trusted training files only.
    config = LMConfig(**saved['config']['model'])
    model = DecoderLM(config)
    model.load_state_dict(saved['model'], strict=True)
    meta = {"path": str(Path(path).resolve()), "step": saved['step'], "config": saved['config'],
            "model": asdict(config), "parameters": sum(p.numel() for p in model.parameters())}
    del saved
    return model.cuda().eval(), meta


def check_pair(left, right, allow_partial=False):
    if left['model']['architecture'] != 'hybrid' or right['model']['architecture'] != 'swa_only':
        raise ValueError("Expected hybrid and SWA-only checkpoints, in that order")
    if left['step'] != right['step']:
        raise ValueError("Compare the same number of optimizer updates")
    for key in ('data', 'tokenizer', 'batch', 'micro_batch', 'steps', 'lr', 'weight_decay', 'warmup', 'seed'):
        if left['config'][key] != right['config'][key]:
            raise ValueError(f"Unmatched training setting: {key}")
    for key in ('vocab_size', 'width', 'layers', 'heads', 'head_dim', 'context', 'window', 'rope_theta', 'softcap'):
        if left['model'][key] != right['model'][key]:
            raise ValueError(f"Unmatched model setting: {key}")
    if abs(left['parameters'] - right['parameters']) / left['parameters'] > .001:
        raise ValueError("Parameter counts differ by more than0.1%")
    if not allow_partial and any(m['step'] != m['config']['steps'] for m in (left, right)):
        raise ValueError("Training incomplete; use --allow-partial only for explicit diagnostics")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--hybrid', required=True, type=Path)
    p.add_argument('--swa', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--batches', type=int, default=100, help='Effective validation batches, matching training batch')
    p.add_argument('--hard-prob', type=float, choices=(0., 1.), default=1.)
    p.add_argument('--seed', type=int, default=779)
    p.add_argument('--head-chunk', type=int, default=256)
    p.add_argument('--reset-context', type=int, nargs='*', default=[], help='Optional independent-block ablations, e.g.128 512')
    p.add_argument('--allow-partial', action='store_true')
    p.add_argument('--plot', action='store_true')
    a = p.parse_args()
    if a.batches <= 0 or a.head_chunk <= 0 or any(c <= 0 for c in a.reset_context):
        p.error('Positive batch/chunk sizes required')
    if a.output.exists() and any(a.output.iterdir()):
        p.error('Choose an empty output directory; existing reports are not overwritten')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    hybrid, hm = load_checkpoint(a.hybrid)
    swa, sm = load_checkpoint(a.swa)
    check_pair(hm, sm, a.allow_partial)
    length = hybrid.config.context
    if any(c >= length for c in a.reset_context):
        p.error('Reset context must be smaller than full context')
    tokenizer = load_tokenizer(hm['config']['tokenizer'])
    _, files = split_files(hm['config']['data'])
    batch, micro = hm['config']['batch'], hm['config']['micro_batch']
    stream = PackedStream(files, tokenizer, context=length, batch_size=micro, repeat=False)
    configs = [(name, model, reset) for name, model in (('hybrid', hybrid), ('swa', swa))
               for reset in (0, *sorted(set(a.reset_context)))]
    key = lambda name, reset: name + ('/full' if reset == 0 else f'/reset{reset}')
    moments = {key(name, reset): PositionMoments(length) for name, _, reset in configs}
    no_eos = {name: PositionMoments(length) for name in moments}
    pairs = {'hybrid_minus_swa/full': PositionMoments(length)}
    for reset in sorted(set(a.reset_context)):
        pairs[f'hybrid_minus_swa/reset{reset}'] = PositionMoments(length)
        for name in ('hybrid', 'swa'):
            pairs[f'{name}/reset{reset}_minus_full'] = PositionMoments(length)
    digest = hashlib.sha256()
    for i in range(a.batches * (batch // micro)):
        x, y = stream.next_batch()
        digest.update(x.numpy().tobytes()); digest.update(y.numpy().tobytes())
        gpu_x, gpu_y = x.cuda(), y.cuda()
        values = {}
        for name, model, reset in configs:
            nll = evaluate_context(model, gpu_x, gpu_y, a.hard_prob, a.seed + i, reset, a.head_chunk)
            label = key(name, reset)
            values[label] = nll
            moments[label].update(nll)
            no_eos[label].update(nll, y != tokenizer.eos_token_id)
        pairs['hybrid_minus_swa/full'].update(values['hybrid/full'] - values['swa/full'])
        for reset in sorted(set(a.reset_context)):
            pairs[f'hybrid_minus_swa/reset{reset}'].update(values[key('hybrid', reset)] - values[key('swa', reset)])
            for name in ('hybrid', 'swa'):
                pairs[f'{name}/reset{reset}_minus_full'].update(values[key(name, reset)] - values[key(name, 0)])
        if (i + 1) % (batch // micro) == 0:
            print(json.dumps({'effective_batch': (i + 1) // (batch // micro), 'of': a.batches}), flush=True)
    report = {'checkpoints': {'hybrid': hm, 'swa': sm}, 'nll': 'accurate FP32 softcap+CE on BF16 head logits, natural logs',
              'examples': a.batches * batch, 'context': length, 'hard_prob': a.hard_prob, 'seed': a.seed,
              'position_convention': 'position1 predicts labels[:,0] from input_ids[:,0]; final position2048 predicts the next token',
              'validation_files': files, 'packed_xy_sha256': digest.hexdigest(), 'head_chunk': a.head_chunk,
              'swa_max_predecessor_distance': swa.config.layers * (swa.config.window - 1),
              'reset_context': a.reset_context,
              'caveats': ['SWA layers extend receptive field beyond one128-token window.',
                          'Reset ablation has1..chunk available input tokens, not a fixed sliding context per position.',
                          'Packed document boundaries confound position with available within-document context.',
                          'stderr treats packed sequences as independent; descriptive, not a significance guarantee.',
                          'Parameter/token budget matched; compute and wall time are not matched.'],
              'results': {k: {'all_tokens': v.result(), 'exclude_eos_targets': no_eos[k].result()} for k, v in moments.items()},
              'paired_differences': {k: v.result() for k, v in pairs.items()}}
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output / 'nll.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    with (a.output / 'positions.csv').open('w') as f:
        writer = csv.writer(f); writer.writerow(['model_context', 'target_position', 'count', 'nll', 'stderr', 'non_eos_count', 'non_eos_nll'])
        for label, r in report['results'].items():
            full, filtered = r['all_tokens'], r['exclude_eos_targets']
            for pos in range(length):
                writer.writerow([label, pos + 1, full['count'][pos], full['mean'][pos], full['stderr'][pos],
                                 filtered['count'][pos], filtered['mean'][pos]])
    with (a.output / 'paired_differences.csv').open('w') as f:
        writer = csv.writer(f); writer.writerow(['comparison', 'target_position', 'count', 'delta_nll', 'stderr'])
        for label, r in report['paired_differences'].items():
            for pos in range(length):
                writer.writerow([label, pos + 1, r['count'][pos], r['mean'][pos], r['stderr'][pos]])
    if a.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        for label, r in report['results'].items():
            axes[0].plot(range(1, length + 1), r['all_tokens']['mean'], label=label, linewidth=.8)
        for label, r in report['paired_differences'].items():
            axes[1].plot(range(1, length + 1), r['mean'], label=label, linewidth=.8)
        axes[0].set_ylabel('NLL (nats)'); axes[1].set_ylabel('Paired delta NLL')
        axes[1].axhline(0, color='gray', linewidth=.5); axes[1].set_xlabel('Target position')
        for ax in axes: ax.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(a.output / 'positions.png', dpi=160); plt.close(fig)
    print(json.dumps({'overall_nll': {k: v['all_tokens']['overall_mean'] for k, v in report['results'].items()},
                      'output': str(a.output)}), flush=True)


if __name__ == '__main__':
    main()
