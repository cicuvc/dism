"""Measure codebook coverage of the original valres checkpoint with the CURRENT probe.

The original `dism-fwedu-valres-72m-2500m` run (fused shortconv, legacy
microbatch-mean objective) logged `qk/q_coverage` with the old probe, which the
later "live-run diagnostic repair" replaced: it had passed the removed
token-level hard mask and did not propagate the first GDN value through the
value-residual layers. Its numbers are therefore not comparable with a current
run's.

If the original model's final codebooks show the same low coverage under the
repaired probe, the original run concentrated too and its probe hid the decay
(method-level). If they are high, the current runs' decay is a real change.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

_root = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(_root / 'dism_v3' / 'python'), str(_root / 'nanochat')]

from nanochat.metrics import probe_qk_vocab_usage                        # noqa: E402
from nanochat.models import build_model, model_spec_from_metadata        # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint-dir', type=Path, required=True)
    ap.add_argument('--step', type=int, default=19074)
    args = ap.parse_args()
    meta = json.loads((args.checkpoint_dir / f'meta_{args.step:06d}.json').read_text())
    state = torch.load(args.checkpoint_dir / f'train_state_{args.step:06d}_rank0.pt',
                       map_location='cpu', weights_only=False)
    weights = torch.load(args.checkpoint_dir / f'model_{args.step:06d}.pt',
                         map_location='cpu', weights_only=True)
    weights = {k.removeprefix('_orig_mod.'): v for k, v in weights.items()}
    spec = model_spec_from_metadata(meta)
    with torch.device('meta'):
        model = build_model(spec)
    model.to_empty(device='cuda')
    model.init_weights()
    model.load_state_dict(weights, strict=True)
    model.eval()
    batch = state['prefetched_batch']
    ids = batch['inputs'].cuda()
    bounds = batch['cu_seqlens'].cuda()
    seq = ids.numel() // ids.shape[0] if ids.dim() > 1 else ids.shape[0]
    stats = probe_qk_vocab_usage(model, ids, bounds, seq)
    print(f'{args.checkpoint_dir.name} step {args.step}: '
          f'tokens={ids.numel()} docs={bounds.numel() - 1}')
    print('  layer  q_cov  q_pooled  q_maxshare  q_entropy | k_cov  k_pooled  k_maxshare')
    for index in sorted(stats):
        row = []
        for tag in ('q', 'k'):
            s = stats[index][tag]
            row.append((s['coverage'], s['coverage_pooled'], s['max_share'], s['entropy']))
        q, k = row
        print(f'  {index:5d}  {q[0]:.3f}   {q[1]:.3f}     {q[2]:.3f}      {q[3]:.3f}   |'
              f' {k[0]:.3f}   {k[1]:.3f}     {k[2]:.3f}')


if __name__ == '__main__':
    main()
