"""Does norm placement change the selector gradient's noise-to-signal profile?

Motivation: the repo's own optimizer analysis (`astra_optimizer_analysis.md`)
argues the gated DISM selectors see a noisier gradient, and that a higher Adam
beta1 (longer momentum) helps. Separately, an architecture search found postnorm
better than prenorm. `modeling_dism.py::DismBlock.forward` puts postnorm's norm on
the identity path, so the gradient decays with depth in postnorm, and a
scale-only decay is invisible to Adam (a per-parameter constant cancels in
m-hat/sqrt(v-hat)). The hypothesis worth testing is therefore not the scale but
the *shape*: does the norm placement change the per-layer noise-to-signal ratio
of the selector gradients?

`post_norm` is read per forward call from the block, so flipping the attribute
compares the two placements on **identical weights**, which isolates the
architecture from the trained parameters.

For each (norm, hard_probability) it draws `--draws` independent hard-flag
configurations on one fixed packed batch (RNG counter offset), computes the
gradient of every selector tensor for each draw, and reports per layer
    mu = mean gradient, Sigma = covariance over draws, r = tr(Sigma)/||mu||^2
plus the per-layer total gradient norm (the depth profile itself).
"""
import argparse
import json
import sys
from pathlib import Path

import torch

_root = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(_root / 'dism_v3' / 'python'), str(_root / 'nanochat')]

from nanochat.models import build_model, model_spec_from_metadata          # noqa: E402

SELECTOR_SUFFIX = ('q_vocab', 'k_vocab', '.q_conv.conv_weight', '.k_conv.conv_weight')


def is_selector(name):
    return name.endswith(SELECTOR_SUFFIX)


def layer_of(name):
    parts = name.split('.')
    return int(parts[1]) if len(parts) > 1 and parts[0] == 'layers' else -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint-dir', type=Path, required=True)
    ap.add_argument('--step', type=int, default=19074)
    ap.add_argument('--draws', type=int, default=16)
    ap.add_argument('--hard-prob', type=float, nargs='+', default=[0.95, 0.5])
    ap.add_argument('--rows', type=int, default=4,
                    help='packed rows to keep; the run needs less memory and the '
                         'hard-flag draw is per row x head, so fewer rows means a '
                         'coarser gating-noise sample')
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
    model.train()
    batch = state['prefetched_batch']
    x, y = batch['inputs'].cuda(), batch['targets'].cuda()
    bounds, segments = batch['cu_seqlens'].cuda(), batch['segment_ids'].cuda()
    if args.rows and args.rows < x.shape[0]:
        # Keep whole documents and a whole number of packed rows: the adapter
        # flattens [rows, T] and samples hard flags per original row.
        flat = x.reshape(-1)
        keep = args.rows * x.shape[1]
        bounds = bounds[bounds <= keep].contiguous()
        if int(bounds[-1]) != keep:
            raise SystemExit(f'the first {keep} tokens are not document aligned')
        x = flat[:keep].reshape(args.rows, x.shape[1]).contiguous()
        y = y.reshape(-1)[:keep].reshape(args.rows, x.shape[1]).contiguous()
        segments = segments.reshape(-1)[:keep].reshape(args.rows, x.shape[1]).contiguous()
    print(f'rows={x.shape[0]} T={x.shape[1]} docs={bounds.numel() - 1} '
          f'tokens={x.numel()} hard flags per draw={x.shape[0] * spec.config["n_head"]}')
    selectors = [(n, p) for n, p in model.named_parameters() if is_selector(n)]
    print(f'{sum(p.numel() for _, p in selectors):,} selector params across '
          f'{len(selectors)} tensors, {args.draws} hard-flag draws per arm')

    for post_norm in (True, False):
        for layer in model.layers:
            layer.post_norm = post_norm
        for hard_prob in args.hard_prob:
            model.hard_probability.fill_(hard_prob)
            grads = {n: [] for n, _ in selectors}
            layer_norm_sq = {}
            for draw in range(args.draws):
                model.zero_grad(set_to_none=True)
                model.rng_counter.fill_(10_000 + 977 * draw)
                loss, _ = model(x, y, cu_seqlens=bounds, segment_ids=segments,
                                loss_reduction='document_sum')
                loss.backward()
                for name, param in selectors:
                    grads[name].append(param.grad.detach().clone())
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        key = layer_of(name)
                        layer_norm_sq[key] = (layer_norm_sq.get(key, 0.0)
                                              + param.grad.detach().float().pow(2).sum().item())
            print(f'\npost_norm={post_norm} hard_prob={hard_prob}')
            print('  layer | sel params | mean||mu|| | sqrt(tr Sigma) | r = trS/||mu||^2'
                  ' | all-param ||g||')
            per_layer = {}
            for name, param in selectors:
                stack = torch.stack([g.double().flatten() for g in grads[name]])
                mu = stack.mean(0)
                sigma = stack - mu
                tr_sigma = sigma.pow(2).sum() / max(1, args.draws - 1)
                mu_sq = mu.pow(2).sum()
                key = layer_of(name)
                count, sum_mu, sum_tr, sum_musq = per_layer.get(key, (0, 0.0, 0.0, 0.0))
                per_layer[key] = (count + param.numel(), sum_mu + mu.norm().item(),
                                  sum_tr + tr_sigma.item(), sum_musq + mu_sq.item())
            for key in sorted(per_layer):
                count, sum_mu, sum_tr, sum_musq = per_layer[key]
                total = torch.tensor(layer_norm_sq.get(key, 0.0)).sqrt().item()
                print(f'  {key:5d} | {count:11,d} | {sum_mu:10.3e} | '
                      f'{sum_tr ** 0.5:14.3e} | {sum_tr / max(sum_musq, 1e-300):19.3e} | '
                      f'{total:11.3e}')


if __name__ == '__main__':
    main()
