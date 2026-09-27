"""Inference-only activation/codebook diagnostics; never writes model weights.

Calibration and paired NLL use disjoint sequence indices from the frozen bundle.
All surgery is temporary and affects every layer. No claims about retraining.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import time
import types

import torch
import torch.nn.functional as F


def correlation(x, y):
    x, y = x.double(), y.double()
    x, y = x-x.mean(), y-y.mean()
    den = x.norm()*y.norm()
    return float((x*y).sum()/den) if den > 0 else None


def effective(counts):
    p = counts.double()/counts.sum()
    return float(torch.exp(-(p*p.clamp_min(1e-300).log()).sum()))


def transformed_vocab(vocab, mode):
    if mode not in ('embedding_silu', 'embedding_silu_norm'):
        return vocab
    out = F.silu(vocab.float())
    if mode == 'embedding_silu_norm':
        out = out * (vocab.float().norm(dim=-1, keepdim=True) /
                     out.norm(dim=-1, keepdim=True).clamp_min(1e-12))
    return out.to(vocab.dtype).contiguous()


@contextmanager
def surgery(model, mode, means):
    handles, restore = [], []
    try:
        for layer, block in enumerate(model.blocks):
            attn = block.dism
            if mode.startswith('embedding_silu'):
                old = attn.expanded_vocabularies
                # Transform the FP32 master before BF16 casting, on both paths.
                def expand(self, dtype, selected=mode):
                    q = transformed_vocab(self.q_voc, selected).to(dtype).contiguous()
                    k = transformed_vocab(self.k_voc, selected).to(dtype).contiguous()
                    return q, k
                attn.expanded_vocabularies = types.MethodType(expand, attn)
                restore.append((attn, 'expanded_vocabularies', old))
            for side, conv in enumerate((attn.qd_conv, attn.kd_conv)):
                if mode == 'no_qk_silu':
                    restore.append((conv, 'activation', conv.activation))
                    conv.activation = None
                elif mode.startswith('center_'):
                    fraction = float(mode.split('_')[1])
                    mu = means[layer, side].flatten().cuda()
                    def hook(_module, _args, output, mu=mu, fraction=fraction):
                        value = (output[0].float()-fraction*mu).to(output[0].dtype)
                        return (value, *output[1:])
                    handles.append(conv.register_forward_hook(hook))
        yield
    finally:
        for handle in handles:
            handle.remove()
        for obj, key, value in reversed(restore):
            setattr(obj, key, value)


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--global-counts', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--calibration', type=int, default=32)
    p.add_argument('--sequences', type=int, default=64)
    p.add_argument('--batch', type=int, default=4)
    a = p.parse_args()
    if a.output.exists() or min(a.calibration, a.sequences, a.batch) < 1:
        p.error('Positive counts and new output required')
    if a.calibration % a.batch or a.sequences % a.batch:
        p.error('Sequence counts must divide into batches')
    a.output.mkdir(parents=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    saved = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    cfg, step = saved['config'], saved['step']
    os.environ['DISM_TILE_LSE'] = cfg['tile_lse']
    os.environ['DISM_BWD_OPT'] = str(cfg['backward_opt'])
    from .lm_model import DecoderLM, LMConfig
    from .eval_lm_positions import evaluate_context
    from .eval_vocab_load import alignment
    from . import embedding
    c = LMConfig(**cfg['model'])
    assert c.architecture == 'hybrid' and not c.dism_tie_qk_vocab and c.dism_vocab_groups is None
    model = DecoderLM(c).cuda().eval()
    model.load_state_dict(saved['model'], strict=True)
    del saved
    # Byte digest detects any accidental mutation of persistent tensors.
    def model_digest():
        h = hashlib.sha256()
        for name, tensor in model.state_dict().items():
            h.update(name.encode()); h.update(tensor.cpu().contiguous().numpy().tobytes())
        return h.hexdigest()
    before = model_digest()
    bundle = torch.load(a.bundle, map_location='cpu', weights_only=False)
    x, y = bundle['x'][:, :c.context].long(), bundle['y'][:, :c.context].long()
    assert len(x) >= a.calibration+a.sequences
    global_counts = torch.load(a.global_counts, weights_only=True)
    sums = torch.zeros(c.layers, 2, c.heads, c.head_dim, device='cuda', dtype=torch.float64)
    squares = torch.zeros(c.layers, 2, c.heads, device='cuda', dtype=torch.float64)
    negative = torch.zeros_like(squares)
    samples = [[[] for _ in range(2)] for _ in range(c.layers)]
    sampled_labels = [[[] for _ in range(2)] for _ in range(c.layers)]
    pre_sums, pre_squares = torch.zeros_like(sums), torch.zeros_like(squares)
    pre_samples = [[[] for _ in range(2)] for _ in range(c.layers)]
    pre_handles, pre_hook_specs = [], []
    # Replay just each stateless convolution at identical baseline inputs,
    # bypassing module hooks. Never invert SiLU (it is not injective).
    for layer, block in enumerate(model.blocks):
        for side, conv in enumerate((block.dism.qd_conv, block.dism.kd_conv)):
            def pre_capture(module, args, output, layer=layer, side=side):
                activation = module.activation
                try:
                    module.activation = None
                    raw = module.forward(*args)[0]
                finally:
                    module.activation = activation
                z = raw.reshape(raw.shape[0], raw.shape[1], c.heads, c.head_dim).transpose(1, 2)
                pre_sums[layer, side] += z.double().sum((0, 2))
                pre_squares[layer, side] += z.double().square().sum((0, 2, 3))
                pre_samples[layer][side].append(z[:, :, ::16].permute(1, 0, 2, 3).reshape(c.heads, -1, c.head_dim).cpu())
            pre_handles.append(conv.register_forward_hook(pre_capture))
            pre_hook_specs.append((conv, pre_capture))
    original = embedding.forward
    call = 0
    def capture(*args, **kwargs):
        nonlocal call
        raw = original(*args, **kwargs)
        layer = call % c.layers
        for side, (z, labels) in enumerate(zip(args[:2], (raw[7], raw[6]))):
            sums[layer, side] += z.double().sum((0, 2))
            squares[layer, side] += z.double().square().sum((0, 2, 3))
            negative[layer, side] += (z < 0).sum((0, 2, 3))
            samples[layer][side].append(z[:, :, ::16].permute(1, 0, 2, 3).reshape(c.heads, -1, c.head_dim).cpu())
            sampled_labels[layer][side].append(labels[:, :, ::16].permute(1, 0, 2).reshape(c.heads, -1).cpu())
        call += 1
        return raw
    started = time.perf_counter()
    try:
        embedding.forward = capture
        for start in range(0, a.calibration, a.batch):
            gx = x[start:start+a.batch].cuda()
            with torch.autocast('cuda', dtype=torch.bfloat16):
                out = model.forward_features(gx, 1., torch.Generator(device='cuda').manual_seed(779))
            assert torch.isfinite(out).all()
            if start == 0:
                embedding.forward = original
                # These diagnostic hooks do not modify returned activations;
                # disable them during the independent baseline to avoid counts.
                for handle in pre_handles:
                    handle.remove()
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    plain = model.forward_features(gx, 1., torch.Generator(device='cuda').manual_seed(779))
                torch.testing.assert_close(out, plain, atol=0, rtol=0)
                pre_handles = []
                # Re-register the same per-module closure after control pass.
                for module, hook in pre_hook_specs:
                    pre_handles.append(module.register_forward_hook(hook))
                embedding.forward = capture
    finally:
        embedding.forward = original
        for handle in pre_handles:
            handle.remove()
    assert call == a.calibration//a.batch*c.layers
    for layer in range(c.layers):
        for side in range(2):
            assert len(pre_samples[layer][side]) == a.calibration//a.batch
            assert len(samples[layer][side]) == a.calibration//a.batch
    means = (sums/(a.calibration*c.context)).float().cpu()
    pre_means = (pre_sums/(a.calibration*c.context)).float().cpu()
    print(json.dumps({'phase': 'calibration_done', 'seconds': time.perf_counter()-started}), flush=True)
    geometry = []
    for layer, block in enumerate(model.blocks):
        for side, master in enumerate((block.dism.q_voc, block.dism.k_voc)):
            z = torch.cat(samples[layer][side], 1).float().cuda()
            pre_z = torch.cat(pre_samples[layer][side], 1).float().cuda()
            labels = torch.cat(sampled_labels[layer][side], 1).cuda()
            e = master.to(torch.bfloat16).float()
            mu = means[layer, side].cuda()
            bias = torch.einsum('hd,hvd->hv', mu, e)
            scores = z @ e.transpose(-1, -2)
            residual = scores-bias[:, None]
            # Ignore the common-across-vocabulary logit component: it cannot
            # affect softmax or argmax. Ratio is descriptive, not causal R2.
            bvar = bias.var(-1, unbiased=False)
            rvar = residual.var(-1, unbiased=False).mean(-1)
            variants = {'fp32_replay': scores.argmax(-1),
                        'no_qk_silu': (pre_z @ e.transpose(-1, -2)).argmax(-1),
                        'center': residual.argmax(-1),
                        'embedding_silu': (z @ F.silu(master).to(torch.bfloat16).float().transpose(-1, -2)).argmax(-1)}
            for head in range(c.heads):
                counts = global_counts[layer, side, head]
                norm = e[head].norm(dim=-1).cpu()
                bb = bias[head].cpu()
                freq = counts.double()/counts.sum()
                top_bias = bb.topk(8).indices
                row = dict(layer=layer+1, side=('q', 'k')[side], head=head+1,
                    corr_log_count_mean_bias=correlation(counts.double().log1p(), bb),
                    corr_log_count_norm=correlation(counts.double().log1p(), norm),
                    corr_log_count_embedding_mean=correlation(counts.double().log1p(), e[head].mean(-1).cpu()),
                    mass_on_top8_mean_bias=float(freq[top_bias].sum()),
                    mean_energy_fraction=float(mu[head].square().sum()/(squares[layer, side, head]/(a.calibration*c.context))),
                    pre_silu_mean_energy_fraction=float(pre_means[layer, side, head].square().sum()/(pre_squares[layer, side, head].cpu()/(a.calibration*c.context))),
                    mean_coordinate=float(mu[head].mean()),
                    pre_silu_mean_coordinate=float(pre_means[layer, side, head].mean()),
                    negative_fraction=float(negative[layer, side, head]/(a.calibration*c.context*c.head_dim)),
                    bias_to_content_rms=float((bvar[head]/rvar[head]).sqrt()),
                    baseline_effective_sample=effective(torch.bincount(labels[head].long(), minlength=c.qk_vocab)))
                for name, ids in variants.items():
                    cc = torch.bincount(ids[head], minlength=c.qk_vocab)
                    row[name] = dict(label_change=float((ids[head] != labels[head]).float().mean()),
                                     effective_vocab=effective(cc), top1_mass=float(cc.max()/cc.sum()))
                geometry.append(row)
        print(json.dumps({'phase': 'geometry', 'layer': layer+1}), flush=True)
    torch.save(dict(means=means, pre_means=pre_means, pre_squares=pre_squares.cpu(), pre_samples=pre_samples,
                    sums=sums.cpu(), squares=squares.cpu(), samples=samples,
                    labels=sampled_labels), a.output/'calibration.pt')
    (a.output/'geometry.json').write_text(json.dumps(geometry, indent=2)+'\n')
    modes = ['baseline', 'embedding_silu', 'embedding_silu_norm', 'no_qk_silu', 'center_0.25', 'center_1.0']
    losses, results = {}, {}
    for mode in modes:
        parts = []
        hist = torch.zeros(c.layers, 2, c.heads, c.qk_vocab, device='cuda', dtype=torch.int64)
        call = 0
        def count(*args, **kwargs):
            nonlocal call
            raw = original(*args, **kwargs)
            for side, labels in enumerate((raw[7], raw[6])):
                for head in range(c.heads):
                    hist[call % c.layers, side, head] += torch.bincount(labels[:, head].long().flatten(), minlength=c.qk_vocab)
            call += 1
            return raw
        try:
            embedding.forward = count
            with surgery(model, mode, means):
                for start in range(a.calibration, a.calibration+a.sequences, a.batch):
                    loss = evaluate_context(model, x[start:start+a.batch].cuda(), y[start:start+a.batch].cuda(), 1., 779)
                    assert torch.isfinite(loss).all()
                    parts.append(loss)
        finally:
            embedding.forward = original
        assert call == a.sequences//a.batch*c.layers
        assert (hist.sum(-1) == a.sequences*c.context).all()
        losses[mode] = torch.cat(parts)
        seq_delta = (losses[mode]-losses['baseline']).double().mean(-1)
        hcpu = hist.cpu()
        results[mode] = dict(nll=float(losses[mode].double().mean()), delta=float(seq_delta.mean()),
            paired_se=float(seq_delta.std()/a.sequences**.5),
            effective_q=sum(effective(t) for t in hcpu[:, 0].flatten(0, 1))/(c.layers*c.heads),
            effective_k=sum(effective(t) for t in hcpu[:, 1].flatten(0, 1))/(c.layers*c.heads),
            q_mass_on_unused_k=sum(alignment(hcpu[l, 0, h], hcpu[l, 1, h])['q_mass_on_unused_k']
                                   for l in range(c.layers) for h in range(c.heads))/(c.layers*c.heads))
        torch.save(hcpu, a.output/f'{mode}_counts.pt')
        print(json.dumps({'phase': 'nll', 'mode': mode, **results[mode], 'seconds': time.perf_counter()-started}), flush=True)
    # Restoration must reproduce the original model including runtime attributes.
    repeat = evaluate_context(model, x[a.calibration:a.calibration+a.batch].cuda(),
                              y[a.calibration:a.calibration+a.batch].cuda(), 1., 779)
    torch.testing.assert_close(repeat, losses['baseline'][:a.batch], atol=0, rtol=0)
    assert model_digest() == before
    torch.save(losses, a.output/'losses.pt')
    report = dict(checkpoint=str(a.checkpoint), step=step, bundle=str(a.bundle),
        calibration_sequences=a.calibration, evaluation_sequences=a.sequences, context=c.context,
        calibration_indices=[0, a.calibration], evaluation_indices=[a.calibration, a.calibration+a.sequences],
        hard_prob=1., results=results, geometry=geometry, no_weight_mutation=True,
        tracing_bitwise_equal=True, restored_nll_bitwise_equal=True,
        caveats=['Checkpoint surgery, not retraining.', 'Mean estimates are from separate calibration sequences.',
                 'Calibration subsample uses every16th token; moments use all calibration tokens.',
                 'Geometry FP32 GEMM is compared against actual CUDA BF16-input labels.',
                 'Global label counts overlap calibration corpus; correlations are descriptive, not causal.',
                 'SiLU changes interpolation too, but pure-hard core uses labels and original V, not interpolated values.',
                 'Correlated packed text; sequence-level standard errors are descriptive.'])
    (a.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print('Done; model tensors unchanged and baseline restored bitwise.', flush=True)


if __name__ == '__main__':
    main()
