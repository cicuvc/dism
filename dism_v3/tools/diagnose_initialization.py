"""Fresh-model initialization probes; no training/checkpoint files are mutated.

Uses two real packed rows from an existing saved batch. Candidate gates are
temporary Python hooks; raw-sigmoid replaces the final SiLU gate as well.
"""
import gc
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'dism_v3/python'), str(ROOT/'nanochat')]
os.environ['DISM_HARD_PLANNER'] = '0'
os.environ['DISM_HARD_BACKWARD'] = 'quadratic'
import torch
from torch.nn import functional as F
from nanochat.models import build_model, ModelSpec
from diagnose_layer_layout import batches, group


def rms(x):
    return x.detach().float().square().mean().sqrt().item()


def stats(x):
    x = x.detach().float().flatten()
    return dict(rms=rms(x), mean=x.mean().item(), min=x.min().item(), max=x.max().item())


class Probe:
    def __init__(self, model, variant):
        self.rows = [{} for _ in model.layers]
        self.hidden = {}
        self.handles = []
        self.capture = True
        self.noise = 0.
        self.embedding_reference = None
        self.actual_noise = 0.
        def embedding(m, args, out):
            if not self.noise:
                self.embedding_reference = out.detach()
                return out
            gen = torch.Generator(device=out.device).manual_seed(991)
            # Explicit BF16-rounded perturbation, report actual first-layer input.
            amplitude = 1. if variant == 'embedding_unit' else .02
            perturbed = (out.float()+torch.randn(out.shape, device=out.device, generator=gen)*self.noise*amplitude).to(out.dtype)
            self.actual_noise = rms(perturbed.float()-out.float())/rms(out)
            return perturbed
        self.handles.append(model.embedding.register_forward_hook(embedding))
        for i, layer in enumerate(model.layers):
            attn = layer.attn
            def pre(m, args, kwargs, i=i):
                if self.capture:
                    self.rows[i]['input_rms'] = rms(args[0])
                return args, kwargs
            self.handles.append(attn.register_forward_pre_hook(pre, with_kwargs=True))
            def output(m, args, out, i=i):
                if self.capture:
                    self.rows[i]['mixer_output_rms'] = rms(out[0])
            self.handles.append(attn.register_forward_hook(output))
            def hidden(m, args, out, i=i):
                self.hidden[i] = out.detach().float().cpu()
            self.handles.append(layer.register_forward_hook(hidden))
            if variant.startswith('sigmoid'):
                self.handles.append(attn.g_proj_up.register_forward_pre_hook(lambda m, args: (args[0].sigmoid(),)))
            def norm(m, args, i=i):
                if self.capture:
                    u, g = args
                    self.rows[i]['raw_mixer_rms'] = rms(u)
                    self.rows[i]['gate_pre'] = stats(g)
                    self.rows[i]['gate_effective'] = stats(g if variant == 'sigmoid_raw' else F.silu(g))
                    gain = torch.rsqrt(u.detach().float().square().mean(-1)+m.eps)
                    self.rows[i]['norm_gain'] = stats(gain)
            self.handles.append(attn.norm.register_forward_pre_hook(norm))
            if variant == 'sigmoid_raw':
                def raw_norm(u, g, m=attn.norm):
                    z = u.float()*torch.rsqrt(u.float().square().mean(-1,keepdim=True)+m.eps)
                    return (z*m.weight.float()*g.float()).to(u.dtype)
                attn.norm.forward = raw_norm
            if not getattr(attn, 'is_pure_mixer', False):
                fn = attn._combine_cuda
                def combine(d, *args, fn=fn, i=i, **kwargs):
                    out = fn(d, *args, **kwargs)
                    if self.capture:
                        # Reconstruct G from rounded sum; diagnostic approximation.
                        df, gf = d.detach().float(), out.detach().float()-d.detach().float()
                        self.rows[i]['dism_rms'] = rms(df)
                        self.rows[i]['gdn_approx_rms'] = rms(gf)
                        self.rows[i]['branch_cos'] = (df*gf).sum().div(df.norm()*gf.norm()+1e-30).item()
                    return out
                attn._combine_cuda = combine
                def selector(m, args, out, i=i, attn=attn):
                    if not self.capture:
                        return
                    with torch.no_grad(), torch.autocast('cuda', enabled=False):
                        q = out.detach().reshape(-1,attn.heads,attn.head_dim)[::8].float()
                        scores = torch.einsum('nhd,hvd->nhv', q, attn.q_vocab.detach().float())
                        prob = scores.softmax(-1)
                        top = scores.topk(2,dim=-1).values
                        self.rows[i]['selector'] = dict(entropy=(-(prob*prob.clamp_min(1e-30).log()).sum(-1)).mean().item(), margin=(top[...,0]-top[...,1]).mean().item(), top1=prob.max(-1).values.mean().item())
                original_conv = attn.q_conv.depthwise_silu
                def conv(*args, fn=original_conv, capture=selector, **kwargs):
                    out = fn(*args, **kwargs)
                    capture(None, (), out)
                    return out
                attn.q_conv.depthwise_silu = conv
                self.rows[i]['tau'] = stats(F.softplus(attn.log_sel_tau))


def gradients(model):
    groups = {}
    for name, p in model.named_parameters():
        key = group(name) or name
        if p.grad is not None:
            g = p.grad.detach().float()
            a = groups.setdefault(key, [0.,0.,0])
            a[0] += g.square().sum().item()
            a[1] += p.detach().float().square().sum().item()
            a[2] += p.numel()
    return {k: dict(grad_norm=a**.5,weight_norm=b**.5,grad_weight_ratio=(a/max(b,1e-30))**.5) for k,(a,b,n) in groups.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--embedding-unit', action='store_true', help='Append a paired seed-42 embedding scale intervention')
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('high')
    base = ROOT/'dism_v3/build/layout_diagnosis'
    directory = Path(json.loads((base/'paths.json').read_text())[0])
    run = Path((base/'new_training_run.txt').read_text().strip())
    spec = ModelSpec.from_dict(json.loads((run/'model_spec.json').read_text()))
    batch = batches(directory, 22888, 2)[0]
    destination = ROOT/'dism_v3/build/initialization_diagnosis.json'
    results = dict(protocol='fresh full-size models; fixed 2x2048 packed rows from final saved prefetch; eager BF16, MXFP8 FFN, BF16 CE; no optimizer updates', runs=[])
    cases = [(42,'baseline'),(43,'baseline'),(44,'baseline'),(42,'tau_narrow'),(42,'sigmoid_silu'),(42,'sigmoid_raw')]
    if args.embedding_unit:
        results = json.loads(destination.read_text())
        results['runs'] = [r for r in results['runs'] if r['variant'] != 'embedding_unit']
        cases = [(42,'embedding_unit')]
    for seed, variant in cases:
        torch.manual_seed(seed)
        with torch.device('meta'):
            model = build_model(spec)
        model.to_empty(device='cuda'); model.init_weights(); model.eval(); model.eval_hard = False
        if variant == 'embedding_unit':
            with torch.no_grad():
                model.embedding.weight.mul_(50.)
        if variant == 'tau_narrow':
            with torch.no_grad():
                for layer in model.layers:
                    if hasattr(layer.attn,'log_sel_tau'):
                        layer.attn.log_sel_tau.mul_(.125)
        probe = Probe(model,variant)
        x,y,cu = batch
        loss = model(x,y,cu_seqlens=cu); loss.backward()
        item = dict(seed=seed,variant=variant,soft_loss=loss.item(),layers=probe.rows,gradients=gradients(model))
        item['top_parameter_gradients'] = sorted([dict(name=n,grad_norm=p.grad.detach().float().norm().item(),weight_rms=rms(p)) for n,p in model.named_parameters() if p.grad is not None],key=lambda row:-row['grad_norm'])[:25]
        baseline = probe.hidden
        probe.hidden = {}; probe.capture = False
        model.zero_grad(set_to_none=True)
        with torch.no_grad():
            for hard, perturb in [(False,0.),(False,.001),(True,0.),(True,.001)]:
                model.eval_hard = hard; probe.noise = perturb
                value = model(x,y,cu_seqlens=cu)
                key = f'{hard=},{perturb=}'
                item[key] = dict(loss=value.item(),hidden_delta={i:rms(probe.hidden[i]-baseline[i])/max(rms(baseline[i]),1e-30) for i in baseline})
                if perturb:
                    item[key]['actual_embedding_relative_noise'] = probe.actual_noise
                if hard and perturb == 0.:
                    hard_baseline = probe.hidden.copy()
                if hard and perturb:
                    item[key]['hard_pair_hidden_delta'] = {i:rms(probe.hidden[i]-hard_baseline[i])/max(rms(hard_baseline[i]),1e-30) for i in baseline}
        results['runs'].append(item)
        destination.write_text(json.dumps(results,indent=2)+'\n')
        print(seed,variant,'loss',item['soft_loss'],'layers',[(i,round(r['mixer_output_rms'],5),round(r['gate_effective']['rms'],5)) for i,r in enumerate(item['layers'])],flush=True)
        for h in probe.handles:h.remove()
        del model, probe, loss, baseline, hard_baseline
        gc.collect();torch.cuda.empty_cache()
    print(destination,flush=True)


if __name__ == '__main__':
    main()
