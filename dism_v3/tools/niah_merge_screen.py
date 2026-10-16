"""Read-only harmful-collision screening with single-row pre-mix interventions.

The source is the intended needle span, not proven exclusive provenance.
Candidate distractors exclude a 32-token needle halo and the last 256 tokens.
Random controls match removed mass within log2-distance bins, using fractional
edge deletion at bin boundaries. They do not also match edge counts/chain length.
"""
import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from niah_circuit_probe import ARMS, RUNS, ROOT, atomic_json, digest, evaluate
from niah_read_rows import hard_row

PATHS = {'old': [(10, 0), (10, 3)], 'restored': [(8, 5), (10, 0)],
         'gqa2': [(8, 1), (8, 5)]}
MODES = ('sham', 'content', 'denominator', 'full', 'half', 'source', 'random0', 'random1')


def normalized(lengths, tau, retain=None):
    """Stable geometric-sum weights, with an unchanged unit fallback."""
    logw = np.full(len(lengths), -np.inf, dtype=np.float64)
    m = lengths > 0
    l = lengths[m].astype(np.float64)
    if tau == 0:
        logw[m] = np.log(l)
    elif tau > 0:
        logw[m] = l*tau + np.log(-np.expm1(-l*tau))-np.log(-np.expm1(-tau))
    else:
        logw[m] = tau + np.log(-np.expm1(l*tau))-np.log(-np.expm1(tau))
    if retain is not None:
        if np.any((retain < 0) | (retain > 1)):
            raise ValueError('retention must be in [0,1]')
        logret = np.full_like(logw, -np.inf)
        np.log(retain, out=logret, where=retain > 0)
        logw += logret
    z = max(0., float(logw.max()))
    w = np.exp(logw-z)
    fallback = np.exp(-z)
    den = w.sum()+fallback
    return w/den, fallback/den, z+np.log(den)


def random_mass_control(mass, candidate, pool, query, seed):
    """Delete equal mass per distance bin, randomizing which eligible keys lose it."""
    rng = np.random.default_rng(seed)
    bins = np.floor(np.log2(np.maximum(1, query-np.arange(len(mass))))).astype(int)
    fraction = np.zeros_like(mass)
    for b in np.unique(bins[candidate]):
        target = mass[candidate & (bins == b)].sum()
        indices = np.flatnonzero(pool & (bins == b) & (mass > 0))
        for j in rng.permutation(indices):
            if target <= 0:
                break
            removed = min(target, mass[j])
            fraction[j] = removed/mass[j]
            target -= removed
        if target > 1e-12:
            raise AssertionError(f'control cannot match mass: {target}')
    return fraction


def interventions(lengths, tau, features, source, candidate, pool, query, seed):
    mass, fallback, logden = normalized(lengths, tau)
    baseline = mass @ features
    u = (mass*candidate) @ features
    retained, _, retained_logden = normalized(lengths, tau, 1.-candidate)
    logscale = logden-retained_logden
    denominator = baseline*np.exp(logscale) if logscale < 700 else np.full_like(baseline, np.nan)
    half, _, _ = normalized(lengths, tau, 1.-.5*candidate)
    source_removed, _, _ = normalized(lengths, tau, 1.-source)
    vectors = dict(sham=baseline, content=baseline-u, denominator=denominator,
                   full=retained@features, half=half@features, source=source_removed@features)
    controls = []
    for rep in range(2):
        fraction = random_mass_control(mass, candidate.astype(bool), pool, query, seed+rep)
        new, _, _ = normalized(lengths, tau, 1.-fraction)
        vectors[f'random{rep}'] = new@features
        controls.append(dict(mass=float(mass@fraction), edges=float(fraction.sum()),
                             source_mass=float((mass*source)@fraction)))
    return {k: v-baseline for k,v in vectors.items()}, dict(
        candidate_mass=float(mass@candidate), source_mass=float(mass@source),
        fallback_mass=float(fallback), log_amplification=float(logscale),
        random_controls=controls)


@contextmanager
def capture(model, sample, arm, statistics, plans, histograms, halo, recent):
    import flash_dism.compiler as compiler
    import flash_dism.kernels.interp_true as interpolation
    orig_voc, orig_emb = compiler.voc_forward, interpolation.emb_fwd_true
    active, labels, handles, calls = [], [], [], []
    for layer in range(6, 12):
        def pre(_m, _args, layer=layer):
            active[:] = [layer]
        handles.append(model.layers[layer].attn.register_forward_pre_hook(pre))
    def embed(*args, **kwargs):
        out = orig_emb(*args, **kwargs)
        labels.append((out[-1][0].cpu().numpy(), out[-2][0].cpu().numpy()))
        return out
    def voc(q,k,sq,sk,v,eq,ek,tau,beta,direction,hard,boundaries,layout=None,*,direction_true=False):
        assert direction_true and bool(hard.all()) and q.shape[0] == 1
        assert boundaries.tolist() == [0, q.shape[1]]
        out = orig_voc(q,k,sq,sk,v,eq,ek,tau,beta,direction,hard,boundaries,layout,direction_true=direction_true)
        assert len(labels) == 1
        iq, ik = labels.pop()
        layer = active[0]
        calls.append(layer)
        def cpu(t):
            return t.detach().double().cpu().numpy()
        sqn, skn, vn, taun, actual = map(cpu, (sq[0],sk[0],v[0],tau,out[0][0]))
        slo, shi = sample['needle_span']
        tlo, thi = sample['target_span']
        for h in range(iq.shape[0]):
            hk = f'L{layer:02d}_slot{h:02d}'
            hist = histograms.setdefault(hk, {'q':np.zeros(512,dtype=np.int64), 'k':np.zeros(512,dtype=np.int64)})
            hist['q'] += np.bincount(iq[h],minlength=512)
            hist['k'] += np.bincount(ik[h],minlength=512)
            for target in range(tlo+1,thi):
                i, offset = target-1, target-tlo
                lengths, mass, fallback = hard_row(iq[h],ik[h],i,taun[h])
                pos = np.arange(i+1)
                source = (pos>=slo)&(pos<shi)
                pool = pos < i-recent+1
                candidate = pool & ~((pos>=slo-halo)&(pos<shi+halo))
                curve=[]
                for length in (1,2,3,4,5,8):
                    eligible=lengths>=length
                    ns=int((eligible&source).sum()); nd=int((eligible&candidate).sum())
                    curve.append(dict(length=length, source=ns, candidate=nd))
                signed=skn[:i+1,h]@sqn[i,h]
                features=signed[:,None]*vn[:i+1,h]
                reference=mass@features
                row=dict(layer=layer,slot=h,target_offset=offset,query_position=i,
                         query_symbol=int(iq[h,i]),source_gate=bool(np.any(lengths[source]>0)),
                         source_digit_gate=bool(lengths[slo+offset]>0),
                         source_mass=float(mass[source].sum()), candidate_mass=float(mass[candidate].sum()),
                         protected_mass=float(mass[~(source|candidate)].sum()),fallback_mass=float(fallback),
                         candidate_edges=int(np.count_nonzero(candidate&(lengths>0))),
                         k_symbol_share=float(np.mean(ik[h,:i+1]==iq[h,i])),curve=curve,
                         reconstruction_rel_l2=float(np.linalg.norm(reference-actual[i,h])/max(np.linalg.norm(actual[i,h]),1e-12)))
                statistics.append(row)
                if (layer,h) in PATHS[arm]:
                    seed=int(digest([sample['trial'],i,layer,h])[:8],16)
                    delta, extra=interventions(lengths,taun[h],features,source.astype(float),candidate.astype(float),pool,i,seed)
                    plans.append(dict(layer=layer,slot=h,target_offset=offset,query_position=i,
                                      delta=delta,diagnostics=extra, original=out[0][0,i,h].clone()))
        return out
    compiler.voc_forward, interpolation.emb_fwd_true = voc, embed
    try:
        yield calls
    finally:
        compiler.voc_forward, interpolation.emb_fwd_true = orig_voc, orig_emb
        for h in handles:
            h.remove()


@contextmanager
def patch_row(model, plan, mode):
    import flash_dism.compiler as compiler
    original=compiler.voc_forward
    delta=torch.as_tensor(plan['delta'][mode],device='cuda',dtype=torch.float32)
    if not bool(torch.isfinite(delta).all()):
        raise FloatingPointError('non-finite intervention delta')
    active, handles, calls=[],[],[]
    for layer in range(6,12):
        def pre(_m,_args,layer=layer):
            active[:]=[layer]
        handles.append(model.layers[layer].attn.register_forward_pre_hook(pre))
    def wrapper(*args,**kwargs):
        out=original(*args,**kwargs)
        if active==[plan['layer']]:
            i,h=plan['query_position'],plan['slot']
            value=out[0].clone()
            if not torch.equal(value[0,i,h],plan['original']):
                raise AssertionError('pre-intervention core changed from capture')
            patched=(value[0,i,h].float()+delta).to(value.dtype)
            if not bool(torch.isfinite(patched).all()):
                raise FloatingPointError('non-finite BF16 patched output')
            value[0,i,h]=patched
            out=[value,*out[1:]]
            calls.append(1)
        return out
    compiler.voc_forward=wrapper
    try:
        yield calls
    finally:
        compiler.voc_forward=original
        for h in handles:
            h.remove()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--p1',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--trials',type=int,default=16)
    p.add_argument('--start-trial',type=int,default=0)
    p.add_argument('--arms',nargs='+',choices=list(ARMS),default=list(ARMS))
    p.add_argument('--halo',type=int,default=32)
    p.add_argument('--recent',type=int,default=256)
    p.add_argument('--modes',nargs='+',choices=MODES,default=list(MODES))
    a=p.parse_args()
    if a.out.exists():p.error('output already exists')
    suite=json.loads((a.p1/'inputs.json').read_text())
    if not 0<=a.start_trial<a.start_trial+a.trials<=suite['trials']:p.error('trial range out of bounds')
    if a.halo<0 or a.recent<1:p.error('invalid protection windows')
    for name in ('flash_dism','cu_flash_dism'):
        origin=Path(importlib.util.find_spec(name).origin).resolve()
        if not origin.is_relative_to(ROOT/'dism_v3'):raise RuntimeError(f'foreign module {origin}')
    a.out.mkdir(parents=True)
    cell=next(c for c in suite['cells'] if c['context']==16384 and c['depth']==.05)
    spec=importlib.util.spec_from_file_location('original_niah',RUNS/ARMS['old'][0]/'diagnostics/niah.py')
    harness=importlib.util.module_from_spec(spec);spec.loader.exec_module(harness)
    torch.cuda.set_per_process_memory_fraction(.30)
    for arm in a.arms:
        subprocess.run(['nvidia-smi','--query-gpu=memory.used,memory.free,utilization.gpu','--format=csv'],check=True)
        if torch.cuda.mem_get_info()[0]<10*1024**3:raise RuntimeError('less than 10 GiB free')
        baseline=json.loads((a.p1/f'{arm}.json').read_text())
        assert baseline['suite_sha256']==digest(suite)
        model,meta=harness.load_ckpt(baseline['checkpoint'])
        model.eval_hard=True;model.config.sequence_len=16384
        buffers={k:v.detach().cpu().clone() for k,v in model.named_buffers()}
        old=next(c for c in baseline['cells'] if c['context']==16384 and c['depth']==.05)['measurements']['baseline']['present']
        result=dict(arm=arm,checkpoint=baseline['checkpoint'],model_spec=meta['model_spec'],
                    suite_sha256=digest(suite),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    paths=PATHS[arm],halo=a.halo,recent=a.recent,modes=a.modes,
                    context=16384,depth=.05,samples=[])
        histograms={}
        for sample in cell['samples'][a.start_trial:a.start_trial+a.trials]:
            begin=time.monotonic();statistics=[];plans=[]
            torch.cuda.reset_peak_memory_stats()
            with capture(model,sample,arm,statistics,plans,histograms,a.halo,a.recent) as calls:
                score=evaluate(model,[sample],'present',1)[0]
            assert calls==list(range(6,12))
            assert score['token_nll']==old[sample['trial']]['token_nll'], 'baseline replay drift'
            record=dict(trial=sample['trial'],baseline=score,statistics=statistics,interventions=[])
            result['samples'].append(record)
            for plan in plans:
                offset=plan['target_offset']
                r={k:v for k,v in plan.items() if k not in ('delta','original')}
                r['measurements']={}
                for mode in a.modes:
                    try:
                        with patch_row(model,plan,mode) as hits:
                            changed=evaluate(model,[sample],'present',1)[0]
                        assert len(hits)==1
                        nll=changed['token_nll'][offset]
                        if mode=='sham':assert changed['token_nll']==score['token_nll']
                        r['measurements'][mode]=dict(nll=nll,hit=changed['token_hits'][offset],
                                                     benefit=score['token_nll'][offset]-nll)
                    except FloatingPointError as error:
                        r['measurements'][mode]=dict(error=str(error))
                record['interventions'].append(r)
            record['seconds']=time.monotonic()-begin
            record['peak_allocated']=torch.cuda.max_memory_allocated()
            result['histograms']={k:{side:v.tolist() for side,v in h.items()} for k,h in histograms.items()}
            atomic_json(a.out/f'{arm}.json',result)
            print(arm,sample['trial'],f"{record['seconds']:.1f}s",flush=True)
        for k,v in model.named_buffers():
            assert torch.equal(v.detach().cpu(),buffers[k]), k
        result['buffers_unchanged']=True
        atomic_json(a.out/f'{arm}.json',result)
        del model
        torch.cuda.empty_cache()


if __name__=='__main__':main()
