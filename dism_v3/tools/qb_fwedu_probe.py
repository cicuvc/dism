"""Offline K-only quantile-price calibration and causal held-out hard evaluation.

Changes only labels returned by the actual interpolation entry, with fixed prices.
No parameters, soft-path math, training state, or checkpoint files are modified.
"""
import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from niah_circuit_probe import ARMS, ROOT, RUNS, atomic_json

DATA = '/media/cicuvc/c63abdf1-0e56-4153-9228-95df5a2f239b/cicuvc/fwedu-100B/sample/100BT'
TOKENIZER = str(RUNS / ARMS['old'][0] / 'tokenizer')


def histogram_stats(counts):
    counts = np.asarray(counts, dtype=np.float64)
    f = counts / max(counts.sum(), 1)
    positive = f[f > 0]
    return dict(active=int((counts > 0).sum()), effective=float(np.exp(-(positive*np.log(positive)).sum())),
                max_frequency=float(f.max()), max_load_ratio=float(f.max()*len(f)), counts=counts.astype(int).tolist())


@torch.no_grad()
def quantile_prices(scores, iterations=32, initial=None):
    """Exact block minimizers of the hinge dual, not an exact assignment solver.

    Row alpha is between largest/second largest adjusted scores. Column price
    is between capacity-th/(capacity+1)-th largest S-alpha. Midpoints reduce
    boundary ties. Best reduced-dual iterate is returned; load error is reported.
    """
    n, vocab = scores.shape
    if n < vocab or n % vocab:
        raise ValueError('calibration sample count must be a positive multiple of vocabulary size')
    capacity = n // vocab
    price = scores.new_zeros(vocab) if initial is None else initial.to(scores).clone()
    price -= price.mean()
    best = price.clone()
    best_dual = float('inf')
    trace = []
    for step in range(iterations + 1):
        adjusted = scores-price
        top = adjusted.topk(2, dim=1).values
        dual = float(top[:, 0].mean()+price.mean())
        labels = adjusted.argmax(1)
        counts = torch.bincount(labels, minlength=vocab)
        record = dict(iteration=step, dual_per_token=dual, active=int((counts>0).sum()),
                      mean_abs_load_error=float((counts.float()/capacity-1).abs().mean()),
                      max_load_ratio=float(counts.max()/capacity))
        trace.append(record)
        if dual < best_dual:
            best_dual = dual
            best = price.clone()
        if step == iterations:
            break
        alpha = top.mean(1)
        columns = (scores-alpha[:, None]).topk(capacity+1, dim=0).values
        price = columns[-2:].mean(0)
        price -= price.mean()
    chosen = (scores-best).argmax(1)
    regret = scores.max(1).values-scores.gather(1, chosen[:, None]).squeeze(1)
    return best, dict(trace=trace, baseline=histogram_stats(torch.bincount(scores.argmax(1), minlength=vocab).cpu()),
                      calibrated=histogram_stats(torch.bincount(chosen, minlength=vocab).cpu()),
                      mean_score_regret=float(regret.mean()), best_dual=best_dual)


def chain_counts(iq, old_k, new_k, valid, segments):
    """Local causal pair retention on eight evenly spaced valid query positions.

    Counts gates/chains of length >=1/2/3 within real segment boundaries. These
    compare old/new labels on the SAME current activations, not two networks.
    """
    h, n = iq.shape
    eligible = np.flatnonzero(valid)
    queries = eligible[np.linspace(0, len(eligible)-1, min(8, len(eligible)), dtype=int)]
    out = np.zeros((h, 3, 3), dtype=np.int64)  # old, new, intersection
    for i in queries:
        keys = np.arange(i+1)
        allowed = valid[:i+1] & (segments[:i+1] == segments[i])
        before = np.broadcast_to(allowed, (h, i+1)).copy()
        after = before.copy()
        for length in range(1, 4):
            offset = length-1
            if i < offset:
                break
            ok = (keys >= offset)
            shifted = np.maximum(0, keys-offset)
            ok &= valid[shifted] & (segments[shifted] == segments[i])
            ok &= bool(valid[i-offset] and segments[i-offset] == segments[i])
            before &= ok[None, :] & (iq[:, i-offset, None] == old_k[:, shifted])
            after &= ok[None, :] & (iq[:, i-offset, None] == new_k[:, shifted])
            out[:, length-1, 0] += before.sum(1)
            out[:, length-1, 1] += after.sum(1)
            out[:, length-1, 2] += (before & after).sum(1)
    return out


class Observer:
    def __init__(self, model, prices=None, scale=0., only=None, collect=False, sample_per_batch=512, analyze=True):
        self.model, self.prices, self.scale, self.only = model, prices, scale, only
        self.collect, self.sample_per_batch, self.analyze = collect, sample_per_batch, analyze
        self.scores = {l: [] for l in range(6,12)}
        self.stats = {}
        self.calls = []

    def set_batch(self, x, y, cu, segments):
        self.valid = (y.flatten() >= 0)
        self.valid_cpu = self.valid.cpu().numpy()
        self.segments = segments.flatten().cpu().numpy()
        self.calls = []

    @contextmanager
    def hooked(self):
        if self.model.training or not self.model.eval_hard:
            raise RuntimeError('this diagnostic supports full-hard evaluation only')
        import flash_dism.kernels.interp_true as interpolation
        original = interpolation.emb_fwd_true
        active = []
        handles = []
        for layer in range(6,12):
            def pre(_module, _args, index=layer):
                active[:] = [index]
            handles.append(self.model.layers[layer].attn.register_forward_pre_hook(pre))

        def embed(q, k, eq, ek, sm_scale=1., tau=None):
            result = original(q, k, eq, ek, sm_scale, tau)
            layer = active[0]
            self.calls.append(layer)
            # Inputs/codebooks are already the real BF16 interpolation operands.
            with torch.autocast('cuda', enabled=False):
                scores = torch.einsum('nhd,hvd->hnv', k[0].float(), ek.float()) * float(sm_scale)
            original_k, iq = result[-2], result[-1]
            reference_labels = scores.argmax(-1)
            chosen = original_k[0]
            if self.prices is not None and self.scale:
                proposed = (scores-self.scale*self.prices[layer][:, None, :]).argmax(-1).to(original_k.dtype)
                if self.only is None:
                    chosen = proposed
                elif layer == self.only[0]:
                    chosen = chosen.clone()
                    chosen[self.only[1]] = proposed[self.only[1]]
                result = (*result[:-2], chosen[None].contiguous(), iq)
            if self.collect:
                positions = self.valid.nonzero().flatten()
                if len(positions) < self.sample_per_batch:
                    raise RuntimeError('too few valid tokens for fixed-size calibration sample')
                positions = positions[torch.linspace(0,len(positions)-1,self.sample_per_batch,device=k.device).long()]
                self.scores[layer].append(scores[:, positions].cpu())
            if self.analyze:
                valid_scores = scores[:, self.valid]
                new = chosen[:, self.valid]
                old = original_k[0, :, self.valid]
                queries = iq[0, :, self.valid]
                for head in range(scores.shape[0]):
                    key = f'L{layer:02d}H{head}'
                    stat = self.stats.setdefault(key, dict(layer=layer, head=head, n=0, k_counts=np.zeros(scores.shape[-1],dtype=np.int64),
                        q_counts=np.zeros(scores.shape[-1],dtype=np.int64), changed=0, fp32_label_disagreement=0, regret_sum=0., chains=np.zeros((3,3),dtype=np.int64)))
                    stat['n'] += int(self.valid.sum())
                    stat['k_counts'] += torch.bincount(new[head].long(), minlength=scores.shape[-1]).cpu().numpy()
                    stat['q_counts'] += torch.bincount(queries[head].long(), minlength=scores.shape[-1]).cpu().numpy()
                    stat['changed'] += int((new[head]!=old[head]).sum())
                    stat['fp32_label_disagreement'] += int((reference_labels[head,self.valid]!=old[head]).sum())
                    regret = valid_scores[head].max(-1).values-valid_scores[head].gather(1,new[head,:,None].long()).squeeze(1)
                    stat['regret_sum'] += float(regret.double().sum())
                chains = chain_counts(iq[0].cpu().numpy(), original_k[0].cpu().numpy(), chosen.cpu().numpy(), self.valid_cpu, self.segments)
                for head in range(scores.shape[0]):
                    self.stats[f'L{layer:02d}H{head}']['chains'] += chains[head]
            return result

        interpolation.emb_fwd_true = embed
        try:
            yield
        finally:
            interpolation.emb_fwd_true = original
            for h in handles:
                h.remove()

    def report(self):
        rows = []
        for key, stat in self.stats.items():
            q = histogram_stats(stat['q_counts'])
            k = histogram_stats(stat['k_counts'])
            chains = stat['chains']
            rows.append(dict(name=key, layer=stat['layer'], head=stat['head'], tokens=stat['n'], k=k, q=q,
                changed_fraction=stat['changed']/stat['n'], fp32_label_disagreement=stat['fp32_label_disagreement']/stat['n'],
                mean_score_regret=stat['regret_sum']/stat['n'],
                sampled_causal_chains=[dict(length=i+1, original=int(c[0]), modified=int(c[1]), intersection=int(c[2]),
                    retention=float(c[2]/c[0]) if c[0] else None) for i,c in enumerate(chains)]))
        return rows


@torch.no_grad()
def run_batches(model, batches, token_bytes, observer=None):
    rows = []
    for index, batch in enumerate(batches):
        x,y,cu,segments = [t.cuda() for t in batch]
        if observer is not None:
            observer.set_batch(x,y,cu,segments)
        loss = model(x,y,cu_seqlens=cu,segment_ids=segments,loss_reduction='none').flatten()
        target = y.flatten()
        sizes = token_bytes[target.clamp_min(0)]
        mask = (target>=0) & (sizes>0)
        rows.append(dict(batch=index, nats=float(loss[mask].double().sum()), tokens=int(mask.sum()), bytes=int(sizes[mask].sum())))
        if observer is not None:
            assert observer.calls == list(range(6,12)), observer.calls
        if (index+1)%16==0:
            print('batches',index+1,flush=True)
    nats=sum(r['nats'] for r in rows); tokens=sum(r['tokens'] for r in rows); sizes=sum(r['bytes'] for r in rows)
    return dict(nll=nats/tokens,bpb=nats/(math.log(2)*sizes),valid_tokens=tokens,bytes=sizes,batches=rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--calibration-batches',type=int,default=32)
    p.add_argument('--eval-batches',type=int,default=128)
    p.add_argument('--iterations',type=int,default=32)
    p.add_argument('--data-dir',default=DATA)
    a=p.parse_args()
    if min(a.calibration_batches,a.eval_batches,a.iterations)<1:
        p.error('batch counts and iterations must be positive')
    if a.out.exists(): p.error('choose a new output directory')
    a.out.mkdir(parents=True)
    subprocess.run(['nvidia-smi','--query-gpu=memory.used,memory.free,utilization.gpu','--format=csv'],check=True)
    if torch.cuda.mem_get_info()[0] < 10*1024**3: raise RuntimeError('less than 10 GiB free')
    torch.cuda.set_per_process_memory_fraction(.30)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.manual_seed(1729)
    from nanochat.tokenizer import HuggingFaceTokenizer
    from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
    from nanochat.dataset import list_parquet_files
    tokenizer=HuggingFaceTokenizer.from_pretrained(TOKENIZER)
    token_bytes=tokenizer.get_token_bytes(device='cuda')
    spec=importlib.util.spec_from_file_location('original_niah',RUNS/ARMS['old'][0]/'diagnostics/niah.py')
    harness=importlib.util.module_from_spec(spec);spec.loader.exec_module(harness)
    run,tag=ARMS['restored']; checkpoint=RUNS/run/'base_checkpoints'/tag/'model_019074.pt'
    model,meta=harness.load_ckpt(checkpoint);model.eval_hard=True
    buffers={k:v.detach().cpu().clone() for k,v in model.named_buffers()}
    def load(split,count):
        loader=tokenizing_distributed_data_loader_bos_bestfit(tokenizer,1,2048,split,device='cpu',
            seq_align=256,data_dir=a.data_dir,shuffle=False,num_workers=0,
            aligned_segment_ends=model.aligned_segment_ends)
        return [tuple(t.clone() for t in next(loader)) for _ in range(count)]
    calibration=load('train',a.calibration_batches); evaluation=load('val',a.eval_batches)
    torch.save(dict(calibration=calibration,evaluation=evaluation),a.out/'inputs.pt')
    paths=list_parquet_files(data_dir=a.data_dir)
    result=dict(checkpoint=str(checkpoint),model_spec=meta['model_spec'],git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),input_sha256=hashlib.sha256((a.out/'inputs.pt').read_bytes()).hexdigest(),
        tokenizer=TOKENIZER, tokenizer_fingerprint=tokenizer.get_fingerprint(), calibration_shard=paths[0],validation_shard=paths[-1],
        calibration_batches=a.calibration_batches,evaluation_batches=a.eval_batches,calibration_sample_per_batch=512,iterations=a.iterations,
        note='All-hard K-label-only frozen-price intervention. No soft-path claim. Calibration uses baseline activations.',arms={})
    print('collect calibration',flush=True)
    observer=Observer(model,collect=True)
    with observer.hooked(): result['calibration_baseline']=run_batches(model,calibration,token_bytes,observer)
    result['calibration_heads']=observer.report()
    prices={}; fits={}
    for layer in range(6,12):
        layer_scores=torch.cat(observer.scores.pop(layer),dim=1)
        prices[layer]=[]
        for head in range(layer_scores.shape[0]):
            scores=layer_scores[head].cuda()
            price,fit=quantile_prices(scores,a.iterations)
            prices[layer].append(price)
            fits[f'L{layer:02d}H{head}']=fit
            print('fit',layer,head,'active',fit['baseline']['active'],'->',fit['calibrated']['active'],
                  'max load',round(fit['calibrated']['max_load_ratio'],3),flush=True)
            del scores
        prices[layer]=torch.stack(prices[layer])
        del layer_scores
    result['fits']=fits
    del observer
    torch.save({l:v.cpu() for l,v in prices.items()},a.out/'prices.pt')
    atomic_json(a.out/'results.json',result)
    worst=min(result['calibration_heads'],key=lambda r:r['k']['effective'])
    result['worst_calibration_head']=[worst['layer'],worst['head']]
    native=run_batches(model,evaluation[:1],token_bytes)
    for name,scale,only in [('baseline',0.,None),('k_price_025',.25,None),('k_price_050',.5,None),('k_price_100',1.,None),
                            ('worst_head_100',1.,tuple(result['worst_calibration_head']))]:
        subprocess.run(['nvidia-smi','--query-gpu=memory.used,memory.free,utilization.gpu','--format=csv'],check=True)
        begin=time.monotonic();torch.cuda.reset_peak_memory_stats()
        observer=Observer(model,prices=prices,scale=scale,only=only)
        with observer.hooked(): metrics=run_batches(model,evaluation,token_bytes,observer)
        if name=='baseline':
            assert metrics['batches'][0]==native['batches'][0],(metrics['batches'][0],native['batches'][0])
        result['arms'][name]=dict(**metrics,heads=observer.report(),seconds=time.monotonic()-begin,
                                 peak_allocated=torch.cuda.max_memory_allocated(),price_scale=scale,only=only)
        atomic_json(a.out/'results.json',result)
        print(name,'NLL',metrics['nll'],'BPB',metrics['bpb'],flush=True)
    for key,value in model.named_buffers():
        assert torch.equal(value.cpu(),buffers[key]),key
    result['buffers_unchanged']=True
    atomic_json(a.out/'results.json',result)


if __name__=='__main__':
    main()
