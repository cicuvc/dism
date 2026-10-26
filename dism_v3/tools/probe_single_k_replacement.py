"""CPU FP64 single-K-token interventions, including committed-history effects.

Reads immutable operand captures; never imports CUDA operators or loads a model.
Rewards are frozen-cotangent output differences, NOT actual LM loss differences.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F


def geometry(tau, n):
    return torch.cat((torch.tensor([-torch.inf], dtype=torch.float64),
                      torch.logcumsumexp(torch.arange(1, n + 1, dtype=torch.float64) * tau, 0)))


def baselines(q, sq, sk, v, eq, pk, iq, ik, tau, positions, gs):
    """Hard/zero-history reads and full native-Jensen K probability tangent.

    O(N^2) scalar storage only; never materializes N^2*DV. Reverse recurrence
    includes all future diagonal uses of each soft matching probability.
    """
    n = len(q); cols = torch.arange(n); lg = geometry(tau, n)
    lengths = torch.empty((n, n), dtype=torch.int32)
    logsoft = torch.empty((n, n), dtype=torch.float64)
    prev = torch.zeros(n, dtype=torch.int32)
    prevsoft = torch.full((n,), -torch.inf, dtype=torch.float64)
    emb = pk @ eq; lq = (q @ eq.T).logsumexp(-1)
    oh, oz, os = [torch.empty_like(v) for _ in range(3)]
    zh, zz = [torch.empty(n, dtype=torch.float64) for _ in range(2)]
    for start in range(0, n, 128):
        end = min(start + 128, n)
        raw = q[start:end] @ emb.T - lq[start:end, None] + tau
        for a, i in enumerate(range(start, end)):
            valid = cols <= i
            prev = (F.pad(prev[:-1], (1, 0)) + 1) * (iq[i] == ik) * valid
            lengths[i] = prev
            prevsoft = (raw[a] + F.softplus(F.pad(prevsoft[:-1], (1, 0), value=-torch.inf))).masked_fill(~valid, -torch.inf)
            logsoft[i] = prevsoft
        sim = sq[start:end] @ sk.T
        for logs, out, den in [(lg[lengths[start:end].long()], oh, zh),
                               (torch.where(lengths[start:end] > 0, tau, -torch.inf), oz, zz),
                               (logsoft[start:end], os, None)]:
            logz = torch.logaddexp(torch.zeros(end-start, dtype=torch.float64), logs.logsumexp(-1))
            out[start:end] = ((logs - logz[:, None]).exp() * sim) @ v
            if den is not None:
                den[start:end] = logz
    # d<g,Osoft>/dpk_j(c), with both normalization and full history differentiated.
    features = torch.zeros((len(gs), len(positions), q.shape[-1]), dtype=torch.float64)
    following = torch.zeros((len(gs), n), dtype=torch.float64)
    for end in range(n, 0, -128):
        start = max(0, end - 128)
        logs = logsoft[start:end]
        logz = torch.logaddexp(torch.zeros(end-start, dtype=torch.float64), logs.logsumexp(-1))
        w = (logs - logz[:, None]).exp()
        sim = sq[start:end] @ sk.T
        vg = gs[:, start:end] @ v.T
        og = (gs[:, start:end] * os[None, start:end]).sum(-1)
        local = w[None] * (sim[None] * vg - og[:, :, None])
        selected = torch.empty((len(gs), end-start, len(positions)), dtype=torch.float64)
        for i in range(end - 1, start - 1, -1):
            following = local[:, i-start] + F.pad(following[:, 1:], (0, 1)) * logsoft[i].sigmoid()
            selected[:, i-start] = following[:, positions]
        features += selected.transpose(1, 2) @ q[start:end]
    # The omitted -logsumexp(zq) term is constant across K candidates and cancels.
    tangent = features @ eq.T
    return dict(lengths=lengths, logz=zh, out=oh, zero_logz=zz, zero_out=oz,
                soft_out=os, tangent=tangent, lg=lg)


def changed_pairs(iq, ik, lengths, j, symbol):
    """Exact sparse cone of a single changed K label, stopping at mismatches."""
    n = len(iq)
    if symbol == int(ik[j]):
        empty = torch.empty(0, dtype=torch.long)
        return empty, empty, empty, empty
    rows = torch.arange(j, n)
    rows = rows[(iq[rows] == ik[j]) | (iq[rows] == symbol)]
    keys = torch.full_like(rows, j)
    old = lengths[rows, keys].long()
    before = lengths[rows-1, keys-1].long() if j else torch.zeros_like(rows)
    new = (before + 1) * (iq[rows] == symbol)
    result = [[], [], [], []]
    while rows.numel():
        for dst, src in zip(result, (rows, keys, old, new)):
            dst.append(src)
        rows, keys = rows + 1, keys + 1
        valid = (rows < n) & (keys < n)
        rows, keys, old, new = rows[valid], keys[valid], old[valid], new[valid]
        valid = iq[rows] == ik[keys]
        rows, keys, old, new = rows[valid], keys[valid], old[valid]+1, new[valid]+1
    return tuple(torch.cat(x) if x else torch.empty(0, dtype=torch.long) for x in result)


def delta_read(rows, keys, old, new, base_out, logz, lg, sq, sk, v):
    n = len(sq)
    dw = (lg[new] - logz[rows]).exp() - (lg[old] - logz[rows]).exp()
    dz = torch.zeros(n, dtype=torch.float64).index_add_(0, rows, dw)
    dx = ((sq[rows] * sk[keys]).sum(-1)[:, None] * v[keys] - base_out[rows]) * dw[:, None]
    numerator = torch.zeros_like(base_out).index_add_(0, rows, dx)
    assert bool((1 + dz > 0).all()), 'unstable cancellation in counterfactual denominator'
    return numerator / (1 + dz[:, None])


def interventions(base, iq, ik, sq, sk, v, gs, positions, ids):
    m, count, candidates = len(gs), len(positions), ids.shape[-1]
    rewards = {name: torch.zeros((m, count, candidates), dtype=torch.float64)
               for name in ('full', 'endpoint', 'zero', 'numerator_only')}
    history_pairs = torch.zeros((count, candidates), dtype=torch.long)
    max_hop = torch.zeros_like(history_pairs)
    active_rows = torch.zeros_like(history_pairs)
    rounded = torch.zeros_like(rewards['full'])
    for a, j in enumerate(positions.tolist()):
        for b, symbol in enumerate(ids[a].tolist()):
            rows, keys, old, new = changed_pairs(iq, ik, base['lengths'], j, symbol)
            full = delta_read(rows, keys, old, new, base['out'], base['logz'], base['lg'], sq, sk, v)
            rewards['full'][:, a, b] = (gs * full).sum((1, 2))
            quant = (base['out'] + full).bfloat16().double() - base['out'].bfloat16().double()
            rounded[:, a, b] = (gs * quant).sum((1, 2))
            endpoint = keys == j
            ep = delta_read(rows[endpoint], keys[endpoint], old[endpoint], new[endpoint],
                            base['out'], base['logz'], base['lg'], sq, sk, v)
            rewards['endpoint'][:, a, b] = (gs * ep).sum((1, 2))
            zero = delta_read(rows[endpoint], keys[endpoint], (old[endpoint] > 0).long(),
                              (new[endpoint] > 0).long(), base['zero_out'], base['zero_logz'],
                              base['lg'], sq, sk, v)
            rewards['zero'][:, a, b] = (gs * zero).sum((1, 2))
            dw = (base['lg'][new]-base['logz'][rows]).exp() - (base['lg'][old]-base['logz'][rows]).exp()
            # Deliberately omit denominator changes to diagnose their importance.
            direct = dw[:, None] * (sq[rows]*sk[keys]).sum(-1)[:, None] * v[keys]
            rewards['numerator_only'][:, a, b] = (gs[:, rows]*direct).sum((1, 2))
            history_pairs[a, b] = int((~endpoint).sum())
            max_hop[a, b] = int((keys-j).max()) if len(keys) else 0
            active_rows[a, b] = rows.unique().numel()
    rewards['rounded'] = rounded
    tangent = base['tangent'].gather(2, ids[None].expand(m, -1, -1))
    rewards['soft_tangent'] = tangent - tangent[:, :, :1]
    return rewards, dict(history_pairs=history_pairs, max_hop=max_hop, active_rows=active_rows)


def moments(p, r, proxy=None, alpha=1., baseline=None):
    score = torch.eye(p.shape[-1], dtype=p.dtype)[None] - p[:, None]
    raw = r[:, :, None]*score
    target = (p[:, :, None]*raw).sum(1)
    est = raw.clone()
    if proxy is not None:
        cv = proxy[:, :, None]*score
        cv -= (p[:, :, None]*cv).sum(1, keepdim=True)
        est -= alpha*cv
    if baseline is not None:
        est -= baseline[:, None, None]*score
    torch.testing.assert_close((p[:, :, None]*est).sum(1), target, atol=1e-10, rtol=1e-9)
    var = (p[:, :, None]*(est-target[:, None]).square()).sum((1, 2))
    return target, var, est


def summarize(p, rewards):
    rows = []
    train = torch.arange(len(p)) % 2 == 0
    test = ~train
    for m, mode in enumerate(('hard', 'soft')):
        r = rewards['full'][m]; target, rawvar, raw = moments(p, r)
        centered = r - (p*r).sum(-1, keepdim=True)
        score = torch.eye(p.shape[-1], dtype=p.dtype)[None] - p[:, None]
        score2 = score.square().sum(-1)
        oracle = (p*r*score2).sum(-1)/(p*score2).sum(-1).clamp_min(1e-300)
        methods = [('paired_argmax', None, 1., None), ('oracle_scalar', None, 1., oracle)]
        for name in ('zero', 'endpoint', 'soft_tangent'):
            s = rewards[name][m]
            cv = s[:, :, None]*score
            cv -= (p[:, :, None]*cv).sum(1, keepdim=True)
            alpha = float((p[train, :, None]*(raw-target[:, None])[train]*cv[train]).sum() /
                          (p[train, :, None]*cv[train].square()).sum().clamp_min(1e-300))
            methods += [(name+'_unit', s, 1., None), (name+'_fit', s, alpha, None)]
        for name, proxy, alpha, baseline in methods:
            _, var, _ = moments(p, r, proxy, alpha, baseline)
            rows.append(dict(mode=mode, method=name, alpha=alpha,
                             signal=float(target[test].square().sum()), variance=float(var[test].sum()),
                             raw_variance=float(rawvar[test].sum()), count=int(test.sum())))
        detail = dict(mode=mode, method='diagnostic', count=int(test.sum()),
                      beneficial_fraction=float((r[test, 1:].min(-1).values < -1e-12).double().mean()),
                      nonzero_fraction=float((r[test, 1:].abs().max(-1).values > 1e-12).double().mean()),
                      reward_rms=float(r[test, 1:].square().mean().sqrt()),
                      gradient_rms=float(target[test].square().sum(-1).mean().sqrt()))
        for name in ('zero', 'endpoint', 'soft_tangent', 'rounded', 'numerator_only'):
            s = rewards[name][m]; sc = s - (p*s).sum(-1, keepdim=True)
            weight = p[test]
            detail[name+'_corr'] = float((weight*centered[test]*sc[test]).sum() /
                ((weight*centered[test].square()).sum()*(weight*sc[test].square()).sum()).sqrt().clamp_min(1e-300))
            detail[name+'_relative_error'] = float(((weight*(r[test]-s[test]).square()).sum() /
                (weight*r[test].square()).sum().clamp_min(1e-300)).sqrt())
            informative = (r[test, 1:].abs() > 1e-12) & (s[test, 1:].abs() > 1e-12)
            detail[name+'_sign_agreement'] = float(((r[test, 1:]*s[test, 1:] > 0)[informative]).double().mean()) if informative.any() else None
        rows.append(detail)
    return rows


def dense_reference(iq, ik, sq, sk, v, tau, pk=None, q=None, eq=None, endpoint=None):
    n = len(iq); previous = torch.zeros(n, dtype=torch.float64); out=[]
    for i in range(n):
        match = (iq[i] == ik).double() if pk is None else ((q[i]@eq.T).log_softmax(-1)@pk.T).exp()
        current = tau.exp()*(1+F.pad(previous[:-1], (1, 0)))*match*(torch.arange(n)<=i)
        if endpoint is None:
            read=current
        else:
            j, symbol = endpoint
            read=current.clone()
            if j <= i:
                history = previous[j-1] if j else previous.new_zeros(())
                read[j]=tau.exp()*(1+history)*(iq[i]==symbol)
        out.append(((read*(sk@sq[i]))[:, None]*v).sum(0)/(1+read.sum()))
        previous=current
    return torch.stack(out)


def self_test():
    from check_single_k_prefix_control import prefix_control
    torch.manual_seed(149); torch.set_default_dtype(torch.float64)
    for tau0 in (-.4, 0., .3):
        n,c,d,r,dv=19,5,4,3,6
        q,sq,sk,v,eq=[torch.randn(*s)*.3 for s in [(n,d),(n,r),(n,r),(n,dv),(c,d)]]
        pk=torch.randn(n,c).softmax(-1);iq=(q@eq.T).argmax(-1);ik=pk.argmax(-1)
        # Include a long common chain to exercise nonlocal propagation.
        iq[:8]=1;ik[:8]=1
        pos=torch.tensor([0,1,5,9,18]);ids=torch.stack([torch.cat((ik[j:j+1],torch.arange(c)[torch.arange(c)!=ik[j]]))[:4] for j in pos])
        gs=torch.randn(2,n,dv);tau=torch.tensor(tau0)
        b=baselines(q,sq,sk,v,eq,pk,iq,ik,tau,pos,gs)
        ref=dense_reference(iq,ik,sq,sk,v,tau)
        torch.testing.assert_close(b['out'],ref,atol=1e-12,rtol=1e-10)
        rewards,_=interventions(b,iq,ik,sq,sk,v,gs,pos,ids)
        zero_out,zero_rewards=prefix_control(iq,ik,sq,sk,v,gs,tau,pos,ids,c)
        torch.testing.assert_close(zero_out,b['zero_out'],atol=1e-12,rtol=1e-10)
        torch.testing.assert_close(zero_rewards,rewards['zero'],atol=1e-12,rtol=1e-9)
        for a,j in enumerate(pos):
            for z,symbol in enumerate(ids[a]):
                mutated=ik.clone();mutated[j]=symbol
                full=dense_reference(iq,mutated,sq,sk,v,tau)
                ep=dense_reference(iq,ik,sq,sk,v,tau,endpoint=(j,symbol))
                for name,out in [('full',full),('endpoint',ep)]:
                    torch.testing.assert_close(rewards[name][:,a,z],(gs*(out-ref)).sum((1,2)),atol=1e-12,rtol=1e-9)
        p=pk.clone().requires_grad_();soft=dense_reference(iq,ik,sq,sk,v,tau,pk=p,q=q,eq=eq)
        torch.testing.assert_close(b['soft_out'],soft,atol=1e-12,rtol=1e-10)
        for m in range(2):
            grad=torch.autograd.grad((soft*gs[m]).sum(),p,retain_graph=True)[0][pos]
            torch.testing.assert_close(b['tangent'][m]-b['tangent'][m,:,:1],grad-grad[:,:1],atol=1e-11,rtol=1e-9)
        pi=pk[pos].gather(1,ids);pi/=pi.sum(-1,keepdim=True)
        summarize(pi,rewards)
    print('PASS: full-history and endpoint interventions vs independent dense recomputation; full soft tangent vs autograd; exact CV expectations; linear prefix/suffix zero-history control',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path);p.add_argument('--output',type=Path);p.add_argument('--model')
    p.add_argument('--heads',type=int,nargs='+',default=[0,1,4,7]);p.add_argument('--positions',type=int,default=64)
    p.add_argument('--threads',type=int,default=2);p.add_argument('--self-test',action='store_true')
    a=p.parse_args();torch.set_num_threads(a.threads);torch.set_num_interop_threads(1)
    if a.self_test:self_test();return
    a.output.mkdir(parents=True,exist_ok=True)
    data=torch.load(a.input,map_location='cpu',weights_only=False);vals=data['inputs'][0]
    assert vals[8]==1. and bool(vals[9].all())
    n=vals[0].shape[1];assert vals[11].tolist()==[0,n]
    pos=torch.linspace(n//16,n*15//16-1,a.positions).long();assert len(pos.unique())==a.positions
    source=json.loads(a.input.with_name('results.json').read_text())
    sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();inputsha=hashlib.sha256(a.input.read_bytes()).hexdigest()
    for h in a.heads:
        dest=a.output/f'{a.model}_L{data["layer"]}_H{h}'
        start=time.perf_counter()
        with torch.no_grad():
            q,k,sq,sk,v=[x[0,:,h].double().contiguous() for x in vals[:5]]
            eq,ek=[x[h].bfloat16().double() for x in vals[5:7]];tau=vals[7][h].double()
            zq,zk=q@eq.T,k@ek.T;iq,ik=zq.argmax(-1),zk.argmax(-1);pk=zk.softmax(-1)
            ids=zk[pos].argsort(dim=-1,descending=True,stable=True)[:,:4];pi=zk[pos].gather(1,ids).softmax(-1)
            gs=torch.stack([data['g_'+m][0,:,h].double()*n for m in ('hard','soft')])
            b=baselines(q,sq,sk,v,eq,pk,iq,ik,tau,pos,gs)
            rewards,extent=interventions(b,iq,ik,sq,sk,v,gs,pos,ids)
            report=dict(model=a.model,layer=data['layer'],head=h,context=n,document=data['document'],
                checkpoint=source['checkpoint'],input=str(a.input.resolve()),input_sha256=inputsha,source_sha256=sha,
                tau=float(tau),positions=a.positions,heldout=a.positions//2,
                fp32_q_label_changes=int((iq!=(q.float()@eq.float().T).argmax(-1)).sum()),
                fp32_k_label_changes=int((ik!=(k.float()@ek.float().T).argmax(-1)).sum()),
                soft_output_relative_error=float((b['soft_out']-data['o_soft'][0,:,h].double()).norm()/data['o_soft'][0,:,h].double().norm()),
                max_history_hop=int(extent['max_hop'].max()),
                history_pair_fraction=float(extent['history_pairs'][:,1:].sum()/extent['active_rows'][:,1:].sum().clamp_min(1)),
                mean_affected_rows=float(extent['active_rows'][:,1:].double().mean()),
                results=summarize(pi,rewards),seconds=time.perf_counter()-start)
            torch.save(dict(report=report,pi=pi,ids=ids,positions=pos,rewards=rewards,extent=extent),dest.with_suffix('.pt'))
            dest.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
            print(dest.name,'seconds',round(report['seconds'],2),'hop',report['max_history_hop'],flush=True)
            del b


if __name__=='__main__':main()
