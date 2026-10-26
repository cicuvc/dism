"""CPU-only, frozen-cotangent diagnostics for sampled local hard Q reads.

Consumes saved probe_selector_surrogate_gradients operands; never loads a model
or touches training state. All numerical work is FP64 at saved BF16 operands.
The Q mainline and K string remain argmax; only the current read is sampled.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F


def weighted_read(logw, values):
    maximum = logw.amax().clamp_min(0)
    w = (logw - maximum).exp()
    return (w @ values) / ((-maximum).exp() + w.sum())


def local_soft(logw, sqsk, v, pk, pq, ids, g):
    """Current-read d<g,Osoft>/dpQ; prior soft history is held fixed."""
    maximum = logw.amax().clamp_min(0)
    w = (logw - maximum).exp()
    w = w / ((-maximum).exp() + w.sum())
    o = (w * sqsk) @ v
    value = o @ g
    t = w * (sqsk * (v @ g) - value)
    tangent = (t @ pk[:, ids]) / pq[ids]
    return o, value, tangent


def log_geometry(tau, n):
    # G(l)=sum_{r=1}^{l+1} exp(r*tau), stable for long/large-tau chains.
    return torch.logcumsumexp(torch.arange(1, n + 2, dtype=torch.float64) * tau, 0)


def controls(p, r, soft_value, proxy=None):
    """Each action's raw estimator, target, and zero-mean controls.

    E[(r_C-alpha*s_C-b)*score_C]+alpha*E[s_C*score_C]=G.
    Values/coefficients/cotangents are constants in this diagnostic.
    """
    score = torch.eye(p.shape[-1], dtype=p.dtype)[None] - p[:, None, :]
    raw = r[:, :, None] * score
    target = (p[:, :, None] * raw).sum(1)
    baseline = soft_value[:, None, None] * score
    out = [baseline]
    if proxy is not None:
        x = proxy[:, :, None] * score
        out.append(x - (p[:, :, None] * x).sum(1, keepdim=True))
    return raw, target, torch.stack(out, -1)


def fit_controls(p, residual, control, train):
    # Solve in rescaled coordinates: no ridge or validation-set tuning.
    x = control[train]
    w = p[train, :, None, None]
    scale = (w * x.square()).sum((0, 1, 2)).sqrt().clamp_min(1e-300)
    xn = x / scale
    a = torch.einsum('nc,ncdj,ncdk->jk', p[train], xn, xn)
    b = torch.einsum('nc,ncdj,ncd->j', p[train], xn, residual[train])
    return (torch.linalg.pinv(a, rtol=1e-12) @ b) / scale


def extract_head(d, h, count):
    vals = d['inputs'][0]
    assert vals[8] == 1.0 and bool(vals[9].all())
    q, k, sq, sk, v = [x[0, :, h].double().contiguous() for x in vals[:5]]
    eq, ek = [x[h].to(torch.bfloat16).double() for x in vals[5:7]]
    n = len(q)
    assert vals[11].tolist() == [0, n], 'Only single-document saved operands supported'
    tau = vals[7][h].double()
    zq, zk = q @ eq.T, k @ ek.T
    iq, ik = zq.argmax(-1), zk.argmax(-1)
    iq32 = (q.float() @ eq.float().T).argmax(-1)
    ik32 = (k.float() @ ek.float().T).argmax(-1)
    pk, pq = zk.softmax(-1), zq.softmax(-1)
    emb, lq = pk @ eq, zq.logsumexp(-1)
    # Calibration queries precede evaluation queries; no fit uses test outcomes.
    positions = torch.cat([
        torch.linspace(n // 4, n // 2 - 1, count // 2).long(),
        torch.linspace(n // 2, n - 1, count - count // 2).long(),
    ])
    assert positions.unique().numel() == count
    wanted = set(positions.tolist())
    ids = zq[positions].argsort(dim=-1, descending=True, stable=True)[:, :4]
    pi = zq[positions].gather(1, ids).softmax(-1)
    logg = log_geometry(tau, n)
    hard_prev = torch.zeros(n, dtype=torch.long)
    soft_prev = torch.full((n,), -torch.inf, dtype=torch.float64)
    rows = []
    j = torch.arange(n)
    cursor = 0
    for start in range(0, n, 128):
        # Bounded pair tile: no N^2 V intermediate and no dense N^2 storage.
        raw = q[start:start + 128] @ emb.T - lq[start:start + 128, None] + tau
        for offset in range(len(raw)):
            i = start + offset
            hard_length = F.pad(hard_prev[:-1], (1, 0))
            loga = logg[hard_length]
            soft_loga = tau + F.softplus(F.pad(soft_prev[:-1], (1, 0), value=-torch.inf))
            soft_row = raw[offset] + soft_loga - tau
            if i in wanted:
                cand = ids[cursor]
                end = i + 1
                similarity = sk[:end] @ sq[i]
                values = similarity[:, None] * v[:end]
                outs, soft_history, zero_history = [], [], []
                for c in cand:
                    mask = ik[:end] == c
                    if not bool(mask.any()):
                        empty = torch.zeros(v.shape[-1], dtype=torch.float64)
                        outs.append(empty); soft_history.append(empty); zero_history.append(empty)
                    else:
                        outs.append(weighted_read(loga[:end][mask], values[mask]))
                        soft_history.append(weighted_read(soft_loga[:end][mask], values[mask]))
                        zero_history.append(weighted_read(tau.expand(int(mask.sum())), values[mask]))
                outs = torch.stack(outs)
                result = dict(hard=outs, soft_history=torch.stack(soft_history),
                              zero_history=torch.stack(zero_history))
                for mode in ('hard', 'soft'):
                    g = d['g_' + mode][0, i, h].double() * n
                    o, value, tangent = local_soft(soft_row[:end], similarity, v[:end],
                                                  pk[:end], pq[i], cand, g)
                    result[mode + '_value'] = value
                    result[mode + '_tangent'] = tangent
                    result[mode + '_r'] = outs @ g
                    result[mode + '_soft_history_r'] = result['soft_history'] @ g
                    result[mode + '_zero_history_r'] = result['zero_history'] @ g
                result['soft_output'] = o
                rows.append(result)
                cursor += 1
            hard_prev = (hard_length + 1) * (iq[i] == ik) * (j <= i)
            soft_prev = soft_row.masked_fill(j > i, -torch.inf)
    out = {key: torch.stack([r[key] for r in rows]) for key in rows[0]}
    out.update(pi=pi, ids=ids, positions=positions, q=q[positions], eq=eq,
               g_hard=d['g_hard'][0, positions, h].double() * n,
               g_soft=d['g_soft'][0, positions, h].double() * n,
               production_soft=d['o_soft'][0, positions, h].double())
    out['meta'] = dict(head=h, layer=d['layer'], document=d['document'], context=n,
                       float32_q_argmax_changes=int((iq32 != iq).sum()),
                       float32_k_argmax_changes=int((ik32 != ik).sum()),
                       tau=float(tau))
    return out


def self_test():
    torch.manual_seed(311)
    dtype = torch.float64
    # Independent autograd check of the frozen-history soft categorical tangent.
    n, c, dv = 13, 7, 5
    p = torch.randn(c, dtype=dtype).softmax(-1).requires_grad_()
    pk = torch.randn(n, c, dtype=dtype).softmax(-1)
    hist = torch.randn(n, dtype=dtype)
    sim = torch.randn(n, dtype=dtype); v = torch.randn(n, dv, dtype=dtype)
    g = torch.randn(dv, dtype=dtype)
    logw = pk @ p.log() + hist
    o = weighted_read(logw, sim[:, None] * v)
    actual = torch.autograd.grad(o @ g, p)[0]
    _, _, found = local_soft(logw.detach(), sim, v, pk, p.detach(), torch.arange(c), g)
    torch.testing.assert_close(found, actual, atol=1e-11, rtol=1e-10)
    # Exhaustive categorical expectation, including arbitrary action-dependent CV.
    p = torch.randn(27, 4, dtype=dtype).softmax(-1)
    r = torch.randn_like(p); s = torch.randn_like(p); b = torch.randn(27, dtype=dtype)
    raw, target, cv = controls(p, r, b, s)
    alpha = torch.tensor([1.73, -2.41], dtype=dtype)
    est = raw - cv @ alpha
    torch.testing.assert_close((p[:, :, None] * est).sum(1), target, atol=1e-12, rtol=1e-11)
    # Logit gradient independent autograd oracle (held-fixed support/values).
    z = p.log().requires_grad_()
    expected = torch.autograd.grad((z.softmax(-1) * r).sum(), z)[0]
    torch.testing.assert_close(target, expected, atol=1e-12, rtol=1e-11)
    # Hard match-length/log-geometric implementation vs ordinary recurrence.
    for tau in (-2., 0., .3, 4.):
        iq = torch.randint(5, (39,)); ik = torch.randint(5, (39,))
        lengths = torch.zeros(39, dtype=torch.long); prev = torch.zeros(39, dtype=dtype)
        lg = log_geometry(torch.tensor(tau, dtype=dtype), 39)
        for i in range(39):
            shifted = F.pad(lengths[:-1], (1, 0))
            a = torch.exp(torch.tensor(tau, dtype=dtype)) * (1 + F.pad(prev[:-1], (1, 0)))
            torch.testing.assert_close(lg[shifted].exp(), a, atol=1e-10, rtol=1e-10)
            valid = (iq[i] == ik) & (torch.arange(39) <= i)
            lengths = (shifted + 1) * valid; prev = a * valid
    # End-to-end extraction vs independent scalar recurrence and readout.
    n, c, dv, dq, rank = 64, 7, 5, 4, 3
    vals = [torch.randn(1, n, 1, dim, dtype=dtype) for dim in (dq, dq, rank, rank, dv)]
    vals += [torch.randn(1, c, dq, dtype=dtype) for _ in range(2)]
    vals += [torch.tensor([.2], dtype=dtype), 1., torch.ones(1, 1, dtype=torch.bool),
             torch.zeros(1, 1, dtype=torch.bool), torch.tensor([0, n])]
    d = dict(inputs=(vals, {}), layer=0, document=0, context=n,
             g_hard=torch.randn(1, n, 1, dv, dtype=dtype),
             g_soft=torch.randn(1, n, 1, dv, dtype=dtype),
             o_soft=torch.zeros(1, n, 1, dv, dtype=dtype))
    found = extract_head(d, 0, 16)
    q, k, sq, sk, v = [x[0, :, 0] for x in vals[:5]]
    eq, ek = [x[0].to(torch.bfloat16).double() for x in vals[5:7]]
    pq, pk = (q @ eq.T).softmax(-1), (k @ ek.T).softmax(-1)
    iq, ik = pq.argmax(-1), pk.argmax(-1)
    hp = torch.zeros(n, dtype=dtype); sp = hp.clone(); cursor = 0
    for i in range(n):
        ah = .2 + torch.log1p(F.pad(hp[:-1], (1, 0)))
        ash = .2 + torch.log1p(F.pad(sp[:-1], (1, 0)))
        ws = (pq[i].log() @ pk.T + ash).exp() * (torch.arange(n) <= i)
        values = (sk @ sq[i])[:, None] * v
        if i == int(found['positions'][cursor]):
            for name, weights in [('hard', ah.exp()), ('soft_history', ash.exp()),
                                  ('zero_history', torch.full((n,), .2, dtype=dtype).exp())]:
                rows = []
                for c0 in found['ids'][cursor]:
                    w = weights * (ik == c0) * (torch.arange(n) <= i)
                    rows.append(w @ values / (1 + w.sum()))
                torch.testing.assert_close(found[name][cursor], torch.stack(rows), atol=1e-11, rtol=1e-10)
            torch.testing.assert_close(found['soft_output'][cursor], ws @ values / (1 + ws.sum()), atol=1e-11, rtol=1e-10)
            cursor += 1
            if cursor == len(found['positions']):
                break
        hp = ah.exp() * (iq[i] == ik) * (torch.arange(n) <= i)
        sp = ws
    print('PASS: soft tangent autograd, corrected categorical expectation, router autograd, hard recurrence, complete candidate extraction', flush=True)


def analyze(paths, output):
    tables = []
    per_head = []
    sources = []
    for path in paths:
        d = torch.load(path, map_location='cpu', weights_only=False)
        sources.append(dict(artifact=str(path), checkpoint=d['checkpoint'],
                            input=d['input'], input_sha256=d['input_sha256'],
                            extraction_source_sha256=d['source_sha256']))
        for mode in ('hard', 'soft'):
            p, r, b = d['pi'], d[mode + '_r'], d[mode + '_value']
            train = d['positions'] < d['meta']['context'] // 2
            raw, target, _ = controls(p, r, b)
            score = torch.eye(4, dtype=p.dtype)[None] - p[:, None, :]
            score2 = score.square().sum(-1)
            oracle_b = (p * r * score2).sum(-1) / (p * score2).sum(-1)
            methods = {'raw': raw, 'soft_baseline': raw - b[:, None, None] * score,
                       'oracle_scalar_baseline': raw - oracle_b[:, None, None] * score}
            fitted = {}
            for name, proxy in [('fit_soft_baseline', None), ('soft_tangent', d[mode + '_tangent']),
                                ('soft_history', d[mode + '_soft_history_r']),
                                ('zero_history', d[mode + '_zero_history_r'])]:
                _, _, cv = controls(p, r, b, proxy)
                alpha = fit_controls(p, raw - target[:, None, :], cv, train)
                methods[name] = raw - cv @ alpha
                fitted[name] = alpha.tolist()
                if proxy is not None:
                    # No outcome-fitted coefficient or scalar baseline.
                    methods[name + '_unit'] = raw - cv[..., 1]
            eq_selected = d['eq'][d['ids']]
            test = ~train
            signal = target.square().sum(-1)
            q_target = torch.einsum('nc,ncd->nd', target, eq_selected)
            signal_q = q_target.square().sum(-1)
            for method, estimate in methods.items():
                mean = (p[:, :, None] * estimate).sum(1)
                torch.testing.assert_close(mean, target, atol=1e-10, rtol=1e-9)
                delta = estimate - target[:, None, :]
                var = (p[:, :, None] * delta.square()).sum((1, 2))
                delta_q = torch.einsum('ncj,njd->ncd', delta, eq_selected)
                var_q = (p[:, :, None] * delta_q.square()).sum((1, 2))
                # Shared Q-codebook: independent per-token draws; variances add.
                dz_full = torch.zeros((len(p), len(d['eq'])), dtype=p.dtype).scatter(1, d['ids'], target)
                codebook_mean = dz_full[test].T @ d['q'][test]
                row = dict(model=d['model'], mode=mode, method=method, **d['meta'],
                           count=int(test.sum()), variance=float(var[test].sum()),
                           signal=float(signal[test].sum()), q_variance=float(var_q[test].sum()),
                           q_signal=float(signal_q[test].sum()),
                           codebook_variance=float((var[test] * d['q'][test].square().sum(-1)).sum()),
                           codebook_signal=float(codebook_mean.square().sum()),
                           coefficients=fitted.get(method))
                tables.append(row)
            mixed = (p[:, :, None] * d['hard']).sum(1)
            output_variance = (p[test, :, None] * (d['hard'][test] - mixed[test, None]).square()).sum()
            corrected_output = {}
            for proxy in ('soft_history', 'zero_history'):
                mean_proxy = (p[:, :, None] * d[proxy]).sum(1)
                out_cv = mean_proxy[:, None] + d['hard'] - d[proxy]
                torch.testing.assert_close((p[:, :, None] * out_cv).sum(1), mixed, atol=1e-11, rtol=1e-10)
                cv_var = (p[test, :, None] * (out_cv[test] - mixed[test, None]).square()).sum()
                corrected_output[proxy] = float(cv_var)
            pair = {}
            for name in ('tangent', 'soft_history_r', 'zero_history_r'):
                s = d[mode + '_' + name]
                rc = r - (p * r).sum(-1, keepdim=True)
                sc = s - (p * s).sum(-1, keepdim=True)
                corr = (p[test] * rc[test] * sc[test]).sum() / (
                    (p[test] * rc[test].square()).sum() * (p[test] * sc[test].square()).sum()).sqrt().clamp_min(1e-300)
                pair[name] = float(corr)
            ref = d['production_soft']
            per_head.append(dict(model=d['model'], mode=mode, **d['meta'],
                                 soft_output_relative_error=float((d['soft_output'] - ref).norm() / ref.norm().clamp_min(1e-30)),
                                 output_variance=float(output_variance), output_signal=float(mixed[test].square().sum()),
                                 corrected_output_variance=corrected_output,
                                 candidate_centered_correlation=pair,
                                 zero_hard_value_fraction=float((r == 0).double().mean())))
    groups = {}
    for row in tables:
        key = (row['model'], row['mode'], row['method'])
        if key not in groups:
            groups[key] = dict(model=key[0], mode=key[1], method=key[2], heads=0,
                               **{s: 0 for s in ('count', 'variance', 'signal', 'q_variance', 'q_signal',
                                                 'codebook_variance', 'codebook_signal')})
        dst = groups[key]; dst['heads'] += 1
        for s in ('count', 'variance', 'signal', 'q_variance', 'q_signal', 'codebook_variance', 'codebook_signal'):
            dst[s] += row[s]
    for key, row in groups.items():
        base = groups[(key[0], key[1], 'raw')]
        for prefix in ('', 'q_', 'codebook_'):
            row[prefix + 'variance_ratio'] = row[prefix + 'variance'] / max(base[prefix + 'variance'], 1e-300)
            row[prefix + 'snr'] = (row[prefix + 'signal'] / max(row[prefix + 'variance'], 1e-300)) ** .5
    report = dict(protocol='FP64 frozen-cotangent local Q read; exact categorical moments. Per-head coefficients fit on first query half, evaluated on later queries. Neither full LM gradient nor sampled-forward upstream.',
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  torch_version=torch.__version__, sources=sources,
                  aggregate=list(groups.values()), per_head=tables, diagnostics=per_head)
    output.write_text(json.dumps(report, indent=2) + '\n')
    for row in groups.values():
        if row['mode'] == 'hard':
            print(row['model'], row['method'], 'variance/raw', round(row['variance_ratio'], 4),
                  'SNR', round(row['snr'], 4), 'Q SNR', round(row['q_snr'], 4), flush=True)


def rounding_audit(paths, output):
    """Output-rounding sensitivity only, not a native BF16 kernel comparison."""
    rows = []
    for path in paths:
        d = torch.load(path, map_location='cpu', weights_only=False)
        p = d['pi']; test = d['positions'] >= d['meta']['context'] // 2
        for quant in ('fp64', 'bf16_output_rounding'):
            cast = (lambda x: x) if quant == 'fp64' else (lambda x: x.to(torch.bfloat16).double())
            hard, base, g = cast(d['hard']), cast(d['zero_history']), d['g_hard']
            r = (hard * g[:, None]).sum(-1); s = (base * g[:, None]).sum(-1)
            raw, target, cv = controls(p, r, torch.zeros(len(p), dtype=p.dtype), s)
            est = raw - cv[..., 1]
            rv = (p[:, :, None] * (raw - target[:, None]).square()).sum((1, 2))
            vc = (p[:, :, None] * (est - target[:, None]).square()).sum((1, 2))
            mean = (p[:, :, None] * hard).sum(1); pm = (p[:, :, None] * base).sum(1)
            ov = (p[:, :, None] * (hard - mean[:, None]).square()).sum((1, 2))
            ocv = (p[:, :, None] * (hard - base - (mean - pm)[:, None]).square()).sum((1, 2))
            rows.append(dict(model=d['model'], quant=quant, layer=d['meta']['layer'], head=d['meta']['head'],
                             raw=float(rv[test].sum()), cv=float(vc[test].sum()),
                             output_raw=float(ov[test].sum()), output_cv=float(ocv[test].sum()),
                             output_signal=float(mean[test].square().sum()),
                             deterministic_proxy_error=float((mean[test] - pm[test]).square().sum())))
    aggregate = []
    for model in sorted({x['model'] for x in rows}):
        for quant in ('fp64', 'bf16_output_rounding'):
            selected = [x for x in rows if x['model'] == model and x['quant'] == quant]
            sums = {k: sum(x[k] for x in selected) for k in ('raw', 'cv', 'output_raw', 'output_cv',
                                                           'output_signal', 'deterministic_proxy_error')}
            aggregate.append(dict(model=model, quant=quant, **sums,
                                  gradient_variance_ratio=sums['cv'] / max(sums['raw'], 1e-300),
                                  corrected_output_relative_rms=(sums['output_cv'] / max(sums['output_signal'], 1e-300)) ** .5,
                                  deterministic_proxy_relative_error=(sums['deterministic_proxy_error'] / max(sums['output_signal'], 1e-300)) ** .5))
    output.write_text(json.dumps(dict(aggregate=aggregate, per_head=rows), indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path)
    parser.add_argument('--model')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--heads', type=int, nargs='+', default=list(range(12)))
    parser.add_argument('--queries', type=int, default=256)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--analyze', action='store_true')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    if args.self_test:
        self_test()
        return
    if args.analyze:
        paths = sorted(args.output.glob('*.pt'))
        analyze(paths, args.output / 'summary.json')
        rounding_audit(paths, args.output / 'rounding_audit.json')
        return
    assert args.input and args.model and args.output
    args.output.mkdir(parents=True, exist_ok=True)
    d = torch.load(args.input, map_location='cpu', weights_only=False)
    source = json.loads(args.input.with_name('results.json').read_text())
    digest = hashlib.sha256(args.input.read_bytes()).hexdigest()
    for h in args.heads:
        dest = args.output / f'{args.model}_L{d["layer"]}_H{h}.pt'
        if dest.exists():
            old = torch.load(dest, map_location='cpu', weights_only=False)
            assert old['input_sha256'] == digest and len(old['positions']) == args.queries
            print('skip', dest.name, flush=True)
            continue
        start = time.perf_counter()
        with torch.no_grad():
            result = extract_head(d, h, args.queries)
        result.update(model=args.model, checkpoint=source['checkpoint'], input=str(args.input.resolve()),
                      input_sha256=digest, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        torch.save(result, dest)
        print(dest.name, 'seconds', round(time.perf_counter() - start, 2), flush=True)


if __name__ == '__main__':
    main()
