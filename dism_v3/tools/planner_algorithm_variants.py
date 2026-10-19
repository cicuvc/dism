"""Algebra variants, exclusively for the paired diagnostic harness.

Source transformations target 8d47ebf; none changes installed dispatch. Keep
failed numerical candidates here so their measurements can be reproduced.
"""

ALGORITHM_VARIANTS = ('tau_vectors', 'tau_compensated', 'tau_rowcol', 'scan32',
                      'persistent_tf32', 'single_pair')


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f'Algorithm variant requires the 8d47ebf source: {old[:80]!r}')
    return source.replace(old, new, 1)


def algorithm_source(source, name):
    tau = '    dtau = tl.sum(tl.sum(ds*score*(dw[:, None]+dw[None, :]), 1), 0)'
    if name == 'tau_rowcol':
        source = replace_once(source, tau, '''    energy = ds*score
    dtau = tl.sum(dw*(tl.sum(energy, 1)+tl.sum(energy, 0)), 0)''')
    elif name in ('tau_vectors', 'tau_compensated'):
        source = replace_once(source, tau, '    dtau = tl.full((), 0., tl.float32)')
        source = replace_once(source, '\n    tl.atomic_add(DSQ +',
                              '\n    dtau += tl.sum(dw*tl.sum(sq*dsq, 1), 0)\n    tl.atomic_add(DSQ +')
        source = replace_once(source, '\n    tl.atomic_add(DSK +',
                              '\n    dtau += tl.sum(dw*tl.sum(sk*dsk, 1), 0)\n    tl.atomic_add(DSK +')
        if name == 'tau_compensated':
            source = replace_once(source, '''    if BF16:
        ds_mma = ds.to(tl.bfloat16)
        dsq = tl.dot(ds_mma, sk.to(tl.bfloat16))''', '''    if COMPENSATED:
        dsq = tl.trans(_bf16_fp32_dot(tl.trans(sk), tl.trans(ds)))
    elif BF16:
        ds_mma = ds.to(tl.bfloat16)
        dsq = tl.dot(ds_mma, sk.to(tl.bfloat16))''')
            source = replace_once(source, '''    if BF16:
        dsk = tl.dot(tl.trans(ds_mma), sq.to(tl.bfloat16))''', '''    if COMPENSATED:
        dsk = tl.trans(_bf16_fp32_dot(tl.trans(sq), ds))
    elif BF16:
        dsk = tl.dot(tl.trans(ds_mma), sq.to(tl.bfloat16))''')
    elif name == 'single_pair':
        start = source.index('    K: tl.constexpr = C if BC == 0 else BC')
        end = source.index('\n\n@tr.jit\ndef _chunk_summaries', start)
        original = source[start:end]
        source = source[:start] + '''    if CHUNKS is None and COMPENSATED:
        length = tl.load(LENGTH + chunk)
        row0, row1 = tl.load(ROWS+chunk*C), tl.load(ROWS+chunk*C+1)
        one_pair = (length == 2) & (row0 >= 0) & (row1 < 0)
    else:
        one_pair = False
        row0, row1 = 0, -1
    if one_pair:
        _single_pair(SQ, SK, V, DO, DSQ, DSK, DV_OUT, TAU, PREFIX, WEIGHT,
                     DLOG, chunk*C, stream, row0, -row1-1, R, DV, BR, BD, BF16)
    else:
''' + ''.join('    '+line+'\n' for line in original.splitlines()) + source[end:]
        source = source.replace('@tr.jit\ndef _chunk_backward', SINGLE_PAIR+'\n\n@tr.jit\ndef _chunk_backward', 1)
    elif name == 'persistent_tf32':
        # Existing register recurrence, accurate mode only. Keep the compact
        # single buckets and consume exactly the same multi event arena.
        source = source.replace("long_mode='three_stage'", "long_mode='scan'")
        source = source.replace('*metadata, *config,\n                STREAM_IDS=',
                                '*metadata, r, v, self.c, max(32, r), 64, False,\n                STREAM_IDS=')
        source = replace_once(source, "                raise ValueError('compact plans require long_mode three_stage')", '''                args = (m['offsets'], m['rows'], m['prefixes'], m['weights'],
                        m['dlog'], m['lengths'], m['resets'], r, v, self.c,
                        max(32, r), 64, False)
                self.last_kernels.append(_query_backward[(self.multi_streams, 1)](
                    sq, sk, value, dout, dsq, partials[0, base:], *args,
                    num_warps=8, num_stages=1))
                self.last_kernels.append(_key_backward[(self.multi_streams, 1)](
                    sq, sk, value, dout, dsk, dv, partials[1, base:], *args,
                    num_warps=8, num_stages=1))''')
    elif name == 'scan32':
        start = source.index('@tr.jit\ndef _summary_scan(')
        end = source.index('\n\nclass FusedHardPrefillBackward', start)
        source = source[:start] + SCAN32 + source[end:]
        source = source.replace('_summary_scan[(self.multi_count, 1)]',
                                '_summary_scan[(self.multi_count, tr.cdiv(r*v, 128))]')
        source = source.replace('_summary_scan[(self.multi_streams, 1)]',
                                '_summary_scan[(self.multi_streams, tr.cdiv(r*v, 128))]')
        source = source.replace('r*v, r*v, num_warps=4)', 'r*v, 128, num_warps=4)')
    else:
        raise ValueError(name)
    return source.replace('sq.shape[1] not in (16, 32)', 'sq.shape[1] not in (16, 32, 64)')


SINGLE_PAIR = '''@tr.jit
def _single_pair(SQ, SK, V, DO, DSQ, DSK, DV_OUT, TAU, PREFIX, WEIGHT,
                 DLOG, pos, stream, ki, qi, R: tl.constexpr, DV: tl.constexpr,
                 BR: tl.constexpr, BD: tl.constexpr, BF16: tl.constexpr):
    r, d = tl.arange(0, BR), tl.arange(0, BD)
    sq = tl.load(SQ+qi*R+r, r < R, 0).to(tl.float32)
    sk = tl.load(SK+ki*R+r, r < R, 0).to(tl.float32)
    v = tl.load(V+ki*DV+d, d < DV, 0).to(tl.float32)
    do = tl.load(DO+qi*DV+d, d < DV, 0).to(tl.float32)
    if BF16:
        sq, sk = sq.to(tl.bfloat16).to(tl.float32), sk.to(tl.bfloat16).to(tl.float32)
        v, do = v.to(tl.bfloat16).to(tl.float32), do.to(tl.bfloat16).to(tl.float32)
    p0, p1 = tl.load(PREFIX+pos), tl.load(PREFIX+pos+1)
    w0, w1 = tl.load(WEIGHT+pos), tl.load(WEIGHT+pos+1)
    pair = tl.exp(tl.minimum(p1-p0, 0.))*w1*w0
    ds = tl.sum(do*v, 0)*pair
    score = tl.sum(sq*sk, 0)
    tau = ds*score*(tl.load(DLOG+pos)+tl.load(DLOG+pos+1))
    sp = score*pair
    if BF16:
        ds, sp = ds.to(tl.bfloat16).to(tl.float32), sp.to(tl.bfloat16).to(tl.float32)
    tl.atomic_add(DSQ+qi*R+r, ds*sk, r < R, sem="relaxed")
    tl.atomic_add(DSK+ki*R+r, ds*sq, r < R, sem="relaxed")
    tl.atomic_add(DV_OUT+ki*DV+d, sp*do, d < DV, sem="relaxed")
    tl.store(TAU+stream, tau)
'''


SCAN32 = '''@tr.jit
def _affine_join(a, b, c, d):
    return c*a, c*b+d


@tr.jit
def _summary_scan(H, G, A, OFFSETS, M: tl.constexpr, BLOCK: tl.constexpr):
    # Each CTA owns a matrix tile over its whole stream. Load both directions
    # before in-place stores; there is no inter-CTA state or polling protocol.
    stream, tile = tl.program_id(0), tl.program_id(1)
    begin, end = tl.load(OFFSETS+stream), tl.load(OFFSETS+stream+1)
    t, x = tl.arange(0, 32), tile*BLOCK+tl.arange(0, BLOCK)
    h, g = tl.full((BLOCK,), 0., tl.float32), tl.full((BLOCK,), 0., tl.float32)
    for base in range(0, end-begin, 32):
        f, b = begin+base+t, end-1-base-t
        valid = base+t < end-begin
        fp, bp = f[:, None].to(tl.int64)*M+x[None, :], b[:, None].to(tl.int64)*M+x[None, :]
        sh = tl.load(H+fp, valid[:, None] & (x[None, :] < M), 0.)
        sg = tl.load(G+bp, valid[:, None] & (x[None, :] < M), 0.)
        ah = tl.broadcast_to(tl.load(A+f, valid, 1.)[:, None], (32, BLOCK))
        ag = tl.broadcast_to(tl.load(A+b, valid, 1.)[:, None], (32, BLOCK))
        ph, ih = tl.associative_scan((ah, sh), 0, _affine_join)
        pg, ig = tl.associative_scan((ag, sg), 0, _affine_join)
        ih, ig = ph*h[None, :]+ih, pg*g[None, :]+ig
        prev = tl.broadcast_to(tl.maximum(t-1, 0)[:, None], (32, BLOCK))
        eh = tl.where(t[:, None] == 0, h[None, :], tl.gather(ih, prev, 0))
        eg = tl.where(t[:, None] == 0, g[None, :], tl.gather(ig, prev, 0))
        tl.store(H+fp, eh, valid[:, None] & (x[None, :] < M))
        tl.store(G+bp, eg, valid[:, None] & (x[None, :] < M))
        h = tl.sum(tl.where(t[:, None] == 31, ih, 0.), 0)
        g = tl.sum(tl.where(t[:, None] == 31, ig, 0.), 0)
'''
