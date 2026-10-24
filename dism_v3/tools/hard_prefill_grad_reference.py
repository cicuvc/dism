"""CPU-only diagnostic for differentiating the native hard-prefill program.

This is NOT a production backend. SymbolicPlanner retains the original whole-tree
centroid construction and integer match lengths. The native planner may use a
different event topology; compare evaluated results and derivatives, not event
arrays. Only the dense oracle is quadratic; event evaluation is not.
"""
from dataclasses import dataclass
import heapq
import math

import numpy as np


class SymbolicPlanner:
    def __init__(self, queries, keys):
        q, k = list(map(int, queries)), list(map(int, keys))
        if not q or len(q) != len(k):
            raise ValueError('one nonempty independent document required')
        self.n = len(q)
        self.link, self.length, self.next = [-1], [0], [{}]
        self.endpoint = []
        last = 0
        for ch in k:
            cur = len(self.link)
            self.link.append(0)
            self.length.append(self.length[last] + 1)
            self.next.append({})
            p = last
            while p >= 0 and ch not in self.next[p]:
                self.next[p][ch] = cur
                p = self.link[p]
            if p >= 0:
                target = self.next[p][ch]
                if self.length[p] + 1 == self.length[target]:
                    self.link[cur] = target
                else:
                    clone = len(self.link)
                    self.link.append(self.link[target])
                    self.length.append(self.length[p] + 1)
                    self.next.append(self.next[target].copy())
                    while p >= 0 and self.next[p].get(ch) == target:
                        self.next[p][ch] = clone
                        p = self.link[p]
                    self.link[cur] = self.link[target] = clone
            last = cur
            self.endpoint.append(last)
        count = len(self.link)
        self.adj = [[] for _ in range(count)]
        for v in range(1, count):
            self.adj[v].append(self.link[v])
            self.adj[self.link[v]].append(v)
        self.owner = [0] * count
        self.parent = [-1] * count
        self.size = [0] * count
        self.removed = [False] * count
        self.lca = [0] * count
        self.qnode, self.matched = [], []
        state = length = 0
        for ch in q:
            while state and ch not in self.next[state]:
                state = self.link[state]
                length = min(length, self.length[state])
            if ch not in self.next[state]:
                state = length = 0
            else:
                state = self.next[state][ch]
                length += 1
            self.qnode.append(state)
            self.matched.append(length)
        self.streams = []
        self._decompose(0, list(range(self.n)), list(range(self.n)))

    def _fill_lca(self, c):
        chain, v = [], c
        while True:
            chain.append(v)
            p = self.link[v]
            if p < 0 or self.removed[p]:
                break
            v = p
        path = set(chain)
        top = chain[-1]
        self.lca[top] = top
        stack = [top]
        while stack:
            v = stack.pop()
            for w in self.adj[v]:
                if w == self.link[v] or self.removed[w]:
                    continue
                self.lca[w] = w if w in path else self.lca[v]
                stack.append(w)

    def _emit(self, queries, keys, c, mode):
        qs, ks = [], []
        # Length -1 means constant factor 1, NOT geometric weight W_1=e^tau.
        for i in queries:
            length = self.matched[i] if mode == 3 else min(self.matched[i], self.length[c])
            if mode == 1:
                length = min(self.matched[i], self.length[self.lca[self.qnode[i]]])
            if mode == 2:
                length = -1
            if length != 0:
                qs.append((i, length))
        if not qs:
            return
        for j in keys:
            if j > qs[-1][0]:
                break
            length = self.length[self.lca[self.endpoint[j]]] if mode == 2 else -1
            if length != 0:
                ks.append((j, length))
        if ks:
            self.streams.append((qs, ks))

    def _cross(self, a, b, c, pg):
        qp = [i for i in a[0] if self.owner[self.qnode[i]] == pg]
        qd = [i for i in a[0] if self.owner[self.qnode[i]] != pg]
        kp = [j for j in b[1] if self.owner[self.endpoint[j]] == pg]
        kd = [j for j in b[1] if self.owner[self.endpoint[j]] != pg]
        assert not (qp and kp)
        self._emit(qd, kd, c, 0)
        self._emit(qp, kd, c, 1)
        self._emit(qd, kp, c, 2)

    def _decompose(self, root, queries, keys):
        if not queries or not keys:
            return
        order = [root]
        self.parent[root] = -1
        for v in order:
            self.size[v] = 1
            for w in self.adj[v]:
                if not self.removed[w] and w != self.parent[v]:
                    self.parent[w] = v
                    order.append(w)
        for v in reversed(order[1:]):
            self.size[self.parent[v]] += self.size[v]
        c, best = root, len(order)
        for v in order:
            largest = max([len(order) - self.size[v]] +
                          [self.size[w] for w in self.adj[v]
                           if not self.removed[w] and self.parent[w] == v])
            if largest < best:
                c, best = v, largest
        self.removed[c] = True
        components, pg = [[c]], -1
        self.owner[c] = 0
        for neighbor in self.adj[c]:
            if self.removed[neighbor]:
                continue
            g = len(components)
            if neighbor == self.link[c]:
                pg = g
            nodes = [neighbor]
            self.parent[neighbor] = c
            for v in nodes:
                self.owner[v] = g
                for w in self.adj[v]:
                    if not self.removed[w] and w != self.parent[v]:
                        self.parent[w] = v
                        nodes.append(w)
            components.append(nodes)
        groups = [([], []) for _ in components]
        for i in queries:
            groups[self.owner[self.qnode[i]]][0].append(i)
        for j in keys:
            groups[self.owner[self.endpoint[j]]][1].append(j)
        if pg >= 0 and (groups[pg][0] or groups[pg][1]):
            self._fill_lca(c)
        work = groups.copy()
        heap = [(len(nodes), g) for g, nodes in enumerate(components)]
        heapq.heapify(heap)
        while len(heap) > 1:
            wa, a = heapq.heappop(heap)
            wb, b = heapq.heappop(heap)
            self._cross(work[a], work[b], c, pg)
            self._cross(work[b], work[a], c, pg)
            joined = tuple(list(heapq.merge(work[a][i], work[b][i])) for i in (0, 1))
            work[a] = work[b] = ([], [])
            heapq.heappush(heap, (wa + wb, len(work)))
            work.append(joined)
        self._emit(*groups[0], c, 3)
        for g in range(1, len(components)):
            self._decompose(components[g][0], *groups[g])

    def compile(self, tau):
        if not math.isfinite(tau):
            raise ValueError('finite tau required')
        # O(N) stable log geometric sums and their analytic derivatives,
        # including tau=0 and negative tau, without subtractive cancellation.
        logs = np.full(self.n + 1, -np.inf)
        derivs = np.zeros(self.n + 1)
        for length in range(1, self.n + 1):
            logs[length], derivs[length] = _logadd(
                logs[length-1], derivs[length-1], length*tau, float(length))
        def coefficient(length):
            return (0., 0.) if length == -1 else (logs[length], derivs[length])
        logden, dlogden = np.zeros(self.n), np.zeros(self.n)
        # Zero-value fallback contributes exp(0)=1, with derivative zero.
        for queries, keys in self.streams:
            cursor, mass, dmass = 0, -np.inf, 0.
            for i, qlength in queries:
                while cursor < len(keys) and keys[cursor][0] <= i:
                    b, db = coefficient(keys[cursor][1])
                    mass, dmass = _logadd(mass, dmass, b, db)
                    cursor += 1
                if mass != -np.inf:
                    a, da = coefficient(qlength)
                    logden[i], dlogden[i] = _logadd(logden[i], dlogden[i], a+mass, da+dmass)
        rows, decay, weight, dlogweight, offsets = [], [], [], [], [0]
        for queries, keys in self.streams:
            cursor, pivot = 0, -np.inf
            for i, qlength in queries:
                while cursor < len(keys) and keys[cursor][0] <= i:
                    j, klength = keys[cursor]
                    b, db = coefficient(klength)
                    nxt = max(pivot, b)
                    rows.append(j)
                    decay.append(math.exp(pivot-nxt))
                    weight.append(math.exp(b-nxt))
                    # Numerical rescaling pivots are stop-gradient constants.
                    # The represented operator is invariant to these pivots;
                    # physical numerator AND denominator derivatives remain.
                    dlogweight.append(db)
                    pivot = nxt
                    cursor += 1
                if pivot != -np.inf:
                    a, da = coefficient(qlength)
                    rows.append(-i-1)
                    decay.append(1.)
                    weight.append(math.exp(a+pivot-logden[i]))
                    dlogweight.append(da-dlogden[i])
            if len(rows) != offsets[-1]:
                offsets.append(len(rows))
        return Program(np.asarray(offsets), np.asarray(rows, dtype=np.int64),
                       np.asarray(decay), np.asarray(weight), np.asarray(dlogweight), logden, dlogden)


def _logadd(a, da, b, db):
    if a == -np.inf:
        return b, db
    if b == -np.inf:
        return a, da
    out = float(np.logaddexp(a, b))
    return out, math.exp(a-out)*da + math.exp(b-out)*db


@dataclass
class Program:
    offsets: np.ndarray
    rows: np.ndarray
    decay: np.ndarray
    weight: np.ndarray
    dlogweight: np.ndarray
    logden: np.ndarray
    dlogden: np.ndarray

    def evaluate(self, sq, sk, value, dout, dtype=np.float64):
        """Analytic backward: forward replay for dSQ, reverse carry for dSK/dV.

        Tau is accumulated from event-local vector gradients; no forward matrix
        snapshots are saved. The additional tangent matrix is a diagnostic
        cross-check only, not needed by the proposed backward implementation.
        """
        sq, sk, value, dout = [np.asarray(x, dtype=dtype) for x in (sq, sk, value, dout)]
        n, r = sq.shape
        dv = value.shape[1]
        out, tangent = np.zeros((n, dv), dtype), np.zeros((n, dv), dtype)
        dsq, dsk, dvalue = np.zeros_like(sq), np.zeros_like(sk), np.zeros_like(value)
        dtau = dtype(0.)
        for begin, end in zip(self.offsets[:-1], self.offsets[1:]):
            state, dstate = np.zeros((r, dv), dtype), np.zeros((r, dv), dtype)
            for e in range(begin, end):
                row = self.rows[e]
                a, b, db = map(dtype, (self.decay[e], self.weight[e], self.dlogweight[e]))
                if row >= 0:
                    update = b * np.outer(sk[row], value[row])
                    state = a*state + update
                    dstate = a*dstate + db*update
                else:
                    i = -row-1
                    contribution = b*(sq[i] @ state)
                    out[i] += contribution
                    tangent[i] += db*contribution + b*(sq[i] @ dstate)
                    local = b*(state @ dout[i])
                    dsq[i] += local
                    dtau += db*np.dot(sq[i], local)
            adjoint = np.zeros((r, dv), dtype)
            for e in range(end-1, begin-1, -1):
                row = self.rows[e]
                a, b, db = map(dtype, (self.decay[e], self.weight[e], self.dlogweight[e]))
                if row < 0:
                    i = -row-1
                    adjoint += b*np.outer(sq[i], dout[i])
                else:
                    local = b*(adjoint @ value[row])
                    dsk[row] += local
                    dvalue[row] += b*(adjoint.T @ sk[row])
                    dtau += db*np.dot(sk[row], local)
                    adjoint *= a
        return out, (dsq, dsk, dvalue, np.asarray(dtau)), tangent


def dense_oracle(queries, keys, sq, sk, value, tau, dout, direction=True):
    """The maintained v3 FP64 oracle + PyTorch autograd, not the event algebra."""
    import torch
    from flash_dism.reference.dism_v3_ref import dism_ref
    n = len(queries)
    vectors = [torch.tensor(x, dtype=torch.float64).reshape(1, n, 1, -1).requires_grad_()
               for x in (sq, sk, value)]
    t = torch.tensor([tau], dtype=torch.float64, requires_grad=True)
    q, k = [torch.zeros((1, n, 1, 32), dtype=torch.float64, requires_grad=True) for _ in range(2)]
    lq, lk = [torch.zeros((1, n, 1), dtype=torch.float64, requires_grad=True) for _ in range(2)]
    out = dism_ref(q, k, vectors[0], vectors[1], lq, lk,
                   torch.tensor(queries).reshape(1, 1, n), torch.tensor(keys).reshape(1, 1, n),
                   torch.tensor([[direction]]), torch.ones((1, 1, n), dtype=torch.bool), vectors[2], t)
    gradients = torch.autograd.grad(out, (*vectors, t, q, k, lq, lk),
                                    torch.tensor(dout, dtype=torch.float64).reshape_as(out))
    assert all(torch.count_nonzero(g) == 0 for g in gradients[4:])
    shapes = (np.shape(sq), np.shape(sk), np.shape(value), ())
    return out.detach().numpy().reshape(n, -1), tuple(g.numpy().reshape(s) for g, s in zip(gradients[:4], shapes))


def dense_exact_oracle(queries, keys, sq, sk, value, tau, dout):
    """Additional smooth FP64 oracle; does NOT replace maintained v3 comparisons.

    torch softplus defaults to its linear branch above 20. Hard event weights
    use exact geometric sums, so a separate logaddexp recurrence distinguishes
    this known mathematical difference from event-backward implementation errors.
    """
    import torch
    sq, sk, value = [torch.tensor(x, dtype=torch.float64, requires_grad=True) for x in (sq, sk, value)]
    t = torch.tensor(tau, dtype=torch.float64, requires_grad=True)
    q, k = torch.tensor(queries), torch.tensor(keys)
    n = len(q)
    previous = torch.full((n,), -torch.inf, dtype=torch.float64)
    rows = []
    for i in range(n):
        predecessor = torch.cat((torch.full((1,), -torch.inf), previous[:-1]))
        current = torch.where(q[i] == k, t+torch.logaddexp(torch.zeros_like(predecessor), predecessor), -torch.inf)
        causal = current.masked_fill(torch.arange(n) > i, -torch.inf)
        p = torch.softmax(torch.cat((torch.zeros(1, dtype=torch.float64), causal)), dim=0)[1:]
        rows.append(((sk @ sq[i])*p) @ value)
        previous = current
    out = torch.stack(rows)
    gradients = torch.autograd.grad(out, (sq, sk, value, t), torch.tensor(dout, dtype=torch.float64))
    return out.detach().numpy(), tuple(g.numpy() for g in gradients)


def assert_equivalent_programs(actual, expected, *, seed=318):
    """Compare different decompositions without imposing an event ordering.

    Use independent vector channels to exercise the numerator, all three vector
    gradients, tau gradient and forward tangent. Preserve the native FP64 math
    tolerance; packing/layout tests remain separate.
    """
    for name in ('logden', 'dlogden'):
        np.testing.assert_allclose(getattr(actual, name), getattr(expected, name),
                                   rtol=2e-9, atol=2e-10)
    rng = np.random.default_rng(seed)
    n = len(expected.logden)
    vectors = [rng.normal(size=(n, width))*.2 for width in (3, 3, 4, 4)]
    out, grads, tangent = actual.evaluate(*vectors)
    ref_out, ref_grads, ref_tangent = expected.evaluate(*vectors)
    for got, want in zip((out, *grads, tangent), (ref_out, *ref_grads, ref_tangent)):
        np.testing.assert_allclose(got, want, rtol=2e-9, atol=2e-10)
