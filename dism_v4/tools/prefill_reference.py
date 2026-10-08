"""CPU numerical prototype for offline full-hard prefill (one sequence/head).

No production imports, no CUDA, no per-SAM-node vector matrices. Dense audit
is explicitly optional. See ../docs/PREFILL_PLAN.md for complexity caveats.
"""
from dataclasses import dataclass
import heapq
import math

import numpy as np


class SAM:
    def __init__(self, keys):
        self.length = [0]
        self.link = [-1]
        self.next = [{}]
        self.endpoints = []
        last = 0
        for token in keys:
            cur = self._node(self.length[last] + 1)
            self.endpoints.append(cur)
            p = last
            while p >= 0 and token not in self.next[p]:
                self.next[p][token] = cur
                p = self.link[p]
            if p < 0:
                self.link[cur] = 0
            else:
                q = self.next[p][token]
                if self.length[p] + 1 == self.length[q]:
                    self.link[cur] = q
                else:
                    clone = self._node(self.length[p] + 1)
                    self.next[clone] = self.next[q].copy()
                    self.link[clone] = self.link[q]
                    while p >= 0 and self.next[p].get(token) == q:
                        self.next[p][token] = clone
                        p = self.link[p]
                    self.link[q] = self.link[cur] = clone
            last = cur

        self.adj = [[] for _ in self.length]
        for node in range(1, len(self.length)):
            parent = self.link[node]
            self.adj[parent].append(node)
            self.adj[node].append(parent)
        self.depth = [0] * len(self.length)
        order = [0]
        for node in order:
            for child in self.adj[node]:
                if child != self.link[node]:
                    self.depth[child] = self.depth[node] + 1
                    order.append(child)
        self.up = [[max(0, p) for p in self.link]]
        for _ in range(len(self.length).bit_length()):
            prev = self.up[-1]
            self.up.append([prev[prev[v]] for v in range(len(prev))])

    def _node(self, length):
        self.length.append(length)
        self.link.append(-1)
        self.next.append({})
        return len(self.length) - 1

    def lca(self, a, b):
        if self.depth[a] < self.depth[b]:
            a, b = b, a
        gap = self.depth[a] - self.depth[b]
        for bit in range(gap.bit_length()):
            if gap & (1 << bit):
                a = self.up[bit][a]
        if a == b:
            return a
        for table in reversed(self.up):
            if table[a] != table[b]:
                a, b = table[a], table[b]
        return self.link[a]

    def scan(self, queries, resets):
        nodes, lengths = [], []
        state = matched = 0
        for token, reset in zip(queries, resets):
            if reset:
                state = matched = 0
            while state and token not in self.next[state]:
                state = self.link[state]
                matched = min(matched, self.length[state])
            if token in self.next[state]:
                state = self.next[state][token]
                matched += 1
            else:
                state = matched = 0
            assert state == 0 or self.length[self.link[state]] < matched <= self.length[state]
            nodes.append(state)
            lengths.append(matched)
        return nodes, lengths


def log_weight(length, tau):
    if length == 0:
        return -math.inf
    if tau == 0:
        return math.log(length)
    # Factor out the largest term, including for negative tau.
    t = abs(tau)
    geometric = math.log(-math.expm1(-length * t)) - math.log(-math.expm1(-t))
    return (length * tau if tau > 0 else tau) + geometric


@dataclass
class Stream:
    queries: list
    keys: list
    query_log: np.ndarray
    key_log: np.ndarray


class PrefillPlan:
    def __init__(self, queries, keys, resets, tau):
        if not (len(queries) == len(keys) == len(resets)) or not len(keys):
            raise ValueError("Expected nonempty equally sized query/key/reset arrays")
        if not math.isfinite(tau):
            raise ValueError("tau must be finite")
        self.n = len(keys)
        self.sam = SAM(keys)
        self.qnode, self.matched = self.sam.scan(queries, resets)
        self.streams = []
        self.tau = tau
        self._decompose(set(range(len(self.sam.length))), list(range(self.n)), list(range(self.n)))

    def _emit(self, queries, keys, query_depth=None, key_depth=None):
        if not queries or not keys:
            return
        qa = np.array([log_weight(query_depth(i), self.tau) for i in queries]) if query_depth else np.zeros(len(queries))
        kb = np.array([log_weight(key_depth(j), self.tau) for j in keys]) if key_depth else np.zeros(len(keys))
        self.streams.append(Stream(queries, keys, qa, kb))

    def _decompose(self, vertices, queries, keys):
        if not queries or not keys:
            return
        # Linear-time centroid search within this component.
        root = next(iter(vertices))
        parent = {root: -1}
        order = [root]
        for v in order:
            for w in self.sam.adj[v]:
                if w in vertices and w != parent[v]:
                    parent[w] = v
                    order.append(w)
        size = dict.fromkeys(vertices, 1)
        for v in reversed(order[1:]):
            size[parent[v]] += size[v]
        total = len(vertices)
        centroid = min(order, key=lambda v: max(
            [total - size[v]] + [size[w] for w in self.sam.adj[v] if parent.get(w) == v]))

        groups = [{centroid}]
        parent_group = -1
        for neighbor in self.sam.adj[centroid]:
            if neighbor not in vertices:
                continue
            component = {neighbor}
            stack = [neighbor]
            for v in stack:
                for w in self.sam.adj[v]:
                    if w in vertices and w != centroid and w not in component:
                        component.add(w)
                        stack.append(w)
            if neighbor == self.sam.link[centroid]:
                parent_group = len(groups)
            groups.append(component)
        owner = {v: g for g, nodes in enumerate(groups) for v in nodes}
        qgroups = [[] for _ in groups]
        kgroups = [[] for _ in groups]
        for i in queries:
            qgroups[owner[self.qnode[i]]].append(i)
        for j in keys:
            kgroups[owner[self.sam.endpoints[j]]].append(j)

        def cross(qids, kids):
            qp = [i for i in qids if owner[self.qnode[i]] == parent_group]
            qd = [i for i in qids if owner[self.qnode[i]] != parent_group]
            kp = [j for j in kids if owner[self.sam.endpoints[j]] == parent_group]
            kd = [j for j in kids if owner[self.sam.endpoints[j]] != parent_group]
            assert not (qp and kp)
            self._emit(qd, kd, query_depth=lambda i: min(self.matched[i], self.sam.length[centroid]))
            self._emit(qp, kd, query_depth=lambda i: min(self.matched[i], self.sam.length[self.sam.lca(self.qnode[i], centroid)]))
            self._emit(qd, kp, key_depth=lambda j: self.sam.length[self.sam.lca(centroid, self.sam.endpoints[j])])

        # Huffman grouping bounds weighted replication and avoids degree^2 work.
        heap = [(len(nodes), g, qgroups[g], kgroups[g]) for g, nodes in enumerate(groups)]
        heapq.heapify(heap)
        serial = len(groups)
        while len(heap) > 1:
            wa, _, qa, ka = heapq.heappop(heap)
            wb, _, qb, kb = heapq.heappop(heap)
            cross(qa, kb)
            cross(qb, ka)
            heapq.heappush(heap, (wa + wb, serial, list(heapq.merge(qa, qb)), list(heapq.merge(ka, kb))))
            serial += 1
        self._emit(qgroups[0], kgroups[0], query_depth=lambda i: self.matched[i])
        for g in range(1, len(groups)):
            self._decompose(groups[g], qgroups[g], kgroups[g])

    def audit(self, queries, keys, resets):
        """Diagnostic O(N^2) pair coverage and independent diagonal match DP."""
        lengths = np.zeros((self.n, self.n), dtype=np.int64)
        for i in range(self.n):
            for j in range(self.n):
                if queries[i] == keys[j]:
                    lengths[i, j] = 1 + (lengths[i-1, j-1] if i and j and not resets[i] else 0)
                actual = min(self.matched[i], self.sam.length[self.sam.lca(self.qnode[i], self.sam.endpoints[j])])
                assert actual == lengths[i, j], (i, j, actual, lengths[i, j])
        count = np.zeros_like(lengths)
        for stream in self.streams:
            for qi, i in enumerate(stream.queries):
                for kj, j in enumerate(stream.keys):
                    if j <= i:
                        count[i, j] += 1
                        expected = log_weight(int(lengths[i, j]), self.tau)
                        actual = stream.query_log[qi] + stream.key_log[kj]
                        assert actual == expected or abs(actual - expected) < 1e-12
        assert np.array_equal(count, np.tri(self.n, dtype=np.int64))

    def evaluate(self, sq, sk, value, dtype=np.float64):
        """Two passes, one live R*DV matrix; no vector partial per event."""
        sq, sk, value = (np.asarray(x, dtype=dtype) for x in (sq, sk, value))
        if sq.shape != sk.shape or sq.shape[0] != self.n or value.shape[0] != self.n:
            raise ValueError("Expected sq/sk [N,R] and value [N,DV]")
        logden = np.zeros(self.n, dtype=np.float64)
        output = np.zeros((self.n, value.shape[1]), dtype=dtype)
        for vector_pass in (False, True):
            for stream in self.streams:
                pivot = -math.inf
                mass = 0.
                matrix = np.zeros((sq.shape[1], value.shape[1]), dtype=dtype) if vector_pass else None
                cursor = 0
                for qi, i in enumerate(stream.queries):
                    while cursor < len(stream.keys) and stream.keys[cursor] <= i:
                        j = stream.keys[cursor]
                        logb = float(stream.key_log[cursor])
                        cursor += 1
                        if logb == -math.inf:
                            continue
                        if logb > pivot:
                            factor = math.exp(pivot - logb)
                            mass *= factor
                            if vector_pass:
                                matrix *= factor
                            pivot = logb
                        factor = math.exp(logb - pivot)
                        mass += factor
                        if vector_pass:
                            matrix += factor * np.outer(sk[j], value[j])
                    loga = float(stream.query_log[qi])
                    if mass == 0 or loga == -math.inf:
                        continue
                    if vector_pass:
                        output[i] += math.exp(loga + pivot - logden[i]) * (sq[i] @ matrix)
                    else:
                        logden[i] = np.logaddexp(logden[i], loga + pivot + math.log(mass))
        return output, logden

    def statistics(self):
        return dict(n=self.n, states=len(self.sam.length), streams=len(self.streams),
                    events=sum(len(s.queries) + len(s.keys) for s in self.streams))
