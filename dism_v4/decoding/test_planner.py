"""Independent dense recurrence validates C++ query plans, not CUDA arithmetic."""
import importlib.util
import pathlib
import subprocess
import sysconfig
from functools import lru_cache

import numpy as np
import torch


@lru_cache(None)
def load_planner():
    root = pathlib.Path(__file__).parent
    build = root / 'build'
    build.mkdir(exist_ok=True)
    output = build / ('_dism_decode_planner' + sysconfig.get_config_var('EXT_SUFFIX'))
    include = pathlib.Path(torch.__file__).parent / 'include'
    if not output.exists() or output.stat().st_mtime < max((root/name).stat().st_mtime for name in ('planner_bind.cpp','planner.hpp')):
        subprocess.run(['c++', '-O3', '-std=c++17', '-shared', '-fPIC',
                        '-I'+str(include), '-I'+sysconfig.get_paths()['include'],
                        str(root/'planner_bind.cpp'), '-o', str(output)], check=True)
    spec = importlib.util.spec_from_file_location('_dism_decode_planner', output)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Planner


def geometric(n, tau):
    return n if tau == 0 else -np.expm1(-tau*n)/-np.expm1(-tau)


def matrices(planner, g, v, tau):
    links,lens,pos,order,mats,samples = planner.topology()
    subtree = np.zeros((len(pos), v.shape[1], g.shape[1]))
    counts = np.zeros(len(pos))
    for n,p in enumerate(pos):
        if p >= 0:
            subtree[n] = np.outer(v[p], g[p])
            counts[n] = 1
    for n in reversed(order[1:]):
        subtree[links[n]] += subtree[n]
        counts[links[n]] += counts[n]
    prefix = np.zeros_like(subtree)
    prefix_count = np.zeros_like(counts)
    for n in order[1:]:
        p = links[n]
        gap = lens[n]-lens[p]
        rho = np.exp(-tau*gap)
        weight = geometric(gap,tau)
        prefix[n] = rho*prefix[p]+weight*subtree[n]
        prefix_count[n] = rho*prefix_count[p]+weight*counts[n]
    return (subtree[mats], counts[mats]), (prefix[samples], prefix_count[samples])


def test():
    Planner = load_planner()
    cases = 0
    for tau in (0., 1e-9, .7, 4.2):
        for b,k,t in ((1,1,4),(7,3,8),(32,12,15)):
            for pattern in ('random','repeat','periodic'):
                rng = np.random.default_rng(812)
                n,r,d = 257,3,5
                s,q = rng.integers(0,5,(2,n))
                if pattern == 'repeat': s[:]=q[:]=0
                if pattern == 'periodic': s=np.arange(n)%3; q=(np.arange(n)+1)%3
                g=rng.normal(size=(n,r)); v=rng.normal(size=(n,d)); x=rng.normal(size=(n,r))
                planner = Planner(r,b,k,t,tau)
                previous = np.empty(0)
                mat = matrices(planner,g,v,tau)
                for i in range(n):
                    if planner.needs_rebuild():
                        planner.rebuild(); mat=matrices(planner,g,v,tau)
                    reset = i%31 in (0,1)
                    tasks = planner.append(int(s[i]),int(q[i]),reset)
                    numerator=np.zeros(d); denominator=planner.fallback
                    for kind,index,c in tasks:
                        if kind==0: a=v[index]*np.dot(g[index],x[i]); z=1.
                        else: a=mat[kind-1][0][index]@x[i]; z=mat[kind-1][1][index]
                        numerator+=c*a; denominator+=c*z
                    # Independent log-domain causal recurrence.
                    pred=np.pad(previous,(1,0),constant_values=-np.inf)
                    if reset: pred[:]=-np.inf
                    previous=np.where(s[:i+1]==q[i],tau+np.logaddexp(0,pred),-np.inf)
                    maximum=max(0.,previous.max())
                    weights=np.exp(previous-maximum)
                    expected=((weights*(g[:i+1]@x[i]))@v[:i+1])/(np.exp(-maximum)+weights.sum())
                    np.testing.assert_allclose(numerator/denominator,expected,atol=2e-10,rtol=2e-10)
                cases+=1
    print(f'PASS: {cases} streams, {cases*257} steps; rebuild/reset/tau-zero/signed readout')


if __name__ == '__main__':
    test()
