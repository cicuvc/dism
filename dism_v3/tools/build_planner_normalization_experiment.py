"""Build isolated CPU normalization experiments without changing production.

Use --native to also build each private extension with inference/build.py.
The baseline is read from a pinned Git revision, never from another worktree.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

OLD = '''                    double term=stream_a[qi]+pivot+std::log(mass), next=logadd(out.logden[i],term);
                    if (gradients) out.dlogden[i]=std::exp(out.logden[i]-next)*out.dlogden[i]
                        +std::exp(term-next)*(stream_da[qi]+dmass/mass);
                    out.logden[i]=next;'''
INIT = '''        // Scaled denominator includes the constant unit baseline.
        static thread_local std::vector<double> norm_pivot, norm_mass, norm_derivative;
        norm_pivot.assign(n,0.); norm_mass.assign(n,1.);
        if (gradients) norm_derivative.assign(n,0.);
'''
ONLINE = '''                    double t=stream_a[qi]+pivot;
                    double d=gradients ? stream_da[qi]*mass+dmass : 0.;
                    if (t>norm_pivot[i]) {
                        double scale=std::exp(norm_pivot[i]-t);
                        norm_mass[i]=norm_mass[i]*scale+mass;
                        if (gradients) norm_derivative[i]=norm_derivative[i]*scale+d;
                        norm_pivot[i]=t;
                    } else {
                        double scale=std::exp(t-norm_pivot[i]);
                        norm_mass[i]+=mass*scale;
                        if (gradients) norm_derivative[i]+=d*scale;
                    }'''
FINALIZE = '''        for (int i=0;i<n;++i) {
            out.logden[i]=norm_pivot[i]+std::log(norm_mass[i]);
            if (gradients) out.dlogden[i]=norm_derivative[i]/norm_mass[i];
        }
'''
CACHE_INIT = '''        static thread_local std::vector<double> query_t, query_mass, query_derivative, query_scale;
        query_t.assign(stream_q.size(),neg_inf); query_mass.resize(stream_q.size());
        query_scale.resize(stream_q.size());
        if (gradients) query_derivative.resize(stream_q.size());
'''
COLLECT = '''                    double t=stream_a[qi]+pivot;
                    query_t[qi]=t; query_mass[qi]=mass;
                    if (gradients) query_derivative[qi]=stream_da[qi]*mass+dmass;
                    norm_pivot[i]=std::max(norm_pivot[i],t);'''
REDUCE = '''        for (int i=0;i<n;++i) norm_mass[i]=std::exp(-norm_pivot[i]);
        for (size_t qi=0;qi<stream_q.size();++qi) {
            if (query_t[qi]==neg_inf) continue;
            int i=stream_q[qi];
            double scale=std::exp(query_t[qi]-norm_pivot[i]);
            query_scale[qi]=scale;
            norm_mass[i]+=query_mass[qi]*scale;
            if (gradients) norm_derivative[i]+=query_derivative[qi]*scale;
        }
'''


def once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f'Expected one source anchor: {old[:90]}')
    return source.replace(old, new, 1)


def variant(source, name):
    if name == 'baseline':
        return source
    source = once(source, '        // Scalar pass.',
                  INIT + (CACHE_INIT if name == 'twopass' else '') + '        // Scalar pass.')
    source = once(source, OLD, ONLINE if name == 'online' else COLLECT)
    anchor = '        // Compile all floating point transcendental work'
    source = once(source, anchor, (REDUCE if name == 'twopass' else '') + FINALIZE + anchor)
    if name == 'twopass':
        source = once(source, 'std::exp(stream_a[qi]+pivot-out.logden[i])',
                      'query_scale[qi]/norm_mass[i]')
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--native', action='store_true')
    parser.add_argument('--baseline', default='a4d131e', help='Git revision with the original scalar pass')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    output = args.output.resolve()
    if not output.is_relative_to(root/'dism_v3/build'):
        parser.error('output must be under this worktree dism_v3/build')
    if output.exists():
        parser.error('use a fresh output directory to preserve earlier results')
    src = root/'dism_v3/python/flash_dism/inference/csrc'
    relative = src.relative_to(root)
    baseline_files = {}
    for path in [*src.glob('*.hpp'), src/'prefill_bind.cpp', src.parent/'build.py']:
        rel = path.relative_to(root)
        baseline_files[rel] = subprocess.check_output(
            ['git', 'show', f'{args.baseline}:{rel}'], cwd=root)
    source = baseline_files[relative/'prefill.hpp'].decode()
    generated = {name: variant(source, name) for name in ('baseline', 'online', 'twopass')}
    output.mkdir(parents=True)
    commands = []
    for name, text in generated.items():
        directory = output/name/'inference'
        (directory/'csrc').mkdir(parents=True)
        for path in [*src.glob('*.hpp'), src/'prefill_bind.cpp']:
            (directory/'csrc'/path.name).write_bytes(baseline_files[path.relative_to(root)])
        (directory/'build.py').write_bytes(baseline_files[(src.parent/'build.py').relative_to(root)])
        (directory/'csrc/prefill.hpp').write_text(text)
        # Avoid GCC treating byte-identical pragma-once files as the same header
        # when a diagnostic includes baseline/candidate in separate namespaces.
        for header in ('planner.hpp', 'prefill_matcher.hpp'):
            path = directory/'csrc'/header
            path.write_text(path.read_text() + f'\n// Isolated {name} normalization experiment.\n')
        cmd = ['/usr/bin/g++', '-O3', '-g', '-fno-omit-frame-pointer', '-std=c++17',
               '-pthread', '-I', str(directory/'csrc'),
               str(root/'dism_v3/tools/profile_planner_native.cpp'),
               '-o', str(output/name/'native')]
        subprocess.run(cmd, check=True)
        commands.append(cmd)
        if args.native:
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='', MAX_JOBS='2',
                       TORCH_EXTENSIONS_DIR=str(output/name/'cache'))
            cmd = [sys.executable, str(directory/'build.py'), '--component', 'prefill',
                   '--output', str(output/name/'_dism_prefill.so')]
            subprocess.run(cmd, env=env, check=True)
            commands.append(cmd)
    sources = [Path(__file__), src.parent/'build.py', src/'prefill_bind.cpp',
               root/'dism_v3/tools/profile_planner_native.cpp', *sorted(src.glob('*.hpp'))]
    manifest = dict(commands=commands, baseline=args.baseline,
                    baseline_sources={str(p): hashlib.sha256(data).hexdigest()
                                      for p, data in baseline_files.items()},
                    sources={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in sources},
                    variants={name: hashlib.sha256(text.encode()).hexdigest()
                              for name, text in generated.items()})
    (output/'build_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')


if __name__ == '__main__':
    main()
