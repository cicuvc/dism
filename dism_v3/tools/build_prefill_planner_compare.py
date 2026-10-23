"""Isolated native planner controls; production sources are never edited.

Builds an instrumented current-header binary plus an inference logweight-LUT
ablation. No new math: the LUT calls the original scalar function once per
length, without computing derivatives. Program hashes check byte equivalence.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def once(s,old,new):
    if s.count(old)!=1:raise ValueError(f'nonunique anchor: {old}')
    return s.replace(old,new,1)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();root=Path(__file__).resolve().parents[1]
    if not args.output.resolve().is_relative_to(root/'build'):ap.error('output must be under dism_v3/build')
    src=root/'python/flash_dism/inference/csrc';args.output.mkdir(parents=True,exist_ok=True)
    commands=[]
    for variant in ('baseline','lut'):
        dest=args.output/variant;dest.mkdir(exist_ok=True)
        for name in ('planner.hpp','prefill.hpp','prefill_matcher.hpp','planning_workers.hpp','prefill_training.hpp','prefill_inference.hpp'):
            text=(src/name).read_text()
            if name=='prefill_training.hpp':
                text=once(text,'    double plan_ms=0.;','    double plan_ms=0.;\n    std::vector<double> diagnostic_job_ms;')
                text=once(text,'        auto begin=std::chrono::steady_clock::now();','        diagnostic_job_ms.resize(jobs.size());\n        auto begin=std::chrono::steady_clock::now();')
                text=once(text,'            const auto &job=jobs[i];','            auto diagnostic_start=std::chrono::steady_clock::now();\n            const auto &job=jobs[i];')
                text=once(text,'            programs[i]=std::move(p);','            programs[i]=std::move(p);\n            diagnostic_job_ms[i]=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-diagnostic_start).count();')
            if name=='prefill.hpp' and variant=='lut':
                assert text.count('gradients ? logweights[length] : logweight(length)')==2
                text=text.replace('gradients ? logweights[length] : logweight(length)','logweights[length]')
                text=once(text,'        // Build the offline SAM', '''        // Diagnostic-only table: same scalar function, no derivatives.
        if (!gradients) {
            logweights.resize(n+1);logweights[0]=neg_inf;
            for (int l=1;l<=n;++l) logweights[l]=logweight(l);
        }
        // Build the offline SAM''')
            (dest/name).write_text(text)
        cmd=['/usr/bin/g++','-O3','-g','-fno-omit-frame-pointer','-std=c++17','-pthread','-I',str(dest),
             str(Path(__file__).with_name('compare_prefill_planner.cpp')),'-o',str(args.output/f'compare_{variant}')]
        subprocess.run(cmd,check=True);commands.append(cmd)
    sources=[*src.glob('*.hpp'),Path(__file__),Path(__file__).with_name('compare_prefill_planner.cpp')]
    (args.output/'build_manifest.json').write_text(json.dumps(dict(commands=commands,sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}),indent=2))


if __name__=='__main__':main()
