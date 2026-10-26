"""Build an isolated sparse-readout planner with this version's build helper."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', required=True, type=Path)
    ap.add_argument('--source-dir', type=Path,
                    help='inference source snapshot before coefficient-cache optimization')
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[2]
    out = args.output.absolute()
    if out.exists() or not out.is_relative_to(root / 'dism_v3/build'):
        ap.error('use a fresh directory under this worktree dism_v3/build')
    src = (args.source_dir or root / 'dism_v3/python/flash_dism/inference').resolve()
    if 'double cached_logweight(' in (src / 'csrc/prefill.hpp').read_text():
        ap.error('use --source-dir with the original uncached inference snapshot')
    dest = out / 'inference/csrc'
    dest.mkdir(parents=True)
    sources = [*sorted((src / 'csrc').glob('*.hpp')), src / 'csrc/prefill_bind.cpp',
               src / 'build.py', Path(__file__), Path(__file__).with_name('planner_sparse_reads.hpp')]
    manifest = {str(p.relative_to(root) if p.is_relative_to(root) else p):
                hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    for p in sources:
        if p.parent == src / 'csrc':
            shutil.copy2(p, dest / p.name)
    shutil.copy2(src / 'build.py', dest.parent / 'build.py')
    shutil.copy2(Path(__file__).with_name('planner_sparse_reads.hpp'), dest)
    p = dest / 'prefill_training.hpp'
    s = p.read_text().replace('#include "prefill.hpp"', '#include "prefill.hpp"\n#include "planner_sparse_reads.hpp"')
    s = s.replace('const int *query_reset=nullptr)', 'const int *query_reset=nullptr, int sparse_keys=0)')
    s = s.replace('builder.build(labels+base', 'builder.cache_forward_weights=(sparse_keys & 8)!=0;\n            builder.lazy_gradient_weights=(sparse_keys & 16)!=0;\n            builder.build(labels+base')
    s = s.replace('auto p=builder.finish();', 'auto p=split_sparse_reads(builder.finish(),width,sparse_keys & 7);')
    p.write_text(s)
    p = dest / 'prefill.hpp'
    s = p.read_text()
    begin = s.index('    int lca(')
    s = s[:begin]+'''    std::vector<double> forward_weights;
    double forward_tau=std::numeric_limits<double>::quiet_NaN();
    double forward_logweight(int length) {
        if (!cache_forward_weights) return logweight(length);
        auto &value=forward_weights[length];
        if (std::isnan(value)) value=logweight(length);
        return value;
    }

'''+s[begin:]
    s = s.replace(' : logweight(length)', ' : forward_logweight(length)')
    s = s.replace('    Builder() = default;', '    bool cache_forward_weights=false;\n    Builder() = default;')
    s = s.replace('    Builder() = default;', '    bool lazy_gradient_weights=false;\n    Builder() = default;')
    s = s.replace('    int lca(', '''    double gradient_logweight(int length) {
        if (lazy_gradient_weights && length>cached_length) {
            logweights.resize(length+1); derivatives.resize(length+1);
            for (int l=cached_length+1;l<=length;++l) {
                logweights[l]=logweight(l);
                derivatives[l]=std::exp(logweights[l-1]-logweights[l])*derivatives[l-1]
                    + std::exp(l*tau-logweights[l])*l;
            }
            cached_length=length;
        }
        return logweights[length];
    }

    int lca(''',1)
    s = s.replace('gradients ? logweights[length]', 'gradients ? gradient_logweight(length)')
    s = s.replace('            int first=cached_length+1;', '            if (!lazy_gradient_weights) {\n            int first=cached_length+1;')
    s = s.replace('cached_length=std::max(cached_length,n);', 'cached_length=std::max(cached_length,n);\n            } else cached_tau=tau;')
    anchor = '        // Build the offline SAM'
    assert anchor in s
    s = s.replace(anchor, '''        if (!gradients && cache_forward_weights) {
            if (forward_tau!=tau) {
                forward_weights.assign(n+1,std::numeric_limits<double>::quiet_NaN());
                forward_tau=tau;
            } else if (forward_weights.size()<size_t(n+1)) {
                forward_weights.resize(n+1,std::numeric_limits<double>::quiet_NaN());
            }
        }
'''+anchor)
    p.write_text(s)
    p = dest / 'prefill_inference.hpp'
    s = p.read_text().replace('int dispatch_threshold)', 'int dispatch_threshold,int sparse_keys=0)')
    s = s.replace('width,false,false,reset);', 'width,false,false,reset,sparse_keys);')
    p.write_text(s)
    p = dest / 'prefill_bind.cpp'
    s = p.read_text().replace('PYBIND11_MODULE(_dism_prefill,', 'PYBIND11_MODULE(TORCH_EXTENSION_NAME,')
    s = s.replace('int width,int threshold)', 'int width,int threshold,int sparse_keys)')
    s = s.replace('reset,width,threshold);', 'reset,width,threshold,sparse_keys);')
    s = s.replace('py::arg("dispatch_threshold")=64)', 'py::arg("dispatch_threshold")=64,py::arg("sparse_keys")=0)')
    s = s.replace('int width,bool compact)', 'int width,bool compact,int sparse_keys)')
    s = s.replace('hard_ptr,width,compact);', 'hard_ptr,width,compact,true,nullptr,sparse_keys);')
    s = s.replace('py::arg("compact")=false);', 'py::arg("compact")=false,py::arg("sparse_keys")=0);')
    s = s.replace('    // Test oracle:', '    m.def("split_sparse",[](Program p,int width,int limit) { return split_sparse_reads(std::move(p),width,limit); });\n    // Test oracle:')
    p.write_text(s)
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    os.environ['MAX_JOBS'] = '2'
    os.environ['TORCH_EXTENSIONS_DIR'] = str(out / 'cache')
    spec = importlib.util.spec_from_file_location('sparse_build', dest.parent / 'build.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    binary = mod._load('_dism_sparse_prefill', ['prefill_bind.cpp'], force=True)
    shutil.copy2(binary.__file__, out / '_dism_sparse_prefill.so')
    print(out, flush=True)


if __name__ == '__main__':
    main()
