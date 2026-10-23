"""Compare current checkpoint features with pre-fix allocation/dispatch behavior.

The reference vocabulary wrapper is read from an explicit local Git revision;
MXFP8 modules temporarily retain the previous training-style forward caches.
No checkpoint tensors or production source files are modified. Driver timing
includes the reference comparison and must not be used as a performance result.
"""
import ast
import json
from pathlib import Path
import subprocess
import sys

import torch
import flash_dism.compiler as compiler
from benchmark_eval_prefill_crossover import main


def validate():
    source = subprocess.check_output(
        ['git', 'show', '52aee67:dism_v3/python/flash_dism/compiler.py'], text=True)
    node = next(n for n in ast.parse(source).body
                if isinstance(n, ast.FunctionDef) and n.name == '_voc_forward_impl')
    namespace = dict(vars(compiler))
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<legacy-voc-wrapper>', 'exec'), namespace)
    legacy = namespace['_voc_forward_impl']
    comparisons = []

    @torch.no_grad()
    def observe(lm, n):
        # The driver already owns tokenized input; read it via a one-shot wrapper
        # on its next measured call rather than tokenizing a second corpus.
        original_features = lm._features

        def compare(x, cu):
            lm._features = original_features
            current = original_features(x, cu).cpu()
            original_impl = compiler._voc_forward_impl
            modules = [m for m in lm.model.modules()
                       if type(m).__module__.startswith('flash_mxfp8.')]
            training = [m.training for m in modules]
            try:
                compiler._voc_forward_impl = legacy
                for module in modules:
                    module.training = True
                expected = original_features(x, cu).cpu()
            finally:
                compiler._voc_forward_impl = original_impl
                for module, value in zip(modules, training):
                    module.training = value
                    for name in ('_forward_states', '_forward_plans'):
                        cache = getattr(module, name, None)
                        if isinstance(cache, dict):
                            cache.clear()
            row = dict(length=n, bitwise_equal=torch.equal(current, expected),
                       max_abs=float((current.float()-expected.float()).abs().max()))
            comparisons.append(row)
            print('legacy reference', json.dumps(row), flush=True)
            assert row['bitwise_equal'], row
            return original_features(x, cu)

        if lm.hard_backend != 'quadratic':
            raise ValueError('run validator with --modes quadratic')
        lm._features = compare

    try:
        main(observer=observe)
    finally:
        output = Path(sys.argv[sys.argv.index('--output')+1])
        output.with_suffix('.comparison.json').write_text(json.dumps(comparisons, indent=2))


if __name__ == '__main__':
    validate()
