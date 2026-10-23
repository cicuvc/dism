"""Instrument native quadratic prefill allocations; timings are diagnostic only.

Uses the crossover driver's local checkpoint/text loading. Stage records retain
only sizes and shapes, never CUDA tensors. Synchronization and resetting peak
counters mean the driver's ordinary timing/peak fields must not be compared
with uninstrumented benchmark results.
"""
import json
from pathlib import Path
import sys

import torch
import cu_flash_dism as cu
from benchmark_eval_prefill_crossover import main


def describe(tensor):
    if tensor is None:
        return None
    return dict(shape=list(tensor.shape), dtype=str(tensor.dtype),
                bytes=tensor.numel()*tensor.element_size())


def instrument():
    records = []
    resident = []
    originals = {}

    def observe(lm, length):
        seen_objects, seen_storage = set(), set()
        tensors = []

        def visit(value, path):
            if id(value) in seen_objects:
                return
            seen_objects.add(id(value))
            if isinstance(value, torch.Tensor):
                if value.device.type == 'cuda':
                    storage = value.untyped_storage()
                    key = (value.device.index, storage.data_ptr())
                    if key not in seen_storage:
                        seen_storage.add(key)
                        tensors.append(dict(path=path, storage_bytes=storage.nbytes(), **describe(value)))
            elif isinstance(value, dict):
                for key, item in value.items():
                    visit(item, f'{path}.{key}')
            elif isinstance(value, (list, tuple)):
                for i, item in enumerate(value):
                    visit(item, f'{path}.{i}')
            elif type(value).__module__.startswith(('flash_mxfp8', 'nanochat')) and not isinstance(value, torch.nn.Module):
                for key, item in vars(value).items():
                    visit(item, f'{path}.{key}')

        for name, tensor in lm.model.named_parameters():
            visit(tensor, f'parameter.{name}')
        for name, tensor in lm.model.named_buffers():
            visit(tensor, f'buffer.{name}')
        for name, module in lm.model.named_modules():
            for key in ('_forward_states', '_forward_plans'):
                if hasattr(module, key):
                    visit(getattr(module, key), f'cache.{name}.{key}')
        resident.append(dict(length=length, allocated_bytes=torch.cuda.memory_allocated(),
                             enumerated_storage_bytes=sum(t['storage_bytes'] for t in tensors), tensors=tensors))
    for name in ('core_forward', 'hard_summary', 'hard_chunk', 'hard_readout'):
        original = getattr(cu, name)
        originals[name] = original

        def wrapped(*args, _name=name, _fn=original, **kwargs):
            torch.cuda.synchronize()
            before = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            result = _fn(*args, **kwargs)
            torch.cuda.synchronize()
            row = dict(stage=_name, before_bytes=before,
                       after_bytes=torch.cuda.memory_allocated(),
                       peak_bytes=torch.cuda.max_memory_allocated())
            if _name == 'core_forward':
                out, norm, state = result
                row.update(save_state=args[3], all_hard=bool(args[0][10].all()),
                           output=describe(out), normalizer=describe(norm),
                           vertical=describe(state.get('vertical')),
                           soft_boundary=describe(state['operands'][12]) if 'operands' in state else None)
            elif _name in ('hard_summary', 'hard_chunk'):
                row['output'] = describe(result)
            else:
                row['outputs'] = [describe(t) for t in result]
            records.append(row)
            return result

        setattr(cu, name, wrapped)
    try:
        main(observer=observe)
    finally:
        for name, fn in originals.items():
            setattr(cu, name, fn)
        output = Path(sys.argv[sys.argv.index('--output')+1])
        output.with_suffix('.allocations.json').write_text(json.dumps(records, indent=2))
        output.with_suffix('.resident.json').write_text(json.dumps(resident, indent=2))


if __name__ == '__main__':
    instrument()
