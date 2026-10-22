"""NVTX graph-origin annotations and Nsight SQLite component attribution.

CUDA graph replay kernels are linked through graphNodeId/ originalGraphNodeId
to capture-time NVTX scopes. Backward scopes use autograd sequence numbers to
recover their forward component, including launches on autograd worker threads.
No timing nodes, module backward hooks or kernel changes are introduced.
"""
import argparse
import collections
import contextlib
import functools
import heapq
import json
from pathlib import Path
import re
import sqlite3


@contextlib.contextmanager
def annotate_capture(model):
    import torch
    import flash_dism.compiler as compiler
    import flash_dism.graph_gdn as graph_gdn
    from nanochat.models.dism import model as model_module
    changes = []

    def mark(owner, name, category, path):
        original = getattr(owner, name)
        had_local = name in vars(owner)
        local_value = vars(owner).get(name)

        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            with torch.cuda.nvtx.range(f'component/{category}/{path}'):
                return original(*args, **kwargs)

        setattr(owner, name, wrapped)
        changes.append((owner, name, had_local, local_value))

    for path, module in model.named_modules():
        if not path:
            category = 'model_glue'
        elif path == 'embedding':
            category = 'embedding'
        elif path == 'norm':
            category = 'final_norm'
        elif '.mlp' in path and '.mlp_norm' not in path:
            category = 'ffn_down' if 'down_proj' in path else 'ffn_swiglu'
        elif '.attn.' in path:
            leaf = path.split('.attn.', 1)[1].split('.')[0]
            if leaf.endswith('_conv'):
                category = 'input_projection'
            elif leaf in ('sq_proj', 'sk_proj'):
                category = 'dism_readout'
            else:
                category = 'attention_gate_output'
        elif path.endswith('.attn'):
            category = 'attention_glue'
        else:
            category = 'block_norm_residual'
        mark(module, 'forward', category, path or 'model')
        if hasattr(module, 'forward_quantized'):
            mark(module, 'forward_quantized', category, path+'.forward_quantized')
        if hasattr(module, 'project_linear'):
            mark(module, 'project_linear', 'input_projection', path+'.project_linear')
        if hasattr(module, 'depthwise_silu'):
            mark(module, 'depthwise_silu', 'short_convolution', path+'.depthwise_silu')
    mark(compiler, 'voc_forward', 'dism_core', 'voc_forward')
    mark(graph_gdn, 'prepared_gdn', 'gdn_core', 'prepared_gdn')
    mark(model_module, 'packed_document_ce', 'head_loss', 'packed_document_ce')
    try:
        yield
    finally:
        for owner, name, had_local, value in reversed(changes):
            if had_local:
                setattr(owner, name, value)
            else:
                delattr(owner, name)


def containing(scopes, points):
    """Innermost same-thread range for each point; no cross-thread guessing."""
    by_thread = collections.defaultdict(list)
    for scope in scopes:
        by_thread[scope['tid']].append(scope)
    queries = collections.defaultdict(list)
    for key, tid, timestamp in points:
        queries[tid].append((timestamp, key))
    result = {}
    for tid, values in queries.items():
        ordered = sorted(by_thread[tid], key=lambda s: s['start'])
        active, cursor = [], 0
        for timestamp, key in sorted(values):
            while cursor < len(ordered) and ordered[cursor]['start'] <= timestamp:
                s = ordered[cursor]
                heapq.heappush(active, (-s['start'], s['end'], s['id'], s))
                cursor += 1
            while active and active[0][1] < timestamp:
                heapq.heappop(active)
            if active:
                result[key] = active[0][3]
    return result


def interval_union(intervals):
    result = []
    for a, b in sorted(intervals):
        if result and a <= result[-1][1]:
            result[-1][1] = max(result[-1][1], b)
        else:
            result.append([a, b])
    return sum(b-a for a, b in result)


def analyze(path):
    db = sqlite3.connect(f'file:{path.resolve()}?mode=ro', uri=True)
    strings = dict(db.execute('select id,value from StringIds'))
    scopes = [dict(id=i, start=a, end=b, tid=t, name=x or strings.get(y, ''))
              for i, (a, b, t, x, y) in enumerate(db.execute(
                  'select start,end,globalTid,text,textId from NVTX_EVENTS where end is not null'))]
    components = [s for s in scopes if s['name'].startswith('component/')]
    seq_scopes = []
    for s in scopes:
        match = re.search(r'\bseq = (\d+)', s['name'])
        if match:
            s['seq'] = int(match[1])
            seq_scopes.append(s)
    seq_components = containing(components, [(s['id'], s['tid'], s['start']) for s in seq_scopes])
    forward_sequences = {s['seq']: seq_components[s['id']]['name'].split('/')[1]
                         for s in seq_scopes if s['id'] in seq_components}
    engines = [s for s in scopes if s['name'].startswith('autograd::engine::evaluate_function:')]
    seq_engines = containing(engines, [(s['id'], s['tid'], s['start']) for s in seq_scopes])
    for s in seq_scopes:
        if s['id'] in seq_engines and s['seq'] in forward_sequences:
            seq_engines[s['id']]['component'] = forward_sequences[s['seq']]
    for s in engines:
        if 'AccumulateGrad' in s['name']:
            s['component'] = 'gradient_accumulation'

    origins, clones = {}, {}
    for timestamp, tid, node, original in db.execute(
            'select start,globalTid,graphNodeId,originalGraphNodeId from CUDA_GRAPH_NODE_EVENTS'):
        if original:
            clones[node] = original
        elif node not in origins:
            origins[node] = (tid, timestamp)
    points = [(node, tid, timestamp) for node, (tid, timestamp) in origins.items()]
    node_forward = containing(components, points)
    node_backward = containing(engines, points)

    def component(node):
        seen = set()
        while node in clones:
            if node in seen:
                raise RuntimeError('graph clone cycle')
            seen.add(node)
            node = clones[node]
        if node in node_backward:
            return 'backward', node_backward[node].get('component', 'backward_unmapped')
        if node in node_forward:
            return 'forward', node_forward[node]['name'].split('/')[1]
        return 'backward', 'graph_glue_unmapped'

    steps = sorted([s for s in scopes if s['name'].startswith('profile/step/')], key=lambda s:s['start'])
    if not steps:
        raise RuntimeError('no measured step ranges')
    tables = {r[0] for r in db.execute("select name from sqlite_master where type='table'")}
    launches = {}
    for table in ('CUPTI_ACTIVITY_KIND_RUNTIME', 'CUPTI_ACTIVITY_KIND_DRIVER'):
        if table in tables:
            for a, tid, corr in db.execute(f'select start,globalTid,correlationId from {table}'):
                launches[corr] = (tid, a)
    activity = []
    for a, b, node, corr, name in db.execute(
            'select start,end,graphNodeId,correlationId,demangledName from CUPTI_ACTIVITY_KIND_KERNEL'):
        activity.append(dict(start=a, end=b, node=node, corr=corr, name=strings[name], kind='kernel'))
    for table, kind in (('CUPTI_ACTIVITY_KIND_MEMCPY', 'copy'), ('CUPTI_ACTIVITY_KIND_MEMSET', 'memset')):
        if table in tables:
            for a, b, node, corr in db.execute(f'select start,end,graphNodeId,correlationId from {table}'):
                activity.append(dict(start=a, end=b, node=node, corr=corr, name=kind, kind=kind))
    outside_scopes = [s for s in scopes if s['name'].startswith('profile/') and '/step/' not in s['name']]
    launch_scopes = containing(outside_scopes, [(corr, tid, a) for corr, (tid, a) in launches.items()])
    totals = collections.defaultdict(lambda: collections.defaultdict(float))
    kernels = collections.defaultdict(lambda: [0, 0.])
    graph_kernel_counts = collections.Counter()
    replay_count = 0
    per_step = []
    for step in steps:
        replay_count += sum(s['name'] == 'profile/graph_forward_backward' and
                            step['start'] <= s['start'] and s['end'] <= step['end']
                            for s in scopes)
        selected = [a for a in activity if step['start'] <= a['start'] and a['end'] <= step['end']]
        spans = []
        for a in selected:
            duration = (a['end']-a['start'])/1e6
            if a['node']:
                phase, group = component(a['node'])
                if a['kind'] == 'kernel':
                    graph_kernel_counts[a['node']] += 1
            else:
                phase = 'outside_graph'
                group = launch_scopes.get(a['corr'], {}).get('name', 'unmapped')
            totals[phase][group] += duration
            kernels[(phase, group, a['name'])][0] += 1
            kernels[(phase, group, a['name'])][1] += duration
            spans.append((a['start'], a['end']))
        wall = (step['end']-step['start'])/1e6
        active = interval_union(spans)/1e6
        per_step.append(dict(name=step['name'], wall_ms=wall, gpu_activity_union_ms=active,
                             no_gpu_activity_ms=wall-active))
    count = len(steps)
    if not graph_kernel_counts or set(graph_kernel_counts.values()) != {replay_count}:
        raise RuntimeError('graph kernel node coverage differs across measured replays')
    cpu_scopes = collections.defaultdict(float)
    for s in outside_scopes:
        if any(step['start'] <= s['start'] and s['end'] <= step['end'] for step in steps):
            cpu_scopes[s['name']] += (s['end']-s['start'])/1e6/count
    return dict(source=str(path), steps=count,
                method='GPU kernel/copy/memset durations mapped by graph creation origin and autograd seq; '
                       'exclusive component attribution; sums are not critical-path wall time',
                graph_replays=replay_count, unique_graph_kernel_nodes=len(graph_kernel_counts),
                unmapped_graph_ms_per_step=sum(t for p, groups in totals.items()
                                              if p != 'outside_graph'
                                              for g,t in groups.items() if 'unmapped' in g)/count,
                component_ms_per_step={p:{g:t/count for g,t in groups.items()} for p,groups in totals.items()},
                cpu_scope_ms_per_step=dict(cpu_scopes), per_step=per_step,
                kernels=sorted([dict(phase=p, component=g, name=n, count=v[0], ms_per_step=v[1]/count)
                                for (p,g,n),v in kernels.items()], key=lambda r:-r['ms_per_step']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]/'build'
    if args.output.exists() or not args.output.resolve().is_relative_to(root):
        parser.error('output must be a fresh path under dism_v3/build')
    report = analyze(args.input)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'kernels'}, indent=2))


if __name__ == '__main__':
    main()
