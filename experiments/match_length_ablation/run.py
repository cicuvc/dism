"""Frozen-checkpoint, paired validation for suffix-order truncation.

Runtime imports come from an isolated copy of the checkpoint's source snapshot.
The local CUDA extension is reused; production modules are not edited.
"""
import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

import torch

from core import hard_output, lengths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--tokens', type=int, default=1048576)
    parser.add_argument('--packs', type=int, default=0)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--metadata', type=Path)
    parser.add_argument('--expected-sha256')
    parser.add_argument('--modes', default='native,full,cap1,cap2,cap4,cap1_far,cap1_near,no_dism')
    parser.add_argument('--collect-max-chains', action='store_true')
    args = parser.parse_args()
    root = args.workspace
    args.output.mkdir(parents=True, exist_ok=True)
    sys.path[:0] = [str(root / 'runtime/nanochat'), str(root / 'runtime/dism_v3/python'),
                   '/home/cicuvc/cs/project/dism-exp/dism_v3/python']
    from nanochat.models import build_model, ModelSpec
    from nanochat.tokenizer import HuggingFaceTokenizer
    from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
    import flash_dism.varlen as varlen

    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    checkpoint = args.checkpoint or root / 'checkpoint/model_006000.pt'
    with checkpoint.open('rb') as f:
        digest = hashlib.file_digest(f, 'sha256').hexdigest()
    expected = args.expected_sha256
    if args.checkpoint is None:
        expected = 'd066a28f1d75032b605c85c7b9df41011d4a7abe02117f86eab03d4d9c76cf71'
    if expected:
        assert digest == expected, (digest, expected)
    metadata = args.metadata or checkpoint.with_name(checkpoint.name.replace('model_', 'meta_')).with_suffix('.json')
    meta = json.loads(metadata.read_text())
    with torch.device('meta'):
        model = build_model(ModelSpec.from_dict(meta['model_spec']))
    model.to_empty(device='cpu')
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
    model.cuda().eval()
    model.eval_hard = True
    modes = {'native': None, 'full': (0, 'all'), 'cap1': (1, 'all'),
             'cap2': (2, 'all'), 'cap4': (4, 'all'),
             'cap1_far': (1, 'far'), 'cap1_near': (1, 'near'), 'no_dism': None,
             'window128': (0, 'window128')}
    modes = {name: modes[name] for name in args.modes.split(',')}
    assert {'native', 'full'} <= modes.keys()
    context = {}
    parity = []
    maxima = {}
    dism_layers = [i+1 for i, layer in enumerate(model.layers)
                   if not getattr(layer.attn, 'is_pure_gdn', False)]
    native_core = varlen.dism_core_varlen

    def collect_lengths(a):
        layer = dism_layers[context['core_call']]
        context['core_call'] += 1
        for start, end in zip(context['cu'][:-1], context['cu'][1:]):
            if start == end:
                continue
            run = lengths(a[7], a[8], start, end)
            run.masked_fill_(~context['valid'][None, start:end, None], 0)
            for kind in ('including_self', 'excluding_self'):
                if kind == 'excluding_self':
                    run.diagonal(dim1=-2, dim2=-1).zero_()
                value = int(run.max())
                key = f'L{layer}/{kind}'
                if value > maxima.get(key, {}).get('length', -1):
                    flat_index = int(run.flatten().argmax())
                    n = end-start
                    head, rest = divmod(flat_index, n*n)
                    query, key_position = divmod(rest, n)
                    maxima[key] = dict(length=value, pack=context['pack'], layer=layer,
                                       head_zero_based=head, document_start_packed=start,
                                       query_local=query, key_local=key_position, distance=query-key_position)

    def wrapped_core(*a, **kw):
        assert a[9].all() and a[10].all(), 'Expected hard=True, direction=True'
        mode = context['mode']
        if mode == 'no_dism':
            return torch.zeros_like(a[4])
        if mode == 'native':
            if args.collect_max_chains:
                collect_lengths(a)
            output = native_core(*a, **kw)
            if context['pack'] == 0:
                exact = hard_output(a[2], a[3], a[4], a[7], a[8], a[11], context['cu'])
                x, y = output.float().flatten(), exact.float().flatten()
                parity.append(dict(relative_rms=((x-y).norm()/x.norm().clamp_min(1e-20)).item(),
                                   cosine=torch.nn.functional.cosine_similarity(x, y, dim=0).item(),
                                   max_abs=(x-y).abs().max().item()))
            return output
        cap, distance = modes[mode]
        return hard_output(a[2], a[3], a[4], a[7], a[8], a[11], context['cu'], cap, distance)

    varlen.dism_core_varlen = wrapped_core
    tokpath = root / 'runtime/nanochat/runs/dism72m_3k/tokenizer'
    tokenizer = HuggingFaceTokenizer.from_pretrained(str(tokpath))
    token_bytes = tokenizer.get_token_bytes(device='cuda')
    loader = tokenizing_distributed_data_loader_bos_bestfit(
        tokenizer, args.batch, 2048, split='val', device='cuda', seq_align=256,
        data_dir=str(args.data), shuffle=False, shuffle_seed=1337,
        aligned_segment_ends=True, tokenizer_threads=2)
    rows = []
    total = 0
    started = time.time()
    input_hash = hashlib.sha256()

    def write(name, value):
        path = args.output / name
        tmp = path.with_suffix('.tmp')
        tmp.write_text(json.dumps(value, indent=2))
        tmp.replace(path)

    def status(state, **extra):
        write('state.json', dict(status=state, pid=os.getpid(), packs=len(rows),
                                valid_tokens=total, elapsed=time.time()-started, **extra))

    protocol = dict(checkpoint=str(checkpoint), checkpoint_sha256=digest,
                    metadata=str(metadata), step=meta['step'], training=meta.get('user_config'),
                    model_spec=meta['model_spec'], torch=torch.__version__, batch=args.batch,
                    target_tokens=args.tokens, modes=modes, hard=True, direction=True,
                    cap_semantics='sum_{m=1..L} exp(m*tau) -> sum_{m=1..min(L,C)} exp(m*tau); retain all matching edges',
                    distance_cutoff=128, tf32=False, diagnostic_accumulation='FP32; BF16 core output',
                    validation='original snapshot loader; val last parquet, shuffle=False, aligned varlen; targets>=0 and positive token bytes',
                    intervention='all DISM layers; recompute downstream activations; signed readout and fallback retained',
                    ci='paired packed-batch bootstrap; not training-seed uncertainty',
                    val_parquet=str(sorted(args.data.rglob('*.parquet'))[-1]))
    protocol['script_sha256'] = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                 for name in ('run.py', 'core.py')}
    write('protocol.json', protocol)
    status('running')
    try:
        with torch.inference_mode():
            for pack, (x, y, cu, segments) in enumerate(loader):
                boundaries = cu.cpu().tolist()
                assert max(b-a for a, b in zip(boundaries[:-1], boundaries[1:])) <= 2048
                flat = y.flatten()
                valid = (flat >= 0) & (token_bytes[flat.clamp_min(0)] > 0)
                count = int(valid.sum())
                for tensor in (x, y, cu):
                    input_hash.update(tensor.cpu().contiguous().numpy().tobytes())
                context.update(pack=pack, cu=boundaries, valid=valid)
                nats = {}
                position = torch.empty(flat.numel(), device='cuda', dtype=torch.int32)
                for start, end in zip(boundaries[:-1], boundaries[1:]):
                    position[start:end] = torch.arange(end-start, device='cuda', dtype=torch.int32)
                groups = [(0, 128), (128, 256), (256, 512), (512, 1024), (1024, 2048)]
                masks = [valid & (position >= lo) & (position < hi) for lo, hi in groups]
                per_position = {}
                for mode in modes:
                    context['mode'] = mode
                    context['core_call'] = 0
                    status('running', mode=mode)
                    loss = model(x, y, cu_seqlens=cu, segment_ids=segments,
                                 loss_reduction='none').flatten().double()
                    assert torch.isfinite(loss[valid]).all()
                    nats[mode] = loss[valid].sum().item()
                    per_position[mode] = [loss[mask].sum().item() for mask in masks]
                rows.append(dict(pack=pack, tokens=count, nats=nats, position_nats=per_position,
                                 position_counts=[int(mask.sum()) for mask in masks]))
                total += count
                write('batches.json', rows)
                write('core_parity.json', parity)
                if args.collect_max_chains:
                    write('chain_maxima.json', dict(valid_tokens=total, per_layer=maxima,
                          definition='Native unmodified hard forward. Consecutive matching symbols ending at causal pairs; query must have a scored positive-byte target. Padding queries excluded; document boundaries reset.'))
                print(json.dumps(dict(pack=pack, tokens=total, seconds=time.time()-started,
                                      nll={k: v/count for k, v in nats.items()})), flush=True)
                if total >= args.tokens or (args.packs and len(rows) >= args.packs):
                    break
        counts = torch.tensor([row['tokens'] for row in rows], dtype=torch.float64)
        generator = torch.Generator().manual_seed(42)
        indices = torch.randint(len(rows), (4000, len(rows)), generator=generator)
        full = torch.tensor([row['nats']['full'] for row in rows], dtype=torch.float64)
        native = torch.tensor([row['nats']['native'] for row in rows], dtype=torch.float64)
        summary = {}
        for mode in modes:
            values = torch.tensor([row['nats'][mode] for row in rows], dtype=torch.float64)
            delta = values - full
            bootstrap = delta[indices].sum(1) / counts[indices].sum(1)
            summary[mode] = dict(nll=values.sum().item()/total, delta_full=delta.sum().item()/total,
                                 delta_native=(values-native).sum().item()/total,
                                 ci95=torch.quantile(bootstrap, torch.tensor([.025, .975], dtype=torch.float64)).tolist())
        positional = []
        for i, (lo, hi) in enumerate(groups):
            count = sum(row['position_counts'][i] for row in rows)
            positional.append(dict(start=lo, end=hi, tokens=count,
                                   nll={mode: sum(row['position_nats'][mode][i] for row in rows)/max(count, 1)
                                        for mode in modes}))
        write('results.json', dict(valid_tokens=total, packs=len(rows), elapsed=time.time()-started,
                                   input_sha256=input_hash.hexdigest(), checkpoint_sha256=digest,
                                   results=summary, position=positional, core_parity=parity))
        status('complete')
        print(json.dumps(summary, indent=2), flush=True)
    except Exception as error:
        status('failed', error=repr(error))
        traceback.print_exc()
        raise
    finally:
        varlen.dism_core_varlen = native_core


if __name__ == '__main__':
    main()
