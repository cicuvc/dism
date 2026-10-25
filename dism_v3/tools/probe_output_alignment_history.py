"""Forward-only temporal raw DISM alignment, with and without upstream drift."""
import argparse
import gc
import json
from pathlib import Path

import pyarrow.parquet as pq
import torch
from transformers import AutoTokenizer

import flash_dism.compiler as compiler
from nanochat.models import ModelSpec, build_model
from probe_soft_hard_heads import measures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.28)
    checkpoint = Path(args.checkpoint)
    meta = json.loads(checkpoint.with_name(checkpoint.name.replace('model_', 'meta_')).with_suffix('.json').read_text())
    with torch.device('meta'):
        model = build_model(ModelSpec.from_dict(meta['model_spec']))
    model.to_empty(device='cuda')
    model.init_weights()
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model.load_state_dict({k.removeprefix('_orig_mod.'): v for k, v in state.items()}, assign=True)
    del state
    model.cuda().eval()
    model.mxfp8_ce = model.mxfp8_ce_norm = None
    from flash_mxfp8 import refresh_fp8_modules
    refresh_fp8_modules(model)
    tokenizer = AutoTokenizer.from_pretrained('/home/cicuvc/downstream-eval/tokenizer', local_files_only=True)
    bos = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
    documents = pq.read_table('/home/cicuvc/downstream-eval/data/pg19/test.parquet').to_pylist()[:2]
    tokens = [tokenizer.encode(d['text'], add_special_tokens=False) for d in documents]
    layers = [i for i, layer in enumerate(model.layers) if not getattr(layer.attn, 'is_pure_mixer', False)]
    original = compiler.voc_forward
    count = 0
    snapshots = {}
    local = {}

    def snapshot(x):
        y = x.detach().cpu().squeeze(0).clone()
        assert torch.isfinite(y).all()
        return y

    def compare(a, b):
        result = measures(a, b)
        a, b = a.double(), b.double()
        result['pooled_cosine'] = float((a*b).sum() / (a.norm()*b.norm()).clamp_min(1e-30))
        result['token_cosine_mean'] = torch.nn.functional.cosine_similarity(a, b, dim=-1).mean(0).tolist()
        result['zero_soft_token_fraction'] = (a.norm(dim=-1) == 0).double().mean(0).tolist()
        result['zero_hard_token_fraction'] = (b.norm(dim=-1) == 0).double().mean(0).tolist()
        return result

    def hooked(*a, **kw):
        nonlocal count
        layer = layers[count]
        count += 1
        result = original(*a, **kw)
        actual = snapshot(result[0])
        counterpart = list(a)
        counterpart[10] = torch.zeros_like(a[10]) if model.eval_hard else torch.ones_like(a[10])
        other = snapshot(original(*counterpart, **kw)[0])
        torch.testing.assert_close(snapshot(result[0]), actual, atol=0, rtol=0)
        snapshots[layer] = actual
        local[layer] = compare(other, actual) if model.eval_hard else compare(actual, other)
        return result

    compiler.voc_forward = hooked
    counter = model.rng_counter.clone()
    records = []
    try:
        for length, doc in [(2048, 0), (2048, 1), (8192, 0)]:
            model.config.sequence_len = length
            t = tokens[doc]
            ids = torch.tensor([[bos] + t[:length-1]], device='cuda')
            target = torch.tensor([t[:length]], device='cuda')
            cu = torch.tensor([0, length], device='cuda', dtype=torch.int32)
            paths = {}
            for hard in [False, True]:
                model.eval_hard = hard
                count = 0
                snapshots, local = {}, {}
                with torch.no_grad():
                    loss = float(model(ids, target, cu_seqlens=cu))
                    compiler.voc_forward = original
                    plain_loss = float(model(ids, target, cu_seqlens=cu))
                    compiler.voc_forward = hooked
                assert count == len(layers)
                assert torch.equal(counter, model.rng_counter)
                paths[hard] = (loss, snapshots, local, plain_loss)
                for module in model.modules():
                    if type(module).__module__.startswith('flash_mxfp8.'):
                        for name in ['_forward_states', '_forward_plans', '_backward_states', '_backward_plans', '_states']:
                            cache = getattr(module, name, None)
                            if isinstance(cache, dict):
                                cache.clear()
                gc.collect()
            for layer in layers:
                records.append(dict(context=length, document=doc, layer=layer,
                    soft_loss=paths[False][0], hard_loss=paths[True][0],
                    plain_soft_loss=paths[False][3], plain_hard_loss=paths[True][3],
                    end_to_end=compare(paths[False][1][layer], paths[True][1][layer]),
                    same_soft_inputs=paths[False][2][layer], same_hard_inputs=paths[True][2][layer]))
            print(length, doc, 'DONE', paths[False][0], paths[True][0], flush=True)
            Path(args.output).write_text(json.dumps(dict(checkpoint=str(checkpoint),
                protocol='Raw DISM before GDN/norm/gate; end-to-end all-soft/all-hard and paired operator paths at identical inputs from each background; BOS PG19; no optimizer or backward', records=records), indent=2)+'\n')
            del paths
    finally:
        compiler.voc_forward = original


if __name__ == '__main__':
    main()
