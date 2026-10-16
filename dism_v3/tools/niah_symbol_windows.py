"""Inspect actual old L10h3 q/k labels around fixed NIAH needles and queries."""
import argparse
import importlib.util
import json
from pathlib import Path

import torch

from niah_circuit_probe import ARMS, RUNS, atomic_json, digest, evaluate
from niah_read_rows import hard_row


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--trials', type=int, default=4)
    a = p.parse_args()
    if a.out.exists():
        p.error('output already exists')
    suite = json.loads(a.inputs.read_text())
    cell = next(c for c in suite['cells'] if c['context'] == 16384 and c['depth'] == .05)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(suite['tokenizer'], local_files_only=True)
    spec = importlib.util.spec_from_file_location('original_niah', RUNS / ARMS['old'][0] / 'diagnostics/niah.py')
    harness = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(harness)
    torch.cuda.set_per_process_memory_fraction(.30)
    if torch.cuda.mem_get_info()[0] < 10*1024**3:
        raise RuntimeError('less than 10 GiB free')
    checkpoint = RUNS / ARMS['old'][0] / 'base_checkpoints' / ARMS['old'][1] / 'model_019074.pt'
    model, _ = harness.load_ckpt(checkpoint)
    model.config.sequence_len = 16384
    model.eval_hard = True
    buffers = {k: v.detach().cpu().clone() for k, v in model.named_buffers()}
    import flash_dism.kernels.interp_true as interpolation
    original = interpolation.emb_fwd_true
    active = []
    captured = []
    def pre(_m, _args):
        active[:] = [True]
    def post(_m, _args, _out):
        active.clear()
    def embed(*args, **kwargs):
        output = original(*args, **kwargs)
        if active:
            captured.append((output[-1][0, 3].cpu().numpy(), output[-2][0, 3].cpu().numpy()))
        return output
    handles = [model.layers[10].attn.register_forward_pre_hook(pre),
               model.layers[10].attn.register_forward_hook(post)]
    interpolation.emb_fwd_true = embed
    result = dict(checkpoint=str(checkpoint), suite_sha256=digest(suite), layer=10, head=3, context=16384, depth=.05, samples=[])
    tau = torch.nn.functional.softplus(model.layers[10].attn.log_sel_tau.float())[3].item()
    result['tau'] = tau
    try:
        for sample in cell['samples'][:a.trials]:
            score = evaluate(model, [sample], 'present', 1)
            assert len(captured) == 1
            iq, ik = captured.pop()
            ids = sample['present']
            def window(lo, hi):
                return [dict(position=i, token_id=ids[i], token=tok.convert_ids_to_tokens(ids[i]),
                             text=tok.decode([ids[i]]), q=int(iq[i]), k=int(ik[i])) for i in range(max(0,lo),min(len(ids),hi))]
            lo, hi = sample['needle_span']
            tlo, thi = sample['target_span']
            matches = []
            for t in range(tlo, thi):
                i, j = t-1, lo+t-tlo
                lengths, mass, fallback = hard_row(iq, ik, i, tau)
                best = int(mass.argmax())
                matches.append(dict(target=tok.convert_ids_to_tokens(ids[t]), query_position=i, source_position=j,
                                    query_symbol=int(iq[i]), source_key_symbol=int(ik[j]), chain=int(lengths[j]),
                                    source_mass=float(mass[j]), needle_mass=float(mass[lo:hi].sum()),
                                    best_key=best, best_mass=float(mass[best]), best_token=tok.convert_ids_to_tokens(ids[best])))
            result['samples'].append(dict(trial=sample['trial'], number=sample['number'], score=score,
                                          needle_span=[lo,hi], query_span=sample['query_span'], target_span=[tlo,thi],
                                          needle_window=window(lo-8,hi+8), query_window=window(sample['query_span'][0]-4,thi), matches=matches))
    finally:
        interpolation.emb_fwd_true = original
        for handle in handles:
            handle.remove()
    for k, v in model.named_buffers():
        assert torch.equal(v.cpu(), buffers[k]), k
    result['buffers_unchanged'] = True
    atomic_json(a.out, result)
    print(a.out, flush=True)


if __name__ == '__main__':
    main()
