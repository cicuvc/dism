"""Real-checkpoint SAM model prefill/decode diagnostic; no downloads."""
import argparse
import sys
import json
import time
from pathlib import Path
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--suite', default='/home/cicuvc/downstream-eval')
    p.add_argument('--length', type=int, default=256)
    p.add_argument('--steps', type=int, default=5)
    p.add_argument('--rebuild-interval', type=int, default=2)
    p.add_argument('--output', required=True)
    p.add_argument('--generation', action='store_true')
    a = p.parse_args()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    sys.path.insert(0, a.suite)
    from adapter_layout_bounded import NanochatLM
    from flash_dism.inference import HardDismPrefill
    from nanochat.models.dism.inference import SamDecodeSession
    lm = NanochatLM(a.checkpoint, str(Path(a.suite)/'tokenizer'), batch_size=1, context=max(2048, ((a.length+a.steps+255)//256)*256))
    text = 'The history of science is a story of discoveries, careful experiments and new ideas. '
    ids = lm.tok_encode(text * (a.length//10+2))[:a.length]
    tokens = torch.tensor([ids], device='cuda')
    records = []
    with torch.inference_mode(), HardDismPrefill(workers=16) as prefill:
        session = SamDecodeSession(lm.model, prefill, lm.max_length, rebuild_interval=a.rebuild_interval)
        for step in range(a.steps+1):
            torch.cuda.synchronize(); start=time.perf_counter()
            hidden = session.features(tokens if step == 0 else tokens[:, -1:])
            torch.cuda.synchronize(); elapsed=time.perf_counter()-start
            x,cu = lm._inputs([tokens[0].tolist()])
            torch.cuda.synchronize(); start=time.perf_counter()
            reference = lm._features(x,cu)[:, tokens.shape[1]-1]
            torch.cuda.synchronize(); reference_ms=(time.perf_counter()-start)*1000
            actual=hidden[:,-1]
            diff=(actual.float()-reference.float())
            record=dict(step=step,position=session.position,ms=elapsed*1000,
                recompute_ms=reference_ms,finite=bool(torch.isfinite(hidden).all()),relative_l2=float(diff.norm()/reference.float().norm()),
                max_abs=float(diff.abs().max()),cached_token=int(lm._greedy(actual)[0]),
                recompute_token=int(lm._greedy(reference)[0]))
            assert record['finite'], 'nonfinite cached hidden'
            print(json.dumps(record), flush=True); records.append(record)
            # Teacher forcing isolates state error from generation divergence.
            tokens=torch.cat((tokens,lm._greedy(reference).view(1,1)),dim=1)
        for state in session.states:
            if state['dism'] is not None:
                assert state['dism'].core.check_status()==session.position
        session.close()
        if a.generation:
            from types import SimpleNamespace
            requests=[SimpleNamespace(args=(prompt, dict(max_gen_toks=16, until=['\n']))) for prompt in ('The capital of France is', 'Once upon a time there was')]
            reference=lm.generate_until(requests)
            lm.decode_backend='sam'; lm.prefill=prefill
            actual=lm.generate_until(requests)
            print(json.dumps(dict(generation_reference=reference, generation_sam=actual)), flush=True)
    Path(a.output).write_text(json.dumps(records,indent=2)+'\n')

if __name__ == '__main__': main()
