"""P2/P3: reconstruct full-hard query rows from actual fixed-true kernel inputs.

For a matching suffix of length l and per-head tau, exp(score) is
sum(exp(k*tau), k=1..l). Only O(N) storage is needed for a requested row.
CPU FP64 reconstruction is diagnostic only; the model runs its original path.
"""
import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from niah_circuit_probe import ARMS, RUNS, ROOT, atomic_json, digest, evaluate


def hard_row(iq, ik, i, tau):
    """Return suffix lengths and stable scores for all causal keys of query i."""
    lengths = np.zeros(i + 1, dtype=np.int32)
    active = np.arange(i + 1)
    for offset in range(i + 1):
        active = active[active >= offset]
        active = active[ik[active - offset] == iq[i - offset]]
        if active.size == 0:
            break
        lengths[active] += 1
    scores = np.full(i + 1, -np.inf, dtype=np.float64)
    matched = lengths > 0
    l = lengths[matched].astype(np.float64)
    if abs(tau) < 1e-12:
        scores[matched] = np.log(l)
    elif tau > 0:
        scores[matched] = l * tau + np.log(-np.expm1(-l * tau)) - np.log(-np.expm1(-tau))
    else:
        scores[matched] = tau + np.log(-np.expm1(l * tau)) - np.log(-np.expm1(tau))
    zmax = max(0., float(scores.max()))
    weight = np.exp(scores - zmax)
    fallback = np.exp(-zmax)
    denominator = weight.sum() + fallback
    return lengths, weight / denominator, fallback / denominator


def describe_rows(iq, ik, sq, sk, v, tau, actual, sample, layer, arm):
    result = []
    lo, hi = sample["target_span"]
    needle_lo, needle_hi = sample["needle_span"]
    for h in range(iq.shape[0]):
        for t in range(lo, hi):
            i = t - 1
            lengths, mass, fallback = hard_row(iq[h], ik[h], i, tau[h])
            signed = sk[:i+1, h] @ sq[i, h]
            contributions = (mass * signed)[:, None] * v[:i+1, h]
            reconstructed = contributions.sum(0)
            reference = actual[i, h]
            span = slice(needle_lo, needle_hi)
            pair_rows = []
            for j in range(needle_lo, needle_hi):
                pair_rows.append(dict(key_position=j, key_symbol=int(ik[h, j]),
                                      chain_length=int(lengths[j]), mass=float(mass[j]),
                                      signed_readout=float(signed[j]),
                                      rank_min=int(1 + np.count_nonzero(mass > mass[j])),
                                      contribution_norm=float(np.linalg.norm(contributions[j]))))
            result.append(dict(layer=layer, slot=h, base_head=h//2 if arm=="gqa2" else h,
                               channel=h%2 if arm=="gqa2" else 0, target_offset=t-lo,
                               target_token=sample["target_tokens"][t-lo], query_position=i,
                               query_symbol=int(iq[h, i]), tau=float(tau[h]),
                               needle_mass=float(mass[span].sum()),
                               other_mass=float(mass.sum()-mass[span].sum()), fallback_mass=float(fallback),
                               needle_contribution_norm=float(np.linalg.norm(contributions[span].sum(0))),
                               core_norm=float(np.linalg.norm(reference)),
                               reconstruction_abs_l2=float(np.linalg.norm(reconstructed-reference)),
                               reconstruction_rel_l2=float(np.linalg.norm(reconstructed-reference)/max(np.linalg.norm(reference),1e-12)),
                               needle_pairs=pair_rows))
    return result


@contextmanager
def capture_rows(model, sample, arm, rows):
    import flash_dism.compiler as compiler
    import flash_dism.kernels.interp_true as interpolation
    original_voc = compiler.voc_forward
    original_emb = interpolation.emb_fwd_true
    active = []
    handles = []
    calls = []
    for layer in range(6, 12):
        def pre(_module, _args, index=layer):
            active[:] = [index]
        handles.append(model.layers[layer].attn.register_forward_pre_hook(pre))
    captured = []

    def embed(*args, **kwargs):
        output = original_emb(*args, **kwargs)
        captured.append((output[-1].detach().cpu().numpy(), output[-2].detach().cpu().numpy()))
        return output

    def voc(q, k, sq, sk, v, eq, ek, tau, beta, direction, hard, boundaries, layout=None, *, direction_true=False):
        assert direction_true and bool(hard.all()) and q.shape[0] == 1
        assert boundaries.tolist() == [0, q.shape[1]], "one independent trial per capture"
        output = original_voc(q, k, sq, sk, v, eq, ek, tau, beta, direction, hard, boundaries, layout, direction_true=direction_true)
        assert len(captured) == 1, "actual interpolation hook must fire exactly once"
        iq, ik = captured.pop()
        def cpu(x):
            return x.detach().double().cpu().numpy()
        rows.extend(describe_rows(iq[0], ik[0], cpu(sq[0]), cpu(sk[0]), cpu(v[0]), cpu(tau), cpu(output[0][0]), sample, active[0], arm))
        calls.append(active[0])
        return output

    compiler.voc_forward = voc
    interpolation.emb_fwd_true = embed
    try:
        yield calls
    finally:
        compiler.voc_forward = original_voc
        interpolation.emb_fwd_true = original_emb
        for handle in handles:
            handle.remove()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--trials", type=int, default=1)
    p.add_argument("--start-trial", type=int, default=0)
    a = p.parse_args()
    if a.out.exists():
        p.error("output already exists")
    suite = json.loads(a.inputs.read_text())
    if a.start_trial < 0 or not 1 <= a.trials <= suite["trials"] - a.start_trial:
        p.error("trials must fit the fixed suite")
    for name in ("flash_dism", "cu_flash_dism"):
        origin = Path(importlib.util.find_spec(name).origin).resolve()
        if not origin.is_relative_to(ROOT / "dism_v3"):
            raise RuntimeError(f"foreign module: {name}: {origin}")
    a.out.mkdir(parents=True)
    spec = importlib.util.spec_from_file_location("original_niah", RUNS / ARMS["old"][0] / "diagnostics/niah.py")
    harness = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(harness)
    torch.cuda.set_per_process_memory_fraction(0.30)
    for arm, (run, tag) in ARMS.items():
        subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.free,utilization.gpu", "--format=csv"], check=True)
        if torch.cuda.mem_get_info()[0] < 10*1024**3:
            raise RuntimeError("less than 10 GiB free")
        checkpoint = RUNS / run / "base_checkpoints" / tag / "model_019074.pt"
        model, meta = harness.load_ckpt(checkpoint)
        model.eval_hard = True
        model.config.sequence_len = max(c["context"] for c in suite["cells"])
        buffers = {k: v.detach().cpu().clone() for k, v in model.named_buffers()}
        result = dict(arm=arm, checkpoint=str(checkpoint), model_spec=meta["model_spec"],
                      suite_sha256=digest(suite), inputs=str(a.inputs.resolve()),
                      probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), cells=[])
        for cell in suite["cells"]:
            for sample in cell["samples"][a.start_trial:a.start_trial+a.trials]:
                begin = time.monotonic()
                rows = []
                with capture_rows(model, sample, arm, rows) as calls:
                    score = evaluate(model, [sample], "present", 1)[0]
                assert calls == list(range(6, 12)), calls
                result["cells"].append(dict(context=cell["context"], depth=cell["depth"], trial=sample["trial"], score=score, rows=rows))
                atomic_json(a.out / f"{arm}.json", result)
                print(arm, cell["context"], cell["depth"], sample["trial"],
                      'max row relative reconstruction error', max(r["reconstruction_rel_l2"] for r in rows),
                      f'{time.monotonic()-begin:.1f}s', flush=True)
        for key, value in model.named_buffers():
            if not torch.equal(value.detach().cpu(), buffers[key]):
                raise AssertionError(f"evaluation changed buffer {key}")
        result["buffers_unchanged"] = True
        atomic_json(a.out / f"{arm}.json", result)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
