"""Opt-in paired NIAH layer ablations; never writes model/checkpoint state.

Retains the original niah.py task (bare number insertion, six scored tokens).
Samples are materialized on CPU with paired present/absent backgrounds. This
is a new input suite, not a reproduction of the historical RNG traversal.
"""
import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "dism_v3/python"), str(ROOT / "nanochat")]
os.environ.setdefault("FLA_FLASH_KDA", "0")
os.environ.setdefault("DISM_HARD_PLANNER", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
import torch.nn.functional as F

RUNS = Path("/media/cicuvc/c63abdf1-0e56-4153-9228-95df5a2f239b/cicuvc/dism-runs")
ARMS = {
    "old": ("dism-fwedu-valres-72m-2500m", "seqtrue72m-fwedu-valres"),
    "restored": ("dism-fwedu-valres-72m-2500m-tokenbal-quadforce-20261015", "seqtrue72m-fwedu-valres-tokenbal-quadforce"),
    "gqa2": ("dism-fwedu-valres-72m-2500m-tokenbal-gqa2-20261015", "seqtrue72m-fwedu-valres-tokenbal-gqa2"),
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def make_suite(contexts, depths, trials, seed, tokenizer, pg19):
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer, local_files_only=True)
    texts = pq.read_table(pg19, columns=["text"]).column("text")
    hay = []
    for i in range(6):
        hay.extend(tok.encode(texts[i].as_py(), add_special_tokens=False))
    query = tok.encode("\nThe special magic number is", add_special_tokens=False)
    cells = []
    for length in contexts:
        if length <= 0 or length % 256:
            raise ValueError("contexts must be positive multiples of 256")
        for depth in depths:
            if not 0 <= depth <= 1:
                raise ValueError("depth must be in [0,1]")
            # Stable under changing the order/subset of requested cells.
            rng = random.Random(f"{seed}:{length}:{depth}")
            samples = []
            for trial in range(trials):
                number = rng.randint(10000, 99999)
                needle = tok.encode(f"\nThe special magic number is {number}.", add_special_tokens=False)
                assert needle[:len(query)] == query
                tail = needle[len(query):]
                budget = length - 6
                haylen = budget - len(query) - len(tail)
                off = rng.randrange(0, len(hay) - haylen - len(tail))
                background = hay[off:off + haylen]
                pos = int(depth * max(1, haylen - 1))
                sequences = {}
                for condition in ("present", "absent"):
                    insert = tail if condition == "present" else hay[off + haylen:off + haylen + len(tail)]
                    head = background[:pos] + insert + background[pos:] + query
                    assert len(head) == budget
                    seq = (head + tail)[:length]
                    seq += [tok.eos_token_id] * (length - len(seq))
                    sequences[condition] = seq
                samples.append(dict(trial=trial, number=number, hay_offset=off,
                                    needle_span=[pos, pos + len(tail)],
                                    query_span=[budget-len(query), budget],
                                    target_span=[budget, min(budget+len(tail), length)],
                                    target_tokens=tail[:6], **sequences))
            cells.append(dict(context=length, depth=depth, samples=samples))
    return dict(schema=1, seed=seed, trials=trials, tokenizer=tokenizer, pg19=pg19,
                task="bare-number insertion; original six-token target budget; paired absence",
                cells=cells)


@contextmanager
def ablate(model, layer):
    """Zero only the DISM core, retaining the real GDN and shared output path."""
    counts = {"calls": 0}
    if layer is None:
        yield counts
        return
    attn = model.layers[layer].attn
    if not hasattr(attn, "gdn_raw"):
        raise ValueError(f"layer {layer} is not a hybrid layer")
    original = attn._combine_cuda
    had_local = "_combine_cuda" in attn.__dict__

    def combine(output, *args, **kwargs):
        counts["calls"] += 1
        return original(torch.zeros_like(output), *args, **kwargs)

    attn._combine_cuda = combine
    try:
        yield counts
    finally:
        if had_local:
            attn._combine_cuda = original
        else:
            del attn._combine_cuda


class StopForward(Exception):
    pass


@torch.no_grad()
def evaluate(model, samples, condition, chunk):
    result = []
    captured = []

    def stop(_module, _args, output):
        captured.append(output.detach().to(torch.bfloat16))
        raise StopForward()

    handle = model.norm.register_forward_hook(stop)
    try:
        for start in range(0, len(samples), chunk):
            batch = samples[start:start + chunk]
            x = torch.tensor([s[condition] for s in batch], device="cuda", dtype=torch.long)
            b, n = x.shape
            cu = torch.arange(b + 1, device="cuda", dtype=torch.int32) * n
            try:
                model(x, cu_seqlens=cu)
            except StopForward:
                pass
            assert len(captured) == 1
            hidden = captured.pop().reshape(b, n, -1)
            for i, sample in enumerate(batch):
                lo, hi = sample["target_span"]
                logits = hidden[i, lo-1:hi-1].float() @ model.lm_head.weight.float().T
                cap = model.config.softcap
                logits = cap * torch.tanh(logits / cap)
                target = x[i, lo:hi]
                losses = F.cross_entropy(logits, target, reduction="none")
                hits = logits.argmax(-1).eq(target)
                result.append(dict(trial=sample["trial"], nll=losses.mean().item(),
                                   token_nll=losses.tolist(), token_hits=hits.tolist(),
                                   tok_acc=hits.float().mean().item(), all_correct=hits.all().item()))
    finally:
        handle.remove()
    return result


def atomic_json(path, obj):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(obj, indent=2) + "\n")
    temp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--contexts", type=int, nargs="+", default=[8192, 16384])
    p.add_argument("--depths", type=float, nargs="+", default=[0.05, 0.95])
    p.add_argument("--trials", type=int, default=16)
    p.add_argument("--chunk", type=int, default=1)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--arms", nargs="+", choices=list(ARMS), default=list(ARMS))
    p.add_argument("--layers", type=int, nargs="*", default=list(range(6, 12)))
    p.add_argument("--tokenizer", default="/home/cicuvc/downstream-eval/tokenizer")
    p.add_argument("--pg19", default="/home/cicuvc/downstream-eval/data/pg19/test.parquet")
    a = p.parse_args()
    if a.trials < 1 or a.chunk < 1:
        p.error("trials and chunk must be positive")
    if a.out.exists():
        p.error("output directory already exists; choose a fresh directory")
    for name in ("flash_dism", "cu_flash_dism"):
        origin = Path(importlib.util.find_spec(name).origin).resolve()
        if not origin.is_relative_to(ROOT / "dism_v3"):
            raise RuntimeError(f"foreign module: {name}: {origin}")
    a.out.mkdir(parents=True)
    suite = make_suite(a.contexts, a.depths, a.trials, a.seed, a.tokenizer, a.pg19)
    atomic_json(a.out / "inputs.json", suite)
    suite_hash = digest(suite)
    spec = importlib.util.spec_from_file_location("original_niah", RUNS / ARMS["old"][0] / "diagnostics/niah.py")
    harness = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(harness)
    torch.manual_seed(a.seed)
    for arm in a.arms:
        subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.free,utilization.gpu", "--format=csv"], check=True)
        free, total = torch.cuda.mem_get_info()
        if free < 10 * 1024**3:
            raise RuntimeError("less than 10 GiB free; defer probe")
        torch.cuda.set_per_process_memory_fraction(0.30)
        run, tag = ARMS[arm]
        ckpt = RUNS / run / "base_checkpoints" / tag / "model_019074.pt"
        model, meta = harness.load_ckpt(ckpt)
        model.eval_hard = True
        model.config.sequence_len = max(a.contexts)
        buffers = {k: v.detach().cpu().clone() for k, v in model.named_buffers()}
        output = dict(arm=arm, checkpoint=str(ckpt), model_spec=meta["model_spec"],
                      suite_sha256=suite_hash, git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                      probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), chunk=a.chunk, cells=[])
        for cell in suite["cells"]:
            samples = cell["samples"]
            row = dict(context=cell["context"], depth=cell["depth"], measurements={})
            for layer in [None, *a.layers]:
                key = "baseline" if layer is None else f"zero_L{layer:02d}"
                torch.cuda.reset_peak_memory_stats()
                begin = time.monotonic()
                with ablate(model, layer) as counts:
                    measurements = {c: evaluate(model, samples, c, a.chunk) for c in ("present", "absent")}
                expected = 2 * ((len(samples) + a.chunk - 1) // a.chunk)
                if layer is not None and counts["calls"] != expected:
                    raise AssertionError((layer, counts, expected))
                measurements.update(hook_calls=counts["calls"], seconds=time.monotonic()-begin,
                                    peak_allocated=torch.cuda.max_memory_allocated(), peak_reserved=torch.cuda.max_memory_reserved())
                row["measurements"][key] = measurements
                print(arm, cell["context"], cell["depth"], key,
                      {c: round(sum(t["nll"] for t in measurements[c])/len(samples), 5) for c in ("present", "absent")},
                      f'{measurements["seconds"]:.1f}s', flush=True)
                # Persist every finished measurement, including a partial cell.
                atomic_json(a.out / f"{arm}.json", dict(output, cells=[*output["cells"], row]))
            output["cells"].append(row)
        for key, value in model.named_buffers():
            if not torch.equal(value.detach().cpu(), buffers[key]):
                raise AssertionError(f"evaluation changed buffer {key}")
        output["buffers_unchanged"] = True
        atomic_json(a.out / f"{arm}.json", output)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
