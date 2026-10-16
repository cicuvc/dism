"""P1 follow-up: zero individual pre-mix DISM slots at L8/L10, far needles."""
import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

import torch

from niah_circuit_probe import ARMS, RUNS, atomic_json, digest, evaluate
from summarize_niah_circuit import estimate


@contextmanager
def zero_slot(model, layer, slot):
    import flash_dism.compiler as compiler
    original = compiler.voc_forward
    active = []
    calls = []
    handles = []
    for index in range(6, 12):
        def pre(_module, _args, index=index):
            active[:] = [index]
        handles.append(model.layers[index].attn.register_forward_pre_hook(pre))

    def wrapper(*args, **kwargs):
        output = original(*args, **kwargs)
        if active == [layer]:
            values = output[0].clone()
            values[:, :, slot] = 0
            output = [values, *output[1:]]
            calls.append(layer)
        return output

    compiler.voc_forward = wrapper
    try:
        yield calls
    finally:
        compiler.voc_forward = original
        for handle in handles:
            handle.remove()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--p1", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--condition", choices=["present", "absent"], default="present")
    p.add_argument("--contexts", type=int, nargs="+")
    p.add_argument("--slots", type=int, nargs="+")
    a = p.parse_args()
    if a.out.exists():
        p.error("output already exists")
    a.out.mkdir(parents=True)
    suite = json.loads((a.p1 / "inputs.json").read_text())
    spec = importlib.util.spec_from_file_location("original_niah", RUNS / ARMS["old"][0] / "diagnostics/niah.py")
    harness = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(harness)
    torch.cuda.set_per_process_memory_fraction(.30)
    for arm, (run, tag) in ARMS.items():
        subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.free,utilization.gpu", "--format=csv"], check=True)
        if torch.cuda.mem_get_info()[0] < 10*1024**3:
            raise RuntimeError("less than 10 GiB free")
        baseline = json.loads((a.p1 / f"{arm}.json").read_text())
        assert baseline["suite_sha256"] == digest(suite)
        model, meta = harness.load_ckpt(RUNS / run / "base_checkpoints" / tag / "model_019074.pt")
        model.eval_hard = True
        model.config.sequence_len = max(c["context"] for c in suite["cells"])
        buffers = {k: v.detach().cpu().clone() for k, v in model.named_buffers()}
        result = dict(arm=arm, condition=a.condition, suite_sha256=digest(suite), checkpoint=baseline["checkpoint"],
                      model_spec=meta["model_spec"], probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), cells=[])
        for ci, cell in enumerate(suite["cells"]):
            if cell["depth"] != .05:
                continue
            if a.contexts is not None and cell["context"] not in a.contexts:
                continue
            samples = cell["samples"]
            base = baseline["cells"][ci]["measurements"]["baseline"][a.condition]
            # Detect numerical drift before comparing to saved baselines.
            replay = evaluate(model, samples[:1], a.condition, 1)[0]
            assert abs(replay["nll"]-base[0]["nll"]) < 1e-6, (arm, replay, base[0])
            row = dict(context=cell["context"], depth=cell["depth"], ablations=[])
            result["cells"].append(row)
            for layer in (8, 10):
                for slot in range(12 if arm == "gqa2" else 6):
                    if a.slots is not None and slot not in a.slots:
                        continue
                    with zero_slot(model, layer, slot) as calls:
                        scores = evaluate(model, samples, a.condition, 1)
                    assert len(calls) == len(samples)
                    delta = estimate([x["nll"]-y["nll"] for x, y in zip(scores, base)])
                    row["ablations"].append(dict(layer=layer, slot=slot, scores=scores, **{f"delta_{a.condition}_nll": delta}))
                    atomic_json(a.out / f"{arm}.json", result)
                    print(arm, cell["context"], layer, slot, delta, flush=True)
        for key, value in model.named_buffers():
            if not torch.equal(value.detach().cpu(), buffers[key]):
                raise AssertionError(f"evaluation changed buffer {key}")
        result["buffers_unchanged"] = True
        atomic_json(a.out / f"{arm}.json", result)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
