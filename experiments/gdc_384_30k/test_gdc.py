"""GDC-arm preflight: parameter count, fp32 finite-difference spot check,
full-model bf16 microbatch forward/backward/AdamW smoke."""
import json
import sys

import torch

from dism_v2.lm_model import DecoderLM, LMConfig, parameter_groups


def main():
    cfg = LMConfig(width=384, heads=6, layers=12, ffn_hidden=1664, softcap=30., gdc=True)
    model = DecoderLM(cfg)
    n = sum(p.numel() for p in model.parameters())
    assert 71_000_000 < n < 73_000_000, n
    print(json.dumps({"event": "parameter_count", "parameters": n}), flush=True)

    # fp32 finite-difference spot check on a single block, tiny width.
    torch.manual_seed(0)
    small = DecoderLM(LMConfig(width=128, heads=2, layers=2, ffn_hidden=352,
                               softcap=30., gdc=True)).cuda()
    x = torch.randint(0, 50257, (2, 256), device="cuda")
    y = torch.randint(0, 50257, (2, 256), device="cuda")
    xg = x.clone()
    loss = small(x, y, 0., None)
    loss.backward()
    g_auto = small.blocks[0].gdc_layer.q_proj.weight.grad.clone()
    w = small.blocks[0].gdc_layer.q_proj.weight
    ok = True
    for i, j in [(0, 0), (3, 7), (100, 100)]:
        old = w[i, j].item()
        with torch.no_grad():
            w[i, j] = old + 1e-2
            lp = small(x, y, 0., None).item()
            w[i, j] = old - 1e-2
            lm = small(x, y, 0., None).item()
            w[i, j] = old
        fd = (lp - lm) / 2e-2
        auto = g_auto[i, j].item()
        match = abs(fd - auto) < 0.05 * max(1., abs(fd))
        ok &= match
        print(json.dumps({"event": "finite_diff", "index": [i, j], "fd": fd,
                          "autograd": auto, "match": bool(match)}), flush=True)
    assert ok
    del small

    # Full-model bf16 microbatch fwd/bwd + AdamW, mirroring the training step.
    torch.manual_seed(777)
    model = DecoderLM(cfg).cuda()
    opt = torch.optim.AdamW(parameter_groups(model, .01), lr=1e-3,
                            betas=(.9, .95), eps=1e-8, fused=True)
    x = torch.randint(0, 50257, (8, 2048), device="cuda")
    y = torch.randint(0, 50257, (8, 2048), device="cuda")
    torch.cuda.synchronize()
    import time
    t0 = time.perf_counter()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = model(x, y, 0., None)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    opt.step()
    torch.cuda.synchronize()
    first = time.perf_counter() - t0
    assert torch.isfinite(loss), loss
    assert torch.isfinite(norm), norm
    t0 = time.perf_counter()
    for _ in range(3):
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(x, y, 0., None)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        opt.step()
    torch.cuda.synchronize()
    steady = (time.perf_counter() - t0) / 3
    print(json.dumps({"event": "gpu_smoke", "loss": loss.item(),
                      "grad_norm": norm.item(),
                      "first_step_seconds": first,
                      "steady_microbatch_seconds": steady,
                      "peak_gib": torch.cuda.max_memory_allocated() / 2**30}), flush=True)


if __name__ == "__main__":
    sys.exit(main())
