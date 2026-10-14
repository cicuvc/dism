"""Chunked fused-linear CE benchmark on the real read-only 72M fixture.

Times the new chunked operator (softcap and ``cu_seqlens`` document weights) and
the existing local fused-CE control on the same resident checkpoint input,
scanning chunk sizes for both objectives required by
``docs/FUSED_CE_TASK.md``: no softcap with token mean, and softcap 30 with
document mean.  Complete compiled autograd is the comparison unit, at upstream
gradient 1 and 1/4; eager autograd, peak extra allocation and a separate
device-duration profile per kernel family are reported alongside.

Read-only: ``bench_cross_entropy.fixture`` captures one eager/no-grad forward of
the saved first prefetched batch.  No optimizer is constructed, no dataloader
position advances, no checkpoint is written and nothing is downloaded.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import torch
import triton
import flash_dism
import cu_flash_dism
from flash_dism.kernels import fused_ce_chunked as chunked_ce
from flash_dism.kernels import fused_cross_entropy as legacy_ce
from bench_cross_entropy import ROOT, fixture
from bench_fla_cross_entropy import diagnostics
from bench_training import command, gpu_status, measure, memory, profile_kernels


_CUOBJDUMP = ("cuobjdump", "/usr/local/cuda/bin/cuobjdump",
              "/usr/local/cuda-13.4/bin/cuobjdump",
              "/usr/local/cuda-13.1/bin/cuobjdump")


MODES = (
    # name, softcap, packed cu_seqlens, reduction
    ("token_nosoftcap", None, False),
    ("document_softcap30", 30.0, True),
)


def build(module, dense_labels, cu_seqlens, max_seqlen, softcap, packed,
          chunk_rows, compile_graph, logits_dtype=None):
    if module == "chunked":
        def entry(x, w):
            return chunked_ce.fused_ce_chunked(
                x, w, dense_labels, ignore_index=-1, softcap=softcap,
                cu_seqlens=cu_seqlens if packed else None,
                max_seqlen=max_seqlen if packed else None,
                chunk_rows=chunk_rows, logits_dtype=logits_dtype)
    else:
        def entry(x, w):
            return legacy_ce.fused_cross_entropy(
                x, w, dense_labels, ignore_index=-1, softcap=softcap,
                cu_seqlens=cu_seqlens if packed else None,
                max_seqlen=max_seqlen if packed else None)
    return torch.compile(entry, fullgraph=True) if compile_graph else entry


def pair_runner(callable_entry, x, w, upstream):
    def pair():
        return torch.autograd.grad(
            callable_entry(x, w), (x, w), grad_outputs=upstream)
    with torch.no_grad():
        loss = float(callable_entry(x, w))
    return pair, loss


def run_mode(args, dense_labels, cu_seqlens, max_seqlen, x, w, softcap, packed,
             name, module, chunk_rows=None, reference=None, logits_dtype=None):
    a = x.detach().requires_grad_()
    b = w.detach().requires_grad_()
    one = torch.ones((), device=x.device, dtype=torch.float32)
    quarter = torch.full((), 0.25, device=x.device, dtype=torch.float32)
    compiled = build(module, dense_labels, cu_seqlens, max_seqlen, softcap,
                     packed, chunk_rows, True, logits_dtype)
    pair, loss = pair_runner(compiled, a, b, one)
    pair_quarter, _ = pair_runner(compiled, a, b, quarter)
    eager = build(module, dense_labels, cu_seqlens, max_seqlen, softcap,
                  packed, chunk_rows, False, logits_dtype)
    eager_pair, eager_loss = pair_runner(eager, a, b, one)
    torch.cuda.synchronize()

    grads = pair()
    quarter_grads = pair_quarter()
    if not all(bool(torch.isfinite(g).all()) for g in grads):
        raise RuntimeError(f"non-finite gradients for {name}")
    if abs(loss - eager_loss) > 1e-5 * max(1.0, abs(loss)):
        raise RuntimeError(
            f"compiled/eager loss mismatch for {name}: {loss} vs {eager_loss}")

    row = {
        "name": name,
        "module": module,
        "reduction": "document_mean" if packed else "token_mean",
        "softcap": softcap,
        "chunk_rows": chunk_rows,
        "backend": "torch.compile fullgraph public autograd",
        "loss": loss,
        "autograd_upstream_1": measure(pair, args.warmup, args.repeats,
                                       args.samples),
        "autograd_upstream_quarter": measure(pair_quarter, args.warmup,
                                             args.repeats, args.samples),
        "autograd_eager_upstream_1": measure(eager_pair, args.warmup,
                                             args.repeats, args.samples),
        "memory": memory(pair),
    }
    row["upstream_quarter_ratio"] = {
        key: diagnostics(g, reference_g.float() * 0.25)
        for key, g, reference_g in zip(("dhidden", "dweight"), quarter_grads,
                                       grads)}
    del quarter_grads
    if module == "chunked":
        state = chunked_ce.chunked_ce_forward(
            x, w, dense_labels, ignore_index=-1, softcap=softcap,
            cu_seqlens=cu_seqlens if packed else None,
            max_seqlen=max_seqlen if packed else None, chunk_rows=chunk_rows,
            logits_dtype=logits_dtype)
        row["normalizer"] = {
            "mean_sum": float(state.mean_sum),
            "count": int(state.count),
            "document_losses": (
                None if state.document_loss is None
                else [float(v) for v in state.document_loss]),
            "reduction": state.reduction,
        }
        row["dtypes"] = {
            "dhidden": str(state.dhidden.dtype),
            "dweight": str(state.dweight.dtype),
        }
    if reference is not None:
        row["versus_local_control"] = {
            "loss_abs": abs(loss - reference["loss"]),
            "dhidden": diagnostics(grads[0], reference["grads"][0]),
            "dweight": diagnostics(grads[1], reference["grads"][1]),
        }
        if module == "chunked":
            # The returned gradients carry the input dtype (one extra BF16
            # rounding).  Compare the internal pre-upstream tensors as well, so
            # the arithmetic difference is visible separately from that cast.
            row["versus_local_control_internal"] = {
                "dhidden": diagnostics(state.dhidden, reference["grads"][0]),
                "dweight": diagnostics(state.dweight, reference["grads"][1]),
                "note": "no upstream scaling, no output-dtype cast",
            }
    if args.profile:
        row["profile"] = profile_kernels(pair, iterations=3)
    print(name, f"loss {loss:.6f}",
          {k: round(v["stream_ms_median"], 3) for k, v in row.items()
           if isinstance(v, dict) and "stream_ms_median" in v},
          f"peak+{row['memory']['peak_increment_bytes'] / 2**20:.1f}MiB",
          flush=True)
    del compiled, eager, pair, eager_pair, pair_quarter
    torch.cuda.empty_cache()
    return row, {"loss": loss, "grads": [g.detach() for g in grads]}


def disassemble(cubin):
    """SASS text for a cubin; the toolkit ships outside PATH on this host."""
    last = None
    for tool in _CUOBJDUMP:
        try:
            return subprocess.run(
                [tool, "-sass", str(cubin)], capture_output=True, text=True,
                check=True).stdout
        except (FileNotFoundError, subprocess.CalledProcessError) as error:
            last = error
    raise last


def kernel_probe(args, data, chunk_rows):
    """Record registers/spills and disassembly for the new Triton kernels."""
    x, w, labels, cu_seqlens = (data[k] for k in ("x", "w", "labels", "cu"))
    hidden = x.reshape(-1, x.shape[-1])
    vocab_size = w.shape[0]
    rows = min(chunk_rows, hidden.shape[0])
    logits = torch.mm(hidden[:rows], w.t())
    lse = torch.empty(rows, device="cuda", dtype=torch.float32)
    nll = torch.empty(hidden.shape[0], device="cuda", dtype=torch.float32)
    scale = torch.full((hidden.shape[0],), 1.0 / rows, device="cuda")
    n_v = triton.cdiv(vocab_size, chunked_ce._CE_STATS_BLOCK_V)
    shape = (rows, n_v)
    partial_max = torch.empty(shape, device="cuda", dtype=torch.float32)
    partial_sum = torch.empty(shape, device="cuda", dtype=torch.float32)
    partial_target = torch.empty(shape, device="cuda", dtype=torch.float32)
    block_m = chunked_ce._CE_BLOCK_M
    record = {}
    launched = {
        "_chunk_partial_stats": chunked_ce._chunk_partial_stats[
            (triton.cdiv(rows, block_m), n_v)](
            logits, labels.reshape(-1), partial_max, partial_sum,
            partial_target, rows, n_v, 0, vocab_size, 30.0,
            IGNORE_INDEX=-1, HAS_SOFTCAP=True, BLOCK_M=block_m,
            BLOCK_V=chunked_ce._CE_STATS_BLOCK_V, num_warps=4),
        "_chunk_row_stats": chunked_ce._chunk_row_stats[
            (triton.cdiv(rows, chunked_ce._CE_ROW_STATS_BLOCK_M),)](
            labels.reshape(-1), lse, nll, partial_max, partial_sum,
            partial_target, rows, n_v, 0, IGNORE_INDEX=-1,
            BLOCK_M=chunked_ce._CE_ROW_STATS_BLOCK_M,
            BLOCK_N=chunked_ce._CE_ROW_STATS_BLOCK_N, num_warps=4),
        "_chunk_dlogits": chunked_ce._chunk_dlogits[
            (triton.cdiv(rows, block_m),
             triton.cdiv(vocab_size, chunked_ce._CE_DLOGITS_BLOCK_V))](
            logits, labels.reshape(-1), lse, scale, rows, 0, vocab_size, 30.0,
            IGNORE_INDEX=-1, HAS_SOFTCAP=True, BLOCK_M=block_m,
            BLOCK_V=chunked_ce._CE_DLOGITS_BLOCK_V, num_warps=4),
    }
    torch.cuda.synchronize()
    for kernel_name, kernel in launched.items():
        entry = {
            "registers_per_thread": kernel.n_regs,
            "spills": kernel.n_spills,
            "shared_bytes": kernel.metadata.shared,
        }
        cubin = args.output.with_suffix("").with_name(
            args.output.stem + kernel_name + ".cubin")
        cubin.write_bytes(kernel.asm["cubin"])
        cubin.with_suffix(".ptx").write_text(kernel.asm["ptx"])
        entry["cubin"] = str(cubin)
        try:
            sass = disassemble(cubin)
        except (FileNotFoundError, subprocess.CalledProcessError) as error:
            entry["sass"] = f"unavailable: {error}"
        else:
            cubin.with_suffix(".sass").write_text(sass)
            entry["sass"] = str(cubin.with_suffix(".sass"))
            entry["sass_call_or_local"] = sorted({
                token for token in
                ("CALL", "LDL", "STL")
                if f"{token} " in sass or f"{token}." in sass})
        record[kernel_name] = entry
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--step", type=int, default=19074)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--chunk-rows", type=int, nargs="+",
                        default=[1024, 2048, 4096])
    parser.add_argument("--no-profile", action="store_true")
    parser.add_argument("--kernel-probe", action="store_true")
    parser.add_argument(
        "--fp32-buffer", action="store_true",
        help="add FP32 chunk-buffer cases to separate GEMM input rounding "
             "from the BF16 chunk-buffer rounding")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.profile = not args.no_profile
    if args.output.exists() or not args.output.resolve().is_relative_to(
            ROOT / "dism_v3" / "build"):
        parser.error("choose a fresh output path under dism_v3/build")
    if min(args.warmup, args.repeats, args.samples,
           *args.chunk_rows) < 1:
        parser.error("counts and chunk rows must be positive")
    for module in (flash_dism, cu_flash_dism):
        if not Path(module.__file__).resolve().is_relative_to(
                ROOT / "dism_v3" / "python"):
            raise RuntimeError(f"foreign module {module.__file__}")
    torch.set_num_threads(1)
    # The scan compiles one graph per (mode, chunk size); the default Dynamo
    # recompile limit counts those as recompiles of one code object.
    torch._dynamo.config.recompile_limit = 64
    torch._inductor.config.compile_threads = 1
    torch._inductor.config.triton.cudagraphs = False

    data, info = fixture(args.checkpoint_dir, args.step)
    x, w, labels, cu_seqlens = (data[k] for k in ("x", "w", "labels", "cu"))
    dense_labels = labels.reshape(1, -1)
    max_seqlen = int(cu_seqlens.diff().max())
    if info["ce_options"] != dict(ignore_index=-1, softcap=30.):
        raise ValueError(f"unexpected CE options: {info['ce_options']}")

    report = {
        "arguments": {k: str(v) if isinstance(v, Path) else v
                      for k, v in vars(args).items()},
        "git_head": command("git", "rev-parse", "HEAD"),
        "git_status": command("git", "status", "--short"),
        "gpu_before": gpu_status(),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "fixture": info,
        "shape": {
            "tokens": x.shape[1],
            "width": x.shape[2],
            "vocabulary": w.shape[0],
            "documents": cu_seqlens.numel() - 1,
            "ignored_labels": int((labels == -1).sum()),
        },
        "sources": {
            module.__name__: {
                "path": module.__file__,
                "sha256": hashlib.sha256(
                    Path(module.__file__).read_bytes()).hexdigest(),
            } for module in (chunked_ce, legacy_ce)},
        "method": (
            "Warm resident real inputs captured from the saved first prefetched "
            "batch; CUDA-event and host-wall intervals; no CUDA graphs; "
            "torch.compile fullgraph for every measured entry (including the "
            "eager column, which is the same entry uncompiled); complete "
            "public autograd is the comparison unit, so each call includes the "
            "chunked forward that precomputes dHidden/dWeight; upstream "
            "gradient 1 and 1/4; memory is peak incremental allocation above "
            "live inputs, not reservation; profiling is a separate intrusive "
            "device-duration attribution; Dynamo recompile limit raised for "
            "the scan only"),
        "cases": [],
        "controls": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    save()
    controls = {}
    for mode_name, softcap, packed in MODES:
        row, outputs = run_mode(
            args, dense_labels, cu_seqlens, max_seqlen, x, w, softcap, packed,
            f"local_control_{mode_name}", "legacy")
        report["controls"].append(row)
        controls[mode_name] = outputs
        save()
    for chunk_rows in args.chunk_rows:
        for mode_name, softcap, packed in MODES:
            row, _ = run_mode(
                args, dense_labels, cu_seqlens, max_seqlen, x, w, softcap,
                packed, f"chunked_{mode_name}_c{chunk_rows}", "chunked",
                chunk_rows=chunk_rows, reference=controls[mode_name])
            report["cases"].append(row)
            save()
    if args.kernel_probe:
        report["kernel_probe"] = kernel_probe(args, data, args.chunk_rows[0])
        save()
    if args.fp32_buffer:
        for mode_name, softcap, packed in MODES:
            row, _ = run_mode(
                args, dense_labels, cu_seqlens, max_seqlen, x, w, softcap,
                packed,
                f"chunked_fp32buffer_{mode_name}_c{args.chunk_rows[0]}",
                "chunked", chunk_rows=args.chunk_rows[0],
                reference=controls[mode_name], logits_dtype=torch.float32)
            row["logits_dtype"] = "torch.float32"
            report["cases"].append(row)
            save()
    report["gpu_after"] = gpu_status()
    save()
    print("saved", args.output, flush=True)


if __name__ == "__main__":
    main()
