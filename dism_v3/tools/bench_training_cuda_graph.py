"""Synthetic packed training throughput; no datasets or checkpoints are accessed.

Run each mode in a fresh process. Timing includes input H2D copies, mutable
packed metadata, accumulated forward/backward, document normalization, clipping
and the production optimizer. Compilation/capture/warmup are excluded, as are
disk loading, evaluation, saving and optional training instrumentation.
"""
import argparse
import contextlib
import json
import math
import os
from pathlib import Path
import statistics
import time

import torch
import cu_flash_dism
import flash_dism
from nanochat.cuda_graph import PackedTrainingGraph
from nanochat.document_loss import DocumentLossAccumulator
from nanochat.models import ModelSpec, build_model, PretrainOptimizerConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-spec', required=True)
    parser.add_argument('--mode', choices=['compile', 'graph', 'eager'], required=True)
    parser.add_argument('--packing', choices=['changing', 'full'], default='changing')
    parser.add_argument('--training-step', type=int, default=11444)
    parser.add_argument('--rows', type=int, default=8)
    parser.add_argument('--microbatches', type=int, default=8)
    parser.add_argument('--warmup', type=int, default=4)
    parser.add_argument('--steps', type=int, default=16)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--nsys', action='store_true',
                        help='Trace graph capture origins and measured steps with Nsight Systems')
    parser.add_argument('--interp-backend', choices=['split','tma'], default='split')
    args = parser.parse_args()
    if args.interp_backend == 'tma':
        from functools import partial
        from flash_dism.kernels import interp_bwd_split
        interp_bwd_split.emb_bwd_split = partial(interp_bwd_split.emb_bwd_split, fuse_dq_tma=True)
    if min(args.rows, args.microbatches, args.warmup, args.steps, args.threads) < 1:
        parser.error('counts must be positive')
    if os.environ.get('DISM_HARD_PLANNER', '0') != '0':
        parser.error('requires DISM_HARD_PLANNER=0')
    if args.nsys and args.mode != 'graph':
        parser.error('--nsys requires --mode graph')

    def scope(name):
        return torch.cuda.nvtx.range(name) if args.nsys else contextlib.nullcontext()
    root = Path(__file__).resolve().parents[2]
    for module in (flash_dism, cu_flash_dism):
        if not Path(module.__file__).resolve().is_relative_to(root/'dism_v3/python'):
            raise RuntimeError(f'foreign import: {module.__file__}')
    torch.set_num_threads(args.threads)
    torch.manual_seed(1741)
    torch.set_float32_matmul_precision('high')
    with open(args.model_spec) as source:
        spec = ModelSpec.from_dict(json.load(source))
    model = build_model(spec).cuda().train()
    model.init_weights()
    model.set_training_step(args.training_step)
    optimizer = model.setup_pretraining_optimizer(
        PretrainOptimizerConfig(.0008, .0008, .0008, .0008, .01))
    length, vocab = model.config.sequence_len, model.config.vocab_size
    if length % 256:
        parser.error('sequence length must be 256-aligned')
    batches = []
    for micro in range(args.microbatches):
        ids = torch.randint(vocab, (args.rows, length), pin_memory=True)
        labels = torch.randint(vocab, ids.shape, pin_memory=True)
        labels[:, ::17] = -1
        parts = ([length], [256, length-256], [512]*(length//512), [256]*(length//256))
        selected = [length] if args.packing == 'full' else parts[micro % 4]
        if any(n <= 0 for n in selected) or sum(selected) != length:
            parser.error('changing packing requires a sequence length divisible by 512')
        ends = [0]
        for _ in range(args.rows):
            for n in selected:
                ends.append(ends[-1]+n)
        bounds = torch.tensor(ends, dtype=torch.int32, pin_memory=True)
        batches.append((ids, labels, bounds))
    ids_gpu = torch.empty_like(batches[0][0], device='cuda')
    labels_gpu = torch.empty_like(ids_gpu)
    bounds_gpu = torch.empty(args.rows*length//256+1, dtype=torch.int32, device='cuda')

    def stage(batch):
        ids, labels, boundaries = batch
        ids_gpu.copy_(ids, non_blocking=True)
        labels_gpu.copy_(labels, non_blocking=True)
        bounds = bounds_gpu[:boundaries.numel()]
        bounds.copy_(boundaries, non_blocking=True)
        return bounds

    setup_start = time.perf_counter()
    graph = None
    runner = model
    if args.mode == 'compile':
        from torch.fx.experimental import _config as shape_config
        shape_config.use_duck_shape = False
        torch._dynamo.config.capture_dynamic_output_shape_ops = True
        torch._inductor.config.compile_threads = 1
        runner = torch.compile(model, dynamic=model.dynamic_compile)
    elif args.mode == 'graph':
        bounds = stage(batches[0])
        if args.nsys:
            from profile_training_cuda_graph import annotate_capture
            torch.cuda.synchronize()
            torch.cuda.profiler.start()
            with annotate_capture(model), torch.autograd.profiler.emit_nvtx():
                graph = PackedTrainingGraph(model, ids_gpu, labels_gpu, bounds,
                                            microbatches=args.microbatches)
            for name in ('_host_boundaries', '_stage'):
                original = getattr(graph, name)

                def marked(*a, _fn=original, _name=name, **kw):
                    with scope('profile/metadata/'+_name):
                        return _fn(*a, **kw)

                setattr(graph, name, marked)
        else:
            graph = PackedTrainingGraph(model, ids_gpu, labels_gpu, bounds,
                                        microbatches=args.microbatches)

    def step():
        model.set_training_step(args.training_step)
        document_step = DocumentLossAccumulator(args.microbatches, 'cuda')
        for batch in batches:
            with scope('profile/input_h2d'):
                bounds = stage(batch)
            if graph is None:
                loss, count = runner(ids_gpu, labels_gpu, cu_seqlens=bounds,
                                     loss_reduction='document_sum')
                document_step.add(loss, count)
                (loss/args.microbatches).backward()
            else:
                with scope('profile/graph_forward_backward'):
                    loss, count = graph.forward_backward(ids_gpu, labels_gpu, bounds)
                with scope('profile/loss_accumulate'):
                    document_step.add(loss, count)
        with scope('profile/document_normalize'):
            loss = document_step.finalize(model.parameters())
        with scope('profile/gradient_clip'):
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.).item()
        if not math.isfinite(norm):
            raise FloatingPointError(f'nonfinite gradient norm: {norm}')
        with scope('profile/optimizer'):
            optimizer.step()
        with scope('profile/gradient_clear'):
            model.zero_grad(set_to_none=True)
        value = loss.item()
        if not math.isfinite(value):
            raise FloatingPointError(f'nonfinite loss: {value}')
        return value

    for _ in range(args.warmup):
        step()
    torch.cuda.synchronize()
    setup_seconds = time.perf_counter()-setup_start
    torch.cuda.reset_peak_memory_stats()
    elapsed = []
    for index in range(args.steps):
        start = time.perf_counter()
        with scope(f'profile/step/{index}'):
            loss = step()
            torch.cuda.synchronize()
        elapsed.append(time.perf_counter()-start)
        print(json.dumps(dict(iteration=index, seconds=elapsed[-1], loss=loss)), flush=True)
    if args.nsys:
        torch.cuda.profiler.stop()
    tokens = args.rows*length*args.microbatches
    print(json.dumps(dict(result=True, **vars(args), tokens_per_step=tokens,
                         hard_probability=model.hard_probability.item(),
                         mean_ms=statistics.mean(elapsed)*1000,
                         median_ms=statistics.median(elapsed)*1000,
                         min_ms=min(elapsed)*1000, max_ms=max(elapsed)*1000,
                         tokens_per_second=tokens/statistics.mean(elapsed),
                         peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                         peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
                         setup_and_warmup_seconds=setup_seconds,
                         torch_version=torch.__version__, gpu=torch.cuda.get_device_name())), flush=True)


if __name__ == '__main__':
    main()
