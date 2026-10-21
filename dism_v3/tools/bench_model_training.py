"""Bounded, read-only-checkpoint 72M training replay; never resumes a dataloader.

Only scratch model/optimizer/RNG copies advance. Replays checkpoint-owned batches
already prefetched on CPU, resident on GPU during timing. No downloads or saves.
Results/profiles must live in this worktree's ignored build directory.
"""
import argparse
import contextlib
import copy
import gc
import hashlib
import functools
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import cu_flash_dism as cu
import flash_dism
from nanochat.models import (build_model, model_spec_from_metadata,
                             PretrainOptimizerConfig, optimizer_schema,
                             optimizer_schema_fingerprint)
from nanochat.metrics import (register_residual_rms_probe, snapshot_parameters,
                              update_rms_max_per_group)
from flash_dism.hard_training import HardPlanTicket, shutdown_hard_planners
from bench_training import command, gpu_status

ROOT = Path(__file__).resolve().parents[2]


def cpu_list(value):
    result=[]
    for part in value.split(',') if value else []:
        pair=part.split('-')
        result.extend(range(int(pair[0]),int(pair[-1])+1))
    if len(set(result))!=len(result):
        raise ValueError('duplicate CPU IDs')
    return result


def thread_snapshot():
    result={}
    for path in Path('/proc/self/task').iterdir():
        try:
            sched=dict(line.split(':',1) for line in (path/'sched').read_text().splitlines() if ':' in line)
            sched={k.strip():v.strip() for k,v in sched.items()}
            result[path.name]=dict(name=(path/'comm').read_text().strip(),
                                  cpus=sorted(os.sched_getaffinity(int(path.name))),
                                  migrations=int(sched['se.nr_migrations']),
                                  runtime_ms=float(sched['se.sum_exec_runtime']))
        except FileNotFoundError:
            pass  # A compiler/helper thread may exit during the snapshot.
    return result


@contextlib.contextmanager
def planner_diagnostics(records):
    original = HardPlanTicket.backward
    def measured(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        # DisabledHardPlanner (DISM_HARD_PLANNER=0) yields future-less quadratic
        # tickets; record them as such instead of dereferencing None.
        future = self.future
        p = future.result() if future is not None and future.done() else None
        if p is None:
            records.append(dict(route=self.route,host_wait_ms=self.host_wait_ms,
                                plan_unavailable=future is not None))
            return result
        records.append(dict(p.timings, route=self.route,host_wait_ms=self.host_wait_ms,
                            native=p.host.timings, statistics=p.host.statistics,
                            single_streams=p.kernel.single_count,
                            multi_streams=p.kernel.multi_count,
                            events=(p.d2h_events, p.h2d_events)))
        return result
    HardPlanTicket.backward = measured
    try:
        yield
    finally:
        HardPlanTicket.backward = original


@contextlib.contextmanager
def planner_nvtx():
    """Extra host timeline labels, only while capturing; no planner math changes."""
    from flash_dism.hard_training import AsyncHardPlanner
    from flash_dism.inference.training_planner import NativeTrainingPlanner, HostTrainingPlan, HostCompactPlan
    patches=[]
    def wrap(cls,name,label):
        original=getattr(cls,name)
        @functools.wraps(original)
        def call(*a,**kw):
            with torch.cuda.nvtx.range(label):
                return original(*a,**kw)
        patches.append((cls,name,original))
        setattr(cls,name,call)
    wrap(AsyncHardPlanner,'submit','planner/submit_D2H')
    wrap(AsyncHardPlanner,'_prepare','planner/coordinator_D2H_wait_prepare_upload')
    wrap(NativeTrainingPlanner,'prepare','planner/native_plan_pack')
    wrap(HostTrainingPlan,'upload','planner/bulk_H2D')
    wrap(HostCompactPlan,'upload','planner/compact_bulk_H2D')
    wrap(HardPlanTicket,'backward','planner/wait_and_event_backward')
    try:
        yield
    finally:
        for cls,name,original in reversed(patches):
            setattr(cls,name,original)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint-dir', type=Path, required=True)
    ap.add_argument('--step', type=int, default=19074)
    ap.add_argument('--batches', type=int, default=4)
    ap.add_argument('--warmup', type=int, default=4)
    ap.add_argument('--phase-steps', type=int, default=5)
    ap.add_argument('--window-steps', type=int, default=10)
    ap.add_argument('--windows', type=int, default=3)
    ap.add_argument('--eager', action='store_true')
    ap.add_argument('--training-probes', action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument('--hard-mode', choices=['scheduled', 'soft', 'hard'], default='scheduled')
    ap.add_argument('--conv-impl', choices=['checkpoint', 'linear_fla'], default='checkpoint',
                    help='explicit scratch-only conversion, including optimizer moments')
    ap.add_argument('--validate-conv', action='store_true', help='compare both frontends to an FP32 convolution oracle')
    ap.add_argument('--profile', action='store_true')
    ap.add_argument('--profile-stack', action='store_true',
                    help='record Python stacks in the torch profile (large trace, host-sync attribution)')
    ap.add_argument('--sync-probe', action='store_true',
                    help='time the FLA chunk-index host round trip and count cache misses per step')
    ap.add_argument('--host-delay-ms', type=float, default=0.0,
                    help='synthetic host stall after gradient accumulation; calibrates host slack')
    ap.add_argument('--nsys', action='store_true', help='one warmed cudaProfilerApi capture with NVTX phases')
    ap.add_argument('--profile-only', action='store_true', help='skip latency/throughput phases after warmup')
    ap.add_argument('--legacy-cuda', type=Path, help='SHA-verified local baseline artifact; compiled benchmark only')
    ap.add_argument('--planner-workers', type=int, default=16, help='process-local planner worker setting')
    ap.add_argument('--hard-backward-policy', choices=['event','ready','quadratic'], default='ready')
    ap.add_argument('--planner-cpus', default='', help='one fixed CPU per native worker, e.g. 2-7,8-17')
    ap.add_argument('--service-cpus', default='', help='CPU set for main/autograd/coordinator; benchmark only')
    ap.add_argument('--launch-cpus', default='', help='After warmup, bind only main/autograd threads; leave planner unchanged')
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    if args.nsys and args.profile:
        ap.error('capture Nsight and PyTorch profiles in separate processes')
    if args.profile_only and not (args.nsys or args.profile):
        ap.error('profile-only requires --profile or --nsys')
    if args.output.exists() or not args.output.resolve().is_relative_to(ROOT/'dism_v3/build'):
        ap.error('output must be a fresh path under dism_v3/build')
    if min(args.batches,args.warmup,args.phase_steps,args.window_steps,args.windows) < 1:
        ap.error('counts must be positive')
    for module in (flash_dism, cu):
        if not Path(module.__file__).resolve().is_relative_to(ROOT/'dism_v3/python'):
            raise RuntimeError(f'foreign import: {module.__file__}')
    torch.set_num_threads(1)
    if args.planner_workers < 1:
        ap.error('planner workers must be positive')
    os.environ['DISM_PLANNER_WORKERS'] = str(args.planner_workers)
    os.environ['DISM_HARD_BACKWARD'] = args.hard_backward_policy
    allowed=os.sched_getaffinity(0)
    worker_cpus,service_cpus=cpu_list(args.planner_cpus),cpu_list(args.service_cpus)
    launch_cpus = cpu_list(args.launch_cpus)
    if not set(launch_cpus) <= allowed or (launch_cpus and service_cpus):
        ap.error('launch CPUs must be allowed and cannot be combined with service CPUs')
    if not set(worker_cpus+service_cpus)<=allowed:
        ap.error('requested CPUs are outside the original allowed CPU set')
    if worker_cpus and len(worker_cpus)!=args.planner_workers:
        ap.error('specify exactly one CPU per planner worker')
    if service_cpus and not worker_cpus:
        ap.error('service CPU isolation requires explicit planner CPUs to avoid inherited worker confinement')
    if set(worker_cpus)&set(service_cpus):
        ap.error('service and worker CPU sets must be disjoint')
    # Set only threads owned by this benchmark process. Later threads inherit
    # the caller's mask; native workers are explicitly repinned after creation.
    if service_cpus:
        for tid in Path('/proc/self/task').iterdir():
            os.sched_setaffinity(int(tid.name),service_cpus)
    from flash_dism.inference.training_planner import NativeTrainingPlanner
    from flash_dism.inference.build import load_prefill
    load_prefill()  # JIT/import is excluded from worker TID discovery.
    original_init=NativeTrainingPlanner.__init__
    worker_tids=[]
    def planner_init(self,workers=8):
        before={int(p.name) for p in Path('/proc/self/task').iterdir()}
        original_init(self,workers)
        after={int(p.name) for p in Path('/proc/self/task').iterdir()}
        tids=sorted(after-before)
        if len(tids)!=workers:
            raise RuntimeError('ambiguous planner worker TIDs; refusing affinity assignment')
        worker_tids.extend(tids)
        for tid,cpu in zip(tids,worker_cpus):
            os.sched_setaffinity(tid,{cpu})
    NativeTrainingPlanner.__init__=planner_init
    legacy_info = None
    if args.legacy_cuda:
        if args.eager:
            ap.error('legacy comparison only overrides the compiled vocabulary operators')
        from bench_legacy_training import install
        legacy_info = install(args.legacy_cuda)
    meta_path = args.checkpoint_dir/f'meta_{args.step:06d}.json'
    meta = json.loads(meta_path.read_text())
    spec = model_spec_from_metadata(meta)
    if spec.config.get('hard_granularity', 'sequence') != 'sequence':
        raise ValueError('only sequence-level checkpoints supported')
    model_state = torch.load(args.checkpoint_dir/f'model_{args.step:06d}.pt',
                             map_location='cpu', weights_only=True)
    model_state = {k.removeprefix('_orig_mod.'):v for k,v in model_state.items()}
    # Trusted local training metadata includes Python/NumPy RNG objects.
    state = torch.load(args.checkpoint_dir/f'train_state_{args.step:06d}_rank0.pt',
                       map_location='cpu', weights_only=False)
    optim_state = torch.load(args.checkpoint_dir/f'optim_{args.step:06d}_rank0.pt',
                             map_location='cpu', weights_only=True)
    if state['model_fingerprint'] != spec.fingerprint():
        raise ValueError('checkpoint model fingerprint mismatch')
    source_schema = meta['optimizer_schema']
    source_fingerprint = optimizer_schema_fingerprint(source_schema)
    if source_fingerprint != meta['optimizer_schema_fingerprint'] or source_fingerprint != state['optimizer_schema_fingerprint']:
        raise ValueError('checkpoint optimizer schema mismatch')
    conversion = None
    validation = None
    if args.validate_conv:
        from bench_conv_frontend import validate_convolution
        validation = validate_convolution(model_state)
        print('convolution validation', json.dumps(validation), flush=True)
    expected_schema = source_schema
    if args.conv_impl == 'linear_fla':
        from bench_conv_frontend import convert
        spec, model_state, optim_state, expected_schema, conversion = convert(
            spec, model_state, optim_state, source_schema)
    if state['grad_scaler'] is not None:
        raise ValueError('benchmark currently requires the BF16/no-scaler baseline')
    batch_keys = ('inputs','targets','cu_seqlens','segment_ids')
    batches = [tuple(state['prefetched_batch'][k] for k in batch_keys)]
    batches.extend(tuple(b[:4]) for b in state['dataloader']['pending'])
    if len(batches) < args.batches:
        raise ValueError('not enough checkpoint-prefetched batches; will not advance loader')
    batches = batches[:args.batches]
    lengths = [b[2].diff().tolist() for b in batches]
    batches = [tuple(t.cuda() for t in b) for b in batches]
    tokens = batches[0][0].numel()
    accum = meta['total_batch_size']//tokens
    if tokens*accum != meta['total_batch_size']:
        raise ValueError('nonintegral accumulation')
    with torch.device('meta'):
        model = build_model(spec)
    model.to_empty(device='cuda')
    model.init_weights()  # nonpersistent buffers initialized before strict restore
    model.load_state_dict(model_state,strict=True)
    model.train()
    u = meta['user_config']
    optimizer = model.setup_pretraining_optimizer(PretrainOptimizerConfig(
        **{k:u[k] for k in ('unembedding_lr','embedding_lr','matrix_lr','scalar_lr','weight_decay')}))
    fingerprint = optimizer_schema_fingerprint(optimizer_schema(model,optimizer))
    if fingerprint != optimizer_schema_fingerprint(expected_schema):
        raise ValueError('optimizer schema mismatch')
    actual_conv_backends = sorted({m.conv.backend for m in model.modules()
                                  if type(m).__name__ == 'LinearShortConvSiLU'})
    if conversion is not None and actual_conv_backends != ['cuda']:
        raise RuntimeError(f'expected CUDA convolution, got {actual_conv_backends}')
    optimizer.load_state_dict(copy.deepcopy(optim_state))
    if args.training_probes:
        handles, probe_buffer = register_residual_rms_probe(model)
    if not args.eager:
        from torch.fx.experimental import _config as shape_config
        shape_config.use_duck_shape = False
        torch._dynamo.config.capture_dynamic_output_shape_ops = True
        torch._inductor.config.compile_threads = 1
    fn = model if args.eager else torch.compile(model,dynamic=True)
    hard_mode = {'scheduled':None,'soft':False,'hard':True}[args.hard_mode]
    cursor = 0
    trace_kind = None

    def mark(name):
        if trace_kind=='torch':
            return torch.profiler.record_function(name)
        if trace_kind=='nsys':
            return torch.cuda.nvtx.range(name)
        return contextlib.nullcontext()

    def restore():
        nonlocal cursor
        torch.cuda.synchronize()
        model.zero_grad(set_to_none=True)
        model.load_state_dict(model_state,strict=True)
        optimizer.load_state_dict(copy.deepcopy(optim_state))
        random.setstate(state['python_rng'])
        np.random.set_state(state['numpy_rng'])
        torch.set_rng_state(state['torch_rng'])
        torch.cuda.set_rng_state(state['cuda_rng'])
        cursor = 0
        gc.collect()
        torch.cuda.synchronize()

    # Optional uninstrumented host-sync accounting. FLA's chunk-index helper is
    # a single-entry identity cache: within one microbatch every mixer reuses
    # the first result, so the blocking device round trip happens once per
    # microbatch. Timing the wrapper measures host blocking directly, without
    # profiler overhead, and distinguishes misses from cache hits.
    sync_counts = {}
    sync_last = None
    if args.sync_probe:
        import fla.ops.gated_delta_rule.chunk as _gdn_chunk
        _sync_real = _gdn_chunk.prepare_chunk_indices

        def _sync_timed(*a, **k):
            nonlocal sync_last
            started = time.perf_counter()
            out = _sync_real(*a, **k)
            hit = out is sync_last
            sync_last = out
            key = 'chunk_index_hit' if hit else 'chunk_index_miss'
            count, total = sync_counts.get(key, (0, 0.0))
            sync_counts[key] = (count + 1, total + (time.perf_counter() - started) * 1000)
            return out

        _gdn_chunk.prepare_chunk_indices = _sync_timed

    def step(measure=False):
        nonlocal cursor
        events, host = [], []
        def begin(name):
            if measure:
                a,b = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                a.record()
                events.append((name,a,b))
                return time.perf_counter()
        def end(t):
            if measure:
                events[-1][2].record()
                host.append((events[-1][0],(time.perf_counter()-t)*1000))
        from nanochat.document_loss import DocumentLossAccumulator
        document_step = DocumentLossAccumulator(accum, model.get_device()) if getattr(model, 'count_normalized_loss', getattr(model, 'document_balanced_loss', False)) else None
        for micro in range(accum):
            x,y,bounds,segments = batches[cursor % len(batches)]
            cursor += 1
            t = begin('forward')
            with mark(f'microbatch_{micro}/forward'):
                if hard_mode is not None:
                    model.hard_probability.fill_(float(hard_mode))
                options = {'loss_reduction': getattr(model, 'loss_sum_reduction', 'document_sum')} if document_step is not None else {}
                result = fn(x,y,cu_seqlens=bounds,segment_ids=segments, **options)
                if document_step is not None:
                    loss, count = result
                    document_step.add(loss, count)
                else:
                    loss = result
            end(t)
            t = begin('backward')
            with mark(f'microbatch_{micro}/backward'):
                (loss/accum).backward()
            end(t)
        if args.host_delay_ms > 0:
            # Synthetic host stall: how much extra host work the step absorbs
            # before it starts costing step time one-for-one.
            deadline = time.perf_counter() + args.host_delay_ms / 1000
            while time.perf_counter() < deadline:
                pass
        t = begin('optimizer_and_probes')
        if document_step is not None:
            loss = document_step.finalize(model.parameters())
        with mark('step/gradient_norm'):
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(),float('inf')).item()
        if not math.isfinite(norm):
            raise FloatingPointError('nonfinite gradient norm')
        with mark('step/optimizer_and_probes'):
            snapshot = snapshot_parameters(model) if args.training_probes else None
            optimizer.step()
            if snapshot is not None:
                update_rms_max_per_group(model,snapshot)
        model.zero_grad(set_to_none=True)
        value = loss.item()
        if not math.isfinite(value):
            raise FloatingPointError('nonfinite loss')
        end(t)
        return value,norm,events,host

    report = dict(arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                  loss_normalization=getattr(model, 'loss_normalization',
                      'global_valid_document_mean_v1' if getattr(model, 'document_balanced_loss', False)
                      else 'legacy_microbatch_mean'),
                  git_head=command('git','rev-parse','HEAD'),git_status=command('git','status','--short'),
                  gpu_before=gpu_status(),torch=torch.__version__,extension=cu.__file__,
                  extension_sha256=hashlib.sha256(Path(cu.__file__).read_bytes()).hexdigest(),
                  benchmark_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  config=spec.to_dict(),parameters=sum(p.numel() for p in model.parameters()),
                  checkpoint_step=args.step,checkpoint_rng_counter=int(model.rng_counter.item()),
                  checkpoint_hard_probability=float(model.hard_probability.item()),
                  legacy_cuda=legacy_info,
                  convolution_conversion=conversion, convolution_validation=validation,
                  convolution_backends=actual_conv_backends,
                  tokens_per_microbatch=tokens,accumulation=accum,document_lengths=lengths,
                  optimizer_groups=[{k:v for k,v in g.items() if k!='params'} for g in optimizer.param_groups],
                  method='Scratch checkpoint model+optimizer replay of resident prefetched batches; '
                  'no dataloader/IO/W&B/checkpoint writes; original RNG and optimizer restored per phase; '
                  'GPU phase events include launch gaps and async waits, no sync between fwd/bwd. '
                  'Sustained windows include real scratch AdamW updates, gradient norm sync and optional training probes.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    def save():
        args.output.write_text(json.dumps(report,indent=2)+'\n')
    restore()
    print(f"Loaded {report['parameters']:,} params; {tokens} tokens x {accum}, "
          f"{args.batches} real batches, direction={spec.config.get('direction_policy', 'true')}; warming up",flush=True)
    for i in range(args.warmup):
        started=time.perf_counter()
        value,norm,_,_=step()
        torch.cuda.synchronize()
        print(f'warmup {i}: {(time.perf_counter()-started)*1000:.1f}ms loss={value:.5f} norm={norm:.4f}',flush=True)
    restore()
    torch.cuda.reset_peak_memory_stats()
    if launch_cpus:
        for path in Path('/proc/self/task').iterdir():
            try:
                if int(path.name) == os.getpid() or (path/'comm').read_text().strip().startswith('pt_autograd'):
                    os.sched_setaffinity(int(path.name), launch_cpus)
            except FileNotFoundError:
                pass
    report['thread_start']=thread_snapshot()
    report['planner_worker_tids']=worker_tids.copy()
    phase=[]
    for i in range(0 if args.profile_only else args.phase_steps):
        started=time.perf_counter()
        value,norm,events,host=step(True)
        torch.cuda.synchronize()
        item=dict(wall_ms=(time.perf_counter()-started)*1000,loss=value,norm=norm,
                  gpu_ms={name:sum(a.elapsed_time(b) for n,a,b in events if n==name)
                          for name in ('forward','backward','optimizer_and_probes')},host_ms=host)
        phase.append(item)
        print('phase',i,item['wall_ms'],item['gpu_ms'],flush=True)
    report['phase_steps']=phase
    report['peak_cuda_bytes']=torch.cuda.max_memory_allocated()
    save()
    restore()
    windows=[]
    sync_windows=[]
    for i in range(0 if args.profile_only else args.windows):
        torch.cuda.synchronize()
        sync_counts.clear()
        started=time.perf_counter()
        for _ in range(args.window_steps):
            value,norm,_,_=step()
        torch.cuda.synchronize()
        elapsed=time.perf_counter()-started
        item=dict(step_ms=elapsed*1000/args.window_steps,
                  tokens_per_second=tokens*accum*args.window_steps/elapsed,loss=value,norm=norm)
        if args.sync_probe:
            item['host_sync_per_step_ms']={
                k:{'calls_per_step':v[0]/args.window_steps,
                   'host_ms_per_step':v[1]/args.window_steps}
                for k,v in sync_counts.items()}
            sync_windows.append(item['host_sync_per_step_ms'])
        windows.append(item)
        print('window',i,item,flush=True)
    report['sustained']=windows
    if args.sync_probe:
        report['host_sync_probe']=dict(
            helper='fla.ops.gated_delta_rule.chunk.prepare_chunk_indices',
            method='wrapper perf_counter around the real helper; host_ms is CPU blocking, '
                   'not added GPU work; miss = the single-entry identity cache did not hit',
            windows=sync_windows)
    report['thread_end']=thread_snapshot()
    save()
    restore()
    records=[]
    with planner_diagnostics(records):
        step()
    torch.cuda.synchronize()
    for r in records:
        events=r.pop('events',None)
        if events is not None:
            d2h,h2d=events
            h2d[1].synchronize()  # diagnostic only, outside timed windows
            r.update(d2h_ms=d2h[0].elapsed_time(d2h[1]),h2d_ms=h2d[0].elapsed_time(h2d[1]))
    report['planner_diagnostic_step']=records
    save()
    if args.profile:
        restore()
        trace_kind='torch'
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA],
                                    record_shapes=True,
                                    with_stack=args.profile_stack) as prof:
            with mark('training_step'):
                step()
                torch.cuda.synchronize()
        trace_kind=None
        prof.export_chrome_trace(str(args.output.with_suffix('.trace.json')))
        kernels={}
        for e in prof.events():
            if e.device_type==torch.autograd.DeviceType.CUDA and not getattr(e,'is_user_annotation',False):
                v=kernels.setdefault(e.name,dict(count=0,total_us=0.))
                v['count']+=1
                v['total_us']+=e.time_range.elapsed_us()
        report['profile_kernels']=sorted([dict(name=k,**v) for k,v in kernels.items()],key=lambda x:-x['total_us'])
        if args.profile_stack:
            # The aggregated call tree is deeper than CPython's default
            # recursion limit once Python frames are recorded; the chrome trace
            # is the supported output for that mode.
            report['profile_cpu_table']='omitted with --profile-stack (see the chrome trace)'
        else:
            report['profile_cpu_table']=prof.key_averages().table(sort_by='self_cpu_time_total',row_limit=40)
    if args.nsys:
        restore()
        trace_kind='nsys'
        with planner_nvtx(), torch.autograd.profiler.emit_nvtx():
            torch.cuda.profiler.start()
            with mark('training_step'):
                step()
                torch.cuda.synchronize()
            torch.cuda.profiler.stop()
        trace_kind=None
    report['gpu_after']=gpu_status()
    report['final_live_plan_handles']=cu.training_plan_handle_count()
    save()
    shutdown_hard_planners()
    print('saved',args.output,flush=True)


if __name__=='__main__':
    main()
