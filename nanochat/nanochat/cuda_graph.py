"""Packed DISM training forward/backward replay with an eager optimizer.

The graph consumes one microbatch and accumulates parameter gradients. Metadata
is validated and staged before replay; input storage, workspace capacities and
launch dimensions stay fixed even when the document count/lengths change.
Nothing here advances a dataloader or enters the model/optimizer state_dict.
"""
import os
import gc
import torch
import cu_flash_dism as cu
from flash_dism.varlen import VarlenLayout


def _workspace_tensors(model):
    """Keep pre-capture MXFP8 workspaces alive across intervening eager evals."""
    found, seen = [], set()

    def visit(value):
        if id(value) in seen:
            return
        seen.add(id(value))
        if isinstance(value, torch.Tensor):
            found.append(value.detach())
        elif isinstance(value, dict):
            for item in value.values():
                visit(item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                visit(item)
        elif type(value).__module__.startswith('flash_mxfp8') and hasattr(value, '__dict__'):
            visit(vars(value))

    for module in model.modules():
        for name in ('_forward_states', '_backward_states', '_forward_plans', '_backward_plans', '_states'):
            visit(getattr(module, name, None))
    return found


class PackedTrainingGraph:
    """One fixed [training rows, row length] graph; documents remain variable.

    Call forward_backward once per consumed microbatch. Use the existing
    DocumentLossAccumulator, clipping, optimizer and checkpoint flow outside.
    Zeroing gradients with either set_to_none value is supported. One instance
    must not be used concurrently, or between a forward and a pending backward.
    With planner=True, drain before normalization/optimizer/eval/checkpoint and
    close on exit (or use a context manager). Runtime state is never serialized.
    """
    def __init__(self, model, input_ids, targets, cu_seqlens, *, microbatches=1, warmup=3,
                 planner=False, planner_workers=None, planner_max_bytes=2**30):
        if not planner and os.environ.get('DISM_HARD_PLANNER', '0') != '0':
            raise ValueError('planner-enabled CUDA Graph requires explicit planner=True / --cuda-graph-planner')
        if not model.training or not input_ids.is_cuda or microbatches < 1 or warmup < 1:
            raise ValueError('expected a CUDA training model and positive accumulation count')
        if getattr(model.config, 'tau_schedule_fraction', 0):
            raise ValueError('CUDA Graph training currently requires a constant host tau scale')
        if input_ids.ndim != 2 or input_ids.shape != targets.shape:
            raise ValueError('expected matching [training rows, row length] inputs/targets')
        if planner:
            if getattr(model.config, 'hard_query_sampled', False):
                raise ValueError('sampled-CV CUDA Graph requires planner=False')
            if torch.distributed.is_initialized() and torch.distributed.get_world_size() != 1:
                raise ValueError('planner CUDA Graph currently supports one training rank')
            if (os.environ.get('DISM_HARD_BACKWARD', 'ready') != 'ready' or
                    os.environ.get('DISM_PLANNER_COMPACT', '1') != '1'):
                raise ValueError('planner CUDA Graph supports only compact C64 with ready policy')
            if (model.config.readout_dim != 32 or model.config.value_dim != 64 or
                    model.config.head_dim not in (32,64)):
                raise ValueError('planner CUDA Graph requires R32, DV64 and D32/64')
        self.planner_enabled = bool(planner)
        self.runtime, self.bindings = None, None
        self.closed, self.error, self.pending = False, None, False
        self.last_routes, self.last_planner_reasons = (), ()
        self.route_counts = {'e': 0, 'q': 0}
        self.model, self.microbatches = model, microbatches
        self.device = input_ids.device
        self.shape = tuple(input_ids.shape)
        self.parameters = list(model.parameters())
        self.parameter_pointers = tuple(p.data_ptr() for p in self.parameters)
        self.temperature = model.vocab_temperature
        self.state_buffers = tuple((name, value, value.data_ptr())
                                   for name, value in model.named_buffers(recurse=False))
        self.stream = torch.cuda.Stream(device=self.device)
        saved_grads = [p.grad for p in self.parameters]
        saved_counter = model.rng_counter.clone()
        saved_rng = torch.get_rng_state()
        saved_cuda_rng = torch.cuda.get_rng_state(self.device)
        from flash_dism.hard_training import HardPlanTicket
        route_start = len(HardPlanTicket._ROUTE_LOG)
        self.hard_layers = sum(not getattr(layer.attn, 'is_pure_mixer', False)
                               for layer in model.layers)
        self.stream.wait_stream(torch.cuda.current_stream(self.device))
        try:
            # MXFP8 plans may retain autograd output tensors from an earlier
            # eager call. Rebuild these runtime-only plans on the capture stream
            # so their AccumulateGrad nodes cannot refer to the default stream.
            for module in model.modules():
                for name in ('_forward_states', '_backward_states', '_forward_plans', '_backward_plans', '_states'):
                    cache = getattr(module, name, None)
                    if cache is not None:
                        cache.clear()
            gc.collect()
            with torch.cuda.stream(self.stream):
                self.ids = torch.empty_like(input_ids)
                self.targets = torch.empty_like(targets)
                total = input_ids.numel()
                self.boundaries = torch.empty(total//256+1, dtype=torch.int32, device=self.device)
                self.chunks = torch.empty((total//64, 2), dtype=torch.int32, device=self.device)
                host = self._host_boundaries(cu_seqlens)
                self.table = cu.make_graph_layout(host, total, self.shape[1], self.shape[1])
                self.layout = VarlenLayout.for_graph(self.table, total)
                self._stage(input_ids, targets, host)
                if self.planner_enabled:
                    from flash_dism.planner_graph import GraphPlannerRuntime
                    if not self.hard_layers:
                        raise ValueError('planner CUDA Graph requires DISM layers')
                    workers = int(os.environ.get('DISM_PLANNER_WORKERS', '16')) if planner_workers is None else planner_workers
                    self.runtime = GraphPlannerRuntime(device=self.device, workers=workers,
                        max_slots=self.hard_layers, max_bytes=planner_max_bytes)
                    self.bindings = tuple(None if getattr(layer.attn, 'is_pure_mixer', False) else
                        self.runtime.slot(rows=self.shape[0], heads=layer.attn.heads, row_tokens=self.shape[1], topk=getattr(model.config, "hard_query_topk", 1))
                            .bind('vocabulary', layout=self.layout) for layer in model.layers)
                for _ in range(warmup):
                    model.zero_grad(set_to_none=True)
                    self._step()
                gc.collect()
                self.grads = [p.grad for p in self.parameters]
                for grad in self.grads:
                    if grad is not None:
                        grad.zero_()
                self.graph = torch.cuda.CUDAGraph(keep_graph=self.planner_enabled)
                with torch.cuda.graph(self.graph, stream=self.stream):
                    self.loss, self.count = self._step()
                if self.planner_enabled:
                    self.graph.instantiate()
                self.workspaces = _workspace_tensors(model)
                self.fp8_buffers = [(module, name, tensor.data_ptr())
                                    for module in model.modules()
                                    for name, tensor in module.named_buffers(recurse=False)
                                    if 'weight_fp8' in name or 'weight_scale' in name]
        except BaseException:
            try:
                if self.runtime is not None:
                    self.runtime.close()
            finally:
                self.stream.synchronize()
            raise
        finally:
            torch.cuda.current_stream(self.device).wait_stream(self.stream)
            model.rng_counter.copy_(saved_counter)
            torch.set_rng_state(saved_rng)
            torch.cuda.set_rng_state(saved_cuda_rng, self.device)
            for p, grad in zip(self.parameters, saved_grads):
                p.grad = grad
            del HardPlanTicket._ROUTE_LOG[route_start:]

    def _step(self):
        loss, count = self.model(
            self.ids, self.targets, cu_seqlens=self.boundaries,
            loss_reduction='document_sum', graph_layout=self.layout,
            gdn_chunk_indices=self.chunks,
            graph_planner_bindings=self.bindings,
        )
        (loss / self.microbatches).backward()
        return loss.detach(), count.detach()

    @staticmethod
    def _host_boundaries(boundaries):
        if boundaries.dtype != torch.int32 or boundaries.ndim != 1:
            raise ValueError('graph cu_seqlens must be int32 [documents+1]')
        return boundaries.detach().to(device='cpu').contiguous()

    def _stage(self, ids, targets, host):
        if tuple(ids.shape) != self.shape or targets.shape != ids.shape:
            raise ValueError('CUDA Graph input shape changed; construct a new graph')
        if ids.device != self.device or targets.device != self.device:
            raise ValueError('CUDA Graph input device changed')
        if ids.dtype != self.ids.dtype or targets.dtype != self.targets.dtype:
            raise ValueError('CUDA Graph input dtype changed')
        cu.update_graph_layout(self.table, host)
        # The native table compacts empty documents and pads unused slots with
        # empty intervals. This preserves document-balanced loss and positions.
        lengths = self.table[:, 1].tolist()
        ends = [0]
        chunks = []
        for doc, length in enumerate(lengths):
            ends.append(ends[-1] + length)
            chunks.extend((doc, index) for index in range(length//64))
        self.boundaries.copy_(torch.tensor(ends, dtype=torch.int32, pin_memory=True), non_blocking=True)
        self.chunks.copy_(torch.tensor(chunks, dtype=torch.int32, pin_memory=True), non_blocking=True)
        self.ids.copy_(ids)
        self.targets.copy_(targets)

    def forward_backward(self, input_ids, targets, cu_seqlens):
        if self.closed:
            raise RuntimeError('training graph is closed')
        self.drain()
        if not self.planner_enabled and os.environ.get('DISM_HARD_PLANNER', '0') != '0':
            raise ValueError('planner-enabled CUDA Graph requires explicit planner=True / --cuda-graph-planner')
        if not self.model.training or self.model.vocab_temperature != self.temperature:
            raise ValueError('graph training mode/host temperature changed; construct a new graph')
        current_parameters = tuple(self.model.parameters())
        if (len(current_parameters) != len(self.parameters) or
                any(a is not b for a,b in zip(current_parameters,self.parameters)) or
                tuple(p.data_ptr() for p in self.parameters) != self.parameter_pointers):
            raise ValueError('graph parameter storage changed; construct a new graph')
        if any(getattr(self.model, name) is not value or value.data_ptr() != pointer
               for name,value,pointer in self.state_buffers):
            raise ValueError('graph state buffer storage changed; construct a new graph')
        if any(getattr(module, name).data_ptr() != pointer for module, name, pointer in self.fp8_buffers):
            raise ValueError('graph FP8 weight storage changed; construct a new graph')
        current = torch.cuda.current_stream(self.device)
        self.stream.wait_stream(current)
        with torch.cuda.stream(self.stream):
            host = self._host_boundaries(cu_seqlens)
            self._stage(input_ids, targets, host)
            # Accumulate into stable graph-owned .grad storage, preserving any
            # existing microbatch contribution and normal set_to_none resets.
            for parameter, grad in zip(self.parameters, self.grads):
                if grad is None:
                    continue
                if parameter.grad is not grad:
                    if parameter.grad is None:
                        grad.zero_()
                    else:
                        grad.copy_(parameter.grad)
                    parameter.grad = grad
            if self.runtime is None:
                self.graph.replay()
            else:
                for slot in self.runtime.slots:
                    slot.arm()
                self.pending = True
                try:
                    self.runtime.replay(self.graph)
                except BaseException as exc:
                    self.error = exc
                    current.wait_stream(self.stream)
                    raise
        current.wait_stream(self.stream)
        if self.runtime is None:
            self._record_routes(('q',)*self.hard_layers)
        return self.loss, self.count

    def _record_routes(self, routes):
        from flash_dism.hard_training import HardPlanTicket
        self.last_routes = tuple(routes)
        for route in routes:
            HardPlanTicket._record_route(route)
            self.route_counts[route] += 1

    def drain(self):
        """Before loss normalization, clipping, optimizer, evaluation or checkpoint.

        A returned loss alone does not imply CPU planning/upload has finished.
        Errors remain sticky; no failed microbatch can authorize an update.
        """
        try:
            if self.runtime is not None:
                self.runtime.drain()
            if self.error is not None:
                raise RuntimeError('training graph failed; rebuild after handling the error') from self.error
            if self.pending:
                routes = tuple('e' if s.choice.item() else 'q' for s in self.runtime.slots)
                self.last_planner_reasons = tuple(s.reason for s in self.runtime.slots)
                self._record_routes(routes)
        except BaseException as exc:
            self.error = exc
            raise
        finally:
            self.pending = False

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            self.drain()
        finally:
            try:
                if self.runtime is not None:
                    self.runtime.close()
            finally:
                self.stream.synchronize()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
