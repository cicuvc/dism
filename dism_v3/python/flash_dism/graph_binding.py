"""P5: explicit core/vocabulary autograd for a planner-owned CUDA graph.

These eager capture entrypoints share the implementations used by the ordinary
custom ops, but keep graph runtime mutations outside their functional schemas.
New binding entrypoints deliberately reject Dynamo/AOT; ordinary entrypoints
retain their existing compilation support. No global registry or model state.
"""
import ctypes
import math

import torch
import cu_flash_dism as cu


class _BindingState:
    def __init__(self, slot, kind, layout):
        if kind not in ('core', 'vocabulary'):
            raise ValueError('binding kind must be core or vocabulary')
        if layout is not None and (layout.tokens != slot.rows*slot.row_tokens or
                                  not cu.graph_layout_maximum(layout.table)):
            raise ValueError('packed binding requires an owned fixed-capacity graph layout')
        self.slot, self.kind, self.layout = slot, kind, layout
        # Fixed native kernels read this tensor; the true-only dispatch flag
        # specializes only packed kernels. Own the semantic override in both.
        self.direction = torch.ones((slot.rows if layout is None else 1, slot.heads),
                                    device=slot.device, dtype=torch.bool)
        self.pending = False
        self.capture_id = None
        self.capture_graph = None
        self.failed = None
        self.stream = None
        self.stream_owner = None
        self.staged_layout = None
        self.get_capture = slot.runtime.writer.library.cuStreamGetCaptureInfo_v2
        self.get_capture.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_uint64), ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        self.get_capture.restype = ctypes.c_int

    def capture(self):
        stream = torch.cuda.current_stream(self.slot.device)
        status, identity, graph = ctypes.c_int(), ctypes.c_uint64(), ctypes.c_void_p()
        with torch.cuda.device(self.slot.device):
            error = self.get_capture(stream.cuda_stream, ctypes.byref(status), ctypes.byref(identity), ctypes.byref(graph), None, None)
        if error or status.value == 2:
            raise RuntimeError(f'graph binding capture query failed: CUDA {error}, status {status.value}')
        if status.value == 1:
            if self.capture_graph is not None and self.capture_graph != graph.value:
                raise RuntimeError('binding belongs to a different CUDA graph')
            self.capture_graph = graph.value
        return stream.cuda_stream, identity.value if status.value == 1 else None

    def check(self):
        self.slot._check_open()
        if self.failed is not None:
            raise RuntimeError('graph binding failed; create a new binding/runtime') from self.failed

    def begin(self, kind, q, sq, value, hard):
        self.check()
        if kind != self.kind:
            raise ValueError('graph binding operator kind mismatch')
        if self.pending:
            raise RuntimeError('a binding cannot have two forwards pending; allocate one slot per layer')
        if self.slot.state != 'idle':
            raise RuntimeError('warmup/capture requires an idle slot; arm only after capture')
        stream, capture_id = self.capture()
        if self.capture_id is not None:
            raise RuntimeError('binding already captured; replay its graph or create a new slot')
        if self.stream is not None and stream != self.stream:
            raise RuntimeError('binding warmup/capture must use its owning stream')
        expected = ((self.slot.rows, self.slot.row_tokens, self.slot.heads) if self.layout is None
                    else (1, self.slot.rows*self.slot.row_tokens, self.slot.heads))
        if (q.ndim != 4 or tuple(q.shape[:3]) != expected or q.shape[-1] not in (32,64)
                or sq.shape != (*expected,32) or value.shape != (*expected,64)
                or hard.shape != (self.slot.rows,self.slot.heads)
                or q.device != self.slot.device):
            raise ValueError('binding requires fixed geometry D32/64, R32, DV64 and original-row hard flags')
        self.pending, self.stream, self.capture_id = True, stream, capture_id
        self.stream_owner = torch.cuda.current_stream(self.slot.device)

    def submit(self, iq, ik, hard, tau, *, layout=None):
        # Called only after the shared forward's operand validation/interpolation.
        self.slot.snapshot(iq, ik, hard, tau)
        return _GraphTicket(self, iq, ik, hard, tau)

    def finish(self, ticket):
        if ticket.binding is not self or not ticket.consumed or not self.pending:
            raise RuntimeError('graph binding forward/backward ownership mismatch')
        self.slot.finish()
        self.pending = False

    def arm_layout(self, layout):
        self.check()
        if self.pending or self.capture_id is None:
            raise RuntimeError('binding requires a completed forward/backward capture before arm')
        stream, capturing = self.capture()
        if capturing is not None or stream != self.stream:
            raise RuntimeError('stage/replay must use the binding owning stream outside capture')
        if layout is not None and (self.layout is None or
                                  layout.table.data_ptr() != self.layout.table.data_ptr()):
            raise ValueError('binding layout cannot be replaced; update its graph table in place')
        return self.layout

    def validate_replay(self, graph):
        self.arm_layout(None)
        if graph.raw_cuda_graph() != self.capture_graph:
            raise RuntimeError('replay requires the captured graph with keep_graph=True')
        if self.layout is not None and not torch.equal(self.layout.table, self.staged_layout):
            raise RuntimeError('packed layout changed after arm; drain and stage this generation again')

    def retire_host_work(self):
        # Warmup/abandoned eager forwards enqueue real D2H without a replay
        # Future. A stream drain covers these, and never waits on an unlaunched
        # external event from a failed capture. It also covers graph work after
        # this layer's finish node before an optimizer/error boundary.
        if self.stream_owner is not None:
            _, capture_id = self.capture()
            if capture_id is not None:
                raise RuntimeError('cannot drain a graph binding during capture')
            self.stream_owner.synchronize()


class _GraphTicket:
    """Saved tensor handle owns this forward's actual labels/masks/tau and slot."""
    def __init__(self, binding, iq, ik, hard, tau):
        self.binding = binding
        self.inputs = iq, ik, hard, tau
        self.consumed = False

    def backward(self, sq, sk, value, dout, *, fallback=None):
        self.binding.check()
        stream, capture_id = self.binding.capture()
        if (self.consumed or stream != self.binding.stream or
                capture_id != self.binding.capture_id):
            raise RuntimeError('bound backward must run once with its forward in the same stream/capture')
        self.consumed = True
        return self.binding.slot.backward(sq, sk, value, dout, *self.inputs,
                                           layout=self.binding.layout)


def _state(handle, kind):
    state = cu.get_training_plan(handle)
    if not isinstance(state, _BindingState) or state.kind != kind:
        raise ValueError('invalid graph binding handle or operator kind')
    return state


class _Core(torch.autograd.Function):
    @staticmethod
    def forward(ctx, handle, absorbed, *raw):
        from .compiler import _core_forward_impl
        state = _state(handle, 'core')
        state.begin('core', raw[0], raw[2], raw[4], raw[10])
        raw = (*raw[:9], state.direction, *raw[10:])
        table = None if state.layout is None else state.layout.table
        try:
            saved = _core_forward_impl(list(raw), table, 0, absorbed, True,
                                       planner=state, direction_true=True)
        except Exception as exc:
            state.failed = exc
            raise
        ctx.absorbed = absorbed
        ctx.save_for_backward(handle, *raw, *saved, *([] if table is None else [table]))
        ctx.packed = table is not None
        return saved[0]

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, dout):
        from .compiler import _core_backward_impl
        saved = ctx.saved_tensors
        state = _state(saved[0], 'core')
        try:
            grads = _core_backward_impl(list(saved[1:13]), list(saved[13:19]),
                saved[19] if ctx.packed else None, 0, ctx.absorbed, dout, direction_true=True)
            state.finish(cu.get_training_plan(saved[18]))
        except Exception as exc:
            state.failed = exc
            raise
        return (None, None, *grads[:7], None, None, None, None, grads[7])


class _Vocabulary(torch.autograd.Function):
    @staticmethod
    def forward(ctx, handle, beta, direction, hard, *raw):
        from .compiler import _voc_forward_impl
        state = _state(handle, 'vocabulary')
        state.begin('vocabulary', raw[0], raw[2], raw[4], hard)
        direction = state.direction
        layout = None if state.layout is None else state.layout.table.flatten()
        try:
            saved = _voc_forward_impl(*raw, beta, direction, hard, None, layout, True,
                                      direction_true=True, planner=state)
        except Exception as exc:
            state.failed = exc
            raise
        ctx.beta, ctx.packed = beta, layout is not None
        ctx.save_for_backward(handle, *raw, direction, hard, *saved,
                              *([] if layout is None else [layout]))
        return saved[0]

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, dout):
        from .compiler import _voc_backward_impl
        saved = ctx.saved_tensors
        state = _state(saved[0], 'vocabulary')
        try:
            grads = _voc_backward_impl([*saved[1:9], ctx.beta, *saved[9:11]],
                list(saved[11:23]), None, dout, saved[23] if ctx.packed else None,
                direction_true=True)
            state.finish(cu.get_training_plan(saved[22]))
        except Exception as exc:
            state.failed = exc
            raise
        return (None, None, None, None, *grads)


class GraphPlannerBinding:
    """One layer/slot/graph, explicit fixed-true-direction training entrypoints.

    Construct through slot.bind('core'|'vocabulary', layout=...). Use a graph with
    keep_graph=True so replay can check its identity. Warm up complete
    forward/backward calls, capture one complete invocation on that same stream,
    then slot.arm() / runtime.replay(graph) / runtime.drain(). Mutable packed
    boundaries must update the bound graph layout before arm(). Never checkpoint
    or clone the handle. This object is not a Module and changes no state_dict.
    """
    def __init__(self, slot, kind, *, layout=None):
        slot._check_open()
        with torch.cuda.device(slot.device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError('create graph bindings before capture')
        if slot._binding is not None or slot.state != 'idle' or slot.epoch:
            raise RuntimeError('binding requires a fresh unbound slot')
        state = _BindingState(slot, kind, layout)
        self.handle = cu.hold_training_plan(state)
        slot._binding = state

    def _training(self, raw):
        if torch.compiler.is_compiling():
            raise RuntimeError('explicit planner bindings support eager CUDA capture, not Dynamo/AOT')
        if not torch.is_grad_enabled() or not any(t.requires_grad for t in raw):
            raise RuntimeError('graph binding requires a gradient-enabled training invocation')

    def core(self, *raw, absorbed=False):
        """Same twelve operands as dism_core; direction is explicitly fixed true."""
        if len(raw) != 12:
            raise ValueError('core binding requires twelve operands')
        self._training(raw)
        return _Core.apply(self.handle, absorbed, *raw)

    def vocabulary(self, q, k, sq, sk, value, eq, ek, tau, *, direction, hard, beta=1.0):
        """Vocabulary interpolation + full DISM, host beta and fixed true direction."""
        raw = (q,k,sq,sk,value,eq,ek,tau)
        self._training(raw)
        if isinstance(beta, torch.Tensor) or not math.isfinite(beta) or beta <= 0:
            raise ValueError('graph vocabulary requires a finite positive host beta')
        return _Vocabulary.apply(self.handle, float(beta), direction, hard, *raw)
