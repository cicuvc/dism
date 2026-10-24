"""Fixed-storage packed FLA attention/RoPE forward+backward capture."""
import gc
import torch
from nanochat.cuda_graph import _workspace_tensors
from nanochat.models.fla_transformer.model import rotary_block_size


class PackedAttentionTrainingGraph:
    def __init__(self, model, input_ids, targets, cu_seqlens, *, microbatches=1,
                 warmup=3, planner=False, planner_max_bytes=None):
        if planner:
            raise ValueError('a standard Transformer has no DISM planner')
        if not model.training or not input_ids.is_cuda or microbatches < 1 or warmup < 1:
            raise ValueError('expected CUDA training with positive warmup/accumulation')
        self.model, self.microbatches = model, microbatches
        self.device, self.shape = input_ids.device, tuple(input_ids.shape)
        self.parameters = list(model.parameters())
        self.pointers = tuple(p.data_ptr() for p in self.parameters)
        self.closed = False
        self.stream = torch.cuda.Stream(device=self.device)
        self.block = rotary_block_size(input_ids.numel(), self.device)
        self.ids, self.targets = torch.empty_like(input_ids), torch.empty_like(targets)
        self.boundaries = torch.empty(input_ids.numel()//256+1, device=self.device, dtype=torch.int32)
        self.chunks = torch.empty((input_ids.numel()//self.block,2), device=self.device, dtype=torch.int32)
        self._stage(input_ids, targets, cu_seqlens)
        saved_grads = [p.grad for p in self.parameters]
        cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state(self.device)
        self.stream.wait_stream(torch.cuda.current_stream(self.device))
        try:
            for module in model.modules():
                for name in ('_forward_states','_backward_states','_forward_plans','_backward_plans','_states'):
                    value = getattr(module, name, None)
                    if isinstance(value, dict):
                        value.clear()
            gc.collect()
            with torch.cuda.stream(self.stream):
                for _ in range(warmup):
                    model.zero_grad(set_to_none=True)
                    self._step()
                gc.collect()
                self.grads = [p.grad for p in self.parameters]
                for grad in self.grads:
                    if grad is not None:
                        grad.zero_()
                self.graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(self.graph, stream=self.stream):
                    self.loss, self.count = self._step()
                self.workspaces = _workspace_tensors(model)
                # Eval can grow/rebind FLA's nonpersistent RoPE tables. Captured
                # forward/backward still reference the original table storage.
                self.rotary_tables = [table for layer in model.layers
                    for table in (layer.attn.rotary._cos_cached, layer.attn.rotary._sin_cached)]
                self.weight_buffers = [(module, name, tensor.data_ptr()) for module in model.modules()
                    for name,tensor in module.named_buffers(recurse=False)
                    if 'weight_fp8' in name or 'weight_scale' in name]
        finally:
            torch.cuda.current_stream(self.device).wait_stream(self.stream)
            torch.set_rng_state(cpu_rng)
            torch.cuda.set_rng_state(cuda_rng, self.device)
            for p, grad in zip(self.parameters, saved_grads):
                p.grad = grad

    def _step(self):
        loss, count = self.model(self.ids, self.targets, cu_seqlens=self.boundaries,
            rotary_chunks=self.chunks, loss_reduction='document_sum')
        (loss/self.microbatches).backward()
        return loss.detach(), count.detach()

    def _stage(self, ids, targets, boundaries):
        if tuple(ids.shape)!=self.shape or targets.shape!=ids.shape:
            raise ValueError('graph input shape changed')
        if ids.device!=self.device or targets.device!=self.device or ids.dtype!=self.ids.dtype or targets.dtype!=self.targets.dtype:
            raise ValueError('graph input device/dtype changed')
        if boundaries.ndim!=1 or boundaries.dtype!=torch.int32:
            raise ValueError('expected int32 document boundaries')
        ends = boundaries.detach().cpu().tolist()
        total = ids.numel()
        if (len(ends)<2 or ends[0]!=0 or ends[-1]!=total or
                any(a>b or a%256 or b%256 or b-a>self.shape[1] for a,b in zip(ends,ends[1:]))):
            raise ValueError('invalid aligned packed boundaries')
        # Compact zero-length documents, then pad with empty intervals at the end.
        compact = [ends[0]]
        for end in ends[1:]:
            if end!=compact[-1]:
                compact.append(end)
        chunks = [(doc,i) for doc,(a,b) in enumerate(zip(compact,compact[1:]))
                  for i in range((b-a)//self.block)]
        compact += [total]*(self.boundaries.numel()-len(compact))
        self.boundaries.copy_(torch.tensor(compact,dtype=torch.int32,pin_memory=True),non_blocking=True)
        self.chunks.copy_(torch.tensor(chunks,dtype=torch.int32,pin_memory=True),non_blocking=True)
        self.ids.copy_(ids)
        self.targets.copy_(targets)

    def forward_backward(self, input_ids, targets, cu_seqlens):
        if self.closed or not self.model.training:
            raise RuntimeError('graph is closed or model is not training')
        params = tuple(self.model.parameters())
        if (len(params)!=len(self.parameters) or any(a is not b for a,b in zip(params,self.parameters))
                or tuple(p.data_ptr() for p in params)!=self.pointers):
            raise ValueError('graph parameters changed')
        if any(getattr(module,name).data_ptr()!=pointer for module,name,pointer in self.weight_buffers):
            raise ValueError('graph quantized weight storage changed')
        current = torch.cuda.current_stream(self.device)
        self.stream.wait_stream(current)
        with torch.cuda.stream(self.stream):
            self._stage(input_ids, targets, cu_seqlens)
            for p,grad in zip(self.parameters,self.grads):
                if grad is not None and p.grad is not grad:
                    if p.grad is None:
                        grad.zero_()
                    else:
                        grad.copy_(p.grad)
                    p.grad=grad
            self.graph.replay()
        current.wait_stream(self.stream)
        return self.loss,self.count

    def drain(self):
        pass  # No asynchronous CPU planner work.

    def close(self):
        if not self.closed:
            self.stream.synchronize()
            self.closed=True
