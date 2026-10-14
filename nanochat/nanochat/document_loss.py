"""Optimizer-step document balancing without advancing the input iterator.

Each microbatch backpropagates S_m / A, where S_m is the sum of its valid
document means and A is the accumulation count. After A backwards, multiply
gradients by A / D, where D counts valid documents over the entire global step.
The temporary division by A controls gradient magnitude; it does NOT average
microbatch means. Empty/ignored-only documents contribute neither loss nor count.
"""
import torch
import torch.distributed as dist


@torch.no_grad()
def _sum_gradients(parameters, bucket_bytes=16 << 20):
    """SUM replicated gradients before clipping, including rank-local unused params.

    DISM uses ordinary AdamW, not a DDP wrapper or an optimizer with reductions.
    Every rank leaves with the same global gradients. If an optimizer later
    averages replicated gradients, that AVG is an identity, not a second scale.
    Parameters unused on every rank stay None (preserves AdamW decay semantics).
    """
    active = torch.tensor([p.grad is not None for p in parameters],
                          device=parameters[0].device, dtype=torch.int32)
    dist.all_reduce(active, op=dist.ReduceOp.MAX)
    groups = {}
    for p, used in zip(parameters, active.tolist()):
        if not used:
            continue
        if p.grad is None:
            p.grad = torch.zeros_like(p)
        if p.grad.is_sparse:
            raise ValueError('document-balanced training requires dense gradients')
        groups.setdefault((p.device, p.grad.dtype), []).append(p.grad)

    def reduce_bucket(bucket):
        flat = torch.cat([g.reshape(-1) for g in bucket])
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        offset = 0
        for grad in bucket:
            grad.copy_(flat[offset:offset+grad.numel()].view_as(grad))
            offset += grad.numel()

    for gradients in groups.values():
        bucket, size = [], 0
        for grad in gradients:
            nbytes = grad.numel()*grad.element_size()
            if bucket and size+nbytes > bucket_bytes:
                reduce_bucket(bucket)
                bucket, size = [], 0
            bucket.append(grad)
            size += nbytes
        if bucket:
            reduce_bucket(bucket)


class DocumentLossAccumulator:
    """Ephemeral step state; never a module buffer or checkpoint schema field.

    Call add once per consumed microbatch; backward its numerator / microbatches.
    Call finalize exactly once BEFORE GradScaler.unscale_ and gradient clipping.
    No data is prefetched here and no RNG/optimizer state is changed.
    """
    def __init__(self, microbatches, device):
        if microbatches < 1:
            raise ValueError('microbatches must be positive')
        self.microbatches = microbatches
        self.seen = 0
        self.total = torch.zeros((), device=device, dtype=torch.float64)
        self.count = torch.zeros((), device=device, dtype=torch.int64)
        self.finished = False

    @torch.no_grad()
    def add(self, numerator, count):
        if self.finished or self.seen >= self.microbatches:
            raise RuntimeError('document loss accumulator already complete')
        self.total.add_(numerator.detach().to(torch.float64))
        self.count.add_(count.detach().to(torch.int64))
        self.seen += 1

    @torch.no_grad()
    def finalize(self, parameters):
        if self.finished or self.seen != self.microbatches:
            raise RuntimeError('finalize requires exactly one full accumulation step')
        params = list(parameters)  # model.parameters() deduplicates tied weights
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(self.total, op=dist.ReduceOp.SUM)
            dist.all_reduce(self.count, op=dist.ReduceOp.SUM)
            if params:
                _sum_gradients(params)
        # Clamping only defines the all-ignored step: its gradients are zero.
        denominator = self.count.clamp_min(1)
        factor = self.microbatches / denominator.to(torch.float64)
        groups = {}
        for p in params:
            if p.grad is not None:
                groups.setdefault((p.device, p.grad.dtype), []).append(p.grad)
        for (_, dtype), gradients in groups.items():
            torch._foreach_mul_(gradients, factor.to(device=gradients[0].device, dtype=dtype))
        self.finished = True
        return (self.total / denominator).float()
