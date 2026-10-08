import torch


def mark_cu_seqlens_dynamic(cu_seqlens, *group_tensors):
    """Mark varlen group dimensions dynamic before calling ``torch.compile``.

    Dynamo's duck-shape inference can accidentally tie ``cu_seqlens.shape[0]``
    to an unrelated static model dimension with the same example value.  Mark
    the first example explicitly so changes in the number of packed documents
    reuse the original graph.  Stateful convolution callers should pass the
    input state as a group tensor because its leading dimension changes with
    the number of documents as well.

    This function must run on the example tensors before their first compiled
    call; ``torch._dynamo.mark_dynamic`` cannot be invoked while tracing.
    """
    if not isinstance(cu_seqlens, torch.Tensor) or cu_seqlens.ndim != 1:
        raise ValueError("cu_seqlens must be a one-dimensional tensor")
    torch._dynamo.mark_dynamic(cu_seqlens, 0)
    for tensor in group_tensors:
        if not isinstance(tensor, torch.Tensor) or tensor.ndim < 1:
            raise ValueError("group tensors must have at least one dimension")
        torch._dynamo.mark_dynamic(tensor, 0)
    return cu_seqlens
