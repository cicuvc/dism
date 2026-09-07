"""Validate blkw's PyTorch/CUDA extension path, not attention correctness."""

from pathlib import Path

import pytest
import torch
from torch.utils.cpp_extension import load


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_sm120_extension_build_and_current_stream():
    if torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("sm120 build probe")
    extension = load(
        name="dism_v2_build_smoke",
        sources=[str(Path(__file__).parent / "csrc" / "dism_v2_build_smoke.cu")],
        extra_cuda_cflags=["-O2", "-gencode=arch=compute_120,code=sm_120"],
        verbose=False,
    )
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream), torch.no_grad():
        source = torch.arange(137, device="cuda", dtype=torch.float32).to(torch.bfloat16)
        actual = extension.copy_probe(source)
        empty = extension.copy_probe(source[:0])
    stream.synchronize()
    assert actual.dtype == torch.bfloat16
    assert actual.device == source.device
    assert torch.equal(actual, source)
    assert empty.numel() == 0
