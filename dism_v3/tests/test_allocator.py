import torch
import pytest
import cu_flash_dism


@pytest.mark.parametrize('offset', [0, 1, 4, 15, 16, 31, 127, 129, 511, 1023])
def test_shared_allocator(offset):
    got = cu_flash_dism.allocator_probe(torch.empty(0, device='cuda'), offset).cpu().tolist()
    aligned16 = (offset + 8 + 15) // 16 * 16
    aligned128 = (aligned16 + 24 + 127) // 128 * 128
    aligned1024 = (aligned128 + 128 + 1023) // 1024 * 1024
    assert got[:6] == [offset, offset + 3, aligned16, aligned128,
                       aligned1024, aligned1024 + 128]
    assert got[6:12] == [1] * 6
    assert got[12:44] == [300 + 2 * (31 - lane) for lane in range(32)]
    assert got[44] == 40
