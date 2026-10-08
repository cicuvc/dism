import pytest
import torch
import cu_flash_dism


@pytest.mark.parametrize('query_lse', [False, True])
def test_key_metadata_layout(query_lse):
    # Labels must preserve every int32 bit, including patterns that would
    # represent NaN/Inf if accidentally interpreted as floating-point values.
    labels = torch.arange(64, device='cuda', dtype=torch.int32) * 7919
    labels[:6] = torch.tensor([0x7fc12345, 0x7f800000, -2147483648,
                              2147483647, -1, 0], device='cuda', dtype=torch.int32)
    bias = torch.arange(64, device='cuda', dtype=torch.float32) * .125 - 3
    actual_labels, actual_bias = cu_flash_dism.key_metadata_probe(labels, bias, query_lse)
    lane = torch.arange(32, device='cuda')[:, None, None, None]
    block = torch.arange(4, device='cuda')[None, :, None, None]
    half = torch.arange(2, device='cuda')[None, None, :, None]
    component = torch.arange(2, device='cuda')[None, None, None, :]
    column = (lane % 4) * 16 + 2 * block + half + 8 * component
    torch.testing.assert_close(actual_labels, labels[column], atol=0, rtol=0)
    expected_bias = torch.zeros_like(actual_bias) if query_lse else -bias[column]
    torch.testing.assert_close(actual_bias, expected_bias, atol=0, rtol=0)
