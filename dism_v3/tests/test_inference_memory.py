"""Inference storage lifetime must preserve outputs, training and graph replay."""
import pytest
import torch

from flash_dism.compiler import _voc_forward_impl, _core_forward_impl, voc_forward
from flash_dism.hard_training import DisabledHardPlanner
from test_planner_graph_binding import inputs, stream

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


@pytest.mark.parametrize('packed', [False, True])
@torch.no_grad()
def test_compiled_inference_changes_hard_mask(packed):
    raw, _, direction, hard, layout = inputs('vocabulary', packed)
    direction.fill_(True)
    table = None if layout is None else layout.table.flatten()

    def run(q):
        return voc_forward(q, *raw[1:], .7, direction, hard, None,
                           table, direction_true=True)[0]

    compiled = torch.compile(run, backend='inductor', fullgraph=True)
    for flag in (True, False, True):
        hard.fill_(flag)
        torch.testing.assert_close(compiled(raw[0]), run(raw[0]), atol=0, rtol=0)


@pytest.mark.parametrize('kind', ['core', 'vocabulary'])
@pytest.mark.parametrize('packed', [False, True])
@pytest.mark.parametrize('mask', ['hard', 'mixed', 'soft'])
@torch.no_grad()
def test_no_backward_output_matches_saved_path(kind, packed, mask, stream):
    raw, _, direction, hard, layout = inputs(kind, packed)
    direction.fill_(True)
    if mask == 'soft':
        hard.zero_()
    elif mask == 'mixed':
        hard[:, 0] = False
    table = None if layout is None else layout.table
    planner = DisabledHardPlanner(raw[0].device)

    def call(save):
        if kind == 'core':
            return _core_forward_impl(list(raw), table, 0, False, save,
                                      planner=planner, direction_true=True)
        return _voc_forward_impl(*raw, .7, direction, hard, None,
                                 None if table is None else table.flatten(), save,
                                 planner=planner, direction_true=True)

    expected = call(True)
    actual = call(False)
    torch.testing.assert_close(actual[0], expected[0], atol=0, rtol=0)
    slots = (1, 2) if kind == 'core' else (7, 8)
    assert all(actual[i].numel() == 0 and expected[i].numel() > 0 for i in slots)
    with torch.cuda.stream(stream):
        call(False)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = call(False)
    for full_hard in (True, False):
        hard.fill_(full_hard)
        graph.replay()
        torch.testing.assert_close(captured[0], call(True)[0], atol=0, rtol=0)


@pytest.mark.parametrize('kind', ['linear', 'mlp', 'norm'])
@torch.no_grad()
def test_mxfp8_eval_releases_workspaces_and_preserves_graph(kind):
    from flash_mxfp8.tk_mxfp8_linear import MXFP8Linear
    from flash_mxfp8.tk_mxfp8_swiglu_layer import MXFP8SwiGLU
    from flash_mxfp8 import FusedAddRMSNormMXFP8
    torch.manual_seed(871)
    x = torch.randn(256, 128, device='cuda', dtype=torch.bfloat16)
    residual = torch.randn_like(x)
    factor = torch.ones(128, device='cuda')
    if kind == 'linear':
        layer = MXFP8Linear(128, 128, device='cuda')
    elif kind == 'mlp':
        layer = MXFP8SwiGLU(128, 256, 128, device='cuda')
    else:
        layer = FusedAddRMSNormMXFP8(128).cuda()

    def call():
        out = layer(x, residual, factor) if kind == 'norm' else layer(x)
        return out[0] if kind == 'norm' else out

    expected = call().clone()
    for module in layer.modules():
        for name in ('_forward_states', '_forward_plans'):
            if hasattr(module, name):
                getattr(module, name).clear()
    layer.eval()
    first = call()
    torch.testing.assert_close(first, expected, atol=0, rtol=0)
    x.add_(.25)
    second = call()
    torch.testing.assert_close(first, expected, atol=0, rtol=0)
    assert first.data_ptr() != second.data_ptr()
    for module in layer.modules():
        assert not getattr(module, '_forward_states', {})
        assert not getattr(module, '_forward_plans', {})
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        call()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = call()
    for _ in range(2):
        x.add_(.125)
        graph.replay()
        torch.testing.assert_close(captured, call(), atol=0, rtol=0)
