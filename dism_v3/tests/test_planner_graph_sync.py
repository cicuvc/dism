"""P1 protocol checks, deliberately independent of the DISM model/planner."""
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from check_planner_graph_sync import SyncProbe

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


@pytest.fixture
def probe():
    torch.set_num_threads(1)
    runtime = SyncProbe(capacity=4096, ctas=8, window_cycles=5_000_000)
    try:
        yield runtime
    finally:
        runtime.close()


@pytest.mark.parametrize('mode', ['ready', 'late_after_sample', 'partial_upload', 'stale'])
def test_controlled_publication_and_immutable_choice(probe, mode):
    # Publish a previous generation first. Then deliberately reuse the flag
    # without zeroing, alternate descriptor payload pointers and change sizes.
    probe.run_control('ready', 1, 4096)
    for epoch, count in enumerate((0, 1, 255, 256, 257, 4087, 4096), 2):
        probe.run_control(mode, epoch, count)


def test_full_replay_d2h_notifications_and_generation(probe):
    for epoch in range(1, 25):
        result = probe.run_async(epoch, (0, 257, 4096)[epoch % 3],
                                 after_graph=epoch % 4 == 0,
                                 split_upload=epoch % 2 == 0)
        assert result['input_matches']
        # Under instrumentation the CPU may always be late. Correct routing
        # and data are required; a latency-dependent route ratio is not.


def test_no_publisher_makes_progress_and_does_not_touch_null_descriptor(probe):
    probe.stage(1, 4096)
    probe.replay(probe.full)
    probe.check(0)


def test_cpu_failure_is_visible_and_gpu_still_finishes(probe):
    probe.stage(1, 257)
    probe.replay(probe.full)
    future = probe.pool.submit(probe.publish_from_snapshot, (2, 257, 19))
    with pytest.raises(AssertionError):
        future.result(timeout=30)
    probe.check(0)


def test_unsigned_stale_tag_does_not_select_current_plan(probe):
    with torch.cuda.stream(probe.upload):
        probe.writer.publish(probe.upload, probe.ready, 2**32-1)
    probe.upload.synchronize()
    probe.run_control('stale', 1, 4096)
