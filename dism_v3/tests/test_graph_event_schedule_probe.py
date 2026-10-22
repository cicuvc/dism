"""Scheduling-probe bounds, inactive branch and repeated graph replay checks."""
from pathlib import Path
import sys
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from graph_event_schedule_probe import ScheduleProbe
from flash_dism.inference.training_planner import NativeTrainingPlanner
from flash_dism.kernels.planner_graph import GraphCompactBackward
from flash_dism.planner_graph import upload_plan
from test_hard_forward import fixture
from test_planner_graph import compare

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


@pytest.mark.parametrize('vocabulary',[8,64,256,512])
def test_schedule_variants_and_inactive_branch(vocabulary):
    x=list(fixture(2048,'random',b=2,h=2))
    for labels in x[3:5]:
        labels.random_(0,vocabulary)
    do=torch.randn_like(x[2])*.2
    flat=[t.reshape(-1,t.shape[-1]) for t in (*x[:3],do)]
    planner=NativeTrainingPlanner(4)
    host=planner.prepare(torch.stack(x[3:5]).cpu(),x[-1].cpu(),
                         hard=x[-2].cpu(),compact=True,chunk_size=64)
    stream=torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        payload=upload_plan(host,device=x[0].device,stream=stream,budget_bytes=2**30)
        descriptor=payload.descriptor.to(x[0].device)
        choice=torch.ones(1,dtype=torch.int32,device=x[0].device)
        reference=GraphCompactBackward(descriptor,choice).backward(*flat,heads=2)
        for direct in (False,True):
            for grid,loop,skip_empty in ((False,True,False),(True,True,False),
                                         (True,False,False),(True,False,True)):
                probe=ScheduleProbe(payload,descriptor,choice,direct=direct,
                                    loop=loop,task_grid=grid,skip_empty=skip_empty)
                # Cover short tails in fixed grids, exact task grids and all
                # bucket widths/multi stages. Payload is immutable until retire.
                probe.backward(*flat,heads=2)
                stream.synchronize()
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph,stream=stream):
                    actual=probe.backward(*flat,heads=2)
                for selected in (1,0,1):
                    choice.fill_(selected)
                    graph.replay()
                    stream.synchronize()
                    if selected:
                        compare(actual,reference)  # existing 1e-4 / 1e-4 contract
                    else:
                        for output in actual.values():
                            assert torch.count_nonzero(output)==0
    stream.synchronize()
