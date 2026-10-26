"""P6/P7 model integration: real routes, failures, resume and bounded replay."""
import copy
import gc
import io
import threading
from types import SimpleNamespace

import pytest
import torch
import cu_flash_dism as cu

from nanochat.cuda_graph import PackedTrainingGraph
from nanochat.models import PretrainOptimizerConfig, optimizer_schema
from test_cuda_graph import model_pair

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def batch(rows=1, n=256):
    ids = torch.randint(128,(rows,n),device='cuda')
    targets = torch.randint(128,ids.shape,device='cuda')
    bounds = torch.arange(rows+1,device='cuda',dtype=torch.int32)*n
    return ids,targets,bounds


def test_middle_dism_layout_replay(monkeypatch):
    from nanochat.models import ModelSpec
    monkeypatch.setenv('DISM_HARD_PLANNER','0')
    layout='gggddddddggg'
    model,reference=model_pair(True,True,layers=12,layout=layout)
    actual=''.join('g' if getattr(l.attn,'is_pure_mixer',False) else 'd' for l in model.layers)
    assert actual==layout
    serialized=ModelSpec.from_dict(model.model_spec.to_dict())
    assert serialized.config['layer_layout']==layout
    legacy=ModelSpec(serialized.architecture,serialized.architecture_version,
                     {k:v for k,v in serialized.config.items() if k!='layer_layout'})
    assert legacy.fingerprint()!=serialized.fingerprint()
    model.set_training_step(5)
    reference.set_training_step(5)
    data=batch(rows=2)
    with PackedTrainingGraph(model,*data,planner=True,planner_workers=2,
                             microbatches=8,warmup=1) as graph:
        assert tuple(b is not None for b in graph.bindings)==tuple(c=='d' for c in layout)
        loss,count=graph.forward_backward(*data)
        graph.drain()
        expected,n=reference(*data[:2],cu_seqlens=data[2],loss_reduction='document_sum')
        (expected/8).backward()
        torch.testing.assert_close(loss,expected,atol=.005,rtol=.02)
        torch.testing.assert_close(count,n,atol=0,rtol=0)
        for (name,p),(_,q) in zip(model.named_parameters(),reference.named_parameters()):
            torch.testing.assert_close(p.grad,q.grad,atol=.001,rtol=.03,msg=name)
        assert model.rng_counter.item()==reference.rng_counter.item()==1
        assert sum(graph.route_counts.values())==6


@pytest.mark.parametrize('topk',[1,2,4])
def test_model_routes_and_snapshots(monkeypatch,topk):
    from flash_dism.planner_graph import upload_plan
    from flash_dism.hard_training import take_route_log
    monkeypatch.setenv('DISM_HARD_PLANNER','0')
    model,reference = model_pair(False,True,layers=4,hard_query_topk=topk)
    model.set_training_step(5)
    reference.set_training_step(5)
    data = batch()
    with PackedTrainingGraph(model,*data,planner=True,planner_workers=2,warmup=1) as graph:
        native = graph.runtime.native if topk==1 else graph.runtime.topk_native
        original = native.prepare
        launch = graph.runtime.replay
        gate = threading.Event()
        def delayed(*args,**kwargs):
            if not gate.wait(60):
                raise TimeoutError('test producer gate')
            return original(*args,**kwargs)
        native.prepare = delayed
        owners = []
        def ready_replay(cuda_graph):
            # Second invocation repeats exactly the first invocation's inputs,
            # weights and selector counter, so these are its real plans.
            for slot in graph.runtime.slots:
                labels,hard,tau = slot.hosts[:3]
                if topk==1:
                    host = original(labels,tau,hard=hard,documents=slot.documents,
                                    chunk_size=64,compact=True)
                    payload = upload_plan(host,device=slot.device,stream=slot.upload,budget_bytes=slot.budget)
                else:
                    from flash_dism.topk_planner import upload_topk
                    host = original(labels,tau,slot.hosts[3],hard,slot.documents,slot.budget//2)
                    payload = upload_topk(host,device=slot.device,stream=slot.upload,budget=slot.budget)
                owners.append(payload)
                with torch.cuda.stream(slot.upload):
                    slot.descriptor.copy_(payload.descriptor,non_blocking=True)
                    graph.runtime.writer.publish(slot.upload,slot.ready,slot.epoch)
                slot.upload.synchronize()
            launch(cuda_graph)
        try:
            take_route_log()
            outputs = []
            for ready in (False,True):
                gate.clear()
                model.rng_counter.zero_()
                model.zero_grad(set_to_none=True)
                graph.runtime.replay = ready_replay if ready else launch
                loss,count = graph.forward_backward(*data)
                graph.stream.synchronize()
                assert [s.choice.item() for s in graph.runtime.slots] == [int(ready)]*2
                gate.set()
                graph.drain()
                assert graph.last_routes == (('e','e') if ready else ('q','q'))
                assert take_route_log() == list(graph.last_routes)
                outputs.append((loss.clone(),count.clone(),[p.grad.clone() for p in model.parameters()]))
            want,count = reference(*data[:2],cu_seqlens=data[2],loss_reduction='document_sum')
            want.backward()
            for loss,n,grads in outputs:
                torch.testing.assert_close(loss,want,atol=.005,rtol=.02)
                torch.testing.assert_close(n,count,atol=0,rtol=0)
                for p,g in zip(reference.parameters(),grads):
                    torch.testing.assert_close(g,p.grad,atol=.001,rtol=.03)
            assert graph.route_counts == {'e':2,'q':2}
        finally:
            gate.set()


@pytest.mark.parametrize('failure',['planner','upload','submit'])
@pytest.mark.parametrize('topk',[1,2,4])
def test_model_errors_retire_before_update(monkeypatch,failure,topk):
    import flash_dism.planner_graph as pg
    monkeypatch.setenv('DISM_HARD_PLANNER','0')
    model,_ = model_pair(False,True,hard_query_topk=topk)
    data = batch()
    graph = PackedTrainingGraph(model,*data,planner=True,planner_workers=2,warmup=1)
    before = {n:p.detach().clone() for n,p in model.named_parameters()}
    def fail(*a,**kw):
        raise ValueError(f'injected {failure} failure')
    if failure=='planner':
        monkeypatch.setattr(graph.runtime.native if topk==1 else graph.runtime.topk_native,'prepare',fail)
    elif failure=='upload':
        if topk==1:monkeypatch.setattr(pg,'upload_plan',fail)
        else:
            import flash_dism.topk_planner as tp
            monkeypatch.setattr(tp,'upload_topk',fail)
    else:
        monkeypatch.setattr(graph.runtime.pool,'submit',fail)
    try:
        with pytest.raises((ValueError,RuntimeError)) as caught:
            graph.forward_backward(*data)
            graph.drain()
        error = caught.value
        while error.__cause__ is not None:
            error = error.__cause__
        assert f'injected {failure}' in str(error)
        assert all(s.state=='idle' and s.future is None and s.payload is None for s in graph.runtime.slots)
        assert graph.stream.query()
        assert sum(graph.route_counts.values()) == 0
        for n,p in model.named_parameters():
            torch.testing.assert_close(p,before[n],atol=0,rtol=0)
        with pytest.raises(RuntimeError,match='failed'):
            graph.forward_backward(*data)
    finally:
        with pytest.raises(RuntimeError,match='failed'):
            graph.close()
    assert graph.runtime.closed


@pytest.mark.parametrize('topk',[1,2,4])
def test_capture_failure_restores_state(monkeypatch,topk):
    from flash_dism.graph_binding import GraphPlannerBinding
    from flash_dism.planner_graph import GraphPlannerRuntime
    monkeypatch.setenv('DISM_HARD_PLANNER','0')
    model,_ = model_pair(False,True,hard_query_topk=topk)
    data = batch()
    model.set_training_step(4)
    model(*data[:2],cu_seqlens=data[2]).backward()
    state = copy.deepcopy(model.state_dict())
    grads = [p.grad for p in model.parameters()]
    rng,cuda_rng = torch.get_rng_state(),torch.cuda.get_rng_state()
    runtimes=[]
    original_init=GraphPlannerRuntime.__init__
    import flash_dism.topk_training as topk_training
    target=GraphPlannerBinding if topk==1 else topk_training
    original_call=target.vocabulary
    def remember(self,**kw):
        original_init(self,**kw)
        runtimes.append(self)
    def fail(self,*args,**kwargs):
        out=original_call(self,*args,**kwargs)
        if torch.cuda.is_current_stream_capturing():
            raise ValueError('injected capture failure after D2H')
        return out
    monkeypatch.setattr(GraphPlannerRuntime,'__init__',remember)
    monkeypatch.setattr(target,'vocabulary',fail)
    with pytest.raises(ValueError,match='injected capture'):
        PackedTrainingGraph(model,*data,planner=True,planner_workers=2,warmup=1)
    assert runtimes[0].closed
    for name,value in model.state_dict().items():
        torch.testing.assert_close(value,state[name],atol=0,rtol=0)
    assert all(p.grad is g for p,g in zip(model.parameters(),grads))
    assert torch.equal(rng,torch.get_rng_state())
    assert torch.equal(cuda_rng,torch.cuda.get_rng_state())


@pytest.mark.parametrize('reason',['budget','cancel'])
@pytest.mark.parametrize('topk',[1,2,4])
def test_model_resource_fallback(monkeypatch,reason,topk):
    monkeypatch.setenv('DISM_HARD_PLANNER','0')
    model,reference=model_pair(False,True,hard_query_topk=topk)
    model.set_training_step(9)
    reference.set_training_step(9)
    data=batch()
    with PackedTrainingGraph(model,*data,planner=True,planner_workers=2,warmup=1,
                             planner_max_bytes=(20_000+(2*4*256*topk*4+256 if topk>1 else 0)) if reason=='budget' else 2**30) as graph:
        if reason=='cancel':
            launch=graph.runtime.replay
            def cancel_before_launch(cuda_graph):
                for slot in graph.runtime.slots:
                    slot.cancel()
                launch(cuda_graph)
            graph.runtime.replay=cancel_before_launch
        loss,count=graph.forward_backward(*data)
        graph.drain()
        assert graph.last_routes==('q',)
        assert all(graph.last_planner_reasons)
        if reason=='cancel':
            assert graph.last_planner_reasons==('cancelled',)
        expected,n=reference(*data[:2],cu_seqlens=data[2],loss_reduction='document_sum')
        expected.backward()
        torch.testing.assert_close(loss,expected,atol=.005,rtol=.02)
        torch.testing.assert_close(count,n,atol=0,rtol=0)
        for p,q in zip(model.parameters(),reference.parameters()):
            torch.testing.assert_close(p.grad,q.grad,atol=.001,rtol=.03)


@pytest.mark.parametrize('topk',[1,2,4])
def test_bounded_model_replay(monkeypatch,topk):
    monkeypatch.setenv('DISM_HARD_PLANNER','0')
    model,_ = model_pair(False,True,hard_query_topk=topk)
    data = batch(rows=2,n=512)
    table_storage = torch.empty(6,device='cuda',dtype=torch.int32)
    gc.collect()
    handles = cu.training_plan_handle_count()
    with PackedTrainingGraph(model,*data,planner=True,planner_workers=2,warmup=1) as graph:
        bounds = ([0,512,1024],[0,256,512,768,1024],[0,0,512,512,1024])
        allocated=[]
        for i in range(36):
            model.set_training_step((0,5,9)[i%3])
            model.zero_grad(set_to_none=i%2==0)
            data[0].random_(128)
            data[1].random_(128)
            b=table_storage[:len(bounds[i%3])]
            b.copy_(torch.tensor(bounds[i%3],device='cuda',dtype=torch.int32))
            loss,count=graph.forward_backward(data[0],data[1],b)
            # Next invocation must drain itself as well; explicit drain at the
            # optimizer boundary remains mandatory for the final microbatch.
            if i%2:
                graph.drain()
            assert torch.isfinite(loss).item() and count.item()>0
            if i>=6 and i%2:
                allocated.append(torch.cuda.memory_allocated())
        graph.drain()
        assert model.rng_counter.item()==36
        assert sum(graph.route_counts.values())==36
        assert max(allocated)-min(allocated) < 16*2**20
        assert cu.training_plan_handle_count()==handles+1
    graph.close()
    with pytest.raises(RuntimeError,match='closed'):
        graph.forward_backward(*data)
    del graph
    gc.collect()
    assert cu.training_plan_handle_count()==handles


@pytest.mark.parametrize('case',['policy','compact','shape','opt_in','rank'])
def test_unsupported_planner_graph_options(monkeypatch,case):
    monkeypatch.setenv('DISM_HARD_PLANNER','0')
    model,_=model_pair(False,case!='shape')
    if case=='policy': monkeypatch.setenv('DISM_HARD_BACKWARD','event')
    if case=='compact': monkeypatch.setenv('DISM_PLANNER_COMPACT','0')
    if case=='opt_in': monkeypatch.setenv('DISM_HARD_PLANNER','1')
    if case=='rank':
        monkeypatch.setattr(torch.distributed,'is_initialized',lambda: True)
        monkeypatch.setattr(torch.distributed,'get_world_size',lambda: 2)
    with pytest.raises(ValueError):
        PackedTrainingGraph(model,*batch(),planner=case!='opt_in',warmup=1)


@pytest.mark.parametrize('case',['parameter','buffer'])
def test_rebound_storage_rejected(monkeypatch,case):
    monkeypatch.setenv('DISM_HARD_PLANNER','0')
    model,_=model_pair(False,True)
    data=batch()
    with PackedTrainingGraph(model,*data,planner=True,planner_workers=2,warmup=1) as graph:
        if case=='parameter':
            model.embedding.weight=torch.nn.Parameter(model.embedding.weight.detach().clone())
        else:
            model.rng_counter=model.rng_counter.clone()
        with pytest.raises(ValueError,match='storage changed'):
            graph.forward_backward(*data)


@pytest.mark.parametrize('topk',[1,2,4])
def test_serialized_resume_with_prefetched_batch(monkeypatch,topk):
    from contextlib import ExitStack
    from nanochat.data_prefetch import StatefulPrefetchLoader
    # Both graphs explicitly opt in, including when eager planner mode is set.
    monkeypatch.setenv('DISM_HARD_PLANNER','1')

    class Source:
        def __init__(self,cursor=0):
            self.cursor=cursor
        def __next__(self):
            x=(torch.arange(256).reshape(1,256)+self.cursor)%128
            self.cursor+=1
            return x,(x+1)%128,torch.tensor([0,256],dtype=torch.int32),torch.zeros_like(x),self.state_dict()
        def state_dict(self):
            return {'cursor':self.cursor}
        progress_state=state_dict

    def loader(state=None):
        tokenizer=SimpleNamespace(pool=SimpleNamespace(shutdown=lambda **kw: None))
        return StatefulPrefetchLoader(Source(state['source']['cursor'] if state else 0),
                                      tokenizer,'cuda',state=state,depth=3)

    def train(model,graph,opt,data,step):
        model.set_training_step(step)
        model.zero_grad(set_to_none=True)
        loss,_=graph.forward_backward(*data[:3])
        graph.drain()
        result=loss.clone()
        opt.step()
        model.zero_grad(set_to_none=True)
        return result

    config=PretrainOptimizerConfig(.001,.001,.001,.001,.01)
    model,restored=model_pair(True,True,hard_query_topk=topk)
    with ExitStack() as stack:
        original_loader=loader()
        stack.callback(original_loader.close)
        data=next(original_loader)
        graph=stack.enter_context(PackedTrainingGraph(model,*data[:3],planner=True,planner_workers=2,warmup=1))
        opt=model.setup_pretraining_optimizer(config)
        train(model,graph,opt,data,4)
        # Same checkpoint boundary as base_train: the next batch is already
        # consumed by the loader but has not yet been used for training.
        data=next(original_loader)
        serialized=io.BytesIO()
        torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),
                        schema=optimizer_schema(model,opt),loader=original_loader.state_dict(),
                        next_batch=tuple(t.cpu() for t in data[:4]),progress=data[4],
                        rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state()),serialized)
        serialized.seek(0)
        saved=torch.load(serialized,map_location='cpu',weights_only=False)
        assert saved['loader']['progress']==saved['progress']=={'cursor':2}
        restored.load_state_dict(saved['model'])
        restored_opt=restored.setup_pretraining_optimizer(config)
        restored_opt.load_state_dict(saved['optimizer'])
        assert optimizer_schema(restored,restored_opt)==saved['schema']
        restored_loader=loader(saved['loader'])
        stack.callback(restored_loader.close)
        restored_data=(*tuple(t.cuda() for t in saved['next_batch']),saved['progress'])
        torch.set_rng_state(saved['rng'])
        torch.cuda.set_rng_state(saved['cuda_rng'])
        restored_graph=stack.enter_context(PackedTrainingGraph(restored,*restored_data[:3],
                                         planner=True,planner_workers=2,warmup=1))
        assert torch.equal(torch.get_rng_state(),saved['rng'])
        assert torch.equal(torch.cuda.get_rng_state(),saved['cuda_rng'])
        assert {n for n,p in restored.named_parameters() if getattr(p,'_no_weight_decay',False)} == {
            n for n,p in model.named_parameters() if getattr(p,'_no_weight_decay',False)}
        for step in (5,9):
            for x,y in zip(data[:4],restored_data[:4]):
                torch.testing.assert_close(x,y,atol=0,rtol=0)
            assert data[4]==restored_data[4]
            actual=train(restored,restored_graph,restored_opt,restored_data,step)
            expected=train(model,graph,opt,data,step)
            torch.testing.assert_close(actual,expected,atol=.005,rtol=.02)
            for name in ('rng_counter','training_step','hard_probability'):
                torch.testing.assert_close(getattr(model,name),getattr(restored,name),atol=0,rtol=0)
            for a,b in zip(opt.state.values(),restored_opt.state.values()):
                assert a.keys()==b.keys() and a['step']==b['step']
                assert a['exp_avg'].dtype==b['exp_avg'].dtype==torch.bfloat16
            data,restored_data=next(original_loader),next(restored_loader)
            assert original_loader.progress_state()==restored_loader.progress_state()
