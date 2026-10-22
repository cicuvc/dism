"""P5 full core/vocabulary CUDA graphs, using actual forward saved tensors."""
import gc
import threading

import pytest
import torch
import cu_flash_dism as cu

from flash_dism.planner_graph import GraphPlannerRuntime, upload_plan
from flash_dism.hard_training import AsyncHardPlanner, DisabledHardPlanner
from flash_dism.varlen import VarlenLayout
from flash_dism.compiler import (_core_forward_impl, _core_backward_impl,
                                 _voc_forward_impl, _voc_backward_impl)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
ACTIVE = (0,1,2,3,4,5,6,11)


@pytest.fixture
def stream():
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        yield stream
    stream.synchronize()


def inputs(kind, packed, *, rows=2, n=512, heads=2, shared=False, d=32):
    torch.manual_seed(257)
    b, length = (1, rows*n) if packed else (rows,n)
    vectors = [(torch.randn(b,length,heads,width,device='cuda')*.2).bfloat16().requires_grad_()
               for width in (d,d,32,32,64)]
    hard = torch.ones(rows,heads,device='cuda',dtype=torch.bool)
    direction = torch.zeros(b,heads,device='cuda',dtype=torch.bool)  # explicit binding overrides to true
    tau = torch.tensor([.25,-.25],device='cuda')[:heads].requires_grad_()
    if kind == 'core':
        lse = [torch.full((b,length,heads),2.,device='cuda',requires_grad=True) for _ in range(2)]
        labels = [torch.randint(8,(b,heads,length),device='cuda',dtype=torch.int32) for _ in range(2)]
        raw = (*vectors,*lse,*labels,direction,hard,tau)
        active = [raw[i] for i in ACTIVE]
    else:
        vocab_shape = (64,d) if shared else (heads,64,d)
        vocab = [(torch.randn(vocab_shape,device='cuda')*.2).requires_grad_() for _ in range(2)]
        raw = (*vectors,*vocab,tau)
        active = list(raw)
    layout = None
    if packed:
        table = cu.make_graph_layout(torch.arange(rows+1,dtype=torch.int32)*n,rows*n,n,n)
        layout = VarlenLayout.for_graph(table,rows*n)
    return raw, active, direction, hard, layout


def invoke(binding, kind, raw, direction, hard, beta=1.):
    return binding.core(*raw) if kind == 'core' else binding.vocabulary(
        *raw,direction=direction,hard=hard,beta=beta)


@torch.no_grad()
def expected(kind, raw, direction, hard, layout, do, route, beta=1.):
    direction = torch.ones_like(direction)
    if kind == 'core':
        raw = (*raw[:9],direction,*raw[10:])
    planner = AsyncHardPlanner(workers=2,backward_policy='event') if route else DisabledHardPlanner(raw[0].device)
    try:
        table = None if layout is None else layout.table
        if kind == 'core':
            saved = _core_forward_impl(list(raw),table,0,False,True,planner=planner,direction_true=True)
        else:
            saved = _voc_forward_impl(*raw,beta,direction,hard,None,None if table is None else table.flatten(),
                                      True,direction_true=True,planner=planner)
        ticket = cu.get_training_plan(saved[-1])
        host = ticket.future.result(timeout=30).host if route else None
        if kind == 'core':
            grads = _core_backward_impl(list(raw),saved,table,0,False,do,direction_true=True)
        else:
            grads = _voc_backward_impl([*raw,beta,direction,hard],saved,None,do,
                None if table is None else table.flatten(),direction_true=True)
        return saved[0],grads,host
    finally:
        planner.close()


@pytest.mark.parametrize('d',[32,64])
@pytest.mark.parametrize('kind',['core','vocabulary'])
@pytest.mark.parametrize('packed',[False,True])
@pytest.mark.parametrize('mode',['soft','mixed','hard'])
def test_full_graph_changes_inputs_and_routes(kind,packed,mode,d,stream):
    raw,active,direction,hard,layout = inputs(kind,packed,d=d)
    beta = .7 if kind == 'vocabulary' else 1.
    with GraphPlannerRuntime(workers=2,max_slots=1,max_bytes=2**29) as runtime:
        slot = runtime.slot(rows=2,heads=2,row_tokens=512)
        binding = slot.bind(kind,layout=layout)
        do = torch.randn_like(raw[4])*.2
        def call():
            out = invoke(binding,kind,raw,direction,hard,beta)
            return out,torch.autograd.grad(out,active,do)
        call()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(graph,stream=stream):
            actual,grads = call()
        original = runtime.native.prepare
        gate = threading.Event()
        def delayed(*args,**kwargs):
            if not gate.wait(30):
                raise TimeoutError('test producer gate')
            return original(*args,**kwargs)
        runtime.native.prepare = delayed
        try:
            for step,route in enumerate((False,True,False,True)):
                with torch.no_grad():
                    for tensor in active:
                        tensor.add_(torch.randn_like(tensor)*.01)
                    hard.fill_(mode=='hard')
                    if mode=='mixed':
                        hard.copy_(torch.tensor([[step%2==0,False],[False,step%2!=0]],device='cuda'))
                    if kind=='core':
                        for label in raw[7:9]:
                            label.zero_() if step%2 else label.random_(8)
                if packed:
                    bounds = ([0,256,512,768,1024] if step%2 else [0,0,512,1024])
                    cu.update_graph_layout(layout.table,torch.tensor(bounds,dtype=torch.int32))
                out,want,host = expected(kind,raw,direction,hard,layout,do,route,beta)
                stream.synchronize()
                slot.arm()
                payload = None
                if route:
                    # Deterministic ready, using a real plan from this exact
                    # forward. The async worker is held until GPU consumption
                    # finishes, so it cannot replace a live prepublished plan.
                    payload = upload_plan(host,device=slot.device,stream=slot.upload,budget_bytes=slot.budget)
                    with torch.cuda.stream(slot.upload):
                        slot.descriptor.copy_(payload.descriptor,non_blocking=True)
                        runtime.writer.publish(slot.upload,slot.ready,slot.epoch)
                    slot.upload.synchronize()
                gate.clear()
                runtime.replay(graph)
                slot.finished.synchronize()
                assert slot.choice.item()==route
                gate.set()
                runtime.drain()
                torch.testing.assert_close(actual,out,atol=0,rtol=0)
                for a,b in zip(grads,want):
                    torch.testing.assert_close(a,b,atol=2e-4,rtol=2e-3)
                if mode=='hard':
                    zero = (0,1,5,6)
                    assert all(not grads[i].count_nonzero() for i in zero)
                del payload
        finally:
            gate.set()


@pytest.mark.parametrize('kind',['core','vocabulary'])
def test_two_layer_actual_async_graph(kind,stream):
    layers = [inputs(kind,False,rows=1,n=256) for _ in range(2)]
    with GraphPlannerRuntime(workers=4,max_slots=2,max_bytes=2**29) as runtime:
        slots = [runtime.slot(rows=1,heads=2,row_tokens=256) for _ in layers]
        bindings = [s.bind(kind) for s in slots]
        do = torch.randn_like(layers[0][0][4])*.2
        active = [p for layer in layers for p in layer[1]]
        def call():
            outputs = [invoke(bind,kind,raw,direction,hard) for bind,(raw,_,direction,hard,_) in zip(bindings,layers)]
            # Captured bounded work creates slack, never waits for a CPU flag.
            torch.cuda._sleep(5_000_000)
            loss_output = outputs[0]+outputs[1]
            return outputs,torch.autograd.grad(loss_output,active,do)
        call()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(graph,stream=stream):
            actual,grads = call()
        for step in range(6):
            for slot in slots:
                slot.arm()
            if step == 0:
                with pytest.raises(RuntimeError,match='every bound slot'):
                    runtime.replay(graph,slots=slots[:1])
            runtime.replay(graph)
            runtime.drain()
            for i,(raw,_,direction,hard,_) in enumerate(layers):
                out,want,_ = expected(kind,raw,direction,hard,None,do,bool(slots[i].choice.item()))
                torch.testing.assert_close(actual[i],out,atol=0,rtol=0)
                for a,b in zip(grads[i*8:(i+1)*8],want):
                    torch.testing.assert_close(a,b,atol=2e-4,rtol=2e-3)


@pytest.mark.parametrize('kind',['core','vocabulary'])
def test_chained_layers_propagate_gradients(kind,stream):
    layers = [inputs(kind,False,rows=1,n=256) for _ in range(2)]
    for _,_,_,hard,_ in layers:
        hard[0,1] = False
    with GraphPlannerRuntime(workers=4,max_slots=2,max_bytes=2**29) as runtime:
        slots = [runtime.slot(rows=1,heads=2,row_tokens=256) for _ in layers]
        bindings = [s.bind(kind) for s in slots]
        active = [*layers[0][1],*[p for i,p in enumerate(layers[1][1]) if i!=4]]
        do = torch.randn_like(layers[0][0][4])*.2
        def call():
            first,_,d1,h1,_ = layers[0]
            second,_,d2,h2,_ = layers[1]
            out1 = invoke(bindings[0],kind,first,d1,h1)
            chained = (*second[:4],out1,*second[5:])
            out2 = invoke(bindings[1],kind,chained,d2,h2)
            torch.cuda._sleep(5_000_000)
            return out2,torch.autograd.grad(out2,active,do)
        call()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(graph,stream=stream):
            actual,grads = call()
        for _ in range(3):
            for slot in slots:
                slot.arm()
            runtime.replay(graph)
            runtime.drain()
            first,_,d1,h1,_ = layers[0]
            second,_,d2,h2,_ = layers[1]
            routes = [bool(s.choice.item()) for s in slots]
            out1,_,_ = expected(kind,first,d1,h1,None,torch.zeros_like(do),routes[0])
            chained = (*second[:4],out1,*second[5:])
            want,g2,_ = expected(kind,chained,d2,h2,None,do,routes[1])
            _,g1,_ = expected(kind,first,d1,h1,None,g2[4],routes[0])
            torch.testing.assert_close(actual,want,atol=0,rtol=0)
            for a,b in zip(grads,[*g1,*[p for i,p in enumerate(g2) if i!=4]]):
                torch.testing.assert_close(a,b,atol=2e-4,rtol=2e-3)


@pytest.mark.parametrize('packed',[False,True])
@pytest.mark.parametrize('mode',['soft','mixed','hard'])
def test_core_fp64_oracle(mode,packed,stream):
    from gradient_acceptance import assert_gradient
    from test_hard_training import oracle
    raw,active,_,hard,layout = inputs('core',packed,rows=1,n=256)
    hard.fill_(mode=='hard')
    if mode=='mixed':
        hard[0,0] = True
    with GraphPlannerRuntime(workers=2,max_slots=1) as runtime:
        binding = runtime.slot(rows=1,heads=2,row_tokens=256).bind('core',layout=layout)
        actual = binding.core(*raw)
        do = torch.randn_like(actual)*.2
        grads = torch.autograd.grad(actual,active,do)
        refs = [t.detach().double().requires_grad_(t.requires_grad) if t.is_floating_point() else t.clone() for t in raw]
        refs[9].fill_(True)
        want = oracle(refs,layout)
        expected_grads = torch.autograd.grad(want,[refs[i] for i in ACTIVE],do.double(),allow_unused=True)
        torch.testing.assert_close(actual.float(),want.float(),atol=.008,rtol=.025)
        for name,a,b in zip(('q_vec','k_vec','sq_vec','sk_vec','v','q_lse','k_lse','rtau'),grads,expected_grads):
            assert_gradient(a,torch.zeros_like(a) if b is None else b,name)


def test_binding_error_channel_and_shared_vocabulary(stream):
    raw,active,direction,hard,_ = inputs('vocabulary',False,rows=1,n=256,shared=True)
    runtime = GraphPlannerRuntime(workers=2,max_slots=1)
    slot = runtime.slot(rows=1,heads=2,row_tokens=256)
    binding = slot.bind('vocabulary')
    do = torch.randn_like(raw[4])*.2
    def call():
        out = binding.vocabulary(*raw,direction=direction,hard=hard,beta=1.3)
        return out,torch.autograd.grad(out,active,do)
    call()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(graph,stream=stream):
        actual,grads = call()
    def fail(*args,**kwargs):
        raise ValueError('injected P5 planner failure')
    runtime.native.prepare = fail
    slot.arm()
    runtime.replay(graph)
    with pytest.raises(ValueError,match='injected P5'):
        runtime.drain()
    want,g,_ = expected('vocabulary',raw,direction,hard,None,do,False,1.3)
    torch.testing.assert_close(actual,want,atol=0,rtol=0)
    for a,b in zip(grads,g):
        torch.testing.assert_close(a,b,atol=2e-4,rtol=2e-3)
    with pytest.raises(RuntimeError,match='failed'):
        slot.arm()
    with pytest.raises(RuntimeError,match='failed'):
        runtime.close()


def test_handle_releases_after_abandoned_forward(stream):
    gc.collect()
    before = cu.training_plan_handle_count()
    raw,_,_,_,_ = inputs('core',False,rows=1,n=256)
    with GraphPlannerRuntime(workers=2,max_slots=1) as runtime:
        binding = runtime.slot(rows=1,heads=2,row_tokens=256).bind('core')
        out = binding.core(*raw)
        assert cu.training_plan_handle_count()==before+2
        runtime.drain()  # no replay Future: retire abandoned warmup D2H too
        del out,binding
        gc.collect()
        assert cu.training_plan_handle_count()==before


def test_binding_ownership_and_rejections(stream):
    raw,active,direction,hard,layout = inputs('core',True)
    with GraphPlannerRuntime(workers=2,max_slots=1) as runtime:
        slot = runtime.slot(rows=2,heads=2,row_tokens=512)
        binding = slot.bind('core',layout=layout)
        with pytest.raises(RuntimeError,match='fresh unbound'):
            slot.bind('core',layout=layout)
        with pytest.raises(RuntimeError,match='completed'):
            slot.arm()
        with pytest.raises(RuntimeError,match='gradient-enabled'):
            with torch.no_grad():
                binding.core(*raw)
        with pytest.raises(RuntimeError,match='invalid DISM plan handle'):
            cu.get_training_plan(binding.handle.clone())
        out = binding.core(*raw)
        with pytest.raises(RuntimeError,match='two forwards'):
            binding.core(*raw)
        torch.autograd.grad(out,active,torch.ones_like(out))
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(graph,stream=stream):
            out = binding.core(*raw)
            torch.autograd.grad(out,active,torch.ones_like(out))
        with pytest.raises(RuntimeError,match='already captured'):
            binding.core(*raw)
        slot.arm()
        other = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(other,stream=stream):
            torch.ones_like(raw[4])
        with pytest.raises(RuntimeError,match='captured graph'):
            runtime.replay(other)
        cu.update_graph_layout(layout.table,torch.tensor([0,256,512,768,1024],dtype=torch.int32))
        with pytest.raises(RuntimeError,match='changed after arm'):
            runtime.replay(graph)
        slot.drain()
        slot.arm()
        runtime.replay(graph)
        runtime.drain()
    del binding,out,graph
    gc.collect()


@pytest.mark.parametrize('mode',['soft','mixed','hard'])
def test_vocabulary_fp64_oracle(mode,stream):
    from gradient_acceptance import assert_gradient
    from flash_dism.reference.dism_v3_ref import dism_wrapper
    raw,active,direction,hard,_ = inputs('vocabulary',False,rows=1,n=256,shared=True)
    hard.fill_(mode=='hard')
    if mode=='mixed':
        hard[0,0] = True
    if mode!='soft':
        # BF16-representable codebooks keep the discrete argmax fixture equal
        # to the FP64 oracle. Arbitrary FP32 tables are quantized by existing
        # interpolation and can change labels; replay equivalence above covers
        # those separately, without changing this oracle or its tolerances.
        with torch.no_grad():
            for table in raw[5:7]:
                table.copy_(table.bfloat16().float())
    with GraphPlannerRuntime(workers=2,max_slots=1) as runtime:
        binding = runtime.slot(rows=1,heads=2,row_tokens=256).bind('vocabulary')
        actual = binding.vocabulary(*raw,direction=direction,hard=hard,beta=.7)
        do = torch.randn_like(actual)*.2
        grads = torch.autograd.grad(actual,active,do)
        refs = [t.detach().double().requires_grad_() for t in raw]
        q,k,sq,sk,v,eq,ek,tau = refs
        want = dism_wrapper(q,k,sq,sk,eq,ek,v,tau,beta=.7,
            direction=torch.ones_like(direction),hard=hard[:,:,None].expand(1,2,256))
        eg = torch.autograd.grad(want,refs,do.double(),allow_unused=True)
        torch.testing.assert_close(actual.float(),want.float(),atol=.008,rtol=.025)
        for name,a,b in zip(('q','k','sq_vec','sk_vec','v','eq','ek','rtau'),grads,eg):
            assert_gradient(a,torch.zeros_like(a) if b is None else b,name)


@pytest.mark.parametrize('case',['retained','compile','stream'])
def test_unsupported_binding_lifecycles(case,stream):
    raw,active,_,_,_ = inputs('core',False,rows=1,n=256)
    with GraphPlannerRuntime(workers=2,max_slots=1) as runtime:
        binding = runtime.slot(rows=1,heads=2,row_tokens=256).bind('core')
        if case=='compile':
            compiled = torch.compile(binding.core,backend='aot_eager',fullgraph=True)
            with pytest.raises(torch._dynamo.exc.Unsupported):
                compiled(*raw)
            assert not runtime.slots[0]._binding.pending
            return
        out = binding.core(*raw)
        do = torch.ones_like(out)
        torch.autograd.grad(out,active,do,retain_graph=True)
        if case=='retained':
            with pytest.raises(RuntimeError,match='must run once'):
                torch.autograd.grad(out,active,do)
        else:
            other = torch.cuda.Stream()
            other.wait_stream(stream)
            with torch.cuda.stream(other), pytest.raises(RuntimeError,match='owning stream'):
                binding.core(*raw)


@pytest.mark.parametrize('kind',['core','vocabulary'])
def test_leaf_backward_accumulation(kind,stream):
    raw,active,direction,hard,_ = inputs(kind,False,rows=1,n=256)
    hard[0,1] = False
    with GraphPlannerRuntime(workers=2,max_slots=1) as runtime:
        slot = runtime.slot(rows=1,heads=2,row_tokens=256)
        binding = slot.bind(kind)
        do = torch.randn_like(raw[4])*.2
        def call():
            out = invoke(binding,kind,raw,direction,hard)
            (out.float()*do.float()).sum().backward()
            return out
        call()
        for p in active:
            p.grad.zero_()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(graph,stream=stream):
            actual = call()
        totals = [torch.zeros_like(p) for p in active]
        for _ in range(3):
            slot.arm()
            runtime.replay(graph)
            runtime.drain()  # also retires leaf accumulation after layer finish
            out,grads,_ = expected(kind,raw,direction,hard,None,do,bool(slot.choice.item()))
            torch.testing.assert_close(actual,out,atol=0,rtol=0)
            for p,total,g in zip(active,totals,grads):
                total.add_(g)
                torch.testing.assert_close(p.grad,total,atol=2e-4,rtol=2e-3)
