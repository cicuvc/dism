"""Compare the full production shape without keeping eager and graph pools together.

All reference tensors live in CPU RAM. Inputs, weights and optimizer states are
synthetic; no dataset or user checkpoint is accessed. Each replay is compared
at the reference step's identical master/quantized weights, copied into existing
storage. Both paths execute optimizer steps and preserve RNG/annealing state.
"""
import argparse
import gc
import json
import torch

from nanochat.models import ModelSpec, build_model, PretrainOptimizerConfig
from nanochat.cuda_graph import PackedTrainingGraph


def optimizer(model):
    return model.setup_pretraining_optimizer(PretrainOptimizerConfig(.0008,.0008,.0008,.0008,.01))


def cpu_state(model):
    return {name:value.detach().cpu().clone() for name,value in model.state_dict().items()}


def references(spec, rows, iterations, microbatches):
    torch.manual_seed(1741)
    model = build_model(spec).cuda().train()
    model.init_weights()
    initial = cpu_state(model)
    opt = optimizer(model)
    length,vocab = model.config.sequence_len,model.config.vocab_size
    records = []
    for index in range(iterations):
        step = int(index/max(1,iterations-1)*(model.config.anneal_steps-1))
        model.set_training_step(step)
        ids = torch.randint(vocab,(rows,length),device='cuda')
        labels = torch.randint(vocab,ids.shape,device='cuda')
        labels[:,::17] = -1
        bounds = [0]
        parts = ([256,length-256] if length>256 else [256]) if index%3==0 else (
            [256]*(length//256) if index%3==1 else [length])
        for _ in range(rows):
            for part in parts:
                bounds.append(bounds[-1]+part)
        cu = torch.tensor(bounds,dtype=torch.int32,device='cuda')
        model.zero_grad(set_to_none=True)
        weights = cpu_state(model)
        buffers = {n:b.detach().cpu().clone() for n,b in model.named_buffers()}
        loss,count = model(ids,labels,cu_seqlens=cu,loss_reduction='document_sum')
        (loss/microbatches).backward()
        records.append(dict(ids=ids.cpu(),labels=labels.cpu(),cu=cu.cpu(),step=step,
                            weights=weights,buffers=buffers,
                            loss=loss.item(),count=count.item(),counter=model.rng_counter.item(),
                            gradients={n:p.grad.cpu() for n,p in model.named_parameters() if p.grad is not None}))
        opt.step()
        print(json.dumps(dict(phase='eager',iteration=index,loss=loss.item(),
                             peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)),flush=True)
    final = {name:model.state_dict()[name].cpu().clone()
             for name in ('rng_counter','training_step','hard_probability')}
    # Cached MXFP8 output tensors can retain their autograd contexts/modules
    # through C++ references; release runtime plans before dropping the model.
    for module in model.modules():
        for name in ('_forward_states','_backward_states','_forward_plans','_backward_plans'):
            cache = getattr(module,name,None)
            if cache is not None:
                cache.clear()
    model.zero_grad(set_to_none=True)
    return initial,records,final


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-spec',required=True)
    parser.add_argument('--rows',type=int,default=8)
    parser.add_argument('--iterations',type=int,default=3)
    parser.add_argument('--microbatches',type=int,default=8)
    args = parser.parse_args()
    torch.set_num_threads(4)
    with open(args.model_spec) as source:
        spec = ModelSpec.from_dict(json.load(source))
    initial,records,final = references(spec,args.rows,args.iterations,args.microbatches)
    gc.collect()
    torch.cuda.empty_cache()
    print(json.dumps(dict(phase='reference_released',allocated_mib=torch.cuda.memory_allocated()/2**20)),flush=True)
    model = build_model(spec).cuda().train()
    model.load_state_dict(initial)
    opt = optimizer(model)
    first = records[0]
    graph = PackedTrainingGraph(model,first['ids'].cuda(),first['labels'].cuda(),first['cu'],
                                microbatches=args.microbatches)
    del initial
    for index,record in enumerate(records):
        model.zero_grad(set_to_none=True)
        # Compare each step at identical master AND quantized weights. Tiny
        # nondeterministic BF16 gradient differences otherwise produce diverging
        # AdamW trajectories, which is not a same-input graph/eager comparison.
        # Copy into existing storage: graph pointers and optimizer bindings stay
        # unchanged, and subsequent replay must read the updated contents.
        with torch.no_grad():
            for name,parameter in model.named_parameters():
                parameter.copy_(record['weights'][name])
            for name,buffer in model.named_buffers():
                buffer.copy_(record['buffers'][name])
        for module in model.modules():
            if callable(getattr(module,'mark_fp8_synced',None)):
                module.mark_fp8_synced()
        model.set_training_step(record['step'])
        loss,count = graph.forward_backward(record['ids'].cuda(),record['labels'].cuda(),record['cu'])
        torch.testing.assert_close(loss.cpu(),torch.tensor(record['loss']),atol=.005,rtol=.02)
        assert count.item()==record['count']
        assert model.rng_counter.item()==record['counter']
        strict=[]
        for name,parameter in model.named_parameters():
            assert (parameter.grad is not None)==(name in record['gradients']),name
            if parameter.grad is None:
                continue
            actual,expected=parameter.grad.cpu(),record['gradients'][name]
            try:
                torch.testing.assert_close(actual,expected,atol=.001,rtol=.03)
            except AssertionError:
                a,b=actual.float().flatten(),expected.float().flatten()
                strict.append(dict(parameter=name,max_error=(a-b).abs().max().item(),
                                   relative_l2=((a-b).norm()/b.norm().clamp_min(1e-30)).item(),
                                   cosine=torch.nn.functional.cosine_similarity(a,b,dim=0).item()))
        if strict:
            print(json.dumps(dict(strict_gradient_failures=strict)),flush=True)
            raise AssertionError('strict gradient diagnostic failed')
        opt.step()
        print(json.dumps(dict(phase='replay',iteration=index,loss=loss.item(),
                             peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)),flush=True)
    # Preserve the counter and annealing buffers exactly along the two paths.
    for name in ('rng_counter','training_step','hard_probability'):
        torch.testing.assert_close(model.state_dict()[name].cpu(),final[name],rtol=0,atol=0)


if __name__=='__main__':
    main()
