"""Read-only paired layout probes on checkpoint-prefetched packed batches.

No optimizer or dataloader is advanced. All interventions affect temporary
in-memory models only. Results are conditional on the chosen trained weights.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'dism_v3/python'), str(ROOT/'nanochat')]
os.environ.setdefault('DISM_HARD_PLANNER', '0')
import torch
from nanochat.models import build_model, model_spec_from_metadata


def load(directory, step):
    meta = json.loads((directory/f'meta_{step:06d}.json').read_text())
    weights = torch.load(directory/f'model_{step:06d}.pt', map_location='cpu', weights_only=True)
    with torch.device('meta'):
        model = build_model(model_spec_from_metadata(meta))
    model.to_empty(device='cuda'); model.init_weights()
    model.load_state_dict({k.removeprefix('_orig_mod.'): v for k,v in weights.items()}, strict=True, assign=True)
    # assign=True with CPU tensors would move parameters back to CPU.
    model.cuda()
    from flash_mxfp8 import refresh_fp8_modules
    refresh_fp8_modules(model)
    return model, meta


def batches(directory, step, rows):
    state = torch.load(directory/f'train_state_{step:06d}_rank0.pt', map_location='cpu', weights_only=False)
    b = state['prefetched_batch']; length = b['inputs'].shape[1]
    result = []
    for start in range(0, b['inputs'].shape[0], rows):
        stop = min(start+rows, b['inputs'].shape[0]); lo,hi=start*length,stop*length
        cu=b['cu_seqlens']; cu=cu[(cu>=lo)&(cu<=hi)]
        assert cu[0]==lo and cu[-1]==hi
        result.append((b['inputs'][start:stop].cuda(),b['targets'][start:stop].cuda(),(cu-lo).cuda()))
    return result


class Probe:
    def __init__(self, model):
        self.model=model
        self.layers=[i for i,l in enumerate(model.layers) if not getattr(l.attn,'is_pure_mixer',False)]
        self.flags={}; self.scales={}; self.hidden={}; self.capture=False
        self.handles=[]; self.original=[]
        for ordinal,index in enumerate(self.layers):
            attn=model.layers[index].attn
            def flags(module,args,kwargs,j=ordinal):
                if j in self.flags: kwargs['hard']=self.flags[j]
                return args,kwargs
            self.handles.append(attn.register_forward_pre_hook(flags,with_kwargs=True))
            original=attn._combine_cuda
            def combine(output,*args,j=ordinal,fn=original,**kwargs):
                return fn(output*self.scales.get(j,1.),*args,**kwargs)
            self.original.append((attn,original));attn._combine_cuda=combine
        for index,layer in enumerate(model.layers):
            def hook(module,args,out,i=index):
                if self.capture:self.hidden[i]=out.detach().clone()
            self.handles.append(layer.register_forward_hook(hook))

    def mask(self, rows, probability, draw=0):
        for j in range(len(self.layers)):
            gen=torch.Generator(device='cuda').manual_seed(19231+draw*977+j*104729)
            self.flags[j]=torch.rand((rows,self.model.config.n_head),device='cuda',generator=gen)<probability

    @torch.no_grad()
    def evaluate(self,batch):
        x,y,cu=batch
        self.model.eval()
        nll=self.model(x,y,cu_seqlens=cu,loss_reduction='none').float().cpu()
        labels=y.flatten().cpu(); bounds=cu.cpu().tolist()
        valid=labels!=-1
        docs=[nll[a:b][valid[a:b]].mean().item() for a,b in zip(bounds[:-1],bounds[1:]) if valid[a:b].any()]
        return dict(token_nll=nll[valid].mean().item(),document_nll=sum(docs)/len(docs),documents=len(docs),valid_tokens=int(valid.sum()))

    def close(self):
        for h in self.handles:h.remove()
        for attn,fn in self.original:attn._combine_cuda=fn
        self.hidden.clear();self.flags.clear()


def group(name):
    if not name.startswith('layers.'):return None
    layer=int(name.split('.')[1]); suffix=name.split('.',2)[2]
    if suffix.endswith('q_vocab'):kind='q_vocab'
    elif suffix.endswith('k_vocab'):kind='k_vocab'
    elif suffix.endswith('log_sel_tau'):kind='tau'
    elif suffix in ('attn.q_conv.weight','attn.k_conv.weight'):kind='shared_qk'
    elif suffix in ('attn.q_conv.conv.weight','attn.k_conv.conv.weight'):kind='selector_conv'
    elif 'sq_proj' in suffix or 'sk_proj' in suffix:kind='readout'
    elif suffix.startswith('mlp.'):kind='mlp'
    else:kind='other'
    return f'{layer:02d}/{kind}'


def noise(probe,batch,probability,draws):
    model=probe.model;model.train();x,y,cu=batch
    params={}
    for n,p in model.named_parameters():
        k=group(n)
        if k:params.setdefault(k,[]).append(p)
    sums={}; squares={}; losses=[]
    for draw in range(draws):
        probe.mask(len(x),probability,draw)
        model.zero_grad(set_to_none=True)
        loss=model(x,y,cu_seqlens=cu,loss_reduction='mean');loss.backward()
        losses.append(loss.item())
        for k,ps in params.items():
            g=torch.cat([(p.grad.detach().float() if p.grad is not None else torch.zeros_like(p,dtype=torch.float32)).flatten() for p in ps])
            if k not in sums:sums[k]=torch.zeros_like(g);squares[k]=0.
            sums[k].add_(g);squares[k]+=g.square().sum().item()
        if (draw+1)%8==0:print(f'noise p={probability} draw={draw+1}/{draws}',flush=True)
    result={}
    for k,total in sums.items():
        mu2=total.square().sum().item()/draws**2
        var=max(0.,(squares[k]-draws*mu2)/(draws-1))
        corrected=mu2-var/draws
        result[k]=dict(mean_norm=mu2**.5,noise_norm=var**.5,r=var/max(mu2,1e-30),
                       corrected_mean_sq=corrected,r_corrected=var/corrected if corrected>0 else None)
    model.zero_grad(set_to_none=True)
    return dict(probability=probability,draws=draws,losses=losses,groups=result)


def backend(probe,batch):
    from nanochat.cuda_graph import PackedTrainingGraph
    model=probe.model; model.train();x,y,cu=batch
    probe.mask(len(x),.95,93)
    model.zero_grad(set_to_none=True)
    loss,count=model(x,y,cu_seqlens=cu,loss_reduction='document_sum');loss.backward()
    reference={n:p.grad.detach().float().clone() for n,p in model.named_parameters() if p.grad is not None}
    expected=loss.item(); model.zero_grad(set_to_none=True)
    del loss, count
    gc.collect()
    def compare():
        errors={}
        for n,p in model.named_parameters():
            if n not in reference:continue
            a=reference[n]; b=p.grad.detach().float(); k=group(n) or 'head_embedding'
            sums=errors.setdefault(k,[0.,0.,0.,0.])
            for i,v in enumerate(((a-b).square().sum(),a.square().sum(),b.square().sum(),(a*b).sum())):sums[i]+=v.item()
        return {k:dict(relative_l2=(e/max(a,1e-30))**.5,cosine=dot/max((a*b)**.5,1e-30)) for k,(e,a,b,dot) in errors.items()}
    result=dict(loss_eager=expected,eager_repeat=[],graphs=[])
    for _ in range(3):
        model.zero_grad(set_to_none=True)
        loss,count=model(x,y,cu_seqlens=cu,loss_reduction='document_sum');loss.backward()
        result['eager_repeat'].append(dict(loss=loss.item(),groups=compare()))
        del loss,count
    model.zero_grad(set_to_none=True);gc.collect()
    for planner in (False,True):
        graph=PackedTrainingGraph(model,x,y,cu,planner=planner,planner_workers=16)
        try:
            for repeat in range(3):
                model.zero_grad(set_to_none=True)
                got,count=graph.forward_backward(x,y,cu);graph.drain()
                result['graphs'].append(dict(planner=planner,repeat=repeat,loss=got.item(),routes=graph.last_routes,
                    reasons=graph.last_planner_reasons,groups=compare()))
                del got,count
        finally:graph.close()
        del graph
        model.zero_grad(set_to_none=True);gc.collect()
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint-dir',type=Path,required=True)
    p.add_argument('--batch-dir',type=Path,required=True)
    p.add_argument('--step',type=int,default=22888)
    p.add_argument('--rows',type=int,default=4)
    p.add_argument('--draws',type=int,default=32)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--backend-only',action='store_true')
    p.add_argument('--noise-only',action='store_true')
    a=p.parse_args();torch.set_num_threads(1);torch.manual_seed(71)
    if not a.output.resolve().is_relative_to(ROOT/'dism_v3/build'):
        p.error('output must be under dism_v3/build')
    if a.rows < 1 or a.draws < 2:
        p.error('rows must be positive and draws >= 2')
    if a.output.exists() and not (a.backend_only or a.noise_only):raise FileExistsError(a.output)
    model,meta=load(a.checkpoint_dir,a.step); data=batches(a.batch_dir,a.step,a.rows)
    probe=Probe(model)
    result=dict(checkpoint=str(a.checkpoint_dir),step=a.step,model_spec=meta['model_spec'],batches=[],noise=[],backend=None)
    def save():a.output.write_text(json.dumps(result,indent=2)+'\n')
    try:
        if a.backend_only:
            result=json.loads(a.output.read_text())
            assert result['checkpoint']==str(a.checkpoint_dir)
            result['backend']=backend(probe,data[0]);save()
            return
        if a.noise_only:
            result=json.loads(a.output.read_text())
            assert result['checkpoint']==str(a.checkpoint_dir)
            result['noise_128']=[]
            for bi,batch in enumerate(data):
                result['noise_128'].append(dict(batch=bi,**noise(probe,batch,.95,a.draws)));save()
            return
        for bi,batch in enumerate(data):
            probe.mask(len(batch[0]),1.);probe.capture=True
            hard=probe.evaluate(batch);base=probe.hidden;probe.hidden={};probe.capture=False
            probe.mask(len(batch[0]),0.);soft=probe.evaluate(batch)
            item=dict(batch=bi,hard=hard,soft=soft,switches=[],ablation=[])
            for j,index in enumerate(probe.layers):
                probe.mask(len(batch[0]),1.);probe.flags[j].fill_(False);probe.capture=True
                metric=probe.evaluate(batch);probe.capture=False
                propagation={}
                for k in range(index,len(model.layers)):
                    ref=base[k].float();delta=probe.hidden[k].float()-ref
                    propagation[k]=dict(relative_rms=(delta.square().sum()/ref.square().sum()).sqrt().item(),delta_rms=delta.square().mean().sqrt().item())
                item['switches'].append(dict(ordinal=j,layer=index,metric=metric,propagation=propagation))
                probe.hidden.clear();probe.mask(len(batch[0]),1.)
                for scale in (0.,.5,1.5):
                    probe.scales[j]=scale;metric=probe.evaluate(batch)
                    item['ablation'].append(dict(ordinal=j,layer=index,scale=scale,metric=metric))
                probe.scales.clear()
            result['batches'].append(item);save();del base
            print('forward probes complete batch',bi,flush=True)
        for bi,batch in enumerate(data):
            for probability in (.5,.95):
                result['noise'].append(dict(batch=bi,**noise(probe,batch,probability,a.draws)));save()
        result['backend']=backend(probe,data[0]);save()
        print('complete',a.output,flush=True)
    finally:probe.close()


if __name__=='__main__':main()
