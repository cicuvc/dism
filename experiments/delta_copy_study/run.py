"""Paired delta discretization study on rl/dism.py copy task; source unchanged."""
import os
os.environ['TORCHDYNAMO_DISABLE']='1'
import sys
sys.path.insert(0,'/home/cicuvc/cs/projects/rl')
import importlib.util
import json
import time
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('delta_copy_legacy',ROOT/'dism_snapshot.py')
legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
STEPS=1000
INIT_BIAS=float(os.environ.get('DELTA_INIT_BIAS','2'))
RESULTS=os.environ.get('DELTA_RESULTS','results-v2')
TASK=os.environ.get('DELTA_TASK','copy')
if TASK not in ('copy','abba'):raise ValueError('Unknown task')
legacy.TOTAL_STEPS=STEPS-1


def make_tokens(x,task=TASK):
    # A/B each32, drawn independently in the same64 random tokens as copy.
    if task=='copy':return torch.cat([x,x],1)
    a,b=x.chunk(2,dim=1)
    return torch.cat([a,b,b,a],1)


class Gate:
    def __init__(self,attn,method):
        self.attn=attn;self.method=method;self.step=0;self.mode='native'
        self.gen=torch.Generator(device='cuda').manual_seed(888)
        attn.delta_proj=nn.Linear(128,4).cuda()
        nn.init.normal_(attn.delta_proj.weight,std=.02)
        nn.init.constant_(attn.delta_proj.bias,INIT_BIAS)
        attn.register_forward_pre_hook(self.capture)
    def capture(self,module,args):
        self.b=module.delta_proj(args[0]).float().transpose(1,2)
    def temperature(self):return .1**min(self.step/(STEPS-1),1.)
    def delta(self):
        b=self.b
        hard=torch.where(b>=0,0.,torch.inf)
        if self.mode=='hard':return hard
        if self.mode=='unit_soft':return F.softplus(-b)
        soft=F.softplus(-b/(self.temperature() if self.method=='temperature' else 1.))
        if self.mode=='soft' or self.method=='temperature':return soft
        mask=(self.row_hard if self.method=='shared' else
              torch.rand(b.shape,device=b.device,generator=self.gen)<min(self.step/(STEPS-1),1.))
        return torch.where(mask,hard,soft)
    def recurrence(self,logm,v):
        delta=self.delta()
        n=logm.shape[-1];rows=[]
        previous=torch.full_like(logm[:,:,0,:],-torch.inf)
        for i in range(n):
            shifted=F.pad(previous[...,:-1],(1,0),value=-torch.inf)
            previous=logm[:,:,i,:]+F.softplus(shifted-delta[:,:,i,None])
            rows.append(previous)
        score=torch.stack(rows,-2).masked_fill(~torch.ones(n,n,device=v.device,dtype=torch.bool).tril(),-torch.inf)
        maximum=score.amax(-1,keepdim=True).clamp_min(0).detach()
        w=(score-maximum).exp();den=w.sum(-1,keepdim=True)+(-maximum).exp()
        return (w.to(v.dtype)@v)/den,w/den


@torch.no_grad()
def evaluate(model,gate):
    rows=[];attn=model[1].attn;saved_step=attn.step
    for score_mode,gate_mode in [('hard','hard'),('hard','soft'),('hard','unit_soft'),('mixed','native'),('mixed','hard')]:
        model.eval();attn.train(score_mode=='mixed');gate.mode=gate_mode
        gd=torch.Generator(device='cuda').manual_seed(424242)
        attn.gen=torch.Generator(device='cuda').manual_seed(1777);gate.gen.manual_seed(1888)
        loss=acc=tail=keep=uncertain=0.
        position_loss=torch.zeros(64,device='cuda');position_acc=torch.zeros_like(position_loss)
        position_keep=torch.zeros_like(position_loss)
        for _ in range(10):
            attn.step=saved_step
            x=torch.randint(0,127,(64,64),device='cuda',generator=gd);tokens=make_tokens(x)
            with torch.autocast('cuda',dtype=torch.bfloat16):y=model(tokens)
            pred=y[:,63:-1].float();target=tokens[:,64:]
            loss+=float(F.cross_entropy(pred.reshape(-1,128),target.flatten()))/10
            correct=pred.argmax(-1)==target
            position_loss+=F.cross_entropy(pred.reshape(-1,128),target.flatten(),reduction='none').reshape(64,64).mean(0)/10
            position_acc+=correct.float().mean(0)/10
            position_keep+=(gate.b[:,:,63:127]>=0).float().mean((0,1))/10
            acc+=float(correct.float().mean())/10;tail+=float(correct[:,8:].float().mean())/10
            p=torch.sigmoid(gate.b/(gate.temperature() if gate.method=='temperature' else 1.))
            keep+=float((gate.b>=0).float().mean())/10
            uncertain+=float(((p>.1)&(p<.9)).float().mean())/10
        rows.append(dict(score=score_mode,gate=gate_mode,loss=loss,accuracy=acc,after8_accuracy=tail,
                         keep_fraction=keep,uncertain_fraction=uncertain,
                         first_segment_accuracy=float(position_acc[:32].mean()),
                         second_segment_accuracy=float(position_acc[32:].mean()),
                         first_segment_after8=float(position_acc[8:32].mean()),
                         second_segment_after8=float(position_acc[40:].mean()),
                         position_loss=position_loss.tolist(),position_accuracy=position_acc.tolist(),
                         position_keep=position_keep.tolist()))
    attn.step=saved_step;model.train();gate.mode='native'
    return rows


def run(method,seed):
    out=ROOT/RESULTS/f'{method}-seed{seed}';out.mkdir(parents=True,exist_ok=False)
    (out/'config.json').write_text(json.dumps(dict(task=TASK,method=method,seed=seed,steps=STEPS,
        init_bias=INIT_BIAS,batch=64,length=128,data_seed=12345,eval_seed=424242),indent=2))
    torch.manual_seed(seed)
    model=nn.Sequential(nn.Embedding(128,128),legacy.DismTransformerBlock(128,mix='random'),
                        nn.RMSNorm(128),nn.Linear(128,128)).cuda()
    gate=Gate(model[1].attn,method);legacy._dism_from_logm=gate.recurrence
    legacy._row_hard_observer=lambda mask:setattr(gate,'row_hard',mask)
    no_decay=[p for p in model.parameters() if hasattr(p,'_no_weight_decay')]
    decay=[p for p in model.parameters() if not hasattr(p,'_no_weight_decay')]
    opt=torch.optim.AdamW([dict(params=no_decay,weight_decay=0.),dict(params=decay,weight_decay=.01)],lr=.005)
    sched=legacy._build_cosine_warmup_scheduler(opt,STEPS,50,.1)
    data=torch.Generator(device='cuda').manual_seed(12345)
    rng=torch.Generator(device='cuda').manual_seed(777);gate_rng=torch.Generator(device='cuda').manual_seed(888)
    start=time.time()
    for step in range(STEPS):
        gate.step=step;gate.gen=gate_rng;model[1].attn.gen=rng
        x=torch.randint(0,127,(64,64),device='cuda',generator=data);tokens=make_tokens(x)
        with torch.autocast('cuda',dtype=torch.bfloat16):y=model(tokens)
        pred=y[:,63:-1];target=tokens[:,64:]
        loss=F.cross_entropy(pred.reshape(-1,128),target.flatten())
        opt.zero_grad(set_to_none=True);loss.backward()
        norm=nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        opt.step();sched.step()
        if (step+1)%50==0:
            row=dict(step=step+1,loss=float(loss),grad_norm=float(norm),seconds=time.time()-start)
            with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            print(method,seed,row,flush=True)
        if step+1 in (250,500,750,1000):
            # Evaluation must not perturb any training RNG stream.
            state=gate_rng.get_state()
            values=evaluate(model,gate);gate_rng.set_state(state)
            with (out/'evaluation.jsonl').open('a') as f:f.write(json.dumps(dict(step=step+1,values=values))+'\n')
            print('EVAL',method,seed,step+1,[{k:v for k,v in row.items() if not k.startswith('position_')} for row in values],flush=True)
    torch.save(dict(model=model.state_dict(),method=method,seed=seed,steps=STEPS,task=TASK,init_bias=INIT_BIAS),out/'final.pt')


if __name__=='__main__':
    torch.set_num_threads(4)
    for seed in (0,1):
        for method in ('temperature','random','shared'):run(method,seed)
