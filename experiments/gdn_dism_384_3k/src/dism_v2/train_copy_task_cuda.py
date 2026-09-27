"""Run copy_task.py's unchanged model/task using the current Dism autograd.

python -m dism_v2.train_copy_task_cuda --steps 1000 --seed 0
The reference script stays untouched. No W&B/network logging or checkpoint writes.
BF16 compute with FP32 master parameters, original scale=1, no rtau clamp.
"""
import argparse
import json
import math
from pathlib import Path
import random
import sys
import time
import torch
from .autograd import voc_dism
from .kernel_config import TILE_LSE


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--steps',type=int,default=1000)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--data-seed',type=int,default=12345)
    parser.add_argument('--layers',type=int,default=1)
    parser.add_argument('--batch',type=int,default=64)
    parser.add_argument('--seq-len',type=int,default=128)
    parser.add_argument('--head-dim',type=int,choices=(32,64,128),default=32)
    parser.add_argument('--qk-vocab',type=int,default=256)
    parser.add_argument('--embedding-backward',choices=['cuda','cuda_symmetric','triton'],default='cuda')
    parser.add_argument('--log-every',type=int,default=25)
    opts=parser.parse_args()
    if min(opts.steps,opts.layers,opts.log_every)<=0: parser.error('positive steps/layers/log-every required')
    if min(opts.batch,opts.qk_vocab)<=0 or opts.seq_len<2 or opts.seq_len%2:
        parser.error('positive batch/vocab and positive even seq-len required')
    # copy_task uses a script-local absolute emb_kernel import.
    sys.path.insert(0,str(Path(__file__).resolve().parent))
    try:
        from . import copy_task as task
    finally:
        sys.path.pop(0)
    calls={'train':0,'hard_eval':0}
    def cuda_dism(q,k,v,rtau,q_voc,k_voc,hard=False,lmb=.5,gen=None,sm_scale=1.):
        calls['hard_eval' if hard else 'train']+=1
        return voc_dism(*(x.to(torch.bfloat16).contiguous() for x in (q,k,v)),
            rtau.float().contiguous(),q_voc.bfloat16().contiguous(),k_voc.bfloat16().contiguous(),
            sm_scale=sm_scale,direction='q_from_k' if hard else 'random',
            hard_prob=1. if hard else lmb,generator=gen,
            embedding_backend='cuda',embedding_backward_backend=opts.embedding_backward)
    task.voc_dism=cuda_dism
    task.TOTAL_STEPS=opts.steps
    cls=task.DismMHAttentionV3
    task.DismMHAttentionV3=lambda dm,heads,vocab,hd,**kw:cls(dm,heads,opts.qk_vocab,hd,**kw)
    torch.set_default_device('cuda')
    torch.manual_seed(opts.seed)
    random.seed(opts.seed)
    b,n,voc,d_model=opts.batch,opts.seq_len,128,4*opts.head_dim
    token_embedding=torch.nn.Embedding(voc,d_model)
    blocks=[task.DismTransformerBlock(d_model) for _ in range(opts.layers)]
    model=torch.nn.Sequential(token_embedding,*blocks,torch.nn.RMSNorm(d_model),torch.nn.Linear(d_model,voc))
    g_data=torch.Generator(device='cuda').manual_seed(opts.data_seed)
    g_drop=torch.Generator(device='cuda').manual_seed(777)
    for block in blocks: block.attn.gen=g_drop
    optim=torch.optim.AdamW([
        {'params':[p for p in model.parameters() if hasattr(p,'_no_weight_decay')],'weight_decay':0.},
        {'params':[p for p in model.parameters() if not hasattr(p,'_no_weight_decay')],'weight_decay':1e-2},
    ],lr=5e-3)
    sched=task._build_cosine_warmup_scheduler(optim,opts.steps,50,.1)
    def emit(kind,**kw): print(json.dumps(dict(kind=kind,**kw)),flush=True)
    def batch(gen):
        x=torch.randint(0,voc-1,(b,n//2),generator=gen)
        return torch.cat((x,x),-1)
    def prediction(tokens):
        with torch.autocast('cuda',torch.bfloat16):
            y=task.softcap_logits(model(tokens))
        logits=y[:,n//2-1:-1,:]
        labels=tokens[:,n//2:]
        loss=torch.nn.functional.cross_entropy(logits.reshape(-1,voc),labels.flatten())
        return loss,logits.argmax(-1)==labels
    def evaluate(step):
        model.eval()
        gen=torch.Generator(device='cuda').manual_seed(task.EVAL_SEED)
        losses=[];accuracies=[];exact=[];positions=[]
        with torch.no_grad():
            for _ in range(10):
                loss,correct=prediction(batch(gen))
                losses.append(float(loss));accuracies.append(float(correct.float().mean()))
                exact.append(float(correct.all(-1).float().mean()))
                positions.append(correct.float().mean(0))
        model.train()
        metrics=dict(step=step,loss=sum(losses)/10,token_accuracy=sum(accuracies)/10,
                     sequence_accuracy=sum(exact)/10,
                     accuracy_by_position=torch.stack(positions).mean(0).tolist())
        emit('eval_hard',**metrics)
        return metrics
    emit('config',**vars(opts),tile_lse=TILE_LSE,gpu=torch.cuda.get_device_name(),n=n,heads=4,d=opts.head_dim,dv=opts.head_dim,
         vocab=opts.qk_vocab,token_vocab=voc,sm_scale=1.,rtau_clamped=False,
         parameters=sum(p.numel() for p in model.parameters()))
    start=time.perf_counter()
    initial=evaluate(-1)
    losses=[];max_grad=0.;evals=[];max_tau=0.
    for step in range(opts.steps):
        loss,correct=prediction(batch(g_data))
        optim.zero_grad(set_to_none=True)
        loss.backward()
        gnorm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        value=float(loss.detach());norm=float(gnorm)
        if not math.isfinite(value): raise RuntimeError(f'nonfinite loss at step {step}')
        taus=[torch.nn.functional.softplus(block.attn.log_sel_tau.detach()).tolist() for block in blocks]
        tau_grads=[block.attn.log_sel_tau.grad.detach().tolist() for block in blocks]
        max_tau=max(max_tau,max(max(t) for t in taus));max_grad=max(max_grad,norm)
        losses.append(value)
        optim.step();sched.step()
        if step%opts.log_every==0 or step==opts.steps-1:
            emit('train',step=step,loss=value,token_accuracy=float(correct.float().mean()),
                 grad_norm=norm,lr=optim.param_groups[0]['lr'],hard_prob=step/opts.steps,
                 rtau=taus,log_sel_tau_grad=tau_grads,elapsed_s=time.perf_counter()-start)
        if step%100==0 or step==opts.steps-1: evals.append(evaluate(step))
    torch.cuda.synchronize()
    emit('result',initial_eval=initial,final_eval=evals[-1],last_100_loss=sum(losses[-100:])/len(losses[-100:]),
         max_grad_norm=max_grad,max_rtau=max_tau,rtau_target_bound=math.log(opts.head_dim),
         calls=calls,elapsed_s=time.perf_counter()-start,
         peak_allocated_bytes=torch.cuda.max_memory_allocated())


if __name__=='__main__': main()
