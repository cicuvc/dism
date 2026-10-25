"""Small v4-only MAD MQAR screen using MAD's generator, backbone and SwiGLU.

Training targets include all next tokens (MAD protocol); test accuracy/CE use
only repeat-key query targets. This is a bounded screen, not the full MAD suite.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from flash_dism.module import DismAttention
from flash_dism_v4.module import V4DismGdnAttention
from nanochat.models.dism.conv import LinearShortConvSiLU


class V4Only(DismAttention):
    forward = V4DismGdnAttention.forward

    def __init__(self, dim, heads, codebook):
        super().__init__(dim, heads, head_dim=32, value_dim=32, readout_dim=16,
                         vocab_size=codebook, conv_size=4, layer_idx=0)
        del self.sq_proj, self.sk_proj
        self.block_size = 64
        self.readout_precision = 'fp32'
        # Library convolution supports odd lengths; no v3 256-token padding.
        # Keep initialized input linears; depthwise filters use the same
        # U[-1/sqrt(4),1/sqrt(4)] distribution as the original module.
        for name in ('q_conv','k_conv','v_conv'):
            old = getattr(self,name)
            conv = LinearShortConvSiLU(old.in_channels,old.out_channels,old.kernel_size)
            conv.weight = old.weight
            setattr(self,name,conv)


class MadV4(nn.Module):
    def __init__(self, dim, max_length, codebook=256, hard_prob=.5, soft_readout='arithmetic', **kwargs):
        super().__init__()
        self.attn = V4Only(dim,4,codebook)
        self.attn.soft_readout = soft_readout
        self.hard_prob = hard_prob
        self.eval_hard = True

    def forward(self,x):
        b,n,d = x.shape
        cu = torch.arange(b+1,device=x.device,dtype=torch.int32)*n
        segments = torch.arange(b,device=x.device,dtype=torch.int32)[:,None].expand(b,n).reshape(1,-1).contiguous()
        if self.training:
            hard = torch.rand(b,4,device=x.device) < self.hard_prob
        else:
            hard = torch.full((b,4),self.eval_hard,device=x.device,dtype=torch.bool)
        out = self.attn(x.reshape(1,b*n,d),cu_seqlens=cu,max_seqlen=n,hard=hard,
                        direction_true=True,conv_seq_idx=segments)[0]
        return out.reshape(b,n,d).to(x.dtype)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mad-root',default=str(Path.home()/'cs/project/mad-lab'))
    parser.add_argument('--vocab',type=int,default=16)
    parser.add_argument('--length',type=int,default=128)
    parser.add_argument('--train-examples',type=int,default=12800)
    parser.add_argument('--test-examples',type=int,default=1280)
    parser.add_argument('--epochs',type=int,default=20)
    parser.add_argument('--stop-after-epochs',type=int,default=None,
                        help='optional short-test budget; cosine horizon remains --epochs')
    parser.add_argument('--batch',type=int,default=128)
    parser.add_argument('--seed',type=int,default=12345)
    parser.add_argument('--codebook',type=int,default=256)
    parser.add_argument('--hard-prob',type=float,default=.5)
    parser.add_argument('--soft-readout',choices=['arithmetic','jensen'],default='arithmetic')
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    if not 0<=args.hard_prob<=1:raise ValueError('hard probability must be in [0,1]')
    if args.stop_after_epochs is not None and args.stop_after_epochs<=0:
        raise ValueError('stop-after-epochs must be positive')
    sys.path.insert(0,args.mad_root)
    from mad.data.instances import generate_in_context_recall_instance as generate
    from mad.model.language_model import LanguageModel
    from mad.model.layers.mlp import SwiGLU
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('highest')
    torch.manual_seed(args.seed)
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    (output/'config.json').write_text(json.dumps(vars(args),indent=2)+'\n')
    started=time.perf_counter()

    def dataset(train,count,seed):
        rng=np.random.default_rng(seed)
        pairs=[generate(vocab_size=args.vocab,seq_len=args.length,is_training=train,
                        rng=rng,multi_query=True,noise_vocab_size=0,frac_noise=0.) for _ in range(count)]
        a,b=(np.stack(x) for x in zip(*pairs))
        return torch.from_numpy(a).long(),torch.from_numpy(b).long()

    train_x,train_y=dataset(True,args.train_examples,args.seed)
    test_x,test_y=dataset(False,args.test_examples,args.seed+1)
    train_hashes={hashlib.sha256(x.numpy().tobytes()).digest() for x in train_x}
    assert all(hashlib.sha256(x.numpy().tobytes()).digest() not in train_hashes for x in test_x)
    data_hash=hashlib.sha256(b''.join(t.numpy().tobytes() for t in (train_x,train_y,test_x,test_y))).hexdigest()
    train_x,train_y,test_x,test_y=(t.cuda() for t in (train_x,train_y,test_x,test_y))
    cfg=dict(dim=128,max_length=args.length,codebook=args.codebook,hard_prob=args.hard_prob,
             soft_readout=args.soft_readout)
    model=LanguageModel(args.vocab,[MadV4,SwiGLU,MadV4,SwiGLU],[cfg.copy() for _ in range(4)],
                        dim=128,max_length=args.length,norm=nn.LayerNorm).cuda()
    mixers=[m for m in model.modules() if isinstance(m,MadV4)]
    optimizer=torch.optim.AdamW(model.parameters(),lr=5e-4,weight_decay=0.,fused=True)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.epochs,eta_min=1e-6)
    generator=torch.Generator().manual_seed(args.seed+2)

    @torch.no_grad()
    def evaluate():
        model.eval();result={}
        for hard in (False,True):
            for m in mixers:m.eval_hard=hard
            totals=torch.zeros(3,device='cuda')
            for start in range(0,args.test_examples,args.batch):
                x,y=test_x[start:start+args.batch],test_y[start:start+args.batch]
                with torch.autocast('cuda',dtype=torch.bfloat16):logits=model(x)
                loss=F.cross_entropy(logits.float().flatten(0,1),y.flatten(),ignore_index=-100,reduction='sum')
                valid=y!=-100
                totals+=torch.stack((loss,((logits.argmax(-1)==y)&valid).sum(),valid.sum()))
            loss,correct,count=totals.tolist()
            result['hard' if hard else 'soft']=dict(loss=loss/count,accuracy=correct/count,queries=int(count))
        return result

    # Check the packing and causal boundaries before interpreting recall scores.
    model.eval()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        small=test_x[:2];reference=model(small)
        perturbed=small.clone();perturbed[0]=(perturbed[0]+1)%args.vocab
        torch.testing.assert_close(model(perturbed)[1],reference[1],atol=0,rtol=0)
        changed=small.clone();changed[:,args.length//2:]=(changed[:,args.length//2:]+1)%args.vocab
        torch.testing.assert_close(model(changed)[:,:args.length//2],reference[:,:args.length//2],atol=0,rtol=0)
    meta=dict(parameters=sum(p.numel() for p in model.parameters()),data_sha256=data_hash,
              baseline_chance=2/args.vocab,hard_sampling='per sample/head',direction=True,
              model='MAD LayerNorm prenorm backbone; [v4-only, SwiGLU] x2; no GDN/SQ/SK',
              mad_generator_sha256=hashlib.sha256((Path(args.mad_root)/'mad/data/instances.py').read_bytes()).hexdigest())
    (output/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps(dict(metadata=meta,initial=evaluate())),flush=True)
    best={'soft':0.,'hard':0.};history=[]
    for epoch in range(1,args.epochs+1):
        model.train();permutation=torch.randperm(args.train_examples,generator=generator).cuda()
        losses=[];start=time.perf_counter()
        for ids in permutation.split(args.batch):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):logits=model(train_x[ids])
            loss=F.cross_entropy(logits.float().flatten(0,1),train_y[ids].flatten())
            loss.backward();optimizer.step()
            losses.append(loss.detach())
        train_loss=float(torch.stack(losses).mean());assert np.isfinite(train_loss)
        metrics=evaluate();scheduler.step()
        row=dict(epoch=epoch,train_loss=train_loss,eval=metrics,seconds=time.perf_counter()-start,
                 elapsed=time.perf_counter()-started,peak_bytes=torch.cuda.max_memory_allocated())
        print(json.dumps(row),flush=True)
        with (output/'history.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        history.append(row)
        for mode in best:
            if metrics[mode]['accuracy']>best[mode]:
                best[mode]=metrics[mode]['accuracy']
                torch.save(dict(model=model.state_dict(),args=vars(args),epoch=epoch,metrics=metrics),output/f'best_{mode}.pt')
        if args.stop_after_epochs is not None and epoch>=args.stop_after_epochs:
            break
    torch.save(dict(model=model.state_dict(),args=vars(args),epoch=history[-1]['epoch'],metrics=metrics),output/'last.pt')
    (output/'summary.json').write_text(json.dumps(dict(config=vars(args),metadata=meta,best_accuracy=best,final=history[-1]),indent=2)+'\n')
    print('complete',json.dumps(best),flush=True)


if __name__=='__main__':
    main()
