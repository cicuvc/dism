"""Single-head FP64 dense smooth oracle: normalization and token-sum cancellation."""
import argparse,json
from pathlib import Path
import torch
from decompose_selector_vjp import compare,norm


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('dump');p.add_argument('--head',type=int,required=True);p.add_argument('--output',required=True)
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.28)
    torch.backends.cuda.matmul.allow_tf32=False
    d=torch.load(a.dump,map_location='cpu',weights_only=False);vals=d['inputs'][0];h=a.head;n=d['context']
    assert vals[8]==1.0
    dev=lambda x:x.to(device='cuda',dtype=torch.float64)
    q,k,sq,sk,v=[dev(x[0,:,h]) for x in vals[:5]]
    eq,ek=[dev(x[h].to(torch.bfloat16)) for x in vals[5:7]]
    tau=dev(vals[7][h]);pq=(q@eq.T).softmax(-1);pk=(k@ek.T).softmax(-1);emb=pk@eq
    lq=(q@eq.T).logsumexp(-1)
    scores=q@emb.T-lq[:,None]+tau
    # Same diagonal recurrence as dism_v3_ref, without retaining autograd graphs.
    for i in range(1,n):scores[i,1:]+=torch.nn.functional.softplus(scores[i-1,:-1])
    scores.masked_fill_(torch.ones((n,n),device='cuda',dtype=torch.bool).triu(1),-torch.inf)
    mx=scores.amax(-1,keepdim=True).clamp_min(0)
    prob=(scores-mx).exp();prob/=prob.sum(-1,keepdim=True)+(-mx).exp()
    readout=sq@sk.T
    output=(prob*readout)@v
    result=dict(layer=d['layer'],head=h,context=n,document=d['document'],precision='FP64 smooth oracle at BF16 input/codebook values; no intermediate BF16 rounding',
        output_vs_production=compare(dev(d['o_soft'][0,:,h]),output),paths={})
    def reverse(t):
        for i in range(n-1,0,-1):
            previous=scores[i-1,:-1]
            alpha=torch.where(previous>20,torch.ones_like(previous),previous.sigmoid())
            t[i-1,:-1]+=t[i,1:]*alpha
        return t
    def selector(t):
        dqdirect=t@emb;demb=t.T@q;dlq=-t.sum(-1)
        logq=dlq[:,None]*pq
        dq_lse=logq@eq
        logk=pk*(demb@eq.T-(demb*emb).sum(-1,keepdim=True))
        deq_direct=pk.T@demb;deq_lse=logq.T@q
        grads=dict(q=dqdirect+dq_lse,k=logk@ek,q_vocab=deq_direct+deq_lse,k_vocab=logk.T@k)
        # Per-token shared-codebook gradient matrices are sums of outer products.
        eq_energy=(pk.square().sum(-1)*demb.square().sum(-1)+logq.square().sum(-1)*q.square().sum(-1)
                   +2*(pk*logq).sum(-1)*(demb*q).sum(-1)).clamp_min(0)
        ek_norm=logk.norm(dim=-1)*k.norm(dim=-1)
        cancellation=dict(q_direct_lse=(norm(dqdirect)+norm(dq_lse))/max(norm(grads['q']),1e-300),
             q_vocab_direct_lse=(norm(deq_direct)+norm(deq_lse))/max(norm(grads['q_vocab']),1e-300),
             q_vocab_token_sum=float(eq_energy.sqrt().sum())/max(norm(grads['q_vocab']),1e-300),
             k_vocab_token_sum=float(ek_norm.sum())/max(norm(grads['k_vocab']),1e-300))
        return {k:v.cpu() for k,v in grads.items()},cancellation
    saved={}
    for name in ['soft','hard']:
        g=dev(d['g_'+name][0,:,h])*n
        tn=prob*readout*(g@v.T)
        td=-prob*(g*output).sum(-1,keepdim=True)
        local=dict(numerator_norm=norm(tn),denominator_norm=norm(td),total_norm=norm(tn+td),cosine=compare(tn,td)['cosine'])
        tn=reverse(tn);td=reverse(td)
        gn,_=selector(tn);gd,_=selector(td);gt,cancel=selector(tn+td)
        result['paths'][name]=dict(local=local,reverse=dict(numerator_norm=norm(tn),denominator_norm=norm(td),total_norm=norm(tn+td)),
            selector={k:dict(numerator_norm=norm(gn[k]),denominator_norm=norm(gd[k]),total_norm=norm(gt[k]),
                  cancellation=(norm(gn[k])+norm(gd[k]))/max(norm(gt[k]),1e-300),numerator_denominator_cosine=compare(gn[k],gd[k])['cosine']) for k in gt},
            token_cancellation=cancel)
        saved[name]=gt
        del tn,td,g
        print('DONE',name,flush=True)
    result['soft_hard']={k:compare(saved['soft'][k],saved['hard'][k]) for k in saved['soft']}
    production_path=out/'production_gradients.pt'
    if production_path.exists():
        prod=torch.load(production_path,weights_only=True)
        result['vs_production']={mode:{k:compare(prod[mode][k],saved[mode][k]) for k in saved[mode]} for mode in saved}
    (out/'dense.json').write_text(json.dumps(result,indent=2)+'\n')
    torch.save(saved,out/'dense_gradients.pt')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
