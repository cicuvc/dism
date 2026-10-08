"""Trained checkpoint: parallel/Triton prefill -> prime -> GPU-planned decode.

Opt-in harness. Checkpoint/model production code remain unchanged. Full-hard
core, FP32 model projections/conv/SWA/MLP, BF16 default core GEMMs/cache payload.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path
import torch
from torch.nn import functional as F
from checkpoint_decode_smoke import (Runner, sample, ROOT, NANO, build_model,
                                     ModelSpec, HuggingFaceTokenizer)
from dism_v4.decoding import HardDismPrefill


def metrics(actual,expected):
    x,y=actual.double().flatten(),expected.double().flatten()
    return dict(max_abs=float((x-y).abs().max()),relative_l2=float((x-y).norm()/y.norm()),
                cosine=float(F.cosine_similarity(x,y,dim=0)),
                top1_agreement=float((actual.argmax(-1)==expected.argmax(-1)).float().mean()),
                kl_mean=float((expected.softmax(-1)*(expected.log_softmax(-1)-actual.log_softmax(-1))).sum(-1).mean()))


def main():
    parser=argparse.ArgumentParser()
    base=Path('/media/cicuvc/c63abdf1-0e56-4153-9228-95df5a2f239b/cicuvc/dism-runs')
    parser.add_argument('--checkpoint',type=Path,default=base/'nanochat-hybrid125m-2500m/base_checkpoints/dism125m/model_019074.pt')
    parser.add_argument('--prefixes',type=int,nargs='+',default=[256,1024])
    parser.add_argument('--steps',type=int,default=64)
    parser.add_argument('--new-tokens',type=int,default=64)
    parser.add_argument('--output',type=Path,default=ROOT/'dism_v4/decoding/results/checkpoint_prefill_decode.json')
    args=parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    meta=json.loads(args.checkpoint.with_name(args.checkpoint.name.replace('model_','meta_')).with_suffix('.json').read_text())
    with torch.device('meta'): model=build_model(ModelSpec.from_dict(meta['model_spec']))
    model.to_empty(device='cpu')
    model.load_state_dict(torch.load(args.checkpoint,map_location='cpu',weights_only=True),strict=True)
    model.cuda().eval()
    tokenizer=HuggingFaceTokenizer.from_pretrained(str(NANO/'runs/dism72m_3k/tokenizer'))
    old_identity='transformers.models.llama.tokenization_llama_fast\0LlamaTokenizerFast\0'
    fingerprint=hashlib.sha256((old_identity+tokenizer.tokenizer.backend_tokenizer.to_str()).encode()).hexdigest()
    assert meta['tokenizer_spec']['fingerprint'] in (tokenizer.get_fingerprint(),fingerprint)
    assert tokenizer.get_bos_token_id()==1 and tokenizer.tokenizer.eos_token_id==2
    prompts=[
        'The solar system consists of the Sun and the objects that orbit it. The largest planet is',
        'Photosynthesis is the process by which plants convert sunlight into chemical energy. During this process,',
        'Once upon a time, a young girl found a mysterious book in the library. When she opened it,',
    ]
    def encode(text):return torch.tensor([[1]+tokenizer.encode(text)],device='cuda',dtype=torch.long)
    report=dict(checkpoint=str(args.checkpoint),step=meta['step'],model_config=meta['model_config'],
                protocol='full-hard; exact Torch reference vs 8-worker CPU/Triton prefill + native prime + GPU-planned decoding; FP32 outer model math',
                parity=[],generations=[])
    with args.checkpoint.open('rb') as f:report['checkpoint_sha256']=hashlib.file_digest(f,'sha256').hexdigest()
    def save():
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2)+'\n')
    text=(' '.join(prompts)+' ')*(1+(max(args.prefixes)+args.steps)//40)
    forced=encode(text)
    assert forced.shape[1]>max(args.prefixes)+args.steps
    with torch.inference_mode():
        for precision,payload in (('tf32x3',torch.float32),('bf16',torch.bfloat16)):
            with HardDismPrefill(mma_precision=precision) as engine:
                for n in args.prefixes:
                    prefix=forced[:,:n]
                    ref=Runner(model,n+args.steps+1,'reference')
                    native=Runner(model,n+args.steps+1,dtype=payload,planner='gpu',prefill_engine=engine)
                    expected=ref(prefix)
                    torch.cuda.synchronize();start=time.perf_counter()
                    actual=native(prefix);torch.cuda.synchronize()
                    prefill_ms=(time.perf_counter()-start)*1000
                    assert torch.isfinite(actual).all()
                    pre=metrics(actual,expected)
                    target=forced[:,1:n+1]
                    pre['reference_nll']=float(F.cross_entropy(expected.flatten(0,1),target.flatten()))
                    pre['native_nll']=float(F.cross_entropy(actual.flatten(0,1),target.flatten()))
                    pre['last_position']=metrics(actual[:,-1],expected[:,-1])
                    del actual,expected
                    steps=[]
                    for t in range(n,n+args.steps):
                        token=forced[:,t:t+1]
                        expected=ref(token)[:,-1];actual=native(token)[:,-1]
                        assert torch.isfinite(actual).all()
                        steps.append(metrics(actual,expected))
                    entry=dict(precision=precision,payload=str(payload),n=n,prefill=pre,
                               cold_or_warm_prefill_ms=prefill_ms,decode_steps=steps,
                               decode_top1_agreement=sum(x['top1_agreement'] for x in steps)/len(steps),
                               decode_relative_l2_max=max(x['relative_l2'] for x in steps),
                               decode_cosine_min=min(x['cosine'] for x in steps),
                               cache_positions=[s['cache'].core.check_status() for s in native.states])
                    report['parity'].append(entry);save()
                    print('PARITY',json.dumps({k:v for k,v in entry.items() if k not in ('decode_steps','cache_positions')}),flush=True)
                    if precision=='tf32x3':
                        assert pre['relative_l2']<1e-3 and pre['cosine']>.99999
                        assert entry['decode_relative_l2_max']<1e-3 and entry['decode_cosine_min']>.99999
                    del ref,native
        with HardDismPrefill() as engine:
            for i,prompt in enumerate(prompts):
                tokens=encode(prompt)
                native=Runner(model,tokens.shape[1]+args.new_tokens+1,dtype=torch.bfloat16,
                              planner='gpu',prefill_engine=engine)
                ref=Runner(model,native.capacity,'reference')
                logits=native(tokens)[:,-1];expected=ref(tokens)[:,-1]
                gen=torch.Generator(device='cuda').manual_seed(20261008+i)
                ref_gen=torch.Generator(device='cuda').manual_seed(20261008+i)
                generated=[];trajectory=[];same_seed=0
                for _ in range(args.new_tokens):
                    assert torch.isfinite(logits).all()
                    trajectory.append(metrics(logits,expected))
                    token=sample(logits,.8,.9,gen)
                    same_seed+=int(sample(expected,.8,.9,ref_gen).item()==token.item())
                    generated.append(token.item())
                    if token.item()==2:break
                    logits=native(token)[:,-1];expected=ref(token)[:,-1]
                entry=dict(prompt=prompt,continuation=tokenizer.decode(generated),tokens=generated,
                           temperature=.8,top_p=.9,seed=20261008+i,
                           same_seed_sampling_agreement=same_seed/len(generated),
                           replay_cosine_min=min(x['cosine'] for x in trajectory),
                           replay_relative_l2_max=max(x['relative_l2'] for x in trajectory))
                report['generations'].append(entry);save()
                print('GENERATION',json.dumps(entry),flush=True)
                del native,ref
    report['status']='completed';save()


if __name__=='__main__':main()
