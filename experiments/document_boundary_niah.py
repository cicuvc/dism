"""Small paired synthetic NIAH boundary intervention, no retraining."""
from experiments.document_boundary_probe import intervention,pointwise_partition
import json
from pathlib import Path
from dataclasses import replace
import torch


@torch.inference_mode()
def main():
    from dism_v2.eval_lm_positions import load_checkpoint
    from dism_v2.train_lm import load_tokenizer
    from dism_v2.lm_data import PackedStream,split_files
    from dism_v2.eval_lm_niah import case_plan,build_prompt
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    root=Path('/media/cicuvc/c63abdf1-0e56-4153-9228-95df5a2f239b/cicuvc/dism-lm-runs/gdn-dism-vocab-silu-384-72m-30k-20260911')
    out=root/'document-boundary-32-canonical/niah.json'
    assert not out.exists()
    model,meta=load_checkpoint(root/'latest.pt');model.config=replace(model.config,context=4096)
    tok=load_tokenizer(meta['config']['tokenizer']);_,files=split_files(meta['config']['data'])
    stream=PackedStream(files,tok,context=8192,batch_size=1,repeat=False)
    encode=lambda s:tok.encode(s,add_special_tokens=False)
    colors=[' red',' blue',' green',' yellow',' white',' black',' orange',' purple']
    color_ids=[encode(c)[0] for c in colors]
    rows=[]
    for trial,(key,target,_,depths) in enumerate(case_plan(64,9321)):
        bg=stream.next_batch()[0][0].tolist()
        prefix=stream.next_batch()[0][0,:1535].tolist()+[tok.eos_token_id]
        suffix='The secret color of the '+key+' is'
        needle=encode('\n\n'+suffix+colors[target]+'.\n\n')
        b,at=build_prompt(bg,2048,needle,encode('\n\n'+suffix),depths[0])
        row=dict(trial=trial,depth=depths[0],key=key,target=colors[target],needle_start=at,arms={})
        ref=None
        for mode in ('alone','intact','dism_core','dism','gdn','both'):
            tokens=b if mode=='alone' else prefix+b;boundary=0 if mode=='alone' else len(prefix)
            x=torch.tensor([tokens],device='cuda')
            with intervention(model,boundary,mode),pointwise_partition(model,boundary,len(tokens)),torch.autocast('cuda',dtype=torch.bfloat16):
                h=model.forward_features(x,1.,torch.Generator(device='cuda').manual_seed(779))
                logits=model.lm_head(h[:,-1]).float()[0]
            logits=30*torch.tanh(logits/30)
            assert torch.isfinite(logits).all()
            if mode=='alone':ref=logits.clone()
            if mode=='both':torch.testing.assert_close(logits,ref,atol=0,rtol=0)
            scores=logits[color_ids]
            row['arms'][mode]=dict(exact=int(logits.argmax())==color_ids[target],
                candidate=int(scores.argmax())==target,nll=float(-logits.log_softmax(-1)[color_ids[target]]))
        rows.append(row)
        if (trial+1)%8==0:print('NIAH pairs',trial+1,flush=True)
    out.write_text(json.dumps(dict(rows=rows,context_b=2048,prefix=1536,
        note='32 paired cases; prefix is unrelated held-out packed text, not necessarily a single document. All pointwise ops canonically partitioned; both-reset logits bitwise equal B alone.'),indent=2))
    for mode in rows[0]['arms']:
        print(mode,{k:sum(r['arms'][mode][k] for r in rows)/len(rows) for k in ('exact','candidate','nll')},flush=True)


if __name__=='__main__':main()
