"""CPU-only exact input bundle matching previously saved hybrid evaluations."""
import argparse
import hashlib
import json
from pathlib import Path
import torch


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('New output path required')
    from .train_lm import load_tokenizer
    from .lm_data import PackedStream,split_files
    from .eval_lm_niah import case_plan,build_prompt
    config=json.loads((a.run/'config.json').read_text())
    tokenizer=load_tokenizer(config['tokenizer'])
    enc=lambda s:tokenizer.encode(s,add_special_tokens=False)
    _,files=split_files(config['data'])
    old=json.loads((a.run/'length-prefill-8192/report.json').read_text())
    stream=PackedStream(files,tokenizer,context=8192,batch_size=4,repeat=False)
    xs,ys=[],[];digest=hashlib.sha256()
    for _ in range(old['sequences']//4):
        x,y=stream.next_batch();xs.append(x);ys.append(y)
        digest.update(x.numpy().tobytes());digest.update(y.numpy().tobytes())
    assert digest.hexdigest()==old['packed_xy_sha256']
    pilot=json.loads((a.run/'niah-256.json').read_text())
    main_rows={(r['trial'],r['length'],r['depth']):r for r in pilot['rows'] if r['condition']=='needle'}
    colors=pilot['candidate_colors'];candidate_ids=[enc(c)[0] for c in colors]
    assert all(len(enc(c))==1 for c in colors)
    stream=PackedStream(files,tokenizer,context=8192,batch_size=1,repeat=False)
    cases=[]
    for trial,(key,target,alternative,depths) in enumerate(case_plan(256,pilot['seed'])):
        background=stream.next_batch()[0][0].tolist()
        prefix=f'The secret color of the {key} is'
        needle=enc('\n\n'+prefix+colors[target]+'.\n\n')
        changed=enc('\n\n'+prefix+colors[alternative]+'.\n\n')
        suffix=enc('\n\n'+prefix)
        for condition,ids in [('near',needle+suffix),('no_information',suffix)]:
            cases.append(dict(trial=trial,condition=condition,length=len(ids),depth=None,
                              target=target,ids=torch.tensor(ids,dtype=torch.int32)))
        for length in (2048,8192):
            for depth in depths:
                ids,at=build_prompt(background,length,needle,suffix,depth)
                sha=hashlib.sha256(torch.tensor(ids).numpy().tobytes()).hexdigest()
                assert sha==main_rows[trial,length,depth]['prompt_sha256']
                counter=ids.copy();counter[at:at+len(needle)]=changed
                absent=ids.copy();absent[at:at+len(needle)]=[enc('\n')[0]]*len(needle)
                cases.append(dict(trial=trial,condition='needle',length=length,depth=depth,
                    target=target,alternative=alternative,prompt_sha256=sha,
                    tokens_after_needle=length-at-len(needle),
                    ids=torch.tensor(ids,dtype=torch.int32),
                    absent=torch.tensor(absent,dtype=torch.int32),counterfactual=torch.tensor(counter,dtype=torch.int32)))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    torch.save(dict(x=torch.cat(xs).int(),y=torch.cat(ys).int(),cases=cases,
                    candidate_ids=candidate_ids,colors=colors,paired_xy_sha256=digest.hexdigest(),
                    training_config={k:config[k] for k in ('model','steps','batch','micro_batch','seed','lr','weight_decay','warmup','data','tokenizer')},
                    lengths=old['lengths']),a.output)
    print('Verified exact hybrid position inputs and all256 NIAH prompt hashes',flush=True)


if __name__=='__main__':main()
