"""Offline restartable final-checkpoint Transformer/native-RoPE/YaRN sweep."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
REPO=Path('/home/cicuvc/cs/project/dism-core')
OUT=ROOT/'results/bos_corrected_20261025'
GROUPS={'mc':'arc_challenge,arc_easy,hellaswag,winogrande,piqa','lambada':'lambada_openai',
        'swde':'swde','fda':'fda','pg19':'pg19','govreport':'govreport'}
SCHEDULE=[('transformer',rope,ctx,g) for rope,ctx in [('native',2048),('native',4096),('yarn',4096),('native',8192),('yarn',8192)] for g in ['pg19','govreport']]
SCHEDULE += [('dism_c','native',ctx,g) for ctx in [2048,4096,8192] for g in ['pg19','govreport']]
SCHEDULE += [('transformer',rope,ctx,g) for rope,ctx in [('native',2048),('native',4096),('yarn',4096)] for g in ['swde','fda']]
SCHEDULE += [('dism_c','native',ctx,g) for ctx in [2048,4096] for g in ['swde','fda']]
ENV=dict(os.environ,PYTHONPATH=f'{REPO}/dism_v3/python:{REPO}/nanochat:/home/cicuvc/lm-evaluation-harness',
         HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',HF_HOME=str(ROOT/'hf-cache'),
         FLA_FLASH_KDA='0',TOKENIZERS_PARALLELISM='false',DISM_HARD_PLANNER='0',OMP_NUM_THREADS='1')
STATE=dict(pid=os.getpid(),status='running',started=time.time(),completed=[],schedule=SCHEDULE,
           skipped=['qasper_bool','squad_completion'],batch_size=4)


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save():
    tmp=OUT/'state.tmp';tmp.write_text(json.dumps(STATE,indent=2)+'\n');tmp.replace(OUT/'state.json')


def collect():
    records=[]
    for path in sorted(OUT.glob('*_ctx*/*/results.json')):
        if not (path.parent/'provenance.json').exists():continue
        data=json.loads(path.read_text())
        for task,metrics in data['results'].items():
            records.append(dict(variant=path.parent.parent.name,group=path.parent.name,task=task,metrics=metrics))
    (OUT/'summary.json').write_text(json.dumps(dict(status=STATE['status'],completed=STATE['completed'],records=records),indent=2)+'\n')


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    run=Path((REPO/'dism_v3/build/fla_transformer_run.txt').read_text().strip())
    assert (run/'exit_code.txt').read_text().strip()=='0'
    checkpoints=list((run/'base_checkpoints').glob('*/model_022888.pt'));assert len(checkpoints)==1
    checkpoint=str(checkpoints[0])
    common=dict(checkpoint=checkpoint,checkpoint_sha256=sha(checkpoint),training_context=2048,
        sources={str(p):sha(p) for p in [ROOT/'adapter_bos_yarn_20261025.py',ROOT/'adapter_bos_base_20261025.py',
            REPO/'nanochat/nanochat/models/fla_transformer/model.py',REPO/'nanochat/nanochat/models/fla_transformer/yarn.py']},
        tasks_sha256={str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'tasks').rglob('*')) if p.suffix in ['.py','.yaml']},
        harness_commit=subprocess.check_output(['git','-C',str(ROOT/'harness'),'rev-parse','HEAD'],text=True).strip(),
        batch_size=4,limit=None,num_fewshot=0,skipped=STATE['skipped'],finetuned_for_extension=False)
    (OUT/'checkpoints.json').write_text(json.dumps(dict(transformer=checkpoint),indent=2)+'\n')
    save()
    c_run=Path((REPO/'dism_v3/build/layout_diagnosis/new_training_run.txt').read_text().strip())
    c_checkpoints=list((c_run/'base_checkpoints').glob('*/model_022888.pt'));assert len(c_checkpoints)==1
    checkpoints_by_model={'transformer':checkpoint,'dism_c':str(c_checkpoints[0])}
    (OUT/'checkpoints.json').write_text(json.dumps(checkpoints_by_model,indent=2)+'\n')
    for model,rope,context,group in SCHEDULE:
        checkpoint=checkpoints_by_model[model]
        common.update(checkpoint=checkpoint,checkpoint_sha256=sha(checkpoint),bos_policy='replace oldest context/input token; targets unchanged')
        name=f'{model}_{rope}_ctx{context}';key=f'{name}/{group}'
        out=OUT/name/group;out.mkdir(parents=True,exist_ok=True)
        expected=dict(common,rope=rope,context=context,tasks=GROUPS[group])
        if (out/'results.json').exists() and (out/'provenance.json').exists():
            prior=json.loads((out/'provenance.json').read_text())
            assert all(prior.get(k)==v for k,v in expected.items()),key
            STATE['completed'].append(key);save();continue
        STATE.update(current=key,current_started=time.time());save()
        command=[sys.executable,'-u',str(ROOT/'adapter_bos_yarn_20261025.py'),
            '--checkpoint',checkpoint,'--tokenizer',str(ROOT/'tokenizer'),'--output',str(out),
            '--tasks',GROUPS[group],'--batch-size','4','--context',str(context),'--rope',rope]
        print('START',key,flush=True)
        with (out/'console.log').open('w') as log:
            child=subprocess.Popen(command,cwd=ROOT,env=ENV,stdout=log,stderr=subprocess.STDOUT)
            STATE['evaluation_pid']=child.pid;save();code=child.wait()
        if code or not (out/'results.json').exists():raise RuntimeError(f'{key}: exit={code}; see console.log')
        expected.update(finished=time.time(),elapsed_seconds=time.time()-STATE['current_started'])
        (out/'provenance.json').write_text(json.dumps(expected,indent=2)+'\n')
        STATE['completed'].append(key);save();collect();print('DONE',key,flush=True)
    STATE.update(status='finished',finished=time.time(),current=None,evaluation_pid=None);save();collect()


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        STATE.update(status='failed',error=repr(exc));save();raise
