"""Frozen transformer evaluation adapter, optionally static inference-only YaRN."""
import argparse
import json
from pathlib import Path
import torch
from adapter_bos_base_20261025 import NanochatLM, evaluator, TaskManager
from nanochat.models.fla_transformer.yarn import install_yarn


class TransformerLM(NanochatLM):
    def __init__(self,checkpoint,tokenizer,batch_size=4,context=2048,rope='native'):
        super().__init__(checkpoint,tokenizer,batch_size,context)
        architecture=self.metadata['model_spec']['architecture']
        if architecture=='fla_transformer':
            del self.model.eval_hard
        elif architecture!='dism_gdn_hybrid' or rope!='native':
            raise ValueError('Expected Transformer, or native DISM control')
        self.rope_protocol=dict(type='native',factor=1.,original_context=self.training_context,
                                target_context=context,finetuned=False)
        if rope=='yarn':
            self.rope_protocol=install_yarn(self.model,self.training_context,context)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True);p.add_argument('--tokenizer',required=True)
    p.add_argument('--output',required=True);p.add_argument('--tasks',default='pg19')
    p.add_argument('--batch-size',type=int,default=4);p.add_argument('--context',type=int,default=2048)
    p.add_argument('--rope',choices=['native','yarn'],default='native')
    p.add_argument('--limit',type=int);p.add_argument('--smoke-only',action='store_true')
    p.add_argument('--memory-fraction',type=float,default=.25)
    args=p.parse_args()
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
    lm=TransformerLM(args.checkpoint,args.tokenizer,args.batch_size,args.context,args.rope)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    try:
        lm.smoke()
        if args.smoke_only:
            # Exercise actual largest packed shape, not just the short smoke.
            with torch.inference_mode():
                x,cu=lm._inputs([[lm.prefix_token_id]*args.context])
                hidden=lm._features(x,cu)
                assert torch.isfinite(hidden).all()
                print('FULL_CONTEXT_PASS',args.context,lm.rope_protocol,'peakGB',torch.cuda.max_memory_allocated()/1e9,flush=True)
        else:
            result=evaluator.simple_evaluate(model=lm,tasks=args.tasks.split(','),
                task_manager=TaskManager(include_path=str(Path(__file__).parent/'tasks')),
                num_fewshot=0,batch_size=args.batch_size,limit=args.limit,bootstrap_iters=1000,log_samples=True)
            (out/'results.json').write_text(json.dumps(result,indent=2,default=str))
        protocol=dict(checkpoint=args.checkpoint,model_spec=lm.metadata['model_spec'],context=lm.max_length,
                      training_context=lm.training_context,rope=lm.rope_protocol,physical_batch_size=lm.batch_size,
                      bos=True,num_fewshot=0,truncated_requests=lm.truncated,softcap=lm.model.config.softcap,
                      hard=getattr(lm.model,'eval_hard',None),direction=(True if lm.metadata['model_spec']['architecture']=='dism_gdn_hybrid' else None),
                      bos_policy='replace oldest input/context token with BOS for every independent sequence; retain all target tokens',likelihood='BF16 fused CE, FP32 accumulation; MXFP8 CE disabled for evaluation',
                      generation='greedy; reserve max_gen_toks; recompute bounded prefixes',
                      padding='256-aligned; fixed batch with independent unscored dummy rows',
                      rolling='same harness disjoint target windows; replace oldest context token with BOS; score every target once; no cross-window state',
                      memory_fraction=args.memory_fraction,peak_allocated_bytes=torch.cuda.max_memory_allocated())
        (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    finally:lm.close()


if __name__=='__main__':main()
