"""Bounded-context nanochat adapter; no changes to frozen training workspaces."""
import argparse, json, math, os
from pathlib import Path
os.environ.setdefault('FLA_FLASH_KDA', '0')
import torch
from transformers import AutoTokenizer
from lm_eval.api.model import TemplateLM
from lm_eval import evaluator, utils
from lm_eval.tasks import TaskManager
from nanochat.models import ModelSpec, build_model
from flash_dism.kernels.fused_cross_entropy import fused_cross_entropy_unreduced

class FeaturesReady(Exception):
    pass

class NanochatLM(TemplateLM):
    def __init__(self, checkpoint, tokenizer, batch_size=8, context=None, *, hard_backend="quadratic", prefill_precision="tf32x3", prefill_workers=16, decode_backend="recompute", decode_rebuild_interval=512, decode_cuda_graph=False, decode_summary_backend="native"):
        super().__init__()
        checkpoint = Path(checkpoint)
        self.metadata = json.loads(checkpoint.with_name(checkpoint.name.replace('model_', 'meta_')).with_suffix('.json').read_text())
        with torch.device('meta'):
            self.model = build_model(ModelSpec.from_dict(self.metadata['model_spec']))
        self.model.to_empty(device='cuda')
        self.model.init_weights()
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        self.model.load_state_dict({k.removeprefix('_orig_mod.'):v for k,v in state.items()}, strict=True, assign=True)
        self.model.cuda().eval()
        if self.model.config.mxfp8:
            from flash_mxfp8 import refresh_fp8_modules
            refresh_fp8_modules(self.model)
        self.model.eval_hard = True
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, local_files_only=True)
        from nanochat.tokenizer import HuggingFaceTokenizer
        expected = self.metadata.get('tokenizer_spec', {}).get('fingerprint')
        actual = HuggingFaceTokenizer(self.tokenizer, str(tokenizer)).get_fingerprint()
        if expected is not None and actual != expected:
            raise ValueError('Evaluation tokenizer does not match checkpoint fingerprint')
        self.batch_size = batch_size
        self.training_context = self.model.config.sequence_len
        if context is not None:
            if context % 256: raise ValueError("Context must be a multiple of 256")
            self.model.config.sequence_len = context
        self.max_length = self.model.config.sequence_len
        self._device = torch.device('cuda')
        self.truncated = 0
        self.weight = self.model.lm_head.weight.to(torch.bfloat16)
        if hard_backend not in ('sam', 'quadratic'):
            raise ValueError('hard_backend must be sam or quadratic')
        if decode_backend not in ('sam', 'recompute'):
            raise ValueError('decode_backend must be sam or recompute')
        if decode_rebuild_interval <= 0: raise ValueError('positive rebuild interval required')
        if decode_cuda_graph and decode_backend != 'sam': raise ValueError('decode CUDA Graph requires SAM decode')
        if decode_summary_backend not in ('native','persistent'): raise ValueError('invalid summary backend')
        if decode_summary_backend!='native' and decode_backend!='sam': raise ValueError('persistent summary requires SAM decode')
        self.decode_summary_backend = decode_summary_backend
        self.decode_cuda_graph = decode_cuda_graph
        self.decode_rebuild_interval = decode_rebuild_interval
        self.decode_backend = decode_backend
        self.hard_backend = hard_backend
        self.prefill = None
        if hard_backend == 'sam' or decode_backend == 'sam':
            from flash_dism.inference import HardDismPrefill
            self.prefill = HardDismPrefill(workers=prefill_workers, mma_precision=prefill_precision)
        self.prefill_precision, self.prefill_workers = prefill_precision, prefill_workers

    def close(self):
        if self.prefill is not None:
            self.prefill.close()

    def _forward_model(self, *args, **kwargs):
        if self.hard_backend == 'quadratic':
            return self.model(*args, **kwargs)
        from flash_dism.inference.model_prefill import use_sam_prefill
        with use_sam_prefill(self.model, self.prefill):
            return self.model(*args, **kwargs)

    @property
    def eot_token_id(self): return self.tokenizer.eos_token_id
    @property
    def prefix_token_id(self):
        return self.tokenizer.bos_token_id if self.tokenizer.bos_token_id is not None else self.eot_token_id
    def tok_encode(self, string, add_special_tokens=None, **kwargs):
        ids = self.tokenizer.encode(string, add_special_tokens=False)
        return [self.prefix_token_id] + ids if add_special_tokens is not False else ids

    def _inputs(self, sequences):
        length = max(256, math.ceil(max(map(len,sequences))/256)*256)
        assert length <= self.max_length
        # Keep the physical batch size fixed, including the final microbatch.
        # MXFP8 + discrete routing can change with GEMM shape; dummy rows are
        # separate packed sequences and never contribute to task scores.
        assert len(sequences) <= self.batch_size
        x = torch.full((self.batch_size,length),self.prefix_token_id,device='cuda',dtype=torch.long)
        for i,s in enumerate(sequences): x[i,:len(s)] = torch.tensor(s,device='cuda')
        cu = torch.arange(self.batch_size+1,device='cuda',dtype=torch.int32)*length
        return x,cu

    def _features(self,x,cu):
        # Eager inference only: bound per-shape FFN workspaces. Long rolling
        # documents have many tail lengths; retaining every shape can OOM.
        shape = (x.device.index, tuple(x.shape))
        if getattr(self, '_workspace_shape', None) != shape:
            assert not self.model.training and not torch.is_grad_enabled()
            assert not torch.cuda.is_current_stream_capturing()
            torch.cuda.synchronize(x.device)
            for module in self.model.modules():
                if type(module).__module__.startswith('flash_mxfp8.'):
                    for name in ('_forward_states', '_forward_plans'):
                        states = getattr(module, name, None)
                        if isinstance(states, dict):
                            states.clear()
            self._workspace_shape = shape
        # Exit after final normalization, avoiding [batch,sequence,vocabulary] logits.
        captured=[]
        def stop(module,args,output):
            captured.append(output.to(torch.bfloat16)); raise FeaturesReady()
        handle=self.model.norm.register_forward_hook(stop)
        try:
            try: self._forward_model(x,cu_seqlens=cu)
            except FeaturesReady: pass
        finally: handle.remove()
        if len(captured)!=1: raise RuntimeError('Final norm was not reached exactly once')
        return captured[0].reshape(*x.shape,-1)

    def _greedy(self,hidden):
        # Match fused CE's fp32 accumulator using bf16-rounded inputs and weights.
        answers=[]
        for chunk in hidden.split(128):
            logits=torch.nn.functional.linear(chunk.float(),self.weight.float())
            cap=self.model.config.softcap
            logits=cap*torch.tanh(logits/cap)
            answers.append(logits.argmax(-1))
        return torch.cat(answers) if answers else torch.empty(0,device='cuda',dtype=torch.long)

    @torch.inference_mode()
    def _loglikelihood_tokens(self,requests,compute_greedy=True,**kwargs):
        result=[None]*len(requests)
        order=sorted(range(len(requests)),key=lambda i:len(requests[i][1])+len(requests[i][2]),reverse=True)
        for offset in range(0,len(order),self.batch_size):
            indices=order[offset:offset+self.batch_size]; examples=[]
            for i in indices:
                _,context,target=requests[i]
                if len(target)>self.max_length: raise ValueError('Continuation exceeds context')
                full=(context or [self.prefix_token_id])+target
                self.truncated+=int(len(full)>self.max_length+1)
                full=full[-self.max_length-1:]
                # Every independent attention sequence was BOS-prefixed in training.
                # Replace only the oldest context token; targets stay unchanged.
                full=[self.prefix_token_id]+full[1:]
                inp=full[:-1] if target else (context or [self.prefix_token_id])[-self.max_length:]
                examples.append((inp,target,len(inp)-len(target)))
            x,cu=self._inputs([e[0] for e in examples]); y=torch.full_like(x,-1)
            for row,(inp,target,start) in enumerate(examples):
                if target: y[row,start:len(inp)]=torch.tensor(target,device='cuda')
            hidden=self._features(x,cu)
            losses=fused_cross_entropy_unreduced(hidden.reshape(1,-1,hidden.shape[-1]),self.weight,y.reshape(1,-1),ignore_index=-1,softcap=self.model.config.softcap).reshape_as(y)
            for row,i in enumerate(indices):
                inp,target,start=examples[row]
                greedy=True
                if target and compute_greedy:
                    greedy=bool((self._greedy(hidden[row,start:len(inp)])==y[row,start:len(inp)]).all())
                result[i]=(-losses[row].double().sum().item(),greedy)
                if requests[i][0] is not None: self.cache_hook.add_partial('loglikelihood',requests[i][0],result[i])
        return result

    def loglikelihood_rolling(self,requests,**kwargs):
        values=[]
        for request in requests:
            tokens=self.tok_encode(request.args[0],add_special_tokens=False)
            windows=[utils.make_disjoint_window(w) for w in utils.get_rolling_token_windows(token_list=tokens,prefix_token=self.prefix_token_id,max_seq_len=self.max_length,context_len=1)]
            assert sum(len(w[1]) for w in windows)==len(tokens)
            values.append(sum(r[0] for r in self._loglikelihood_tokens([(None,a,b) for a,b in windows],compute_greedy=False)))
            if len(values)==1 or len(values)%10==0:
                print(f'ROLLING {len(values)}/{len(requests)} allocated_GB={torch.cuda.memory_allocated()/1e9:.3f} peak_GB={torch.cuda.max_memory_allocated()/1e9:.3f}',flush=True)
        return values

    @torch.inference_mode()
    def generate_until(self,requests,**kwargs):
        if self.decode_backend == "sam":
            from nanochat.models.dism.inference import generate_until_sam
            return generate_until_sam(self, requests)
        results=[]
        # Recompute bounded prefixes; works for every mixer, including independent conv.
        for offset in range(0,len(requests),self.batch_size):
            batch=requests[offset:offset+self.batch_size]; contexts=[]; outputs=[[] for _ in batch]; done=[False]*len(batch); stops=[]; limits=[]
            for r in batch:
                prompt,kw=r.args
                if kw.get('do_sample',False) or kw.get('temperature',0) not in (0,None): raise ValueError('Greedy evaluation only')
                limit=int(kw.get('max_gen_toks',48))
                if not 0<limit<self.max_length: raise ValueError(limit)
                ids=self.tok_encode(prompt); budget=self.max_length-limit
                self.truncated+=int(len(ids)>budget)
                retained=ids[-budget:]
                contexts.append([self.prefix_token_id]+retained[1:])
                until=kw.get('until',[]); stops.append([until] if isinstance(until,str) else until); limits.append(limit)
            for step in range(max(limits)):
                active=[i for i in range(len(batch)) if not done[i]]
                if not active: break
                seqs=[contexts[i]+outputs[i] for i in active]
                x,cu=self._inputs(seqs); hidden=self._features(x,cu)
                next_ids=self._greedy(torch.stack([hidden[row,len(seq)-1] for row,seq in enumerate(seqs)])).tolist()
                for i,token in zip(active,next_ids):
                    if token==self.eot_token_id: done[i]=True; continue
                    outputs[i].append(token)
                    text=self.tokenizer.decode(outputs[i],skip_special_tokens=True)
                    done[i]=len(outputs[i])>=limits[i] or any(s and s in text for s in stops[i])
            for i in range(len(batch)):
                text=self.tokenizer.decode(outputs[i],skip_special_tokens=True)
                for stop in stops[i]:
                    if stop: text=text.split(stop)[0]
                results.append(text)
        return results

    @torch.inference_mode()
    def smoke(self):
        t=self.tok_encode('The capital of France is Paris.')
        req=(None,t[:4],t[4:]); a=self._loglikelihood_tokens([req])[0][0]
        b=self._loglikelihood_tokens([req,(None,t*30,t[-2:])])[0][0]
        x,cu=self._inputs([t[:-1]]); y=torch.full_like(x,-1); y[0,3:len(t)-1]=torch.tensor(t[4:],device='cuda')
        direct=-self._forward_model(x,y,cu_seqlens=cu,loss_reduction='none').double().sum().item()
        assert abs(a-direct)<1e-5,(a,direct)
        # With fixed physical batch shape, adding an independent example must
        # preserve this short request (both examples fit the same 256 bucket).
        assert abs(a-b)<.08,(a,b)
        xb,cub=self._inputs([t[:-1],t*30]);yb=torch.full_like(xb,-1)
        yb[0,3:len(t)-1]=torch.tensor(t[4:],device='cuda')
        direct_b=-self._forward_model(xb,yb,cu_seqlens=cub,loss_reduction='none').double().sum().item()
        assert abs(b-direct_b)<1e-5,(b,direct_b)
        from types import SimpleNamespace
        generated=self.generate_until([SimpleNamespace(args=('The capital of France is',{'max_gen_toks':2,'until':['\n']}))])
        assert len(generated)==1
        print('SMOKE PASS',dict(ll=a,direct=direct,padded=b,generated=generated),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--tokenizer',required=True);p.add_argument('--output',required=True);p.add_argument('--tasks',default='arc_challenge,arc_easy,hellaswag,winogrande,piqa,swde,pg19');p.add_argument('--batch-size',type=int,default=8);p.add_argument('--limit',type=int);p.add_argument('--context',type=int);p.add_argument('--smoke-only',action='store_true');p.add_argument('--hard-backend',choices=('sam','quadratic'),default='quadratic');p.add_argument('--prefill-precision',choices=('tf32x3','bf16'),default='tf32x3');p.add_argument('--prefill-workers',type=int,default=16);p.add_argument('--decode-backend',choices=('recompute','sam'),default='recompute');p.add_argument('--decode-rebuild-interval',type=int,default=512);p.add_argument('--decode-cuda-graph',action='store_true');p.add_argument('--decode-summary-backend',choices=('native','persistent'),default='native');args=p.parse_args()
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    lm=NanochatLM(args.checkpoint,args.tokenizer,args.batch_size,args.context,hard_backend=args.hard_backend,prefill_precision=args.prefill_precision,prefill_workers=args.prefill_workers,decode_backend=args.decode_backend,decode_rebuild_interval=args.decode_rebuild_interval,decode_cuda_graph=args.decode_cuda_graph,decode_summary_backend=args.decode_summary_backend)
    try:
        lm.smoke()
        out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
        if not args.smoke_only:
            result=evaluator.simple_evaluate(model=lm,tasks=args.tasks.split(','),task_manager=TaskManager(include_path=str(Path(__file__).parent/'tasks')),num_fewshot=0,batch_size=args.batch_size,limit=args.limit,bootstrap_iters=1000,log_samples=True)
            (out/'results.json').write_text(json.dumps(result,indent=2,default=str))
        (out/'protocol.json').write_text(json.dumps(dict(checkpoint=args.checkpoint,model_spec=lm.metadata['model_spec'],context=lm.max_length,hard_backend=lm.hard_backend,prefill_precision=lm.prefill_precision if lm.prefill else None,prefill_workers=lm.prefill_workers if lm.prefill else None,training_context=lm.training_context,bos=True,hard=True,direction=True,FLA_FLASH_KDA=os.environ['FLA_FLASH_KDA'],num_fewshot=0,truncated_requests=lm.truncated,softcap=lm.model.config.softcap,generation=('greedy; reserve max_gen_toks by left truncation; SAM cached GPU decode, serial unpadded requests' if lm.decode_backend=='sam' else 'greedy; reserve max_gen_toks by left truncation; recompute prefixes'),decode_backend=lm.decode_backend,decode_rebuild_interval=lm.decode_rebuild_interval,decode_cuda_graph=lm.decode_cuda_graph,decode_summary_backend=lm.decode_summary_backend,likelihood='fused CE, bf16 operands/fp32 accumulation',physical_batch_size=lm.batch_size,padding='256-aligned; fixed physical batch padded by independent unscored rows',workspace_cache='MXFP8 forward workspaces cleared when input shape changes; eager inference only',rolling='harness disjoint windows, context_len=1, score each token once; no cross-window recurrent state'),indent=2))
    finally:
        lm.close()
