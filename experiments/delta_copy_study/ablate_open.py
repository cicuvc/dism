"""Check whether learned delta actually helps final ABBA hard inference."""
import json
import torch
from torch import nn
from torch.nn import functional as F
from experiments.delta_copy_study.run import ROOT,Gate,legacy,make_tokens


@torch.no_grad()
def main():
    torch.set_num_threads(4);report=[]
    for seed in (0,1):
        for method in ('temperature','random','shared'):
            path=ROOT/'results-abba-bias0'/f'{method}-seed{seed}'/'final.pt'
            saved=torch.load(path,map_location='cpu',weights_only=False)
            model=nn.Sequential(nn.Embedding(128,128),legacy.DismTransformerBlock(128,mix='random'),nn.RMSNorm(128),nn.Linear(128,128)).cuda()
            gate=Gate(model[1].attn,method);gate.step=999;gate.mode='hard'
            model.load_state_dict(saved['model']);model.eval()
            legacy._dism_from_logm=gate.recurrence
            legacy._row_hard_observer=lambda mask:setattr(gate,'row_hard',mask)
            original=gate.delta
            for mode in ('learned','always_open'):
                gate.delta=original if mode=='learned' else lambda:torch.zeros_like(gate.b)
                gen=torch.Generator(device='cuda').manual_seed(424242)
                losses=[];correct=[]
                for _ in range(10):
                    tokens=make_tokens(torch.randint(0,127,(64,64),device='cuda',generator=gen),'abba')
                    with torch.autocast('cuda',dtype=torch.bfloat16):pred=model(tokens)[:,63:-1].float()
                    target=tokens[:,64:]
                    losses.append(F.cross_entropy(pred.reshape(-1,128),target.flatten(),reduction='none').reshape(64,64).cpu())
                    correct.append((pred.argmax(-1)==target).cpu())
                loss=torch.cat(losses);acc=torch.cat(correct).float()
                report.append(dict(method=method,seed=seed,mode=mode,loss=float(loss.mean()),accuracy=float(acc.mean()),
                                   position_loss=loss.mean(0).tolist(),position_accuracy=acc.mean(0).tolist()))
            print(method,seed,[(r['mode'],r['loss'],r['accuracy']) for r in report[-2:]],flush=True)
    (ROOT/'abba-open-ablation.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
