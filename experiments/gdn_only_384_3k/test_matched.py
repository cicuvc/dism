import torch
from dism_v2.lm_model import LMConfig,DecoderLM,parameter_count,parameter_groups


def test_matched_full_microbatch():
    torch.manual_seed(777)
    c=LMConfig(width=384,heads=6,layers=12,ffn_hidden=1392,softcap=30.,gdn_only=True)
    assert parameter_count(c)==72_184_080
    model=DecoderLM(c).cuda()
    assert all(b.swa is None for b in model.blocks)
    assert all(b.dism is None and b.gdn is not None for b in model.blocks)
    assert model.embedding.weight is not model.lm_head.weight
    optimizer=torch.optim.AdamW(parameter_groups(model,.01),lr=.001,fused=True)
    x=torch.randint(50257,(8,2048),device='cuda')
    with torch.autocast('cuda',dtype=torch.bfloat16):
        loss=model(x,x.roll(-1,1),.5,torch.Generator(device='cuda').manual_seed(778))
    assert torch.isfinite(loss)
    loss.backward()
    for name,p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(),name
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
    optimizer.step()
    model.eval()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        assert torch.isfinite(model.forward_features(x[:1,:65],1.)).all()
    print(dict(loss=loss.item(),grad_norm=norm.item(),peak_gib=torch.cuda.max_memory_allocated()/2**30),flush=True)
