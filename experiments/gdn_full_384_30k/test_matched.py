import torch
from torch.nn import functional as F
from dism_v2.lm_model import LMConfig, DecoderLM, parameter_count, parameter_groups


def test_full_attention_matches_dense_causal():
    from dism_v2.lm_model import SlidingAttention
    torch.manual_seed(777)
    c=LMConfig(width=384,heads=6,layers=12,architecture='full_attention',window=-1)
    layer=SlidingAttention(c).cuda().eval()
    x=torch.randn(1,257,384,device='cuda')
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        actual=layer(x)
        q,k,v=layer.qkv(x).reshape(1,257,3,6,64).unbind(2)
        q,k=layer.rope(q).transpose(1,2).float(),layer.rope(k).transpose(1,2).float()
        score=q@k.transpose(-1,-2)/8
        score.masked_fill_(~torch.ones(257,257,device='cuda',dtype=torch.bool).tril(),float('-inf'))
        out=score.softmax(-1)@v.transpose(1,2).float()
        expected=layer.out(out.transpose(1,2).reshape(1,257,384).bfloat16())
        torch.testing.assert_close(actual,expected,atol=.006,rtol=.03)
        # Future tokens cannot change earlier outputs (crosses old128 window).
        short=layer(x[:,:193])
        torch.testing.assert_close(actual[:,:193],short,atol=.006,rtol=.03)


def test_matched_full_microbatch():
    torch.manual_seed(777)
    c=LMConfig(width=384,heads=6,layers=12,ffn_hidden=1522,softcap=30.,gdn_full=True)
    assert parameter_count(c)==72_182_172
    model=DecoderLM(c).cuda()
    assert all(b.dism is None for b in model.blocks)
    assert [i+1 for i,b in enumerate(model.blocks) if b.swa is not None]==[4,8,12]
    assert sum(b.gdn is not None for b in model.blocks)==9
    assert all(b.swa.c.architecture=='full_attention' and b.swa.c.window==-1
               for b in model.blocks if b.swa is not None)
    assert model.embedding.weight is not model.lm_head.weight
    optimizer=torch.optim.AdamW(parameter_groups(model,.01),lr=.001,fused=True)
    x=torch.randint(50257,(8,2048),device='cuda')
    with torch.autocast('cuda',dtype=torch.bfloat16):
        loss=model(x,x.roll(-1,1),0.,torch.Generator(device='cuda').manual_seed(778))
    assert torch.isfinite(loss)
    loss.backward()
    for name,p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(),name
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
    optimizer.step()
    model.eval()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        assert torch.isfinite(model.forward_features(x[:1,:65],0.)).all()
    print(dict(loss=loss.item(),grad_norm=norm.item(),peak_gib=torch.cuda.max_memory_allocated()/2**30),flush=True)
