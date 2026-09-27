"""12-head wide-attention forward/backward and paired initialization checks."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import hashlib
import torch
import torch.nn.functional as F


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--architecture', choices=('swa_only', 'full_attention'), required=True)
    p.add_argument('--port', type=int, required=True)
    p.add_argument('--secret', type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(4)
    from dism_v2.lm_model import LMConfig, DecoderLM, SlidingAttention, parameter_count
    from dism_v2.lm_token_stream import fetch, RemotePackedStream
    c = LMConfig(architecture=a.architecture, heads=12, ffn_hidden=1050,
                 softcap=30., window=-1 if a.architecture == 'full_attention' else 128)
    assert parameter_count(c) == 49_675_276
    torch.manual_seed(777)
    model = DecoderLM(c)
    torch.manual_seed(777)
    other = DecoderLM(replace(c, architecture='full_attention' if c.architecture == 'swa_only' else 'swa_only',
                             window=-1 if c.architecture == 'swa_only' else 128))
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, other.state_dict()[name], rtol=0, atol=0)
    del other
    # Compare module including projections/RoPE against dense FP32 attention,
    # including the window edge and all parameter/input gradients.
    for n in (127, 129, 257):
        module = SlidingAttention(c).cuda()
        reference = SlidingAttention(c).cuda()
        reference.load_state_dict(module.state_dict())
        x = torch.randn(1, n, 256, device='cuda', requires_grad=True)
        rx = x.detach().clone().requires_grad_()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            actual = module(x)
            q, k, v = reference.qkv(rx).reshape(1, n, 3, 12, 64).unbind(2)
            q, k = reference.rope(q).transpose(1, 2).float(), reference.rope(k).transpose(1, 2).float()
        delta = torch.arange(n, device='cuda')[:, None]-torch.arange(n, device='cuda')[None, :]
        mask = (delta >= 0) & ((delta < 128) if c.window == 128 else True)
        prob = ((q @ k.transpose(-1, -2)/8).masked_fill(~mask, -torch.inf)).softmax(-1)
        out = (prob @ v.transpose(1, 2).float()).transpose(1, 2).reshape(1, n, 768)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            expected = reference.out(out.to(torch.bfloat16))
        weight = torch.randn_like(actual)
        (actual*weight).sum().backward(); (expected*weight).sum().backward()
        torch.testing.assert_close(actual, expected, atol=.004, rtol=.03)
        torch.testing.assert_close(x.grad, rx.grad, atol=.008, rtol=.06)
        for p0, p1 in zip(module.parameters(), reference.parameters()):
            torch.testing.assert_close(p0.grad, p1.grad, atol=.08, rtol=.06)
    url = f'http://127.0.0.1:{a.port}'
    secret = a.secret.read_text().strip()
    meta = json.loads(fetch(url, secret, '/meta')[0])
    assert meta['steps'] == 3000
    first = fetch(url, secret, '/batch/train/0')[0]
    assert hashlib.sha256(first).hexdigest() == 'af5119c4ca9491b2b947eaaae517783b8431627ba0e6342b23d20a5be98fb94e'
    stream = RemotePackedStream(url, secret, 'train', meta, batch_size=8)
    x, y = stream.next_batch(); stream.close()
    model.cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-6, fused=True)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        loss = model(x.cuda(), y.cuda(), 0., None)
    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None and torch.isfinite(param.grad).all(), name
    opt.step()
    print(json.dumps(dict(passed=True, architecture=a.architecture, parameters=parameter_count(c),
                         loss=loss.item(), first_batch_hash_matches=True, paired_initial_weights_equal=True)), flush=True)


if __name__ == '__main__':
    main()
